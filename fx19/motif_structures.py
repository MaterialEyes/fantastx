from dataclasses import dataclass
from typing import List, Optional, Dict, Union
import numpy as np
import os, random, copy

from ase import Atoms
from ase.io import read, write

from pymatgen.io.ase import AseAtomsAdaptor


from sklearn.cluster import DBSCAN
 
from scipy.spatial import Voronoi
from scipy.spatial.transform import Rotation as R

from fx19 import structure_record


@dataclass
class ClusterMotif:
    """
    Data container for identified structural motifs.
     Attributes:
        motif_type: A string identifier for the motif type (e.g., "B12").
        size: Number of atoms in the motif.
        center: 3D coordinates of the motif center in Cartesian space.
        atom_indices: List of atom indices in the original structure that belong to this motif.
    """
    motif_type: str
    size: int
    center: np.ndarray
    atom_indices: List[int]
    
    def __repr__(self):
        return f"<ClusterMotif {self.motif_type} (n={self.size}) at {self.center.round(2)}>"
    

class ClusterAnalyzer:
    def __init__(self, structure: Atoms):
        self.structure = structure
        
    def _unwrap_cluster(self, indices: List[int]) -> Atoms:
        """
        Helper to extract atoms and unwrap them so they are contiguous 
        in space (removing PBC splits).
        Args:
            indices: List of atom indices belonging to the cluster.
        Returns:
            An Atoms object containing the unwrapped cluster.
        """
        if not indices:
            return Atoms()
            
        # 1. Extract the raw atoms (they might be wrapped apart)
        cluster_atoms = self.structure[indices]
        
        # 2. Use the first atom as a reference anchor
        ref_pos = self.structure.positions[indices[0]]
        
        # 3. Calculate vectors from reference to all other atoms respecting MIC
        # get_distances(vector=True) returns vectors from i to j
        diff_vectors = self.structure.get_distances(
            indices[0], 
            indices, 
            mic=True, 
            vector=True
        )
        
        # 4. Reconstruct positions relative to the reference
        # new_pos = ref_pos + vector_from_ref
        unwrapped_positions = ref_pos + diff_vectors
        
        # 5. Update positions of the extracted cluster
        cluster_atoms.positions = unwrapped_positions
        return cluster_atoms

    def find_by_dbscan(self,
                       species_to_cluster: str = None,
                       eps: float = 2.5, 
                       min_samples: int = 8, 
                       allowed_sizes: Optional[List[int]] = None
                    ) -> List[ClusterMotif]:
        """
        Identify clusters of a specific species using DBSCAN, robust to PBC wraparounds.
        Args:
            species_to_cluster: The atomic symbol of the species to cluster (e.g., 'B').
            eps: The maximum distance between two species to be considered
                 as in the same cluster.
            min_samples: The minimum number of species required to form a cluster.
            allowed_sizes: List of allowed cluster sizes to consider as motifs. If None, defaults to [10, 12].
        Returns:
            A list of ClusterMotif objects representing the identified motifs.
        """
        structure = self.structure
        if allowed_sizes is None:
            allowed_sizes = [10, 12]

        # 1. Filter Species
        if species_to_cluster is None:
            raise ValueError("Species to cluster must be specified (e.g., 'B').")
            
        species_indices = [atom.index for atom in structure if atom.symbol == species_to_cluster]
        if not species_indices:
            return []
            
        # 2. Distance Matrix
        full_dist_matrix = structure.get_all_distances(mic=True)
        species_dist_matrix = full_dist_matrix[np.ix_(species_indices, species_indices)]

        # 3. Clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = clustering.fit_predict(species_dist_matrix)
        motifs = []
        unique_labels = set(labels) - {-1} 

        for label in unique_labels:
            local_cluster_mask = (labels == label)
            abs_indices = np.array(species_indices)[local_cluster_mask]
            
            cluster_size = len(abs_indices)
            if cluster_size not in allowed_sizes:
                continue

            # --- PBC-Aware Center Calculation ---
            ref_idx = abs_indices[0]
            ref_pos = structure.positions[ref_idx]
            
            diff_vectors = structure.get_distances(
                ref_idx, 
                abs_indices, 
                mic=True, 
                vector=True
            )
            
            center_unwrapped = np.mean(ref_pos + diff_vectors, axis=0)
            
            center_wrapped = structure.cell.scaled_positions_to_cartesian(
                structure.cell.cartesian_to_scaled_positions(center_unwrapped) % 1.0
            )

            motifs.append(ClusterMotif(
                motif_type=f"{species_to_cluster}{cluster_size}",
                size=cluster_size,
                center=center_wrapped,
                atom_indices=abs_indices.tolist()
            ))

        return motifs
        
    def find_by_seeds(self, 
                      centers: np.ndarray, 
                      diameter: float,
                      constraints: Dict[str, Union[int, str]] = None,
                ) -> List[ClusterMotif]:
        """
        Create motifs by selecting atoms within a radius (diameter/2) of known centers.
        Args:
            centers: An array of shape (N, 3) containing the Cartesian coordinates of seed centers.
            diameter: The diameter of the motif to extract around each center.
            constraints: A dictionary of constraints to filter motifs. Possible keys:
                - 'motif_n_atoms': int, exact number of atoms required in the motif.
                - 'motif_composition': str or list, required species in the motif.
        Returns:
            A list of ClusterMotif objects representing the identified motifs.
        """
        structure = self.structure
        radius = diameter / 2.0
        motifs = []
        
        # Ensure constraints dict exists
        if constraints is None:
            constraints = {}

        # Loop through each provided center
        for i, center in enumerate(centers):
            # --- Logic to simulate Pymatgen's get_atoms_in_sphere using ASE ---
            # We use a dummy atom method to utilize ASE's robust MIC distance calc
            
            # 1. Create a dummy atom at the center point
            dummy = structure.copy()
            dummy.append('X') # Dummy species
            dummy.positions[-1] = center
            dummy_idx = len(dummy) - 1
            
            # 2. Calculate distances from this dummy to all real atoms
            # distances is an array of length N_atoms
            distances = dummy.get_distances(dummy_idx, range(len(structure)), mic=True)
            
            # 3. Select indices within radius
            matched_indices = np.where(distances <= radius)[0]
            
            if len(matched_indices) == 0:
                continue

            # --- Check Constraints ---
            
            # Constraint 1: Number of atoms
            if 'motif_n_atoms' in constraints:
                if len(matched_indices) != constraints['motif_n_atoms']:
                    continue
            
            # Constraint 2: Composition (e.g. ensure only 'B' atoms are selected)
            # You can expand this logic for complex compositions if needed
            selected_symbols = [structure[idx].symbol for idx in matched_indices]
            if 'motif_composition' in constraints:
                # Assumes constraint is a simple string like 'B' or list ['B', 'H']
                required_species = constraints['motif_composition']
                if isinstance(required_species, str):
                    if any(s != required_species for s in selected_symbols):
                        continue

            # Create Motif
            # Note: For seeded motifs, we assume the provided center is the ground truth
            
            # Generate a motif type string (e.g. "B12_H2")
            unique_species = sorted(list(set(selected_symbols)))
            species_str = "".join(unique_species)
            
            motifs.append(ClusterMotif(
                motif_type=f"{species_str}{len(matched_indices)}",
                size=len(matched_indices),
                center=np.array(center),
                atom_indices=matched_indices.tolist()
            ))
        
        return motifs

    def write_cluster_motifs(self, 
                            motifs: List[ClusterMotif], 
                            fmt: str = 'vasp',
                            output_dir: str = '.',
                            padding: float = 5.0
                            ):
        """
        Write each identified cluster motif to separate files.
        Args:
            motifs: List of ClusterMotif objects to write.
            fmt: File format to write (e.g., 'vasp', 'xyz').
            output_dir: Directory to save the motif files.
            padding: Extra space to add around the cluster in the box.
        Returns:
            None
        """
        if not motifs:
            print("No motifs to write.")
            return

        # 1. Find the maximum cluster diameter to set a uniform box size
        max_diameter = 0.0
        
        for motif in motifs:
            # Unwrap to get true geometric size
            cluster = self._unwrap_cluster(motif.atom_indices)
            if len(cluster) > 1:
                # Get max pairwise distance in the cluster
                # This is a safe approximation for diameter
                dists = cluster.get_all_distances(mic=False) # No mic needed, already unwrapped
                current_diam = np.max(dists)
                if current_diam > max_diameter:
                    max_diameter = current_diam
            else:
                # Fallback for single atoms
                if 2.0 > max_diameter: max_diameter = 2.0

        # Create cubic lattice length
        lattice_length = max_diameter + padding
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # 2. Process and write each motif
        for i, motif in enumerate(motifs):
            # Extract and unwrap the cluster so it is contiguous
            cluster = self._unwrap_cluster(motif.atom_indices)
            
            # Create a new box
            cluster.set_cell([lattice_length] * 3)
            cluster.set_pbc(True)
            
            # Center the cluster in the new box
            # We calculate the geometric center of the unwrapped atoms
            cluster_center_of_mass = cluster.get_center_of_mass()
            box_center = np.array([lattice_length / 2] * 3)
            
            # Shift atoms
            translation = box_center - cluster_center_of_mass
            cluster.positions += translation
            
            # Construct filename
            # Example: ./cluster_0_B12.poscar
            filename = f"cluster_{i}_{motif.motif_type}.{fmt}"
            filepath = os.path.join(output_dir, filename)
            
            write(filepath, cluster, format=fmt)
            
        print(f"Successfully wrote {len(motifs)} motifs to {output_dir}")


class NNOCStructureGenerator:
    def __init__(self, motifs_path: str = "", 
                 min_dist_nb: float = 4.5, 
                 min_dist_cl_cl: float = 3.0,
                 na_exclusion_radius: float = 2.0,
                 min_dist_na_na: float = 3.0,
                 max_dist_na_cl: float = 3.0,
                 target_na_ratio: float = 1.0,
                 max_attempts: int = 1000,
                 probabilities: Dict[str, float] = None):
        
        self.motifs_path = motifs_path
        self.min_dist_nb = min_dist_nb
        self.min_dist_cl_cl = min_dist_cl_cl
        self.na_exclusion_radius = na_exclusion_radius
        self.min_dist_na_na = min_dist_na_na
        self.max_dist_na_cl = max_dist_na_cl
        self.target_na_ratio = target_na_ratio
        self.max_attempts = max_attempts
        self.probabilities = probabilities or {
            'perturb_lattice': 0.2,
            'perturb_motifs': 0.4,
            'rotate_motifs': 0.4,
            'add_remove_motifs': 0.0,
        }

        # Cache motifs with their Cl-plane pre-aligned to X-axis
        self.motifs_cache = self._load_motifs()
        
    def _load_motifs(self):
        """Loads and pre-aligns motifs so their Cl4 plane is normal to the X-axis."""
        motifs = []
        if not os.path.exists(self.motifs_path):
            raise FileNotFoundError(f"Directory {self.motifs_path} not found.")
            
        files = [f for f in os.listdir(self.motifs_path) if f.startswith('POSCAR')]
        
        for f in files:
            path = os.path.join(self.motifs_path, f)
            try:
                motif = read(path)
                aligned_motif = self._align_motif_to_x_axis(motif)
                motifs.append(aligned_motif)
            except Exception as e:
                print(f"Skipping {f}: {e}")
        return motifs

    def _align_motif_to_x_axis(self, motif: Atoms) -> Atoms:
        """Rotates motif so Cl4 plane normal aligns with [1,0,0]."""
        cl_indices = [atom.index for atom in motif if atom.symbol == 'Cl']
        if len(cl_indices) < 3: return motif
            
        cl_pos = motif.positions[cl_indices]
        # Center points for SVD
        centered = cl_pos - np.mean(cl_pos, axis=0)
        u, s, vh = np.linalg.svd(centered)
        normal = vh[2, :] # Normal vector

        target = np.array([1.0, 0.0, 0.0])
        
        # Calculate rotation
        normal = normal / np.linalg.norm(normal)
        cross = np.cross(normal, target)
        dot = np.dot(normal, target)
        
        if np.linalg.norm(cross) < 1e-6:
            if dot < 0: # Anti-parallel
                r = R.from_euler('y', 180, degrees=True)
                motif.positions = motif.positions @ r.as_matrix().T
        else:
            angle = np.arccos(np.clip(dot, -1.0, 1.0))
            axis = cross / np.linalg.norm(cross)
            r = R.from_rotvec(axis * angle)
            motif.positions = motif.positions @ r.as_matrix().T
            
        return motif

    def generate_random_structure(self) -> Atoms:
        """Generates a random structure based on cached motifs and constraints.
        Returns: Atoms object """

        min_dist_nb, min_dist_cl_cl, na_exclusion_radius, min_dist_na_na, max_dist_na_cl, target_na_ratio = (
            self.min_dist_nb, self.min_dist_cl_cl, self.na_exclusion_radius,
            self.min_dist_na_na, self.max_dist_na_cl, self.target_na_ratio
        )

        # --- Step 1: Lattice Generation ---
        a = np.random.uniform(3.5, 4.5)
        b = np.random.uniform(10.0, 30.0)
        c = np.random.uniform(10.0, 30.0)
        cell = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
        
        # --- Step 2: Motif Count Logic ---
        volume = np.linalg.det(cell)
        # Density input interpreted as Volume per Motif for calculation consistency
        vol_per_motif = np.random.uniform(160, 200)
        n_motifs = int(round(volume / vol_per_motif))
        if n_motifs < 1: n_motifs = 1

        structure = Atoms(cell=cell, pbc=True)
        placed_nb_centers = []
        
        #(f"Generating lattice {cell.diagonal().round(2)}. Target motifs: {n_motifs}")

        # --- Step 3: Place Motifs with Constraints ---
        for i in range(n_motifs):
            motif_template = random.choice(self.motifs_cache).copy()
            
            # Identify internal indices
            nb_idx_local = [a.index for a in motif_template if a.symbol == 'Nb'][0]
            cl_indices_local = [a.index for a in motif_template if a.symbol == 'Cl']
            
            # Center template at origin relative to Nb
            shift = motif_template.positions[nb_idx_local]
            motif_template.positions -= shift
            
            # NEW: Random Rotation in BC Plane (around X axis)
            angle_x = np.random.uniform(0, 360)
            motif_template.rotate(angle_x, 'x', center=(0,0,0))
            
            max_attempts = 100
            #success = False
            for attempt in range(max_attempts):
                # Random location
                rand_scaled = np.random.random(3)
                rand_cart = np.dot(rand_scaled, cell)
                
                # Temp positions for candidates
                candidate_pos = motif_template.positions + rand_cart
                
                # Constraint A: Nb-Nb Distance (using centers)
                if placed_nb_centers:
                    # Quick check against centers
                    dists = self._get_pbc_distances(rand_cart, np.array(placed_nb_centers), cell)
                    if np.min(dists) < min_dist_nb:
                        continue 

                # Constraint B: Cl-Cl Distance > 3.0 A
                # Only check if there are already Cl atoms in structure
                existing_cl_indices = [a.index for a in structure if a.symbol == 'Cl']
                if existing_cl_indices:
                    # Current Cl positions
                    # existing_cl_pos = structure.positions[existing_cl_indices]
                    
                    # Candidate Cl positions (Apply shift + PBC wrap might be complex, 
                    # easiest to check unwrapped dists via MIC)
                    cand_cl_pos = candidate_pos[cl_indices_local]
                    
                    # We check every candidate Cl against every existing Cl
                    # Using ASE logic for robustness with MIC
                    cl_overlap = False
                    for c_pos in cand_cl_pos:
                        # Create dummy atom to measure distance to existing Cls
                        dummy = structure.copy()
                        dummy.append('X')
                        dummy.positions[-1] = c_pos
                        dists_cl = dummy.get_distances(
                            len(dummy)-1, 
                            existing_cl_indices, 
                            mic=True
                        )
                        if np.min(dists_cl) < min_dist_cl_cl:
                            cl_overlap = True
                            break
                    if cl_overlap:
                        continue
                
                # If we pass all checks
                placed_nb_centers.append(rand_cart)
                motif_to_add = motif_template.copy()
                motif_to_add.positions = candidate_pos
                structure += motif_to_add
                #success = True
                break
            
            #if not success:
                #print(f"  Warning: Failed to place motif {i+1} after {max_attempts} attempts.")
        
        structure.wrap()

        # --- Step 4: Place Na Atoms ---
        self._place_na_atoms(
            structure, 
            na_exclusion_radius, 
            min_dist_na_na,
            max_dist_na_cl, 
            target_count=int(len(placed_nb_centers) * target_na_ratio)
        )
        
        return structure

    def random_model(self, reg_id) -> structure_record.model:
        """Generates a random model for Basin Hopping."""
        generated = False
        attempts = 0
        while not generated and attempts < 100:
            new_structure = self.generate_random_structure()

            cell_volume = new_structure.get_volume()
            nb_count = sum(1 for atom in new_structure if atom.symbol == 'Nb')
            ratio = cell_volume / nb_count if nb_count > 0 else float('inf')
            if ratio > 220:
                attempts += 1
                continue
            else:
                generated = True

            new_structure = AseAtomsAdaptor().get_structure(new_structure)

        if attempts == 100:
            print("Failed to generate a suitable structure after 100 attempts.")
            return None
        
        rand_model = structure_record.model(new_structure, reg_id)
        rand_model.inheritance = 'random'
        rand_model.made_by = 'random'
        return rand_model
    
    def _get_pbc_distances(self, point, targets, cell):
        """Vectorized distance check for points."""
        diff = targets - point
        box_diag = np.diag(cell)
        diff = diff - box_diag * np.round(diff / box_diag)
        return np.linalg.norm(diff, axis=1)

    def _place_na_atoms(self, structure: Atoms, 
                        exclusion_radius: float, 
                        min_dist_na_na: float, 
                        max_dist_na_cl: float,
                        target_count: int):
            
        if target_count <= 0: return
        
        # 1. Voronoi Decomposition
        cl_indices = [a.index for a in structure if a.symbol == 'Cl']
        if len(cl_indices) < 4: return
        
        cl_pos = structure.positions[cl_indices]
        cell = structure.cell
        
        # Supercell logic (same as before)
        shifts = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    shifts.append(i * cell[0] + j * cell[1] + k * cell[2])
        super_points = np.vstack([cl_pos + s for s in shifts])
        
        try:
            vor = Voronoi(super_points)
        except Exception:
            return 
            
        nodes = vor.vertices
        
        # 2. Filter: Inside Box
        box_diag = np.diag(cell)
        in_box_mask = np.all((nodes >= 0) & (nodes < box_diag), axis=1)
        candidates = nodes[in_box_mask]
        
        # 3. Pre-filter candidates for Static Constraints (Exclusion vs Structure & Bonding)
        # We do NOT check Na-Na here yet.
        pre_validated_sites = []
        
        for site in candidates:
            dummy = structure.copy()
            dummy.append('X')
            dummy.positions[-1] = site
            
            # A. Exclusion Check (Too close to atoms ALREADY in structure)
            dists_all = dummy.get_distances(len(dummy)-1, range(len(structure)), mic=True)
            if np.min(dists_all) < exclusion_radius:
                continue 
                
            # B. Bonding Check (Must be close to Cl)
            dists_cl = dummy.get_distances(len(dummy)-1, cl_indices, mic=True)
            if np.min(dists_cl) > max_dist_na_cl:
                continue 
            
            pre_validated_sites.append(site)
            
        # 4. Sequential Placement with Dynamic Na-Na Check
        random.shuffle(pre_validated_sites)
        placed_count = 0
        
        # Track indices of Na atoms we actally place in this batch
        newly_placed_indices = []
        
        for site in pre_validated_sites:
            if placed_count >= target_count:
                break
            
            # --- DYNAMIC CHECK START ---
            # Check distance against OTHER Na atoms (both old and newly placed)
            # We can't rely on the loop above because structure updates
            
            # Create a dummy again to check against the CURRENT updated structure
            dummy = structure.copy()
            dummy.append('Na')
            dummy.positions[-1] = site
            
            # Find all Na atoms currently in structure (including ones we just added)
            na_indices = [a.index for a in structure if a.symbol == 'Na']
            
            if na_indices:
                dists_na = dummy.get_distances(len(dummy)-1, na_indices, mic=True)
                if np.min(dists_na) < min_dist_na_na:
                    continue # Skip this site, it's too close to a Na we just placed
            # --- DYNAMIC CHECK END ---

            # Place the atom
            structure.append('Na')
            structure.positions[-1] = site
            placed_count += 1
            
        #print(f"  Placed {placed_count}/{target_count} Na atoms.")


class NNOCBasinhopping(NNOCStructureGenerator):
    """
    Implements mutation operators for Basin Hopping / Genetic Algorithms
    specifically tailored for Nb-Cl motif structures.
    """
    def __init__(self, **kwargs):
        """
        Initializes the NNOCBasinhopping class with default probabilities for mutation operators.
        """
        super().__init__(**kwargs)

        if self.probabilities is None:
            self.probabilities = {
                'perturb_lattice': 0.2,
                'perturb_motifs': 0.4,
                'rotate_motifs': 0.2,
                'add_remove_motifs': 0.2,
            }

    def select_and_apply_operator(self, structure: Atoms) -> Atoms:
        """
        Main entry point. Selects an operator based on probabilities and applies it.
        
        Args:
            structure: The parent Atoms object.
        """
        probabilities = self.probabilities
        ops = list(probabilities.keys())
        probs = list(probabilities.values())
        
        # Normalize probabilities just in case
        total = sum(probs)
        probs = [p / total for p in probs]
        
        chosen_op = np.random.choice(ops, p=probs)
        
        #print(f"--- Applying Operator: {chosen_op} ---")
        
        if chosen_op == 'perturb_lattice':
            result = self.op_perturb_lattice(structure)
        elif chosen_op == 'perturb_motifs':
            result = self.op_perturb_motifs(structure)
        elif chosen_op == 'rotate_motifs':
            result = self.op_rotate_motifs(structure)
        elif chosen_op == 'add_remove_motifs':
            # Assuming you have this defined or return original for now
            result = getattr(self, 'op_add_remove_motifs', lambda x: x)(structure)
        else:
            result = structure.copy()

        return result, chosen_op

    def get_model(self, select, pool, reg_id):
        """
        Selects a mutation operator and applies it to the given model's structure.
        
        Args:
            select: A function to select a model from the pool.
            pool: A list of models to select from.
            reg_id: Registration ID for the new model.
        """
        parent_model = select.get_a_parent(pool)
        # make a copy
        parent = copy.deepcopy(parent_model)

        parent_structure = parent.astr
        parent_structure = AseAtomsAdaptor().get_atoms(parent_structure)
        
        # Apply mutation operator
        child_structure, chosen_op = self.select_and_apply_operator(parent_structure)
        
        if child_structure is None:
            return None
        
        # Convert back to pymatgen structure
        child_pmg = AseAtomsAdaptor().get_structure(child_structure)
        
        # Create a new model record
        child_model = structure_record.model(child_pmg, reg_id)
        child_model.inheritance = [parent.label] 
        child_model.made_by = chosen_op
        
        return child_model



    # =========================================================================
    # OPERATOR 1: PERTURB LATTICE
    # =========================================================================
    def op_perturb_lattice(self, structure: Atoms) -> Atoms:
        """
        Randomly perturb lattice lengths a, b, c within -10% to +20%.
        Keeps fractional coordinates constant (scale_atoms=True).
        """
        new_struct = structure.copy()
        
        # Factors between 0.9 (-10%) and 1.2 (+20%)
        factors = np.random.uniform(0.9, 1.2, size=3)
        
        current_cell = new_struct.get_cell()
        new_cell = current_cell * factors[:, np.newaxis] # Scale diagonal/vectors
        
        # scale_atoms=True keeps fractional coordinates, effectively stretching the bonds
        # Note: This might violate bond lengths, but is standard for lattice moves.
        new_struct.set_cell(new_cell, scale_atoms=True)
        
        return new_struct

    # =========================================================================
    # OPERATOR 2: PERTURB MOTIFS
    # =========================================================================
    def op_perturb_motifs(self, structure: Atoms) -> Optional[Atoms]:
        """
        1. Remove Na.
        2. Perturb Nb positions in a new lattice sequentially.
           - For each Nb, try up to 100 times to find a valid perturbed position.
           - If ANY Nb cannot be placed validly, fail and return None.
        3. Place motifs at valid positions.
        4. Re-add Na.
        """
        # 1. Strip Na
        parent = structure.copy()
        nb_indices = [a.index for a in parent if a.symbol == 'Nb']
        
        if not nb_indices: return parent 
        
        # Create new empty lattice
        child = Atoms(cell=parent.cell, pbc=True)
        
        # 2. Get original Nb positions
        nb_positions = parent.positions[nb_indices]
        placed_nb_centers = []
        
        max_attempts = 100
        perturb_limit = 1.0 # Angstroms
        
        # Iterate through every Nb atom from the parent
        for i, orig_pos in enumerate(nb_positions):
            success = False
            
            for attempt in range(max_attempts):
                # Generate a random perturbation vector for THIS atom only
                perturb_vector = np.random.uniform(-perturb_limit, perturb_limit, size=3)
                cand_pos = orig_pos + perturb_vector
                
                # Check 1: Nb-Nb Constraints
                # If child is empty, first one is always accepted (regarding Nb constraints)
                nb_clash = False
                if placed_nb_centers:
                    dists = self._get_pbc_distances(cand_pos, np.array(placed_nb_centers), child.cell)
                    if np.min(dists) < self.min_dist_nb:
                        nb_clash = True
                
                if nb_clash:
                    continue # Try next perturbation
                
                # Check 2: Prepare Motif & Check Cl-Cl Constraints
                # We must ensure the attached Cl atoms also fit
                motif_template = random.choice(self.motifs_cache).copy()
                nb_idx_local = [a.index for a in motif_template if a.symbol == 'Nb'][0]
                
                # Center and shift motif to candidate position
                shift = motif_template.positions[nb_idx_local]
                motif_template.positions -= shift 
                motif_template.positions += cand_pos 
                
                # Check against existing Cl atoms in child
                if self._check_cl_constraints(child, motif_template):
                    # --- SUCCESS ---
                    child += motif_template
                    placed_nb_centers.append(cand_pos)
                    success = True
                    break # Break retry loop, move to next Nb
            
            # If we exhausted max_attempts for this specific Nb atom without success:
            if not success:
                #print(f"Failed to place motif {i} after {max_attempts} attempts. Aborting operator.")
                return None
        
        child.wrap()
        
        # 4. Re-add Na Atoms
        # (Only happens if all motifs were successfully placed)
        self._place_na_atoms(
            child,
            self.na_exclusion_radius,
            self.min_dist_na_na,
            self.max_dist_na_cl,
            target_count=int(len(placed_nb_centers) * self.target_na_ratio)
        )
        
        return child

    # =========================================================================
    # OPERATOR 3: ROTATE MOTIFS
    # =========================================================================
    def op_rotate_motifs(self, structure: Atoms) -> Optional[Atoms]:
        """
        1. Remove Na.
        2. Identify existing motifs (Nb+Cl+O) using ClusterAnalyzer around Nb centers.
        3. Create a new empty lattice.
        4. For each motif:
           - Extract the specific atoms from the parent.
           - Rotate them in the b-c plane (around the A-axis/X-axis) centered at Nb.
           - Check for collisions with previously placed motifs.
           - If collision, retry with a new angle.
        5. Re-add Na.
        """
        # 1. Strip Na to isolate the framework
        parent_no_na = structure[[a.index for a in structure if a.symbol != 'Na']]
        
        # Get Nb positions (Seeds)
        nb_indices = [a.index for a in parent_no_na if a.symbol == 'Nb']
        if not nb_indices: return structure.copy()
        
        nb_positions = parent_no_na.positions[nb_indices]
        
        # 2. Identify Motifs using ClusterAnalyzer
        # We assume a diameter sufficient to catch Nb-Cl and Nb-O bonds (approx 6.0 Angstroms)
        analyzer = ClusterAnalyzer(parent_no_na)
        try:
            motifs = analyzer.find_by_seeds(
                centers=nb_positions,
                diameter=6.0, 
                constraints=None # We accept whatever is attached to Nb
            )
        except Exception as e:
            print(f"Clustering failed: {e}")
            return None

        # 3. Create Child Lattice
        child = Atoms(cell=structure.cell, pbc=True)
        placed_nb_centers = []
        
        # 4. Process each identified motif
        # We loop through the motifs found by the analyzer
        for motif in motifs:
            # Unwrap the cluster to ensure atoms are contiguous in space (handles PBC)
            # This method returns an Atoms object of just the motif
            original_motif_atoms = analyzer._unwrap_cluster(motif.atom_indices)
            
            # Identify the Nb atom index within this small local cluster object
            # (It should be the one closest to the center, or just find symbol Nb)
            nb_local_indices = [a.index for a in original_motif_atoms if a.symbol == 'Nb']
            if not nb_local_indices: 
                continue # Should not happen if seeded correctly
            nb_idx_local = nb_local_indices[0]
            
            # The current center of this motif
            current_center = original_motif_atoms.positions[nb_idx_local]
            
            success = False
            max_attempts = 50
            
            for _ in range(max_attempts):
                # Work on a copy so we don't mutate the original if we need to retry
                cand_motif = original_motif_atoms.copy()
                
                # A. Shift to Origin (relative to Nb)
                cand_motif.positions -= current_center
                
                # B. Rotate in b-c plane (About X-axis)
                angle = np.random.uniform(0, 360)
                cand_motif.rotate(angle, 'x', center=(0,0,0))
                
                # C. Shift back to original position
                cand_motif.positions += current_center
                
                # D. Constraint Check
                # Nb positions didn't change, so Nb-Nb is valid by definition (if parent was valid).
                # We only need to check if the rotated Cl/O atoms hit existing atoms in 'child'.
                if self._check_cl_constraints(child, cand_motif):
                    child += cand_motif
                    placed_nb_centers.append(current_center)
                    success = True
                    break
            
            if not success:
                # If we can't rotate it without hitting neighbors, we can:
                # 1. Fail the whole operator (Strict)
                # 2. Keep the original un-rotated orientation (Lenient)
                # Let's be lenient: keep original orientation if rotation fails
                if self._check_cl_constraints(child, original_motif_atoms):
                    child += original_motif_atoms
                    placed_nb_centers.append(current_center)
                else:
                    # If even the original doesn't fit (rare, implies overlapping seeds), fail.
                    return None

        child.wrap()

        # 5. Re-add Na Atoms
        self._place_na_atoms(
            child,
            self.na_exclusion_radius,
            self.min_dist_na_na,
            self.max_dist_na_cl,
            target_count=int(len(placed_nb_centers) * self.target_na_ratio)
        )

        return child

    # =========================================================================
    # HELPERS
    # =========================================================================
    def _check_cl_constraints(self, existing_struct: Atoms, new_motif: Atoms) -> bool:
        """
        Checks if Cl atoms in new_motif overlap with Cl atoms in existing_struct.
        """
        existing_cl = [a.index for a in existing_struct if a.symbol == 'Cl']
        if not existing_cl:
            return True
            
        new_cl_pos = new_motif.positions[[a.index for a in new_motif if a.symbol == 'Cl']]
        
        for pos in new_cl_pos:
            dummy = existing_struct.copy()
            dummy.append('X')
            dummy.positions[-1] = pos
            
            dists = dummy.get_distances(len(dummy)-1, existing_cl, mic=True)
            if np.min(dists) < self.min_dist_cl_cl:
                return False
        return True