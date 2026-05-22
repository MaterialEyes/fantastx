from dataclasses import dataclass
from typing import List, Optional, Dict, Union, Tuple
import numpy as np
import os, random, copy

from ase import Atoms, Atom
from ase.io import read, write

from pymatgen.core import Lattice, Structure
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


class NBHStructureGenerator:
    """
    Generator for Sodium Borohydride (NBH) structures using a reference lattice map
    and closo-borohydride cage motifs (BUs).
    """
    def __init__(self, 
                 ref_presets_path: str,
                 b10_path: str,
                 b12_path: str,
                 lattice_range: Tuple[float, float] = (8.5, 9.5),
                 max_bu_deformation_pct: float = 5.0,
                 min_dist_cation_anion: float = 3.1,
                 min_dist_cation_cation: float = 3.0,
                 max_attempts: int = 100,
                 **kwargs):
        """
        Args:
            ref_presets_path: Path to directory containing POSCAR preset files acting as maps (e.g., POSCAR_preset_bcc_24g).
                                IMPORTANT: Assumes 'Cl' = Anion sites, 'Na' = Candidate Cation sites.
            b10_path: Path to B10H10 motif POSCAR.
            b12_path: Path to B12H12 motif POSCAR.
            lattice_range: (min, max) for the cubic lattice parameter a0.
            max_bu_deformation_pct: Max percentage to stretch/compress BUs (e.g., 5.0 for 5%).
            min_dist_cation_anion: Exclusion radius around BU centers (Å).
            min_dist_cation_cation: Minimum Na-Na distance (Å).
            max_attempts: Max retries to generate a valid structure.
            **kwargs: Additional overrides for internal attributes.
        """
        
        # Configuration Attributes
        self.lattice_range = lattice_range
        self.max_bu_deformation_pct = max_bu_deformation_pct
        self.min_dist_cation_anion = min_dist_cation_anion
        self.min_dist_cation_cation = min_dist_cation_cation
        self.max_attempts = max_attempts
        
        # Paths
        self.ref_presets_path = ref_presets_path
        self.b10_path = b10_path
        self.b12_path = b12_path

        # Handle any extra kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Cache Resources
        self.ref_atoms_list, self.b10_motif, self.b12_motif = self._load_resources()

        # Attributes to add water molecules if needed (can be set via kwargs)
        self.num_waters_per_cage_motif = kwargs.get('num_waters_per_cage_motif', [])
        self.min_dist_h2os = kwargs.get('min_dist_h2os', 2.5)
        self.min_dist_h2o_host = kwargs.get('min_dist_h2o_host', 2.0) # Minimum distance from water to any host atom (Na, B, H))

    def _load_resources(self) -> Tuple[Atoms, Atoms, Atoms]:
        """Loads reference structure and motifs into memory."""
        required_paths = [self.ref_presets_path, self.b10_path, self.b12_path]
        for p in required_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Resource file not found: {p}")
            
        ref_poscars = [i for i in os.listdir(self.ref_presets_path) if i.startswith("POSCAR_preset")]
        if not ref_poscars:
            print (f"Note: POSCAR_presets needed with Na and Cl species for mapping to cation and anion sites, respectively.")
            raise FileNotFoundError(f"No POSCAR_preset files found in {self.ref_presets_path}")
        
        # create atoms for each preset and store in a list for random selection later
        ref_atoms_list = []
        for poscar in ref_poscars:
            ref_atoms_i = read(os.path.join(self.ref_presets_path, poscar))
            ref_atoms_list.append(ref_atoms_i)
        
        return (ref_atoms_list, read(self.b10_path), read(self.b12_path))

    def _transform_bu(self, bu: Atoms) -> Atoms:
        """Applies random rotation and deformation to a motif."""
        bu = bu.copy()
        
        # 1. Random Rotation
        axis = random.choice(['x', 'y', 'z'])
        angle = random.uniform(0, 360)
        # Using 'COM' (Center of Mass) for rotation center
        bu.rotate(angle, axis, center='COM')
        
        # 2. Random Deformation
        deform_factor = random.uniform(0, self.max_bu_deformation_pct) / 100.0
        scale = 1.0 + deform_factor * random.choice([-1, 1])
        
        # Apply scaling matrix to positions manually to deform along one axis
        M = np.eye(3)
        idx = 'xyz'.index(axis)
        M[idx, idx] = scale
        bu.set_positions(bu.get_positions() @ M.T)
        
        return bu

    def _pbc_distance(self, p1: np.ndarray, p2: np.ndarray, cell_array: np.ndarray) -> float:
        """
        Calculates distance between two cartesian points under PBC using numpy.
        """
        diff = p1 - p2
        # Round to nearest image
        # This assumes an orthogonal box (diagonal matrix) which fits the cubic logic
        box_diag = np.diag(cell_array)
        diff = diff - box_diag * np.round(diff / box_diag)
        return np.linalg.norm(diff)
    
    def _wrap_point(self, point: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        """Wraps a single point into the unit cell defined by cell and pbc."""
        wrapped = np.copy(point)
        for i in range(3):
            if pbc[i]:
                wrapped[i] = wrapped[i] % cell[i, i]
        return wrapped

    def _create_h2o_template(self) -> Atoms:
        """
        Creates a standardized H2O molecule centered at its geometric center.
        Uses typical geometry: O-H = 0.96 Å, H-O-H = 104.5°.
        """
        bond_len = 0.96
        angle_rad = np.radians(104.5) / 2
        
        # Define relative positions (O at top, H's below)
        pos = [
            [0.0, 0.0, 0.0],                                      # O
            [bond_len * np.sin(angle_rad), -bond_len * np.cos(angle_rad), 0.0], # H1
            [-bond_len * np.sin(angle_rad), -bond_len * np.cos(angle_rad), 0.0] # H2
        ]
        
        h2o = Atoms('OHH', positions=pos)
        # Center positions at (0,0,0) for correct rotation later
        h2o.positions -= np.mean(h2o.positions, axis=0)
        return h2o

    def add_water_molecules(self, 
                            structure: Atoms, 
                            num_waters: int, 
                            max_attempts: int = 5000) -> Atoms:
        """
        Inserts randomly rotated H2O molecules into existing structure.
        
        Args:
            structure: The host structure (NBH).
            num_waters: Number of H2O molecules to add.
        """
        if num_waters <= 0:
            return structure

        # 1. Prepare Template
        h2o_template = self._create_h2o_template()
        
        # 2. Identify Existing Atom Positions (Host)
        host_positions = structure.get_positions()
        cell = structure.get_cell()
        
        # List to track centers of newly placed waters for self-overlap check
        placed_water_centers = []
        
        count_added = 0
        
        for i in range(num_waters):
            success = False
            for attempt in range(max_attempts):
                # A. Generate Random Candidate Point (Cartesian)
                # Using fractional allows uniform sampling in non-cubic cells too
                rand_frac = np.random.random(3)
                candidate_center = np.dot(rand_frac, cell)
                
                # B. Constraint 1: Check against Host Structure
                # We check the distance from Candidate Center -> All Host Atoms
                # (Assuming water roughly spherical ~2.5A radius is sufficient)
                dists_host = self._get_pbc_distances_array(candidate_center, host_positions, cell)
                
                if np.min(dists_host) < self.min_dist_h2o_host:
                    continue
                
                # C. Constraint 2: Check against previously placed Waters
                if placed_water_centers:
                    dists_h2o = self._get_pbc_distances_array(
                        candidate_center, 
                        np.array(placed_water_centers), 
                        cell
                    )
                    if np.min(dists_h2o) < self.min_dist_h2os:
                        continue
                
                # --- Placement is Valid ---
                
                # D. Rotate and Place
                new_h2o = h2o_template.copy()
                
                # Random 3D rotation
                angles = np.random.uniform(0, 360, 3)
                new_h2o.rotate(angles[0], 'x')
                new_h2o.rotate(angles[1], 'y')
                new_h2o.rotate(angles[2], 'z')
                
                # Shift to candidate spot
                new_h2o.positions += candidate_center
                
                # Add to structure
                structure += new_h2o
                
                # Update tracking lists
                placed_water_centers.append(candidate_center)
                success = True
                count_added += 1
                break
            
            if not success:
                print(f"Warning: Could only place {count_added}/{num_waters} water molecules.")
                break

            structure.wrap()
            
        return structure

    def _get_pbc_distances_array(self, point: np.ndarray, targets: np.ndarray, cell: np.ndarray) -> np.ndarray:
        """
        Vectorized PBC distance calculation for a single point vs an array of targets.
        (Added here to ensure the method above is self-contained within the class)
        """
        diff = targets - point
        # MIC for Orthorhombic cells (assumed based on cubic/diag lattice logic)
        box_diag = np.diag(cell)
        diff = diff - box_diag * np.round(diff / box_diag)
        return np.linalg.norm(diff, axis=1)

    def generate_random_structure(self) -> Optional[Atoms]:
        """
        Generates a single random NBH structure satisfying all constraints.
        Returns None if max_attempts is reached.
        """
        for attempt in range(self.max_attempts):
            
            # --- Step 1: Define Lattice ---
            a0 = random.uniform(*self.lattice_range)
            cell = np.eye(3) * a0
            
            # --- Step 2: Map Reference Sites to New Lattice ---
            # We use fractional coords from the cached reference to scale to new a0
            ref_atoms = random.choice(self.ref_atoms_list) # In case we want to extend to multiple refs later
            ref_frac_coords = ref_atoms.get_scaled_positions()
            
            # Pre-calculate site indices from reference map
            ref_symbols = np.array(ref_atoms.get_chemical_symbols())
            ref_anion_indices = np.where(ref_symbols == "Cl")[0]
            # We are targeting sites which are labeled 'Na' in the ref file
            ref_cation_candidate_indices = np.where(ref_symbols == "Na")[0]

            anion_sites_cart = ref_frac_coords[ref_anion_indices] @ cell
            cation_candidates_cart = ref_frac_coords[ref_cation_candidate_indices] @ cell
            
            new_atoms = Atoms(cell=cell, pbc=True)
            
            # --- Step 3: Place Anion Cages (BUs) ---
            num_bus = len(anion_sites_cart)
            
            for i in range(num_bus):
                # Alternate between B10 and B12 or randomize
                # Here we stick to the logic: even=B10, odd=B12
                template = self.b10_motif if i % 2 == 0 else self.b12_motif
                bu = self._transform_bu(template)
                
                # Center BU at the target site
                bu_center = np.mean(bu.get_positions(), axis=0)
                shift = anion_sites_cart[i] - bu_center
                bu.positions += shift
                
                new_atoms += bu
            
            # --- Step 4: Select and Place Cations (Na) ---
            # Probability Logic (Uniform for now based on snippet, but extensible)
            n_cations = 2 * num_bus # Assuming 2 Na per BU as in NBH
            if len(ref_cation_candidate_indices) < n_cations:
                print("Warning: Not enough candidate sites for target cation count.")
                continue

            # Randomly select indices
            selected_indices_local = np.random.choice(
                range(len(ref_cation_candidate_indices)),
                size=n_cations,
                replace=False
            )
            
            # Create temporary list of proposed Na positions for validation
            proposed_na_positions = cation_candidates_cart[selected_indices_local]
            
            # --- Step 5: Constraint Validation ---
            
            # A. Anion-Cation Distance Check
            # Check if any proposed Na is too close to any Anion center
            valid_anion_dist = True
            for na_pos in proposed_na_positions:
                for anion_pos in anion_sites_cart:
                    dist = self._pbc_distance(na_pos, anion_pos, cell)
                    if dist < self.min_dist_cation_anion:
                        valid_anion_dist = False
                        break
                if not valid_anion_dist: break
            
            if not valid_anion_dist:
                continue # Retry structure generation
            
            # B. Cation-Cation Distance Check
            valid_cation_dist = True
            for i in range(len(proposed_na_positions)):
                for j in range(i + 1, len(proposed_na_positions)):
                    dist = self._pbc_distance(proposed_na_positions[i], 
                                              proposed_na_positions[j], 
                                              cell)
                    if dist < self.min_dist_cation_cation:
                        valid_cation_dist = False
                        break
                if not valid_cation_dist: break
            
            if not valid_cation_dist:
                continue # Retry structure generation

            # --- Step 6: Finalize Structure ---
            for pos in proposed_na_positions:
                new_atoms.append(Atom("Na", position=pos))

            # --- Step 7: Optional Add waters ---
            if hasattr(self, 'num_waters_per_cage_motif') and len(self.num_waters_per_cage_motif) > 0:
                num_waters = round(random.choice(self.num_waters_per_cage_motif) * num_bus)
                new_atoms = self.add_water_molecules(
                    new_atoms, 
                    num_waters=num_waters, 
                )
            
            # Sort and wrap (optional but good practice)
            # Standardize using pymatgen adaptor if sorting is complex, 
            # but ASE wrap is usually sufficient for bounding box.
            new_atoms.wrap()
            
            return new_atoms

        print(f"Failed to generate valid NBH structure after {self.max_attempts} attempts.")
        return None

    def random_model(self, reg_id) -> structure_record.model:
        """Generates a random model for Basin Hopping."""
        new_structure = self.generate_random_structure()
        if new_structure is None:
            print(f"[NBHStructureGenerator] Failed to generate a valid structure after {self.max_attempts} attempts. Returning None.")
            return None
        new_structure = AseAtomsAdaptor().get_structure(new_structure)
        new_structure.sort()

        rand_model = structure_record.model(new_structure, reg_id)
        rand_model.inheritance = 'random'
        rand_model.made_by = 'random'
        return rand_model


class NBHBasinhopping(NBHStructureGenerator):
    """
    Implements mutation operators for Basin Hopping on Borohydride structures.
    Uses DBSCAN to identify B10/B12 cages and Voronoi tessellation for 
    intelligent Na+ placement.
    """
    def __init__(self, **kwargs):
        """
        Args:
            probabilities (dict): Probabilities for mutation operators.
            max_perturbation (float): Max translation in Å for motif perturbation.
            min_dist_anion_anion (float): Min distance between cage centers.
            motif_radius (float): Radius of the exclusion sphere around cages for Na placement.
            min_dist_cation_H (float): Minimum distance between Na and H.
            **kwargs: Passed to NBHStructureGenerator.
        """
        super().__init__(**kwargs)
        
        # 1. Load attributes from kwargs or set defaults
        self.probabilities = kwargs.get('probabilities', {
            'perturb_lattice': 0.2,
            'perturb_motifs': 0.25,
            'rotate_motifs': 0.2,
            'perturb_and_rotate': 0.25,
            'reshuffle_cations': 0.1,
        })
        
        # Geometrical constraints for mutations 
        # NOTE: anion is the motif or cage (B10/B12 + H), cation is Na+
        self.max_perturbation = kwargs.get('max_perturbation', 0.5)
        self.min_dist_anion_anion = kwargs.get('min_dist_anion_anion', 7.0)
        self.min_dist_cation_H = kwargs.get('min_dist_cation_H', 1.5)
        self.motif_radius = kwargs.get('motif_radius', 3.0) # Default radius for motif exclusion in Na placement
        self.min_dist_cation_anion = kwargs.get('min_dist_cation_anion', 3.1) # Default cation-anion min distance

        self.eps_dbscan = kwargs.get('eps_dbscan', 2.0) # DBSCAN eps for clustering B atoms into cages
        self.min_samples_dbscan = kwargs.get('min_samples_dbscan', 6) # DBSCAN min_samples for cage identification

        self.verbose = kwargs.get('verbose', False) 

    def select_and_apply_operator(self, structure: Atoms) -> Tuple[Optional[Atoms], str]:
        """Selects an operator based on probabilities and applies it."""
        ops = list(self.probabilities.keys())
        probs = list(self.probabilities.values())
        
        # Normalize probabilities
        total = sum(probs)
        probs = [p / total for p in probs]
        
        chosen_op = str(np.random.choice(ops, p=probs))
        
        if chosen_op == 'perturb_lattice':
            result = self.op_perturb_lattice(structure)
        elif chosen_op == 'perturb_motifs':
            result = self.op_perturb_motifs(structure)
        elif chosen_op == 'rotate_motifs':
            result = self.op_rotate_motifs(structure)
        elif chosen_op == 'perturb_and_rotate':
            result = self.op_perturb_and_rotate_motifs(structure)
        elif chosen_op == 'reshuffle_cations':
            result = self.op_reshuffle_cations(structure)
        else:
            print (f"Unknown operator: {chosen_op}")
            return None, chosen_op

        return result, chosen_op
    
    def get_model(self, select, pool, reg_id):
        """
        Adapter for the Basin Hopping engine.
        Workflow:
        1. Get Parent
        2. IF WATER SEARCH: Strip Water (Check for H3O+ outliers). If outlier, abort.
        3. Apply Mutation
        4. IF WATER SEARCH: Re-add Water
        5. Return Child
        """
        parent_model = select.get_a_parent(pool)
        parent = copy.deepcopy(parent_model)

        # Convert to ASE
        parent_structure = parent.astr
        if not isinstance(parent_structure, Atoms):
            parent_structure = AseAtomsAdaptor().get_atoms(parent_structure)
        
        # Default: treat parent as the clean starting point
        clean_structure = parent_structure
        
        # --- STEP A: Handle Water Logic (Conditional) ---
        # Only trigger if this is a "Water Search" (num_waters > 0)
        is_water_search = hasattr(self, 'num_waters_per_cage_motif') and len(self.num_waters_per_cage_motif) > 0
        
        if is_water_search:
            # Strip waters and check for stability
            clean_structure = self._strip_water_molecules(parent_structure, model_label=parent.label)
            
            # If H3O/instability was found, _strip returns None. We skip this parent.
            if clean_structure is None:
                return None
        
        # --- STEP B: Apply Mutation to Framework ---
        child_structure, chosen_op = self.select_and_apply_operator(clean_structure)
        
        if child_structure is None:
            return None
        
        # --- STEP C: Re-add Waters (Conditional) ---
        if is_water_search:
            try:
                # Identify motifs to calculate how many waters to add
                motifs = self._identify_motifs_with_dbscan(child_structure, 
                                                           eps=self.eps_dbscan, 
                                                           min_samples=self.min_samples_dbscan)
                num_bus = len(motifs)
                
                if num_bus > 0:
                    child_structure = self.add_water_molecules(
                        child_structure,
                        num_waters=round(random.choice(self.num_waters_per_cage_motif) * num_bus),
                    )
            except Exception as e:
                print(f"[NBHBasinhopping] Warning: Failed to re-add water to child of {parent.label}: {e}")
                return None

        # Convert back to Pymatgen
        child_pmg = AseAtomsAdaptor().get_structure(child_structure)
        child_pmg.sort() 
        
        # Create record
        child_model = structure_record.model(child_pmg, reg_id)
        child_model.inheritance = [parent.label] 
        child_model.made_by = chosen_op
        
        return child_model

    # =========================================================================
    # CORE LOGIC: Strip waters with strict stability check
    # =========================================================================
    def _strip_water_molecules(self, structure: Atoms, model_label: int | str = "Unknown") -> Optional[Atoms]:
        """
        Identifies and removes H2O molecules.
        Strict Check: If Oxygen has > 2 H neighbors (e.g. H3O), it is an outlier.
        Returns None if unstable.
        """
        structure = structure.copy()
        
        o_indices = [atom.index for atom in structure if atom.symbol == 'O']
        if not o_indices:
            return structure
            
        h_indices = [atom.index for atom in structure if atom.symbol == 'H']
        if not h_indices:
            del structure[o_indices]
            print (f"Warning: No H atoms found in {model_label} while stripping waters. Removing O atoms without stability check.")
            return structure

        indices_to_remove = set(o_indices)
        
        # Calculate distances from all O to all H
        for o_idx in o_indices:
            dists = structure.get_distances(o_idx, h_indices, mic=True)
            
            # Find H's closer than 1.2 A
            nearby_h_local_indices = np.where(dists < 1.2)[0]

            # --- STRICT STABILITY CHECK ---
            if len(nearby_h_local_indices) > 2:
                # Found H3O or worse -> Cage collapsed or reaction occurred.
                print(f"Skipping outlier parent {model_label}: Oxygen {o_idx} has {len(nearby_h_local_indices)} H neighbors (unstable/H3O).")
                return None 

            for local_idx in nearby_h_local_indices:
                global_h_idx = h_indices[local_idx]
                indices_to_remove.add(global_h_idx)
                
        # Delete atoms
        del structure[list(indices_to_remove)]
        
        return structure

    # =========================================================================
    # CORE LOGIC: DBSCAN MOTIF IDENTIFICATION
    # =========================================================================
    def _identify_motifs_with_dbscan(self, structure: Atoms, eps: float = 2.0, min_samples: int = 6) -> List[Dict]:
        """
        Identifies B10/B12 cages AND attaches associated H atoms.
        Returns a list of dicts: {'indices': [all_atom_indices], 'center': np.array, 'type': str}
        """
        # 1. Cluster Boron Atoms
        b_indices = [i for i, a in enumerate(structure) if a.symbol == 'B']
        if not b_indices: 
            print("No Boron atoms found in structure for motif identification.")
            return []
        
        b_pos = structure.positions[b_indices]
        
        full_dist_matrix = structure.get_all_distances(mic=True)
        b_dist_matrix = full_dist_matrix[np.ix_(b_indices, b_indices)]
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = clustering.fit_predict(b_dist_matrix)

        # --- Add safety check for weird clusters ---
        unique_labels = set(labels)
        if -1 in unique_labels and len(unique_labels) == 1:
             if self.verbose: 
                 print("[Debug] DBSCAN found only noise (-1). Check eps/min_samples.")
             return []
        
        motifs = {} # Map label -> {'b_indices': [], 'h_indices': []}
        
        # 2. Group B atoms
        for local_idx, label in enumerate(labels):
            if label == -1: continue
            if label not in motifs:
                motifs[label] = {'b_indices': [], 'h_indices': []}
            motifs[label]['b_indices'].append(b_indices[local_idx])
            
        # 3. Associate H atoms to nearest B cluster
        h_indices = [i for i, a in enumerate(structure) if a.symbol == 'H']
        
        for h_idx in h_indices:
            # Find distances from this H to ALL B atoms
            dists_to_b = full_dist_matrix[h_idx, b_indices]
            nearest_b_local_idx = np.argmin(dists_to_b)
            nearest_b_label = labels[nearest_b_local_idx]
            
            if nearest_b_label != -1 and nearest_b_label in motifs:
                motifs[nearest_b_label]['h_indices'].append(h_idx)

        # 4. Format Output
        results = []
        cell = structure.get_cell()
        pbc = structure.get_pbc()

        for label, data in motifs.items():
            all_indices = data['b_indices'] + data['h_indices']
            
            # --- FIX START: Handle PBC for Center Calculation ---
            b_pos_subset = structure.positions[data['b_indices']]
            
            # Use the first atom as a reference anchor
            ref_pos = b_pos_subset[0]
            
            # Find the minimum image of all other atoms relative to the reference
            # geometric_center = mean( reference + min_image_vector(rest - reference) )
            diffs = b_pos_subset - ref_pos
            
            # Apply Minimum Image Convention to the diffs manually or via ASE
            # (assuming orthogonal/simple cells for simplicity, but ASE's find_mic is safer)
            diffs_mic = structure.get_distances(
                data['b_indices'][0], 
                data['b_indices'], 
                mic=True, 
                vector=True
            )
            
            # Reconstruct positions relative to the anchor, then average
            unwrapped_subset = ref_pos + diffs_mic
            center = np.mean(unwrapped_subset, axis=0)
            
            # Optional: Wrap the center back into the cell for consistency
            # (Only needed if you want the center point strictly inside the box)
            center = self._wrap_point(center, cell, pbc) 
            # --- FIX END ---
            
            cluster_size = len(data['b_indices'])
            
            results.append({
                'indices': all_indices, 
                'center': center,       
                'type': f'B{cluster_size}'
            })
            
        return results

    # =========================================================================
    # OPERATOR 1: PERTURB LATTICE 
    # =========================================================================
    def op_perturb_lattice(self, structure: Atoms) -> Atoms:
        """Standard lattice perturbation preserving fractional coordinates."""
        new_struct = structure.copy()
        factors = np.random.uniform(0.9, 1.2, size=3) # -10% to +20%
        current_cell = new_struct.get_cell()
        new_cell = current_cell * factors[:, np.newaxis]
        new_struct.set_cell(new_cell, scale_atoms=True)
        return new_struct

    # =========================================================================
    # OPERATOR 2: PERTURB MOTIFS (Translation)
    # =========================================================================
    def op_perturb_motifs(self, structure: Atoms) -> Optional[Atoms]:
        """
        Translates cages slightly.
        """
        # Work on copy
        child = structure.copy()
        
        # Identify Motifs
        try:
            motifs = self._identify_motifs_with_dbscan(child, eps=self.eps_dbscan, 
                                                       min_samples=self.min_samples_dbscan)
        except Exception as e:
            print(f"DBSCAN failed: {e}")
            return None
            
        if not motifs: return None

        # Track updated centers to prevent collisions
        updated_centers = [m['center'] for m in motifs]
        
        for i, motif in enumerate(motifs):
            indices = motif['indices']
            current_center = motif['center']
            
            success = False
            for _ in range(100): # Max retries
                move_vec = np.random.uniform(-self.max_perturbation, self.max_perturbation, 3)
                new_center = current_center + move_vec
                
                # Check collision with OTHER cages
                collision = False
                for j, other_center in enumerate(updated_centers):
                    if i == j: continue 
                    dist = self._pbc_distance(new_center, other_center, child.cell)
                    if dist < self.min_dist_anion_anion: 
                        collision = True
                        break
                
                if not collision:
                    child.positions[indices] += move_vec
                    updated_centers[i] = new_center
                    success = True
                    break
            
            if not success:
                pass # Leave at original position

        child.wrap()
        
        # Remove and Repopulate Cations using Voronoi
        del child[[atom.index for atom in child if atom.symbol == 'Na']]
        return self._repopulate_cations_voronoi(child, updated_centers)

    # =========================================================================
    # OPERATOR 3: ROTATE MOTIFS (Corrected for PBC)
    # =========================================================================
    def op_rotate_motifs(self, structure: Atoms) -> Optional[Atoms]:
        """
        Rotates cages in place, ensuring rigid body rotation even across boundaries.
        """
        child = structure.copy()
        try:
            motifs = self._identify_motifs_with_dbscan(child, eps=self.eps_dbscan,
                                                       min_samples=self.min_samples_dbscan)
        except Exception as e:
            print(f"[NBHBasinhopping.op_rotate_motifs] DBSCAN failed: {e}")
            return None

        if not motifs: return None

        centers = []
        for motif in motifs:
            indices = motif['indices']
            # We don't use the 'center' from DBSCAN here because we need 
            # the center of the SPECIFIC unwrapped image we act on below.
            
            # --- 1. Unwrap atoms to ensure they are contiguous ---
            ref_idx = indices[0]
            ref_pos = child.positions[ref_idx]
            
            # Get vectors from Reference -> All other atoms in motif (MIC aware)
            # This handles the boundary crossing correctly.
            diffs_mic = child.get_distances(
                ref_idx, 
                indices, 
                mic=True, 
                vector=True
            )
            
            # Reconstruct contiguous motif relative to the reference atom
            unwrapped_pos = ref_pos + diffs_mic
            
            # --- 2. Calculate Center of THIS unwrapped cluster ---
            # This is the critical fix. We define the center based on these specific coordinates.
            local_center = np.mean(unwrapped_pos, axis=0)
            
            # --- 3. Rotate around this local center ---
            # Shift to origin
            rel_pos = unwrapped_pos - local_center
            
            # Create temp atoms for rotation
            temp_atoms = Atoms('X'*len(indices), positions=rel_pos)
            angle = np.random.uniform(0, 360)
            axis = random.choice(['x', 'y', 'z']) 
            
            # Rotate
            temp_atoms.rotate(angle, axis, center=(0,0,0))
            
            # Shift back to the local center
            new_pos = temp_atoms.positions + local_center

            # Update positions in the child structure
            child.positions[indices] = new_pos
            
            # Save this center for Voronoi exclusion later
            # (It is safe to use this center because Voronoi distances 
            # should be calculated using PBC-aware distance functions anyway)
            centers.append(local_center)

        # Wrap everything at the end to put atoms back into the box
        child.wrap()
        
        # Remove and Repopulate Cations
        del child[[atom.index for atom in child if atom.symbol == 'Na']]
        
        return self._repopulate_cations_voronoi(child, centers)

    # =========================================================================
    # OPERATOR 4: COMBO (TRANSLATE + ROTATE)
    # =========================================================================
    def op_perturb_and_rotate_motifs(self, structure: Atoms) -> Optional[Atoms]:
        """
        Combined operator: Translates motifs first, then rotates them.
        """
        # 1. Perturb (Translate)
        translated_structure = self.op_perturb_motifs(structure)
        
        if translated_structure is None:
            return None
            
        # 2. Rotate
        # Pass the translated structure (which has regenerated Na) to the rotation operator.
        # It will strip Na, rotate the cages (which are now in new spots), and regen Na again.
        rotated_structure = self.op_rotate_motifs(translated_structure)
        
        return rotated_structure
    
    # =========================================================================
    # OPERATOR 5: RESHUFFLE CATIONS
    # =========================================================================
    def op_reshuffle_cations(self, structure: Atoms) -> Optional[Atoms]:
        """
        Keeps the anion lattice as is, but re-generates Na positions.
        """
        child = structure.copy()

        try:
            motifs = self._identify_motifs_with_dbscan(child, eps=self.eps_dbscan,
                                                       min_samples=self.min_samples_dbscan)
        except Exception as e:
            print(f"[NBHBasinhopping.op_reshuffle_cations] DBSCAN failed: {e}")
            return None

        centers = [m['center'] for m in motifs]

        del child[[atom.index for atom in child if atom.symbol == 'Na']]
        
        return self._repopulate_cations_voronoi(child, centers)

    # =========================================================================
    # NEW CATION PLACEMENT LOGIC (VORONOI)
    # =========================================================================
    def _repopulate_cations_voronoi(self, structure_no_na: Atoms, motif_centers: List[np.ndarray]) -> Optional[Atoms]:
        """
        Wrapper to call the Voronoi sampling method and append atoms.
        """
        # Ensure we have centers. If not passed (e.g. from lattice perturb), find them.
        if not motif_centers:
            motifs = self._identify_motifs_with_dbscan(structure_no_na, eps=self.eps_dbscan, 
                                                       min_samples=self.min_samples_dbscan)
            motif_centers = [m['center'] for m in motifs]

        new_na_coords = self._sample_na_sites_from_voronoi(
            structure=structure_no_na,
            motif_centers=motif_centers
        )
        
        if new_na_coords is None:
            return None
            
        for pos in new_na_coords:
            # pos is Cartesian, derived from Voronoi vertices
            structure_no_na.append(Atom('Na', position=pos))
            
        structure_no_na.wrap() # Ensure all new atoms are wrapped into the cell

        return structure_no_na

    def _sample_na_sites_from_voronoi(self,
                                      structure: Atoms,
                                      motif_centers: List[np.ndarray],
                                      max_attempts: int = 5000) -> Optional[List[np.ndarray]]:
        """
        Sample Na positions from Voronoi vertices of H atoms.
        """
        # Determine number of Na atoms from motifs
        num_na = 2 * len(motif_centers)
            
        motif_radius = self.motif_radius
        min_na_na_dist = self.min_dist_cation_cation
        min_na_h_dist = self.min_dist_cation_H

        # Get all H atom positions
        H_positions = [atom.position for atom in structure if atom.symbol == "H"]
        if len(H_positions) < 4:
            # Need at least 4 points for 3D Voronoi
            return None

        # Compute Voronoi vertices
        try:
            vor = Voronoi(H_positions)
        except Exception as e:
            print(f"[NBHBasinhopping] Voronoi decomposition failed on H positions: {e}")
            return None
            
        candidate_sites = vor.vertices
        cell = structure.get_cell()

        # 1. Filter out vertices that are inside any BU/Motif sphere
        filtered_by_motif = []
        for pt in candidate_sites:
            too_close = False
            for c in motif_centers:
                if self._pbc_distance(pt, c, cell) < motif_radius:
                    too_close = True
                    break
            if not too_close:
                filtered_by_motif.append(pt)
        
        filtered_by_motif = np.array(filtered_by_motif)
        if len(filtered_by_motif) < num_na:
            if self.verbose:
                print(f"[Debug] Not enough Voronoi sites outside motifs. Found {len(filtered_by_motif)}, need {num_na}.")
            return None
        
        # 2. Filter out vertices too close to any H atom
        # (This is distinct from the motif center check; checks actual atoms)
        final_candidates = []
        for pt in filtered_by_motif:
            too_close = False
            # Optimization: check against H_positions using array math if possible, 
            # but using loop for PBC safety as per existing pattern
            dists = self._get_pbc_distances_array(pt, np.array(H_positions), cell)
            if np.min(dists) < min_na_h_dist:
                too_close = True
            
            if not too_close:
                final_candidates.append(pt)
        
        if len(final_candidates) < num_na:
            return None

        # 3. Select well-separated Na sites from valid candidates
        selected = []
        attempts = 0
        
        while len(selected) < num_na and attempts < max_attempts:
            # Pick random index from final_candidates
            idx = np.random.randint(len(final_candidates))
            pt = final_candidates[idx]
            
            # Check against already selected Na
            valid = True
            for existing in selected:
                if self._pbc_distance(pt, existing, cell) < min_na_na_dist:
                    valid = False
                    break
            
            if valid:
                selected.append(pt)
            
            attempts += 1

        if len(selected) < num_na:
            if self.verbose:
                print(f"[Debug] Failed to place Na. Placed {len(selected)}/{num_na} after {max_attempts} attempts.")
            return None
            
        return selected


class NNOCStructureGenerator:
    def __init__(self, motifs_path: str = "", 
                 min_dist_nb: float = 4.5, 
                 min_dist_cl_cl: float = 3.0,
                 na_exclusion_radius: float = 2.0,
                 min_dist_na_na: float = 3.0,
                 max_dist_na_cl: float = 3.0,
                 target_na_ratio: float = 1.0,
                 probabilities: Dict[str, float] = None):
        
        self.motifs_path = motifs_path
        self.min_dist_nb = min_dist_nb
        self.min_dist_cl_cl = min_dist_cl_cl
        self.na_exclusion_radius = na_exclusion_radius
        self.min_dist_na_na = min_dist_na_na
        self.max_dist_na_cl = max_dist_na_cl
        self.target_na_ratio = target_na_ratio
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
        except Exception as e:
            print(f"[NNOCStructureGenerator] Voronoi decomposition failed on anion supercell: {e}")
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
        super().__init__(**kwargs)

        if 'probabilities' not in kwargs:
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