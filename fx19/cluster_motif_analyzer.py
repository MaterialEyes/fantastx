from dataclasses import dataclass
from typing import List, Optional, Dict, Union
import numpy as np
import os

from ase import Atoms
from ase.io import write

from sklearn.cluster import DBSCAN


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