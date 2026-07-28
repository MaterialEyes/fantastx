"""
mol_crystals.py — Perturbation operators for molecular crystal structure search.

Design principle
----------------
All operators perturb only the crystallographic asymmetric unit (the true
independent degrees of freedom), then apply the space-group symmetry operations
read directly from the CIF to regenerate the full unit cell.  The full cell is
what gets stored in the FANTASTX model record and passed to the XRD evaluator.

CSD export quirk handled automatically
---------------------------------------
CSD-exported CIFs for molecules sitting on crystallographic inversion centres
often list both halves of the molecule with a 'B'-suffix on the copy labels
(Cu1B, N2B, …).  The loader strips these duplicates so only the true asymmetric
unit atoms are kept.

Class hierarchy
---------------
MolCrystalGenerator
    └── MolCrystalBasinhopping
            ├── Compound1Ops   (P2₁/n, bis-hydroxo dicopper diperchlorate)
            └── Compound3Ops   (P2₁/c, terephthalato-dicopper diperchlorate)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import copy
import re

import numpy as np
from scipy.spatial.transform import Rotation as SciRot

from pymatgen.core import Lattice, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.io.cif import CifParser

from fx19 import structure_record


# ── module-level helpers ───────────────────────────────────────────────────────

def _strip_uncertainty(s: str) -> float:
    """Parse a CIF numeric field such as '8.619(4)' → 8.619."""
    return float(re.sub(r'\(.*?\)', '', s.strip()))


# Covalent radii (Å) used for bond detection in _identify_rigid_bodies.
_COVALENT_RADII: Dict[str, float] = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66,
    'Cl': 1.02, 'Cu': 1.32, 'S': 1.05, 'P': 1.07,
    'F': 0.57, 'Br': 1.20, 'I': 1.39,
}
_DEFAULT_COVALENT_RADIUS = 1.00

# Operators that return a full unit cell directly (not an asymmetric unit).
# get_model skips _expand_to_full_cell for these.
_WHOLE_CELL_OPS: frozenset = frozenset({'whole_cell_counterion',
                                        'whole_cell_rigid_bodies'})


def _min_inter_dist(elem_i: str, elem_j: str, scale: float) -> float:
    """Minimum allowed intermolecular distance for an element pair (Å).
    Uses covalent radii (same convention as _min_intra_dist), not vdW radii:
    vdW-radii sums reject legitimate hydrogen-bond contacts (~1.7-2.2 Å,
    routinely shorter than vdW sums) as false clashes. A scale of ~0.95-1.0
    on covalent radii catches genuine overlaps while tolerating contacts
    that are merely a bit close — relaxation pulls those back to their
    ideal distance afterward unless they're close enough to trigger strong
    repulsion."""
    ri = _COVALENT_RADII.get(elem_i, _DEFAULT_COVALENT_RADIUS)
    rj = _COVALENT_RADII.get(elem_j, _DEFAULT_COVALENT_RADIUS)
    return scale * (ri + rj)


def _min_intra_dist(elem_i: str, elem_j: str, scale: float) -> float:
    """Minimum allowed intramolecular distance for an element pair (Å).
    Uses covalent radii — smaller than vdW, so only true atom overlaps are
    caught without constraining physically valid bond lengths."""
    ri = _COVALENT_RADII.get(elem_i, _DEFAULT_COVALENT_RADIUS)
    rj = _COVALENT_RADII.get(elem_j, _DEFAULT_COVALENT_RADIUS)
    return scale * (ri + rj)


# ══════════════════════════════════════════════════════════════════════════════
# Generator — CIF loading, symmetry expansion, pool seeding
# ══════════════════════════════════════════════════════════════════════════════

class MolCrystalGenerator:
    """
    Loads a CSD-exported CIF, extracts the true asymmetric unit by stripping
    symmetry-duplicate 'B'-suffix atoms, reads the space-group operations, and
    provides pool-seeding via small atomic displacements followed by full-cell
    expansion.

    The asymmetric unit is stored as a pymatgen Structure in self.asym_unit.
    The list of SymmOp objects is stored in self.sym_ops.
    """

    def __init__(
        self,
        starting_structure_path: str,
        max_seed_displacement: float = 0.05,
        seed_fraction: float = 0.30,
        max_attempts: int = 50,
        **kwargs,
    ):
        """
        Args:
            starting_structure_path: Path to a CIF file (CSD-exported or
                pre-cleaned).  B-suffix symmetry duplicates are stripped
                automatically.
            max_seed_displacement: Maximum Cartesian displacement (Å) per atom
                when generating seed structures for the initial pool.
            seed_fraction: Fraction of asymmetric-unit atoms displaced during
                pool seeding.
            max_attempts: Maximum retries for generating a valid structure.
            **kwargs: Forwarded to subclasses and stored as instance attributes.
        """
        self.starting_structure_path = starting_structure_path
        self.max_seed_displacement = max_seed_displacement
        self.seed_fraction = seed_fraction
        self.max_attempts = max_attempts

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.asym_unit, self.sym_ops = self._load_asymmetric_unit()
        self.n_asym = self.asym_unit.num_sites

    # ── CIF loading ────────────────────────────────────────────────────────────

    def _load_asymmetric_unit(self) -> Tuple[Structure, List[SymmOp]]:
        """
        Reads the CIF, strips CSD-generated symmetry-duplicate atoms (labels
        matching the pattern '<digits>B' at the end, e.g. Cu1B, H22B), and
        returns:
            asym_unit — pymatgen Structure containing only the true asymmetric
                        unit atoms, with the lattice read from the CIF.
            sym_ops   — list of pymatgen SymmOp built from the CIF's
                        _symmetry_equiv_pos_as_xyz block.
        """
        import os
        if not os.path.exists(self.starting_structure_path):
            raise FileNotFoundError(
                f"CIF not found: {self.starting_structure_path}"
            )

        parser = CifParser(self.starting_structure_path)
        raw = list(parser.as_dict().values())[0]

        # ── lattice ───────────────────────────────────────────────────────────
        a     = _strip_uncertainty(raw['_cell_length_a'])
        b     = _strip_uncertainty(raw['_cell_length_b'])
        c     = _strip_uncertainty(raw['_cell_length_c'])
        alpha = _strip_uncertainty(raw['_cell_angle_alpha'])
        beta  = _strip_uncertainty(raw['_cell_angle_beta'])
        gamma = _strip_uncertainty(raw['_cell_angle_gamma'])
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

        # ── atom sites ────────────────────────────────────────────────────────
        labels  = raw['_atom_site_label']
        symbols = raw['_atom_site_type_symbol']
        xs = [_strip_uncertainty(v) for v in raw['_atom_site_fract_x']]
        ys = [_strip_uncertainty(v) for v in raw['_atom_site_fract_y']]
        zs = [_strip_uncertainty(v) for v in raw['_atom_site_fract_z']]

        # A B-suffix copy has a label ending in a non-zero-leading number then 'B',
        # e.g. Cu1B, N3B, C1B, H22B.  SHELXL hex-count labels (C00B, H00B, H01B)
        # have a leading-zero number and are NOT matched — they are real atoms.
        # No boron in any compound handled here, so a trailing B is unambiguous.
        _b_copy = re.compile(r'[1-9]\d*B$')

        asym_species, asym_coords = [], []
        for label, sym, x, y, z in zip(labels, symbols, xs, ys, zs):
            if _b_copy.search(label):
                continue
            asym_species.append(sym)
            asym_coords.append([x, y, z])

        asym_unit = Structure(
            lattice, asym_species, asym_coords, coords_are_cartesian=False
        )

        # ── symmetry operations ───────────────────────────────────────────────
        sym_xyz = raw.get(
            '_symmetry_equiv_pos_as_xyz',
            raw.get('_space_group_symop_operation_xyz', [])
        )
        if not sym_xyz:
            raise ValueError(
                "No symmetry operations found in CIF — check key names."
            )
        sym_ops = [SymmOp.from_xyz_str(s.strip()) for s in sym_xyz]

        print(
            f"[MolCrystalGenerator] {os.path.basename(self.starting_structure_path)}: "
            f"{asym_unit.num_sites}-atom asymmetric unit, "
            f"{len(sym_ops)} symmetry operations."
        )
        return asym_unit, sym_ops

    # ── symmetry expansion ─────────────────────────────────────────────────────

    def _expand_to_full_cell(self, asym_unit: Structure) -> Structure:
        """
        Applies every crystallographic symmetry operation to the asymmetric
        unit, wraps fractional coordinates to [0, 1), and returns the full
        unit cell as a sorted pymatgen Structure.
        """
        lattice = asym_unit.lattice
        species_all, coords_all = [], []

        for site in asym_unit.sites:
            for op in self.sym_ops:
                new_frac = op.operate(site.frac_coords) % 1.0
                species_all.append(site.specie)
                coords_all.append(new_frac)

        full_cell = Structure(
            lattice, species_all, coords_all, coords_are_cartesian=False
        )
        full_cell.sort()
        return full_cell

    # ── internal perturbation helper ───────────────────────────────────────────

    def _perturb_asym_unit(
        self,
        asym_unit: Structure,
        fraction: float,
        max_displacement: float,
    ) -> Structure:
        """
        Returns a copy of asym_unit with random Cartesian displacements applied
        to a random fraction of its atoms.  No distance checks are performed —
        this is intended for small (seed-level) perturbations only.
        """
        n = asym_unit.num_sites
        n_perturb = max(1, int(n * fraction))
        indices = np.random.choice(n, n_perturb, replace=False)

        cart = asym_unit.cart_coords.copy()
        for idx in indices:
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            cart[idx] += direction * np.random.uniform(0.0, max_displacement)

        return Structure(
            asym_unit.lattice, asym_unit.species, cart,
            coords_are_cartesian=True,
        )

    # ── generator interface ────────────────────────────────────────────────────

    def generate_random_structure(self) -> Optional[Structure]:
        """
        Applies a small random perturbation to the stored asymmetric unit and
        expands to the full unit cell.  Used to populate the initial pool.
        """
        perturbed = self._perturb_asym_unit(
            self.asym_unit, self.seed_fraction, self.max_seed_displacement
        )
        return self._expand_to_full_cell(perturbed)

    def random_model(self, reg_id) -> Optional[structure_record.model]:
        """
        Generates an initial-population model using the same basin-hopping
        operators as get_model(), treating the reference starting structure as
        the implicit parent.  This gives the same structural diversity as
        evolved candidates and avoids the duplicate-fingerprint problem that
        arises when many near-identical tiny perturbations fill the initial pool.
        """
        child_asym, chosen_op = self.select_and_apply_operator(self.asym_unit)
        if child_asym is None:
            return None

        if chosen_op in _WHOLE_CELL_OPS:
            child_full = child_asym
            store_asym = None
        else:
            child_full = self._expand_to_full_cell(child_asym)
            store_asym = child_asym

        rand_model = structure_record.model(child_full, reg_id)
        rand_model.inheritance = 'random'
        rand_model.made_by = chosen_op
        rand_model.asym_unit = store_asym
        return rand_model


# ══════════════════════════════════════════════════════════════════════════════
# Basin-hopping operators
# ══════════════════════════════════════════════════════════════════════════════

class MolCrystalBasinhopping(MolCrystalGenerator):
    """
    Five perturbation operators for molecular crystal structure search, all
    acting on the crystallographic asymmetric unit:

        perturb_atoms          — nudge individual atomic positions.
        perturb_lattice        — strain cell lengths; perturb free angles.
        rigid_translate        — translate one molecular fragment rigidly.
        rigid_rotate           — rotate one molecular fragment about its centroid.
        rigid_translate_rotate — combined translation + rotation.

    Rigid molecular fragments are detected automatically via covalent-bond
    connectivity (BFS).  ClO4⁻ counterions are identified by the presence of
    a Cl atom in the connected component.

    Every operator returns a perturbed asymmetric unit.  get_model() expands
    it to the full cell before building the model record.
    """

    def __init__(self, **kwargs):
        """
        Args:
            probabilities (dict): Operator selection weights (auto-normalised).
                Keys: 'perturb_atoms', 'perturb_lattice', 'rigid_translate',
                'rigid_rotate', 'rigid_translate_rotate'.
            max_atom_displacement (float): Max per-atom displacement in
                op_perturb_atoms (Å).  Default 0.15.
            perturb_atom_fraction (float): Fraction of asymmetric-unit atoms
                displaced in op_perturb_atoms.  Default 0.30.
            max_lattice_strain (float): Maximum fractional strain per cell
                length (e.g. 0.03 = 3 %).  Default 0.03.
            max_angle_perturbation (float): Maximum absolute perturbation of
                each free cell angle (degrees).  Angles within 0.1° of 90° are
                treated as fixed by symmetry and never perturbed.  Default 1.0.
            max_translation (float): Maximum rigid-body translation distance
                (Å).  Default 0.50.
            max_rotation_angle (float): Maximum rigid-body rotation angle
                (degrees).  Default 10.0.
            bond_tolerance (float): Tolerance added to the sum of covalent
                radii when detecting covalent bonds (Å).  Default 0.40.
            inter_scale (float): Intermolecular clash threshold expressed as a
                fraction of the covalent radii sum for each atom pair.
                E.g. 0.95 means atoms of different rigid bodies must be at
                least 0.95*(r_cov_i + r_cov_j) apart.  Default 0.95.
                Covalent (not vdW) radii are used deliberately: vdW-radii
                sums reject legitimate hydrogen-bond contacts as false
                clashes. A bit of closeness beyond the ideal distance is
                tolerated here — relaxation pulls it back out — only
                genuine overlap (strong repulsion) is rejected.
                Typical pair thresholds: H–H 0.59 Å, C–H 1.02 Å,
                C–C 1.44 Å, N–O 1.30 Å.
            intra_scale (float): Intramolecular overlap threshold expressed as
                a fraction of the covalent radii sum for each atom pair.
                Catches only true atom overlaps within a fragment, not normal
                bond compression.  Default 0.85.
                Typical pair thresholds: C–H 0.91 Å, C–C 1.29 Å,
                N–H 0.87 Å, Cu–N 1.73 Å.
            **kwargs: Forwarded to MolCrystalGenerator.
        """
        super().__init__(**kwargs)

        self.probabilities: dict = kwargs.get('probabilities', {
            'perturb_atoms':          0.20,
            'perturb_lattice':        0.20,
            'rigid_translate':        0.25,
            'rigid_rotate':           0.20,
            'rigid_translate_rotate': 0.15,
            'whole_cell_counterion':  0.00,
            'whole_cell_rigid_bodies': 0.00,
        })
        self.max_atom_displacement: float      = kwargs.get('max_atom_displacement', 0.15)
        self.whole_cell_max_translation: float = kwargs.get('whole_cell_max_translation', 1.0)
        self.perturb_atom_fraction: float  = kwargs.get('perturb_atom_fraction', 0.30)
        self.max_lattice_strain: float     = kwargs.get('max_lattice_strain', 0.03)
        self.max_angle_perturbation: float = kwargs.get('max_angle_perturbation', 1.0)
        self.max_translation: float        = kwargs.get('max_translation', 0.50)
        self.max_rotation_angle: float     = kwargs.get('max_rotation_angle', 10.0)
        self.bond_tolerance: float         = kwargs.get('bond_tolerance', 0.40)
        self.inter_scale: float            = kwargs.get('inter_scale', 0.95)
        self.intra_scale: float            = kwargs.get('intra_scale', 0.85)
        self._perturbable_elements: frozenset = frozenset(
            kwargs.get('perturbable_elements', ['H'])
        )

        # Cache rigid body decomposition of the reference asymmetric unit.
        # Keys are fragment names; values are sorted atom-index lists.
        self._rigid_bodies_cache: Optional[Dict[str, List[int]]] = None
        self._rigid_bodies_cache_species: List[str] = []
        self._adj_cache: Optional[List[List[int]]] = None
        self._protected_cache: Optional[frozenset] = None

    # ── main entry point ───────────────────────────────────────────────────────

    def get_model(self, select, pool, reg_id) -> Optional[structure_record.model]:
        """
        Called by run_ops.make_model().  Selects a parent from the pool,
        retrieves its asymmetric unit, applies a perturbation operator, expands
        to the full cell, and returns a FANTASTX model record.
        """
        parent_model = select.get_a_parent(pool)
        parent = copy.deepcopy(parent_model)

        # Prefer the cached asymmetric unit stored on the parent model;
        # fall back to the reference starting structure.
        if hasattr(parent, 'asym_unit') and parent.asym_unit is not None:
            parent_asym = parent.asym_unit
        else:
            parent_asym = self.asym_unit

        child_asym, chosen_op = self.select_and_apply_operator(parent_asym)
        if child_asym is None:
            return None

        if chosen_op in _WHOLE_CELL_OPS:
            # Operator already returned the full cell (symmetry broken).
            # Skip _expand_to_full_cell.  Offspring of this model revert to
            # the reference asymmetric unit because P1 → ASU recovery is not
            # implemented (asym_unit = None triggers the fallback in get_model).
            child_full = child_asym
            store_asym = None
        else:
            child_full = self._expand_to_full_cell(child_asym)
            store_asym = child_asym

        child_model = structure_record.model(child_full, reg_id)
        child_model.inheritance = [parent.label]
        child_model.made_by = chosen_op
        child_model.asym_unit = store_asym
        return child_model

    # ── operator dispatcher ────────────────────────────────────────────────────

    def select_and_apply_operator(
        self, asym_unit: Structure
    ) -> Tuple[Optional[Structure], str]:
        """Selects an operator by normalised probability weight and applies it."""
        ops   = list(self.probabilities.keys())
        probs = np.array(list(self.probabilities.values()), dtype=float)
        probs /= probs.sum()
        chosen_op = str(np.random.choice(ops, p=probs))

        dispatch = {
            'perturb_atoms':          self.op_perturb_atoms,
            'perturb_lattice':        self.op_perturb_lattice,
            'rigid_translate':        self.op_rigid_translate,
            'rigid_rotate':           self.op_rigid_rotate,
            'rigid_translate_rotate': self.op_rigid_translate_rotate,
            'whole_cell_counterion':  self.op_whole_cell_counterion,
            'whole_cell_rigid_bodies': self.op_whole_cell_rigid_bodies,
        }
        fn = dispatch.get(chosen_op)
        if fn is None:
            print(f"[MolCrystalBasinhopping] Unknown operator: {chosen_op}")
            return None, chosen_op

        result = fn(asym_unit)
        if result is None:
            print(f"[MolCrystalBasinhopping] {chosen_op} failed to find a "
                  f"valid move after {self.max_attempts} attempts "
                  f"(clash/overlap rejected every try) — resampling.")
        return result, chosen_op

    # ── operators ──────────────────────────────────────────────────────────────

    def op_perturb_atoms(self, asym_unit: Structure) -> Optional[Structure]:
        """
        Nudges a random fraction of asymmetric-unit atoms by small random
        Cartesian vectors.  Only intermolecular clashes (atoms in *different*
        rigid bodies) are checked using element-pair covalent-radii thresholds.
        Intramolecular distances are not checked: the displacement is at most
        max_atom_displacement (≤ 0.15 Å by default), which cannot create an
        overlap within a covalently bonded fragment starting from a valid
        crystal structure.
        """
        rigid_bodies = self._identify_rigid_bodies(asym_unit)
        atom_to_frag = {
            idx: name
            for name, indices in rigid_bodies.items()
            for idx in indices
        }
        species = [s.symbol for s in asym_unit.species]

        protected = self._protected_atom_indices()
        candidate_indices = [i for i in range(asym_unit.num_sites) if i not in protected]
        if not candidate_indices:
            return None
        n_perturb = max(1, int(len(candidate_indices) * self.perturb_atom_fraction))
        to_perturb = list(
            np.random.choice(candidate_indices,
                             min(n_perturb, len(candidate_indices)),
                             replace=False)
        )

        n = asym_unit.num_sites
        cart = asym_unit.cart_coords.copy()
        lattice = asym_unit.lattice

        for idx in to_perturb:
            my_frag = atom_to_frag[idx]
            inter = [j for j in range(n) if atom_to_frag[j] != my_frag]
            intra = [j for j in range(n) if atom_to_frag[j] == my_frag and j != idx]
            my_elem = species[idx]

            for _ in range(self.max_attempts):
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                candidate = cart[idx] + direction * np.random.uniform(
                    0.0, self.max_atom_displacement
                )

                ok = all(
                    self._cart_dist_pbc(candidate, cart[j], lattice)
                    >= _min_inter_dist(my_elem, species[j], self.inter_scale)
                    for j in inter
                )
                if ok:
                    ok = all(
                        self._cart_dist_pbc(candidate, cart[j], lattice)
                        >= _min_intra_dist(my_elem, species[j], self.intra_scale)
                        for j in intra
                    )
                if ok:
                    cart[idx] = candidate
                    break
            # If no valid position found within max_attempts, atom stays put.

        return Structure(
            asym_unit.lattice, asym_unit.species, cart,
            coords_are_cartesian=True,
        )

    def op_perturb_lattice(self, asym_unit: Structure) -> Optional[Structure]:
        """
        Independently strains each cell length by a random fraction and
        perturbs each *free* cell angle by a small random delta.  Angles
        within 0.1° of 90° are treated as fixed by symmetry (monoclinic
        α=γ=90° constraint).  Fractional coordinates are preserved so atoms
        follow the cell deformation.
        """
        lat = asym_unit.lattice
        a, b, c         = lat.a, lat.b, lat.c
        alpha, beta, gamma = lat.alpha, lat.beta, lat.gamma

        strains = np.random.uniform(
            -self.max_lattice_strain, self.max_lattice_strain, 3
        )
        a_new = a * (1.0 + strains[0])
        b_new = b * (1.0 + strains[1])
        c_new = c * (1.0 + strains[2])

        def _perturb_angle(angle: float) -> float:
            if abs(angle - 90.0) < 0.1:      # fixed by symmetry
                return 90.0
            delta = np.random.uniform(
                -self.max_angle_perturbation, self.max_angle_perturbation
            )
            return float(np.clip(angle + delta, 60.0, 120.0))

        alpha_new = _perturb_angle(alpha)
        beta_new  = _perturb_angle(beta)
        gamma_new = _perturb_angle(gamma)

        frac_coords = asym_unit.frac_coords.copy()
        new_lattice = Lattice.from_parameters(
            a_new, b_new, c_new, alpha_new, beta_new, gamma_new
        )
        return Structure(
            new_lattice, asym_unit.species, frac_coords,
            coords_are_cartesian=False,
        )

    def op_rigid_translate(self, asym_unit: Structure) -> Optional[Structure]:
        """
        Picks one rigid-body fragment at random and translates all its atoms
        by the same random Cartesian vector.  Checks that no atom of the moved
        fragment clashes with any other atom in the asymmetric unit.
        Returns None if no valid translation is found within max_attempts.
        """
        rigid_bodies = self._identify_rigid_bodies(asym_unit)
        frag_name = np.random.choice(list(rigid_bodies.keys()))
        frag_idx  = rigid_bodies[frag_name]
        other_idx = [i for i in range(asym_unit.num_sites) if i not in set(frag_idx)]

        cart    = asym_unit.cart_coords.copy()
        lattice = asym_unit.lattice
        frag_cart   = self._unwrap_fragment_cart(asym_unit, frag_idx)
        other_cart  = cart[other_idx]
        frag_species  = [asym_unit.species[i].symbol for i in frag_idx]
        other_species = [asym_unit.species[i].symbol for i in other_idx]

        for _ in range(self.max_attempts):
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            delta = direction * np.random.uniform(0.0, self.max_translation)
            new_frag = frag_cart + delta

            if self._check_inter_dists(new_frag, frag_species, other_cart, other_species, lattice):
                cart[frag_idx] = new_frag
                inv = np.linalg.inv(lattice.matrix)
                frac_all = cart @ inv % 1.0
                return Structure(
                    lattice, asym_unit.species, frac_all, coords_are_cartesian=False
                )
        return None

    def op_rigid_rotate(self, asym_unit: Structure) -> Optional[Structure]:
        """
        Picks one rigid-body fragment at random and rotates all its atoms by
        a random small rotation about the fragment's Cartesian centroid.
        Returns None if no valid orientation is found within max_attempts.
        """
        rigid_bodies = self._identify_rigid_bodies(asym_unit)
        frag_name = np.random.choice(list(rigid_bodies.keys()))
        frag_idx  = rigid_bodies[frag_name]
        other_idx = [i for i in range(asym_unit.num_sites) if i not in set(frag_idx)]

        cart    = asym_unit.cart_coords.copy()
        lattice = asym_unit.lattice
        frag_cart  = self._unwrap_fragment_cart(asym_unit, frag_idx)
        other_cart = cart[other_idx]
        centroid   = frag_cart.mean(axis=0)
        frag_species  = [asym_unit.species[i].symbol for i in frag_idx]
        other_species = [asym_unit.species[i].symbol for i in other_idx]

        for _ in range(self.max_attempts):
            angle = np.random.uniform(0.0, np.deg2rad(self.max_rotation_angle))
            axis  = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            rot   = SciRot.from_rotvec(axis * angle)
            rotated = rot.apply(frag_cart - centroid) + centroid

            if self._check_inter_dists(rotated, frag_species, other_cart, other_species, lattice):
                cart[frag_idx] = rotated
                inv = np.linalg.inv(lattice.matrix)
                frac_all = cart @ inv % 1.0
                return Structure(
                    lattice, asym_unit.species, frac_all, coords_are_cartesian=False
                )
        return None

    def op_rigid_translate_rotate(
        self, asym_unit: Structure
    ) -> Optional[Structure]:
        """
        Picks one rigid-body fragment at random and applies both a random
        rotation (about its centroid) and a random translation simultaneously.
        The combined move explores a larger region of configuration space than
        either operator alone.
        Returns None if no valid pose is found within max_attempts.
        """
        rigid_bodies = self._identify_rigid_bodies(asym_unit)
        frag_name = np.random.choice(list(rigid_bodies.keys()))
        frag_idx  = rigid_bodies[frag_name]
        other_idx = [i for i in range(asym_unit.num_sites) if i not in set(frag_idx)]

        cart    = asym_unit.cart_coords.copy()
        lattice = asym_unit.lattice
        frag_cart  = self._unwrap_fragment_cart(asym_unit, frag_idx)
        other_cart = cart[other_idx]
        centroid   = frag_cart.mean(axis=0)
        frag_species  = [asym_unit.species[i].symbol for i in frag_idx]
        other_species = [asym_unit.species[i].symbol for i in other_idx]

        for _ in range(self.max_attempts):
            # rotation
            angle = np.random.uniform(0.0, np.deg2rad(self.max_rotation_angle))
            axis  = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            rot   = SciRot.from_rotvec(axis * angle)
            rotated = rot.apply(frag_cart - centroid) + centroid

            # translation
            t_dir = np.random.randn(3)
            t_dir /= np.linalg.norm(t_dir)
            new_frag = rotated + t_dir * np.random.uniform(0.0, self.max_translation)

            if self._check_inter_dists(new_frag, frag_species, other_cart, other_species, lattice):
                cart[frag_idx] = new_frag
                inv = np.linalg.inv(lattice.matrix)
                frac_all = cart @ inv % 1.0
                return Structure(
                    lattice, asym_unit.species, frac_all, coords_are_cartesian=False
                )
        return None

    # ── whole-cell counterion/solvent operator ────────────────────────────────

    def op_whole_cell_counterion(self, asym_unit: Structure) -> Optional[Structure]:
        """
        Rigid-body translate / rotate / both on a randomly selected ClO4⁻ or
        H2O unit in the *full* unit cell.

        Unlike the other operators, this one:
          - Expands the asymmetric unit to the full cell internally.
          - Operates independently on each symmetry copy of ClO4⁻ / H2O,
            intentionally breaking the space-group symmetry (P2₁/c → P1).
          - Returns the modified full cell directly.  get_model() detects this
            via _WHOLE_CELL_OPS and skips _expand_to_full_cell.

        Scientific rationale
        --------------------
        H atoms are invisible to XRD (Z=1, negligible scattering factor).
        Rotating a water molecule or translating a perchlorate does not directly
        improve the XRD score.  However, the subsequent MLIP relaxation (no
        space-group constraint in ASE) moves ALL atoms to a new local minimum
        shaped by the new counterion/solvent environment.  This can shift the
        heavy-atom framework into a basin that ASU-level operators cannot reach
        because the crystallographic constraint ties all 4 copies together.

        Parameters (all configurable via YAML)
        ---------------------------------------
        whole_cell_max_translation : float  max translate distance in Å (default 3.0)
        max_rotation_angle is not used here; rotation is always drawn from [0°, 360°).
        """
        full    = self._expand_to_full_cell(asym_unit)
        n       = full.num_sites
        species = [s.symbol for s in full.species]
        frac    = full.frac_coords.copy()
        lat_mat = full.lattice.matrix
        inv_mat = np.linalg.inv(lat_mat)
        lattice = full.lattice

        # Build full-cell bond graph using vectorised distance computation.
        cov_r   = np.array([_COVALENT_RADII.get(s, _DEFAULT_COVALENT_RADIUS)
                            for s in species])
        cutoffs = (cov_r[:, None] + cov_r[None, :] + self.bond_tolerance)  # (n,n)

        df      = frac[:, None, :] - frac[None, :, :]          # (n,n,3)
        df     -= np.round(df)
        dists   = np.linalg.norm(df @ lat_mat, axis=2)          # (n,n)
        np.fill_diagonal(dists, np.inf)
        bonded  = dists < cutoffs                                # (n,n) bool

        adj: List[List[int]] = [list(np.where(bonded[i])[0]) for i in range(n)]

        # BFS: connected components.
        visited: set = set()
        components: List[List[int]] = []
        for start in range(n):
            if start in visited:
                continue
            comp: List[int] = []
            queue = [start]
            while queue:
                idx = queue.pop(0)
                if idx in visited:
                    continue
                visited.add(idx)
                comp.append(idx)
                queue.extend(x for x in adj[idx] if x not in visited)
            components.append(sorted(comp))

        # Label mobile units: perchlorate (has Cl) and water (O+H only, ≤3 atoms).
        mobile: Dict[str, List[int]] = {}
        perc_n = water_n = 0
        for comp in components:
            comp_sp = {species[i] for i in comp}
            if 'Cl' in comp_sp:
                perc_n += 1
                mobile[f'perchlorate_{perc_n}'] = comp
            elif comp_sp <= {'O', 'H'} and len(comp) <= 3:
                water_n += 1
                mobile[f'water_{water_n}'] = comp

        if not mobile:
            return None

        # Select all mobile units — each gets independent random parameters.
        # Moving all copies maximises diversity per call, compensating for the
        # fact that offspring of whole_cell_counterion models revert to the
        # reference ASU (symmetry-broken arrangement is not inherited).
        frag_names = list(mobile.keys())
        n_select   = max(1, len(frag_names))
        selected   = list(np.random.choice(frag_names, n_select, replace=False))

        selected_idx_set = set(idx for k in selected for idx in mobile[k])
        other_idx        = [i for i in range(n) if i not in selected_idx_set]
        cart_other       = full.cart_coords[other_idx]
        species_other    = [species[i] for i in other_idx]

        # Same sub-operation for all selected fragments; each gets independent
        # random parameters (direction / axis / angle).
        sub_op = str(np.random.choice(['translate', 'rotate', 'both']))

        for _ in range(self.max_attempts):
            new_frac           = frac.copy()
            moved_cart_parts   : List[np.ndarray] = []
            moved_species_parts: List[str]         = []

            for fname in selected:
                frag_idx = mobile[fname]

                # Unwrap fragment to contiguous Cartesian via BFS minimum-image.
                g2l: Dict[int, int] = {g: l for l, g in enumerate(frag_idx)}
                nf        = len(frag_idx)
                unwrapped = np.empty((nf, 3))
                unwrapped[0] = frac[frag_idx[0]]
                placed: set  = {frag_idx[0]}
                queue        = [frag_idx[0]]
                while queue:
                    gi = queue.pop(0)
                    li = g2l[gi]
                    for gj in adj[gi]:
                        if gj in g2l and gj not in placed:
                            lj            = g2l[gj]
                            d             = frac[gj] - unwrapped[li]
                            d            -= np.round(d)
                            unwrapped[lj] = unwrapped[li] + d
                            placed.add(gj)
                            queue.append(gj)
                new_cart = unwrapped @ lat_mat

                # Independent random parameters per fragment, same sub-op type.
                if sub_op in ('translate', 'both'):
                    direction  = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    new_cart  += direction * np.random.uniform(0.0, self.whole_cell_max_translation)

                if sub_op in ('rotate', 'both'):
                    axis      = np.random.randn(3)
                    axis     /= np.linalg.norm(axis)
                    angle_deg = np.random.uniform(0.0, 360.0)
                    R         = SciRot.from_rotvec(np.radians(angle_deg) * axis).as_matrix()
                    centroid  = new_cart.mean(axis=0)
                    new_cart  = centroid + (R @ (new_cart - centroid).T).T

                new_frac_frag = (new_cart @ inv_mat) % 1.0
                moved_cart_parts.append(new_frac_frag @ lat_mat)
                moved_species_parts.extend(species[i] for i in frag_idx)
                for li, gi in enumerate(frag_idx):
                    new_frac[gi] = new_frac_frag[li]

            # Clash check: all moved atoms vs everything not selected.
            moved_cart = np.vstack(moved_cart_parts)
            if self._check_inter_dists(
                moved_cart, moved_species_parts, cart_other, species_other, lattice
            ):
                return Structure(lattice, full.species, new_frac, coords_are_cartesian=False)

        return None

    def op_whole_cell_rigid_bodies(self, asym_unit: Structure) -> Optional[Structure]:
        """
        Rigid-body translate / rotate / both on ALL connected fragments in the
        full unit cell — including the main Cu-complex copies, not just counterions.

        Identical in mechanics to op_whole_cell_counterion but the mobile-unit
        filter accepts every connected component, so the Cu-complex copies are
        moved independently of one another.  This breaks translational symmetry
        across the full cell and lets the MLIP explore packings where adjacent
        complex copies have different orientations — the key hypothesis for
        compounds where peak positions are correct but long-range periodicity
        is broken (e.g., 60°-offset stacking).

        Like whole_cell_counterion, returns a P1 full cell; offspring revert to
        the reference ASU.
        """
        full    = self._expand_to_full_cell(asym_unit)
        n       = full.num_sites
        species = [s.symbol for s in full.species]
        frac    = full.frac_coords.copy()
        lat_mat = full.lattice.matrix
        inv_mat = np.linalg.inv(lat_mat)
        lattice = full.lattice

        cov_r   = np.array([_COVALENT_RADII.get(s, _DEFAULT_COVALENT_RADIUS)
                            for s in species])
        cutoffs = (cov_r[:, None] + cov_r[None, :] + self.bond_tolerance)

        df      = frac[:, None, :] - frac[None, :, :]
        df     -= np.round(df)
        dists   = np.linalg.norm(df @ lat_mat, axis=2)
        np.fill_diagonal(dists, np.inf)
        bonded  = dists < cutoffs

        adj: List[List[int]] = [list(np.where(bonded[i])[0]) for i in range(n)]

        visited: set = set()
        components: List[List[int]] = []
        for start in range(n):
            if start in visited:
                continue
            comp: List[int] = []
            queue = [start]
            while queue:
                idx = queue.pop(0)
                if idx in visited:
                    continue
                visited.add(idx)
                comp.append(idx)
                queue.extend(x for x in adj[idx] if x not in visited)
            components.append(sorted(comp))

        # All fragments are mobile — no element filter.
        mobile: Dict[str, List[int]] = {
            f'frag_{k}': comp for k, comp in enumerate(components)
        }

        if not mobile:
            return None

        frag_names = list(mobile.keys())
        n_select   = max(1, len(frag_names))
        selected   = list(np.random.choice(frag_names, n_select, replace=False))

        selected_idx_set = set(idx for k in selected for idx in mobile[k])
        other_idx        = [i for i in range(n) if i not in selected_idx_set]
        cart_other       = full.cart_coords[other_idx]
        species_other    = [species[i] for i in other_idx]

        sub_op = str(np.random.choice(['translate', 'rotate', 'both']))

        for _ in range(self.max_attempts):
            new_frac           = frac.copy()
            moved_cart_parts   : List[np.ndarray] = []
            moved_species_parts: List[str]         = []

            for fname in selected:
                frag_idx = mobile[fname]

                g2l: Dict[int, int] = {g: l for l, g in enumerate(frag_idx)}
                nf        = len(frag_idx)
                unwrapped = np.empty((nf, 3))
                unwrapped[0] = frac[frag_idx[0]]
                placed: set  = {frag_idx[0]}
                queue        = [frag_idx[0]]
                while queue:
                    gi = queue.pop(0)
                    li = g2l[gi]
                    for gj in adj[gi]:
                        if gj in g2l and gj not in placed:
                            lj            = g2l[gj]
                            d             = frac[gj] - unwrapped[li]
                            d            -= np.round(d)
                            unwrapped[lj] = unwrapped[li] + d
                            placed.add(gj)
                            queue.append(gj)
                new_cart = unwrapped @ lat_mat

                if sub_op in ('translate', 'both'):
                    direction  = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    new_cart  += direction * np.random.uniform(0.0, self.whole_cell_max_translation)

                if sub_op in ('rotate', 'both'):
                    axis      = np.random.randn(3)
                    axis     /= np.linalg.norm(axis)
                    angle_deg = np.random.uniform(0.0, 360.0)
                    R         = SciRot.from_rotvec(np.radians(angle_deg) * axis).as_matrix()
                    centroid  = new_cart.mean(axis=0)
                    new_cart  = centroid + (R @ (new_cart - centroid).T).T

                new_frac_frag = (new_cart @ inv_mat) % 1.0
                moved_cart_parts.append(new_frac_frag @ lat_mat)
                moved_species_parts.extend(species[i] for i in frag_idx)
                for li, gi in enumerate(frag_idx):
                    new_frac[gi] = new_frac_frag[li]

            moved_cart = np.vstack(moved_cart_parts)
            if self._check_inter_dists(
                moved_cart, moved_species_parts, cart_other, species_other, lattice
            ):
                return Structure(lattice, full.species, new_frac, coords_are_cartesian=False)

        return None

    # ── rigid-body detection ───────────────────────────────────────────────────

    def _identify_rigid_bodies(
        self, asym_unit: Structure
    ) -> Dict[str, List[int]]:
        """
        Detects covalently connected fragments in the asymmetric unit by BFS
        over a bond graph built from element covalent radii.  Distances are
        computed directly in Cartesian space (no PBC), which is correct because
        the asymmetric unit is a compact molecular fragment.

        Connected components containing a Cl atom are labelled 'perchlorate_N';
        all others are labelled 'complex_fragment_N'.

        Results are cached against the reference asymmetric unit; the cache is
        invalidated whenever the species list changes (which never happens in
        normal operation).
        """
        # Return cached result if available for this species set
        species = [s.symbol for s in asym_unit.species]
        if (self._rigid_bodies_cache is not None and
                len(self._rigid_bodies_cache_species) == len(species) and
                self._rigid_bodies_cache_species == species):
            return self._rigid_bodies_cache

        n    = asym_unit.num_sites
        frac = asym_unit.frac_coords
        lat_mat = asym_unit.lattice.matrix

        # Build adjacency list using minimum-image distances so that bonds
        # straddling a periodic boundary are correctly detected.
        adj: List[List[int]] = [[] for _ in range(n)]
        for i in range(n):
            ri = _COVALENT_RADII.get(species[i], _DEFAULT_COVALENT_RADIUS)
            for j in range(i + 1, n):
                rj = _COVALENT_RADII.get(species[j], _DEFAULT_COVALENT_RADIUS)
                cutoff = ri + rj + self.bond_tolerance
                diff_frac = frac[i] - frac[j]
                diff_frac -= np.round(diff_frac)   # minimum-image in fractional
                diff_cart = diff_frac @ lat_mat
                if np.linalg.norm(diff_cart) < cutoff:
                    adj[i].append(j)
                    adj[j].append(i)

        # BFS to find connected components
        visited: set = set()
        components: List[List[int]] = []
        for start in range(n):
            if start in visited:
                continue
            comp: List[int] = []
            queue = [start]
            while queue:
                idx = queue.pop(0)
                if idx in visited:
                    continue
                visited.add(idx)
                comp.append(idx)
                queue.extend(nb for nb in adj[idx] if nb not in visited)
            components.append(sorted(comp))

        # Label components
        rigid_bodies: Dict[str, List[int]] = {}
        perc_count, frag_count = 0, 0
        for comp in components:
            comp_species = {species[i] for i in comp}
            if 'Cl' in comp_species:
                perc_count += 1
                rigid_bodies[f'perchlorate_{perc_count}'] = comp
            else:
                frag_count += 1
                rigid_bodies[f'complex_fragment_{frag_count}'] = comp

        print(
            f"[MolCrystalBasinhopping] Rigid bodies detected: "
            + ", ".join(f"{k}({len(v)} atoms)" for k, v in rigid_bodies.items())
        )

        # Cache both body assignments and adjacency (needed for unwrapping)
        self._rigid_bodies_cache = rigid_bodies
        self._rigid_bodies_cache_species = species
        self._adj_cache = adj
        self._protected_cache = None   # invalidate when bond graph is rebuilt
        return rigid_bodies

    def _protected_atom_indices(self) -> frozenset:
        """
        Returns indices of asymmetric-unit atoms that op_perturb_atoms must
        not displace individually.

        Perturbable elements are controlled by self._perturbable_elements
        (default: {'H'}).  Any atom whose element is NOT in that set is
        protected.  Configure via 'perturbable_elements' in the YAML
        mol_crystal_constraints block, e.g.:
            perturbable_elements: ['Cu', 'N', 'H']

        Result is cached on first call.
        """
        if self._protected_cache is not None:
            return self._protected_cache
        species = [s.symbol for s in self.asym_unit.species]
        self._protected_cache = frozenset(
            i for i, elem in enumerate(species)
            if elem not in self._perturbable_elements
        )
        return self._protected_cache

    # ── fragment unwrapping ────────────────────────────────────────────────────

    def _unwrap_fragment_cart(
        self, asym_unit: Structure, frag_idx: List[int]
    ) -> np.ndarray:
        """
        Returns Cartesian coordinates of a fragment's atoms gathered into a
        single contiguous image with no periodic breaks.

        BFS over the cached bond graph: starting from the first atom, each
        bonded neighbour is placed in the image closest to its already-placed
        parent rather than its default wrapped position.  This guarantees that
        the centroid is physically meaningful and rotations/translations act on
        a coherent molecular unit even when the fragment straddles a cell
        boundary.
        """
        frac    = asym_unit.frac_coords
        lat_mat = asym_unit.lattice.matrix
        n_frag  = len(frag_idx)
        g2l     = {g: l for l, g in enumerate(frag_idx)}   # global → local idx

        unwrapped = np.empty((n_frag, 3))
        unwrapped[0] = frac[frag_idx[0]]
        visited  = {frag_idx[0]}
        queue    = [frag_idx[0]]

        while queue:
            g_par = queue.pop(0)
            l_par = g2l[g_par]
            for g_ch in self._adj_cache[g_par]:
                if g_ch not in g2l or g_ch in visited:
                    continue
                visited.add(g_ch)
                l_ch = g2l[g_ch]
                diff = frac[g_ch] - unwrapped[l_par]
                diff -= np.round(diff)          # minimum-image in fractional
                unwrapped[l_ch] = unwrapped[l_par] + diff
                queue.append(g_ch)

        return unwrapped @ lat_mat              # fractional → Cartesian

    # ── distance utilities ─────────────────────────────────────────────────────

    def _cart_dist_pbc(
        self, p1: np.ndarray, p2: np.ndarray, lattice: Lattice
    ) -> float:
        """PBC-aware Cartesian distance using the minimum-image convention."""
        diff = p1 - p2
        frac = lattice.get_fractional_coords(diff)
        frac -= np.round(frac)
        return float(np.linalg.norm(lattice.get_cartesian_coords(frac)))

    def _check_inter_dists(
        self,
        moved: np.ndarray,
        moved_species: List[str],
        others: np.ndarray,
        others_species: List[str],
        lattice: Lattice,
    ) -> bool:
        """
        Returns True iff every atom in `moved` is beyond its element-pair
        covalent-radii threshold from every atom in `others` under PBC.
        Threshold for pair (i, j) = inter_scale * (r_cov_i + r_cov_j).
        Uses minimum-image arithmetic for PBC.
        """
        if len(others) == 0:
            return True

        inv_mat = np.linalg.inv(lattice.matrix)
        others_arr = np.asarray(others)

        for i, p in enumerate(moved):
            diffs = others_arr - p
            frac  = diffs @ inv_mat
            frac -= np.round(frac)
            cart  = frac @ lattice.matrix
            dists = np.linalg.norm(cart, axis=1)
            for j, d in enumerate(dists):
                if d < _min_inter_dist(moved_species[i], others_species[j], self.inter_scale):
                    return False
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Compound-specific subclasses
# ══════════════════════════════════════════════════════════════════════════════

class Compound1Ops(MolCrystalBasinhopping):
    """
    Perturbation operators for Compound 1 (CUXJEK):
    bis(μ₂-hydroxo)-bis(TACN)-dicopper(II) diperchlorate.
    Space group P2₁/n (setting of No. 14), Z=2.
    Asymmetric unit: 41 atoms (36 non-B atoms + 5 ClO4⁻ atoms).

    All operator logic is inherited from MolCrystalBasinhopping.
    This subclass exists to carry compound-appropriate parameter defaults and
    to give a clear, named entry point for YAML wiring.
    """

    def __init__(self, **kwargs):
        """
        All kwargs are forwarded to MolCrystalBasinhopping.__init__.
        Compound-1-specific defaults are set here and can be overridden via
        the input YAML.
        """
        # Compound 1 has shorter cell axes than compound 3 — tighter lattice
        # perturbation keeps moves physically reasonable.
        kwargs.setdefault('max_lattice_strain',     0.03)
        kwargs.setdefault('max_angle_perturbation', 1.0)
        kwargs.setdefault('max_atom_displacement',  0.15)
        kwargs.setdefault('max_translation',        0.50)
        kwargs.setdefault('max_rotation_angle',     10.0)
        kwargs.setdefault('inter_scale', 0.95)
        kwargs.setdefault('intra_scale', 0.85)
        super().__init__(**kwargs)


class Compound3Ops(MolCrystalBasinhopping):
    """
    Perturbation operators for Compound 3 (GECNOR):
    (μ₂-terephthalato)-diaqua-bis(TACN)-dicopper(II) diperchlorate.
    Space group P2₁/c (setting of No. 14), Z=2.
    Asymmetric unit: 48 atoms (43 Cu-complex atoms + 5 ClO4⁻ atoms).

    All operator logic is inherited from MolCrystalBasinhopping.
    This subclass exists to carry compound-appropriate parameter defaults and
    to give a clear, named entry point for YAML wiring.
    """

    def __init__(self, **kwargs):
        """
        All kwargs are forwarded to MolCrystalBasinhopping.__init__.
        Compound-3-specific defaults are set here and can be overridden via
        the input YAML.
        """
        kwargs.setdefault('max_lattice_strain',     0.03)
        kwargs.setdefault('max_angle_perturbation', 1.0)
        kwargs.setdefault('max_atom_displacement',  0.15)
        kwargs.setdefault('max_translation',        0.50)
        kwargs.setdefault('max_rotation_angle',     10.0)
        kwargs.setdefault('inter_scale', 0.95)
        kwargs.setdefault('intra_scale', 0.85)
        super().__init__(**kwargs)


class Compound2Ops(MolCrystalBasinhopping):
    """
    Perturbation operators for Compound 2:
    [Cu4(Me3tacn)4(μ-dbc)(μ-O)2](ClO4)4 · 2H2O  (two ClO4 or ClO5 variants).
    Space group Pbca (No. 61), Z=4, Z'=0.5.
    Asymmetric unit: ~90 atoms (half the Cu4 complex + 2 Cl species + 1 H2O).

    All operator logic is inherited from MolCrystalBasinhopping:
    - op_whole_cell_counterion detects ClO4 and ClO5 (any Cl-containing fragment)
      and H2O (O+H, ≤3 atoms) automatically via bond connectivity.
    - op_perturb_atoms perturbs H atoms only (non-H protected).
    - op_perturb_lattice perturbs only a, b, c lengths; all angles are 90° in
      Pbca so the angle-skip logic (|angle - 90°| < 0.1°) leaves them fixed.
    - Z'=0.5: the ASU holds half the Cu4 complex; rigid_translate/rotate moves
      this fragment and _expand_to_full_cell generates the symmetry-related half.

    This subclass exists only to carry compound-appropriate parameter defaults.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('max_lattice_strain',     0.03)
        kwargs.setdefault('max_angle_perturbation', 1.0)
        kwargs.setdefault('max_atom_displacement',  0.15)
        kwargs.setdefault('max_translation',        0.50)
        kwargs.setdefault('max_rotation_angle',     10.0)
        kwargs.setdefault('inter_scale', 0.95)
        kwargs.setdefault('intra_scale', 0.85)
        super().__init__(**kwargs)