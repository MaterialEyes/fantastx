"""
This module contains classes which handle all genetic structure
operations for FANTASTX. Four basic types of mutations have been
included:

1. **Basin-hopping**: Perturb a fraction of atoms in a structure, each
 by a random distance in a random direction. Atoms cannot overlap, and
 the distance of perturbation cannot exceed a pre-determined threshhold.
2. **Cut-and-splice**: Take 2 structures, cut each structure along a plane,
 and create a child structure by splicing the left piece of structure A
 with the right piece of structure B.
3. **Composition mutation**: Perturb the composition of a structure by
 either adding or removing atoms. The number of atoms added or removed will
 either be random (up to 3 atoms per species), or will correspond to a set
 composition (e.g. AlO4).
4. **Mating by swap**: Take 2 structures and mate them by randomly combining
 sites from each parent.

All basin-hopping mutations are handled by the basinhopping class.
Other mutations are handled differently depending on the class of the
parent structure. Currently, these parent structures can be:

1. **Clusters** (handled by `Evolve`)
2. **Grain Boundaries** (handled by `gb_ops`)
3. **Surfaces** (handled by `surface_ops`)

!!! important
    **Molecules** are also supported in a limited fashion, as a sub-class
    of clusters. The only structural operation currently implemented
    for molecules is basin-hopping.
"""

from __future__ import division, unicode_literals, print_function
from typing import AnyStr
from pymatgen.core.structure import Structure, Lattice
from pymatgen.core.composition import Composition
from pymatgen.transformations.standard_transformations import \
    RotationTransformation
# from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from numpy.random import uniform as unif
import numpy as np
import random
import copy
import math
from math import asin, cos, sqrt, tan, pi

from fx19 import structure_record
from fx19 import distance_check as dc
import traceback


class Evolve(object):
    """
    A wrapper around mating and basinhopping classes. This is used to decide
    whether to do mating (GA) or mutation (basinhopping) to generate a new
    structure and calls either of these classes to generate a single model.
    This class is used in bulk, nanocluster and molecule geometry searches.
    """

    def __init__(self, mate, hop, evolve_params):
        """
        Args:

        mate (obj): a mating object

        hop (obj): a basinhopping object

        evolve_params (dict): The keys are the following:
                              (sum of the three fractions should be 1)
                              'num_species': (int) number of species
        """
        self.mate = mate
        self.hop = hop
        self.num_species = evolve_params['num_species']
        self.shape = evolve_params['shape']

        # Make species dicts as attributes
        # DU
        # If storing all species in a list, and they are in order
        self.species = []
        for sp in range(1, self.num_species+1):
            self.species.append(evolve_params['species' + str(sp)])

        # If storing as variables, and are labeled in order
        # for sp in range(1, self.num_species + 1):
        #     setattr(self, 'species' + str(sp),
        #             evolve_params['species' + str(sp)])

    def get_model(self, select, pool, reg_id):
        """
        Returns a new model made using either mating or basinhopping

        Args:

        select (obj): selection.Select object

        pool (obj): selection.Pool object

        reg_id (obj): structure_record.register_id() object
        """
        hop = self.hop
        mate = self.mate
        operator = np.random.choice(
            select.operators, p=select.operator_frequencies)

        correct_comp = False
        tries = 0
        if operator == "perturb_sites" or operator == "perturb_comp":
            print(f"Selecting a parent for {operator}")
            parent_model = select.get_a_parent(pool)
            label = parent_model.label
        while correct_comp is False and tries <= 10:
            tries += 1
            try:
                print(f"Trying operator {operator} on try {tries}")
                if operator == "perturb_sites":
                    new_astr, inheritance = hop.perturb_sites(
                        select, pool, model_id=label)

                elif operator == "perturb_comp":
                    new_astr, inheritance = hop.perturb_comp(
                        select, pool, model=parent_model)

                elif operator == "fraction_slice_same_cluster":
                    new_astr, inheritance = mate.mate_by_slicing(
                        select, pool, same_cluster=True)

                elif operator == "fraction_slice_dif_cluster":
                    new_astr, inheritance = mate.mate_by_slicing(
                        select, pool, same_cluster=False)

                elif operator == "fraction_slice":
                    new_astr, inheritance = mate.mate_by_slicing(select, pool)

                elif operator == "mate_by_swap":
                    new_astr, inheritance = mate.mate_by_random_swap(
                        select, pool)

                if self.shape == 'cluster' or self.shape == 'molecule':
                    new_astr = mate.move_atoms_to_within_cluster(new_astr)
            except:
                print(
                    "Exception! Unable to conduct mating operation. "
                    f"Operator is: {operator}.")
                traceback.print_exc()
                if operator == "perturb_sites" or operator == "perturb_comp":
                    parent_model = select.get_a_parent(pool)
                    label = parent_model.label
                continue
            if new_astr is None:
                continue
            if any(np.isnan(new_astr.cart_coords.flatten())):
                continue

            new_astr.sort()
            new_comp = new_astr.composition
            # DU
            all_ok = True
            for sp in range(self.num_species):
                species = self.species[sp]
                sym = species['name']
                min_sp = species['min_num']
                max_sp = species['max_num']
                if not min_sp <= new_comp[sym] <= max_sp:
                    all_ok = False
                    break
            if all_ok:
                correct_comp = True

        if not correct_comp:
            print('Failed to produce model in 10 attempts '
                  'with {} operator'.format(operator))
            return None

        new_model = structure_record.model(new_astr, reg_id)
        new_model.inheritance = inheritance
        new_model.made_by = operator

        return new_model


class mating(object):

    def __init__(self, mating_params):
        """
        This class is used in bulk, molecule and cluster geometry searches.
        This is used to create a child structure by mating 2 or more parents.

        Input parameters are provided in the form of a dictionary. Such a
        dictionary would be:

        ```python
            {'shape': 'cluster'  # taken from input file directly
            'mirror_slice_before_join': True
            'min_dist_dict': dictionary of minimum bond distances
                            {'sp1_sp1': 2.3, 'sp1_sp2': 1.5, 'sp2_sp2': 1.2},
            'species_dict': # dictionary of species
                            {'species1': {'name': 'Al',
                            'min_num': 36,
                            'max_num': 36,
                            'mu': -3.35958515625},
                            'species2': {'name': 'O',
                            'min_num': 30,
                            'max_num': 30,
                            'mu': -6.76069604253}}
        ```

        Arguments:

            mating_params (dict): the dictionary of different parameters that
             are required for performing mating on parents
        """
        # Defaults for the parameters
        self.mirror_slice_before_join = True

        if 'mirror_slice_before_join' in mating_params:
            if isinstance(mating_params['mirror_slice_before_join'], bool):
                self.mirror_slice_before_join = \
                    mating_params['mirror_slice_before_join']
            else:
                print('mirror_slice_before_join parameter should be a boolean.'
                      ' Setting to defaults True')

        self.shape = mating_params['shape']
        if self.shape == 'cluster' or\
                self.shape == 'molecule':
            self.max_dia = mating_params['max_dia']
            self.box_abc = np.array(mating_params['box_abc'])
            self.origin = np.array(mating_params['origin'])

        # probabilities of translating a bulk structure before mating along
        # 1) the lattice vector of mating
        # 2) the orthogonal lattice vectors
        self.translation_probs = [0.9, 0.05]

        self.num_species = mating_params['num_species']
        self.species_dict = mating_params['species_dict']
        self.min_dist_dict = mating_params['min_dist_dict']

        self.mating_attempts = 1000  # ensure that most orientations are explored

        # DU
        # If storing all species in a list, and they are all in order
        self.species = []
        for i in range(1, self.num_species+1):
            self.species.append(mating_params['species' + str(i)])

        # If storing as variables, and are labeled in order
        # for sp in range(1, self.num_species + 1):
        #     setattr(self, 'species' + str(sp),
        #             mating_params['species' + str(sp)])

        self.element_syms = mating_params['element_syms']

    def get_attach_type(self):
        """
        Function to get the attach type - mirror and attach, or direct attach.
        Probability of selecting each type is 50%. 

        Returns:
            (str): 'direct' if attaching the slice as is, or 'mirror'
             if attaching the slice after rotating it 180 deg (mirrored)
        """
        attach_type = 'direct'

        # if True, return either mirror and direct attach type
        if self.mirror_slice_before_join:
            if random.random() > 0.5:
                attach_type = 'mirror'

        return attach_type

    def mate_by_slicing(self, select, pool, same_cluster=None):
        """
        Function to create a child (cluster) structure by -
        slicing each parent after a random rotation and attaching two slices
        from both parents.

        !!! note "TODO"
            Make the lattice scaling account for different lattice angles

        Arguments:

            select (obj): fantastx `Select` object

            pool (obj): fantastx `Pool` object

            same_cluster (bool): Whether to mate from same cluster or not.
                                If None, then ignored.

        Returns:
            (obj, list):
             - the pymatgen `Structure` object of the new child
             - the labels of the parent models which were mated
        """
        # Get num_parents and select them parents
        num_parents = 2
        # NOTE: deepcopy already done in get_a_parent()
        parents = select.get_parents(pool, num_parents,
                                     same_cluster=same_cluster)
        parent1, parent2 = parents[0], parents[1]
        inheritance = [parent1.label, parent2.label]

        not_attached = True

        if self.shape == "cluster" or self.shape == "molecule":
            parent_one_astr = parent1.astr.copy()
            parent_two_astr = parent2.astr.copy()
            # rotate all parents randomly and slice them
            temp1 = self.rotate_astr(parent_one_astr)
            temp1_slices = self.fraction_slice(temp1, axis=2)
            temp2 = self.rotate_astr(parent_two_astr)
            temp2_slices = self.fraction_slice(temp2, axis=2)

            # Attach two slices at a time
            if random.randint(0, 1) == 0:
                attach_type = 'direct'
            else:
                attach_type = 'mirror'
            child = self.attach_slices(temp1_slices,
                                       temp2_slices,
                                       attach_type=attach_type,
                                       axis=2,
                                       lattice=None,
                                       remove_overlaps=False)
        else:
            tries = 0
            while not_attached and tries < self.mating_attempts:
                tries += 1
                slice_axis = random.randint(0, 2)

                parent_one_astr = parent1.astr.copy()
                parent_two_astr = parent2.astr.copy()
                child_lattice = None

                # check to make sure that the parents have the same lattice
                # parameters. If not, then make the smaller one have the same
                # lattice parameters as the larger one
                if not np.allclose(parent_one_astr.lattice.matrix,
                                   parent_two_astr.lattice.matrix):
                    parent_one_astr, parent_two_astr = self._orient_two_astrs(
                        parent_one_astr,
                        parent_two_astr
                    )
                    child_lattice = self._generate_child_lattice(
                        parent_one_astr.lattice,
                        parent_two_astr.lattice,
                        False
                    )

                # translate the bulk structures
                translation_probs = [self.translation_probs[1]]*3
                translation_probs[slice_axis] = self.translation_probs[0]
                self._translate_along_latt_vectors(
                    parent_one_astr,
                    translation_probs)
                self._translate_along_latt_vectors(
                    parent_two_astr,
                    translation_probs)

                # determine the cutting point for the two structures
                average_coord_one = np.average(
                    parent_one_astr.frac_coords[:, slice_axis])
                average_coord_two = np.average(
                    parent_two_astr.frac_coords[:, slice_axis])
                cut_point = 0.5 * (average_coord_one + average_coord_two)

                # cut the two structures
                temp1_slices = self.fraction_slice(
                    parent_one_astr, slice_axis, cut_point)
                temp2_slices = self.fraction_slice(
                    parent_two_astr, slice_axis, cut_point)

                # Attach the two slices either with mirroring or directly
                if random.randint(0, 1) == 0:
                    attach_type = 'direct'
                else:
                    attach_type = 'mirror'
                child = self.attach_slices(
                    temp1_slices,
                    temp2_slices,
                    attach_type=attach_type,
                    axis=slice_axis,
                    lattice=child_lattice,
                    remove_overlaps=True)
                if child is None:
                    print(f"On attempt {tries} the child was none")
                    continue
                else:
                    not_attached = False
            if not_attached:
                return None, inheritance

        return child, inheritance

    def _translate_along_latt_vectors(self, astr, selection_probs):
        """
        Translate the sites in a structure by random fractions of its
        lattice vectors. This procedure is the same as in the
        [USPEX]{https://doi.org/10.1016/j.cpc.2006.07.020} paper, where
        first the lattice vectors to translate along are chosen at random,
        before the fractions of the lattice vectors are chosen. The
        cutoffs for choosing the lattice vectors are assigned by the user.

        Arguments:
            astr (obj): pymatgen `Structure` object which will be translated
            selection_probs (iterable): the probabilities of selecting each
             lattice vector for translating along.
        """

        translation_vector = np.zeros(3)
        for i in range(3):
            r1 = np.random.uniform()
            if r1 < selection_probs[i]:
                r2 = np.random.uniform()
                translation_vector[i] = r2

        all_sites = [i for i in range(astr.num_sites)]
        # first, ensure that all sites lie in the unit cell already
        astr.translate_sites(all_sites, [0, 0, 0],
                             frac_coords=True,
                             to_unit_cell=True)
        # then translate by the vector
        astr.translate_sites(all_sites, translation_vector,
                             frac_coords=True,
                             to_unit_cell=True)

    def _align_cell_with_principal_axes(self, astr):
        """
        Given an arbitrary unit cell, orient it so that lattice vector
        `a` lies along the x-axis, lattice vector `b` lies in the
        x-y plane and the z-component of lattice vector `c` is
        positive.
        """
        # align lattice vector a with the x-axis
        rotation = RotationTransformation(
            [0, 0, 1],
            -np.arctan2(astr.lattice.matrix[0][1], astr.lattice.matrix[0][0]),
            True)
        astr = rotation.apply_transformation(astr)
        rotation = RotationTransformation(
            [0, 1, 0],
            np.arctan2(astr.lattice.matrix[0][2], astr.lattice.matrix[0][0]),
            True)
        astr = rotation.apply_transformation(astr)

        # align lattice vector b with the x-y plane
        rotation = RotationTransformation(
            [1, 0, 0],
            -np.arctan2(astr.lattice.matrix[1][2], astr.lattice.matrix[1][1]),
            True)
        astr = rotation.apply_transformation(astr)

        # make sure lattice vector `c` pointing in +z direction
        if astr.lattice.matrix[2][2] < 0:
            a = astr.lattice.matrix[0]
            b = astr.lattice.matrix[1]
            cx = astr.lattice.matrix[2][0]
            cy = astr.lattice.matrix[2][1]
            cz = -1*astr.lattice.matrix[2][2]
            astr.lattice = Lattice([a, b, [cx, cy, cz]])
        return astr

    def _orient_two_astrs(self, astr1, astr2):
        """
        Given two structures which have different lattices, orient the
        first structure to have the same orientation as the second
        structure. This "orientation" is taken to be the norm of the lattice
        vectors, such that the two lattices will have their first, second,
        and third largest lattice vectors in the same positions.

        Given two structures, re-orient them to have a uniform orientation.
        This is conducted by first orienting the first structure such that
        each of its lattice vectors are as close in norm to the respective
        lattice vectors of the second structure as possible. Then, both
        structures are oriented to 

        Arguments:
            astr1 (obj): pymatgen `Structure` object which is the subject of
             orientation.
            astr2 (obj): pymatgen `Structure` object which is the target of
             orientation.
        """
        latt1 = astr1.lattice.matrix
        latt2 = astr2.lattice.matrix

        args1 = np.argsort(np.linalg.norm(latt1, axis=1))
        args2 = np.argsort(np.linalg.norm(latt2, axis=1))

        new_latt1_matrix = [[], [], []]
        new_coords1 = [[], [], []]
        for i in range(3):
            new_latt1_matrix[args2[i]] = latt1[args1[i]]
            new_coords1[args2[i]] = astr1.frac_coords[:, args1[i]]

        # new_latt1_matrix = [[], [], []]
        # new_coords1 = [[], [], []]
        # for i in range(3):
        #     latt1_row = latt1[i]
        #     dists = np.linalg.norm(latt2 - latt1_row, axis=1)
        #     sorted_rows = np.argsort(dists)
        #     for j in range(3):
        #         if len(new_latt1_matrix[sorted_rows[j]]) == 0:
        #             new_latt1_matrix[sorted_rows[j]] = latt1_row
        #             new_coords1[sorted_rows[j]] = astr1.frac_coords[:, i]
        #             break

        new_latt1 = Lattice(new_latt1_matrix)
        new_coords1 = np.array(new_coords1).T
        species1 = [site.specie.symbol for site in astr1.sites]

        astr1 = Structure(new_latt1, species1, new_coords1)

        aligned_astr1 = self._align_cell_with_principal_axes(astr1)
        aligned_astr2 = self._align_cell_with_principal_axes(astr2)

        return aligned_astr1, aligned_astr2

    def _generate_child_lattice(self, latt1, latt2, random=False):
        """
        Given the lattices of the two parents, create a new child lattice.
        This child lattice will be the average of the two parents lattices.
        Alternatively, it can be chosen to be a random linear combination of
        the two parent lattices.

        Arguments:
            latt1 (obj): pymatgen `Lattice` object corresponding to the first
             parent
            latt2 (obj): pymatgen `Lattice` object corresponding to the 2nd
             parent
            random (bool): whether the lattice will be the average of the two
             parent lattices (False), or a random linear combination of the two
             parent lattices (True).
        """
        if not random:
            child_latt_matrix = (latt1.matrix + latt2.matrix) / 2.0
            child_latt = Lattice(child_latt_matrix)
        else:
            r = np.random.uniform()
            child_latt_matrix = r * latt1.matrix + (1 - r) * latt2.matrix
            child_latt = Lattice(child_latt_matrix)
        return child_latt

    def rotate_astr(self, astr, rotate_type='random', mirror_axis=2):
        """
        Given a structure, rotates it at a random angle (0:360) along a
        random lattice vector ([0, 0, 0]:[3, 3, 3]). This lattice vector
        is aligned with the center of the structure's box.

        Alternately, "mirror" the structure by rotating it 180 degrees
        with respect to an arbitrary x-y vector. In this case, ensure that
        the z-bounds remain constant.

        !!! note "TODO"
            Need to make mirror operation an actual mirroring, rather than
            a 180 degree rotation. The two operations are not equivalent.

        Arguments:

            astr (obj): pymatgen `Structure` object

            rotate_type (str): 'random' or 'mirror' for rotation of structure

        Returns:
            (obj): the pymatgen `Structure` object of the rotated structure
        """

        # if bulk, translate all sites such that the center of the
        # box is the origin
        trans_vector = np.array([-0.5, -0.5, -0.5])
        # otherwise, find geometric center and make that it is the origin
        if self.shape == "cluster" or self.shape == "molecule":
            fc = astr.frac_coords
            range_x, range_y, range_z = fc[:, 0], fc[:, 1], fc[:, 2]
            cent_x, cent_y, cent_z = (max(range_x) + min(range_x))/2, \
                                     (max(range_y) + min(range_y))/2, \
                                     (max(range_z) + min(range_z))/2
            cent = np.array([cent_x, cent_y, cent_z])
            trans_vector = -cent

        all_inds = [i for i in range(len(astr.cart_coords))]
        temp_astr = astr.copy()
        species = temp_astr.species
        first_coords = temp_astr.cart_coords
        if rotate_type == 'random':
            temp_astr.translate_sites(all_inds, trans_vector,
                                      frac_coords=True, to_unit_cell=False)
            # perfrom random rotation transformation
            hkl = [random.randint(0, 3), random.randint(
                0, 3), random.randint(0, 3)]
            while hkl[0] == 0 and hkl[1] == 0 and hkl[2] == 0:
                hkl = [random.randint(0, 3), random.randint(
                    0, 3), random.randint(0, 3)]
            rotate = RotationTransformation(hkl, unif(0, 360))
            temp_astr = rotate.apply_transformation(temp_astr)
            # NOTE: The lattice is rotated, but coords are still same
            # Place old cart_coords in temp_parent lattice
            temp_astr.remove_sites(all_inds)
            for specie, coord in zip(species, first_coords):
                temp_astr.append(specie, coord, coords_are_cartesian=True)

            # reverse the prior translation
            temp_astr.translate_sites(all_inds, -trans_vector)

            # Get conventional structure (2 ways)
            # 1. modify_lattice from parent1 (straight forward)
            # 2. use SpacegroupAnalyzer
            temp_astr.lattice = astr.lattice
            # If the above causes any issues, use this approach 2
            # sp = SpacegroupAnalyzer(temp_parent)
            # prepped_parent = sp.get_conventional_standard_structure()
        elif rotate_type == 'mirror':
            # calculate mirror shift needed to make mirroring occur in place
            coord_max = max(temp_astr.frac_coords[:, mirror_axis])
            coord_min = min(temp_astr.frac_coords[:, mirror_axis])
            mirrored_max = 1 - coord_min
            mirror_shift = coord_max - mirrored_max

            frac_coords = temp_astr.frac_coords.copy()
            for i in frac_coords:
                i[mirror_axis] = 1 - i[mirror_axis] + mirror_shift
            temp_astr.remove_sites(all_inds)
            for specie, coord in zip(species, frac_coords):
                temp_astr.append(specie, coord, coords_are_cartesian=False)

        return temp_astr

    def fraction_slice(self, astr, axis, cut_point=None):
        """
        For a given astr, this function slices it from the bottom leaving
        a fraction of the structure either corresponding to 1 divided by the
        number of parents (e.g. 1/2 for 2 parents), or corresponding to the
        provided cut_point. This slice is made along a designated lattice
        vector.

        !!! note
            Currently the number of slices is fixed to two.

        Args:

            astr (obj): pymatgen `Structure` object

            axis (int): the lattice vector to make slices along

            cut_point (float): the fractional point along the axis to make
             slices

        Returns:
            (obj): the input pymatgen `Structure` object with the slice removed
        """
        # Fixed num_parents to 2.
        num_parents = 2

        # make sure that all coordinates are within the box
        sites = [i for i in range(astr.num_sites)]
        astr.translate_sites(sites, [0, 0, 0],
                             frac_coords=True,
                             to_unit_cell=True)

        coords = astr.frac_coords[:, axis]
        # Determine z_cut to get ~ equal fractions from all parents
        if cut_point is None:
            if self.shape == "cluster" or self.shape == "molecule":
                cut_point = (max(coords) + min(coords)) / num_parents
            else:
                cut_point = 1. / num_parents

        slices = []
        for i in range(num_parents):
            rm_inds = []
            slice_astr = astr.copy()
            for n, coord in enumerate(slice_astr.frac_coords[:, axis]):
                if coord <= cut_point*i or coord >= cut_point*(i + 1):
                    rm_inds.append(n)
            # Remove these atoms from the parent
            slice_astr.remove_sites(rm_inds)
            slices.append(slice_astr)

        return slices

    def attach_slices(self, p1_slices, p2_slices,
                      attach_type='mirror', axis=2, lattice=None,
                      remove_overlaps=False):
        """
        Given two sets of slices, rotates and attaches slices. Functionality
        is included to mirror the 2nd slice before attaching it to the first
        slice.

        !!! note
            Currently the number of slices is hard-coded to be two.

        Args:

            p1_slices (list): set of pymatgen `Structure` objects which are
             slices from parent one along the designated axis

            p2_slices (list): set of pymatgen `Structure` objects which are
             slices from parent two along the designated axis

            attach_type (str): how to attach the slices, 'mirror' for
             mirrored or 'direct' if no mirroring should be performed. 

            axis (int): the cartesian axis along which to attach the slices

            lattice (obj): the pymatgen `Lattice` object which will belong to
             the spliced structure.

            remove_overlaps (bool): whether overlapping atoms should be removed
             or should instead be cause for ending the attachment process

        Returns:
            (obj): pymatgen `Structure` object of the merged slices
        """
        new_lattice = p1_slices[0].lattice
        if lattice is not None:
            new_lattice = lattice

        # choose slice 1 and the opposite side slice 2
        slice_one_index = random.randint(0, 1)
        slice1 = p1_slices[slice_one_index].copy()
        slice_two_index = 1 - slice_one_index
        slice2 = p2_slices[slice_two_index].copy()

        discarded_slice1 = p1_slices[slice_two_index]
        discarded_slice2 = p2_slices[slice_one_index]

        # mirror the 2nd slice if desired
        if attach_type == 'mirror':
            slice2 = self.rotate_astr(slice2,
                                      rotate_type='mirror',
                                      mirror_axis=axis)

        slice1_coords = slice1.frac_coords
        slice2_coords = slice2.frac_coords

        attach_axis_length = new_lattice.abc[axis]
        if self.shape == "cluster" or self.shape == "molecule":
            trans_vec = [0, 0, 0]
            if slice_one_index == 0:
                trans_vec[axis] = 1 / attach_axis_length
            else:
                trans_vec[axis] = - 1 / attach_axis_length
            coord_trans = np.array(trans_vec)
            add_coords = slice2_coords + coord_trans
        else:
            add_coords = slice2_coords

        if self.shape == "cluster" or self.shape == "molecule":
            axis1 = axis - 1
            axis2 = (axis + 1) % 3
            # center add_coords on slice1 i.e., align centers along plane
            range_a1, range_b1 = slice1_coords[:, axis1],
            slice1_coords[:, axis2]
            range_a2, range_b2 = slice2_coords[:, axis1],
            slice2_coords[:, axis2]
            cent_a1, cent_b1 = (max(range_a1) + min(range_a1))/2, \
                (max(range_b1) + min(range_b1))/2
            cent_a2, cent_b2 = (max(range_a2) + min(range_a2))/2, \
                (max(range_b2) + min(range_b2))/2
            trans_vec = [0, 0, 0]
            trans_vec[axis1] = (cent_a1 - cent_a2)
            trans_vec[axis2] = (cent_b1 - cent_b2)
            add_coords = add_coords + np.array(trans_vec)

        # species and coordinates of both slices
        add_species = np.concatenate((slice1.species, slice2.species))
        add_coords = np.concatenate((slice1_coords, add_coords))

        # add each of the coordinates to the new child structure
        child = Structure(new_lattice, [], [])
        failed_dc = False
        attached = False
        for specie, coord in zip(add_species, add_coords):
            if child.num_sites > 0:
                child_species = [i.specie.symbol for i in child.sites]
                inv_syms = {v: 'sp' + str(k)
                            for k, v in self.element_syms.items()}
                if dc.satisfies_all_dists_quick(coord,
                                                child.frac_coords,
                                                specie.symbol,
                                                child_species,
                                                inv_syms,
                                                self.min_dist_dict,
                                                lattice=new_lattice,
                                                coords_are_cartesian=False):
                    child.append(specie, coord, coords_are_cartesian=False)
                elif not remove_overlaps:
                    failed_dc = True
                    break
            else:
                child.append(specie, coord, coords_are_cartesian=False)

        if not failed_dc:
            child.sort()
            child_comp = child.composition
            # DU
            failed_cc = False
            discarded_species = np.concatenate((
                discarded_slice1.species,
                discarded_slice2.species))
            discarded_coords = np.concatenate((
                discarded_slice1.frac_coords,
                discarded_slice2.frac_coords
            ))
            for sp in range(self.num_species):
                species = self.species[sp]
                sym = species['name']
                min_sp = species['min_num']
                max_sp = species['max_num']

                # attempt to correct the composition
                while child.composition[sym] > max_sp:
                    # if too many of a specie, remove atoms at random
                    poppable_sites = [i for i in range(child.num_sites)
                                      if child.sites[i].specie.symbol == sym]
                    pop_index = np.random.randint(0, len(poppable_sites))
                    child.pop(poppable_sites[pop_index])

                if child.composition[sym] < min_sp:
                    # if too few of a specie, attempt to add atoms from the
                    # slices that were discarded
                    discarded_indices = [i for i in range(
                        len(discarded_species)) if discarded_species[i].symbol == sym]
                    max_cc = len(discarded_indices)
                    cc_tries = 0
                    while child.composition[sym] < min_sp and\
                            cc_tries < max_cc:
                        ds = discarded_indices[cc_tries]
                        specie = discarded_species[ds].symbol
                        coord = discarded_coords[ds]
                        child_species = [i.specie.symbol for i in child.sites]
                        if dc.satisfies_all_dists_quick(coord,
                                                        child.frac_coords,
                                                        specie,
                                                        child_species,
                                                        inv_syms,
                                                        self.min_dist_dict,
                                                        new_lattice,
                                                        False):
                            child.append(specie, coord,
                                         coords_are_cartesian=False)

                        cc_tries += 1

                if not min_sp <= child_comp[sym] <= max_sp:
                    failed_cc = True
                    break
            if not failed_cc:
                attached = True

        if attached:
            return child
        else:
            return None

    def mate_by_random_swap(self, select, pool, same_cluster=None):
        """
        Function to combine two parents and keep only required number of atoms
        from each of the parent. In other words, randomly select few atoms
        from two parents to create a child structure.

`       !!! note "TODO"
            Still need to add a distance check

        Arguments:

            select (obj): Select object

            pool (obj): Pool object

            same_cluster (bool): `True` if mating structures from the same
             fingerprint cluster, `False` if not. If `None`, then ignored.

        Returns:
            (obj, list):
             - the pymatgen `Structure` object of the child
             - the labels of the parent models which were mated
        """
        # Get num_parents and select them parents
        num_parents = 2
        parents = select.get_parents(pool, num_parents,
                                     same_cluster=same_cluster)
        parent1, parent2 = parents[0], parents[1]
        inheritance = [parent1.label, parent2.label]
        # p1_sites = parent1.astr.sites
        # p2_sites = parent2.astr.sites
        # list_of_p_sites = [p1_sites, p2_sites]

        child = copy.deepcopy(parent1.astr)
        all_inds = [i for i in range(len(child.cart_coords))]
        child.remove_sites(all_inds)

        # add atoms from both parents in to one structure
        child_sites = parent1.astr.sites + parent2.astr.sites
        species = [i.species for i in child_sites]
        coords = [i.coords for i in child_sites]
        latt = parent1.astr.lattice
        child = Structure(latt, species, coords, coords_are_cartesian=True)

        # merge sites
        child.merge_sites(tol=1, mode='delete')

        # get composition of child within th range of both parents
        p1_comp = parent1.astr.composition.as_dict()
        p2_comp = parent2.astr.composition.as_dict()
        # get child elements such that the species exist in both parents
        # NOTE: If elements are different in both parents, child will only get
        # common elements in subsequent generations. So, make sure elements are
        # same in both parents
        child_elems = [i for i in p1_comp.keys() if i in p2_comp.keys()]

        # get child composition
        child_comp = {}
        for k in child_elems:
            l, h = min([p1_comp[k], p2_comp[k]]), max([p1_comp[k], p2_comp[k]])
            child_comp[k] = np.random.randint(l, h+1)

        # get all child sites
        all_child_sites = child.sites
        child_elem_sites = []
        for k in child_elems:
            elem_sites = [i for i in all_child_sites if i.specie.name == k]
            if len(elem_sites) > child_comp[k]:
                random.shuffle(elem_sites)
                child_elem_sites += elem_sites[:child_comp[k]]
            else:
                return None, None

        # replace child with new atoms
        child_sps = [i.specie for i in child_elem_sites]
        child_coords = [i.coords for i in child_elem_sites]
        child = Structure(latt, child_sps, child_coords,
                          coords_are_cartesian=True)

        return child, inheritance

    def get_point_on_sphere(self, r):
        """
        Returns a random point on a sphere of radius `r`

        Arguments:

            r (float): radius of the sphere

        Returns:
            (array): cartesian coordinates of the point
        """

        # get random point (x, y, z) using normal distribution
        point = np.random.randn(3)
        # normalize the point
        point_mag = np.linalg.norm(point)
        point = point / point_mag
        # multiply by radius
        point = point * r

        return point

    def move_atoms_to_within_cluster(self, child):
        """
        In cluster or molecule geometries, after a child is generated by
        mating, check if any of the atoms are outside the maximum radius of
        the cluster or molecule. If so, move them randomly to somewhere
        within the radius.

        Arguments:

            child (obj): pymatgen `Structure` object

        Returns:
            (obj): pymatgen `Structure` object corresponding to the modified
             child object. `None` if 4 or more atoms are outside the max
             diameter.
        """
        radius, abc = self.max_dia/2, self.box_abc

        # get atom indices that needs to be moved
        child_sites = child.sites
        species = child.species
        move_inds = [i for i, site in enumerate(child_sites)
                     if dc.dist(self.origin, site.coords) > radius]
        if len(move_inds) > 3:
            print('More than 3 atoms lie outside max diameter of cluster. '
                  'Rejecting this model and continuing to make new model')
            return None

        # move those atoms in same way as in perturb_sites
        remove_inds = []
        for i in move_inds:
            tries = 0
            replaced = False
            while not replaced and tries < 1000:
                tries += 1
                new_cart = unif(self.origin[0] - radius,
                                self.origin[0] + radius, size=(3,))
                if dc.dist(self.origin, new_cart) > radius:
                    continue
                # Check distance and replace with new coords
                if dc.satisfies_all_dists(new_cart,
                                          child,
                                          self.element_syms,
                                          self.min_dist_dict,
                                          atom_index_in_astr=i):
                    child.replace(i, species[i], new_cart,
                                  coords_are_cartesian=True)
                    replaced = True
            # remove those atoms that cannot be replaced
            if not replaced:
                remove_inds.append(i)
        child.remove_sites(remove_inds)
        return child


class basinhopping(object):
    """
    Class that handles making child models using basinhopping methods
    """

    def __init__(self, basinhopping_params):
        """
        Args:

        basinhopping_params (dict): dictionary with all the required
        basinhopping parameters

        Eg:{'indices_fraction': 0.8,     # fraction of total atoms to perturb
            'max_perturbation': 0.5,     # maximum perturbation distance in Å
            'min_dist_dict': {           # dictionary of minimum bond distances
                 'sp1_sp1': 2.3,
                 'sp1_sp2': 1.5,
                 'sp2_sp2': 1.2},
            'species_dict': {            # dictionary of species information
              'species1': {
                 'name': 'Al',
                 'min_num': 36,
                 'max_num': 36,
                 'mu': -3.35958515625},
              'species2': {
                 'name': 'O',
                 'min_num': 30,
                 'max_num': 30,
                 'mu': -6.76069604253}}}
        """
        # default indices_fraction is 1 ; perturb all atoms (indices)
        self.indices_fraction = 1
        self.max_perturbation = 0.15
        self.min_dist_dict = basinhopping_params['min_dist_dict']
        self.max_dist_dict = basinhopping_params['max_dist_dict']
        self.species_dict = basinhopping_params['species_dict']
        self.element_syms = basinhopping_params['element_syms']
        self.shape = basinhopping_params['shape']

        if 'indices_fraction' in basinhopping_params:
            if 0 < basinhopping_params['indices_fraction'] <= 1:
                self.indices_fraction = basinhopping_params['indices_fraction']
            else:
                print('Provided indices_fraction out of range (0,1]. '
                      'Using default..')

        if 'max_perturbation' in basinhopping_params:
            if not 0 < basinhopping_params['max_perturbation'] <= 0.5:
                print('max_perturbation should be between (0, 0.5]. '
                      'More than 0.5 would be throw the atoms too far.'
                      ' Check the jump distance by lattice vectors *'
                      ' max_perturbation. Using default value of 0.15')
            else:
                self.max_perturbation = basinhopping_params['max_perturbation']

        # maximum diameter and box lattice parameters of the cluster (geometry)
        if self.shape == 'cluster' or self.shape == 'molecule':
            # default is 8 Å
            self.max_dia = basinhopping_params['max_dia']
            # default a=b=c=20 Å
            self.box_abc = np.array(
                basinhopping_params['box_abc'])
            self.origin = np.array(basinhopping_params['origin'])
        if self.shape == 'molecule':
            self.fixed_species = basinhopping_params['fixed_species']
            print(f"Atomic species held fixed: {self.fixed_species}")
        else:
            self.fixed_species = []

        # delta_comps list to modify the composition
        # Ex: ['AlO', 'Al2O', 'Al3O', 'Al4O', 'O2', 'Al2', 'Al3']
        self.delta_comps = []
        if 'delta_comps' in basinhopping_params:
            self.delta_comps = basinhopping_params['delta_comps']

        self.add_rem_comp_frac = 0.6  # add unit comp 60% of the time
        if 'add_rem_comp_frac' in basinhopping_params:
            self.add_rem_comp_frac = basinhopping_params['add_rem_comp_frac']

    def perturb_sites(self, select, pool,
                      surface_thickness=None, model_id=None):
        """
        Displaces atoms in a parent (cluster or gb_iface or surface layer)
        using uniform distribution within max_perturbation
        Returns (child structure, inheritance) or (None, None) if fails.

        !!! note "TODO"
            Still need to add a minimum perturbation to avoid risk of
            redundancy

        Arguments:

            select (obj): Select object

            pool (obj): Pool object

            surface_thickness (float): how many angstroms thick the active
             surface region is

            model_id (int): If given, basinhopping is done on this specific
             model
        """
        if 'good_pool' in pool.__dict__.keys():
            all_models = pool.good_pool
        if 'population' in pool.__dict__.keys():
            all_models = pool.population.models

        if self.shape == 'surface':
            return self.perturb_surface(select, pool,
                                        surface_thickness=surface_thickness,
                                        model_id=None)

        indices_fraction = self.indices_fraction
        if model_id is None:
            parent_model = select.get_a_parent(pool)
            # make a copy
            parent = copy.deepcopy(parent_model)
            inheritance = [parent.label]
        else:
            inheritance = None
            for model in all_models:
                # for model in pool.good_pool:
                if model.label == model_id:
                    parent = copy.deepcopy(model)
                    inheritance = [parent.label]
                    break
            if inheritance is None:
                parent_model = select.get_a_parent(pool)
                # make a copy
                parent = copy.deepcopy(parent_model)
                inheritance = [parent.label]

        # grab target structure. By default, this will be the entire parent
        target_astr = parent.astr
        if self.shape == 'gb':
            target_astr = parent.gb_iface

        # Get the cart_coords to be perturbed
        cart_coords = target_astr.cart_coords
        species = target_astr.species

        # Get frac_coords to perturb
        total_num_atoms = len(cart_coords)
        # use indices_fraction; default to 1.
        # Adjust if any species need to be held fixed.
        num_atoms_to_perturb = int(total_num_atoms * indices_fraction)
        if len(self.fixed_species) > 0:
            cap_reduction = 0
            for specie in self.fixed_species:
                cap_reduction += parent.astr.composition.as_dict()[specie]
            if num_atoms_to_perturb > total_num_atoms - cap_reduction:
                num_atoms_to_perturb = total_num_atoms - cap_reduction

        D_inds = random.sample(range(0, total_num_atoms), num_atoms_to_perturb)
        if len(self.fixed_species) > 0:
            # Grab species which correspond to D_inds
            bh_species = [parent.astr.sites[i].specie.name for i in D_inds]
            occupancy = [i in self.fixed_species for i in bh_species]
            problematic_basinhopping = np.any(occupancy)
            while problematic_basinhopping:
                D_inds = random.sample(
                    range(0, total_num_atoms), num_atoms_to_perturb)
                bh_species = [parent.astr.sites[i].specie.name for i in D_inds]
                occupancy = [i in self.fixed_species for i in bh_species]
                problematic_basinhopping = np.any(occupancy)

        D_coords = [cart_coords[i] for i in D_inds]

        num_perturbed = 0
        jumps_needed = int(0.5 * len(D_coords))
        for i, one_coords in zip(D_inds, D_coords):
            replaced = False
            tries = 0
            while not replaced and tries < 1000:
                tries += 1
                # Max perturbation in Å
                jump = self.max_perturbation
                perturb = self.get_point_on_sphere(jump)
                new_cart = one_coords + perturb

                if self.shape == 'cluster' or self.shape == "molecule":
                    # check if new cart is inside the cluster radius
                    if dc.dist(self.origin, new_cart) > self.max_dia/2:
                        # print(f"between origin and new_cart is too large")
                        continue

                if dc.satisfies_all_dists(new_cart,
                                          target_astr,
                                          self.element_syms,
                                          self.min_dist_dict,
                                          atom_index_in_astr=i):
                    target_astr.replace(i, species[i], new_cart,
                                        coords_are_cartesian=True)
                    replaced = True
                    num_perturbed += 1
            # print(target_astr)

        if num_perturbed >= jumps_needed:
            return target_astr, inheritance
        else:
            return None, None

    def perturb_surface(self, select, pool,
                        surface_thickness=1, model_id=None):
        """
        Displaces atoms in a surface layer using uniform distribution

        Returns slab structure with a perturbed surface layer, its inheritance

        Args:

        select (obj): Select object

        pool (obj): Pool object

        surface_thickness (float): The thickness of the surface layer from top
                                   of the surface

        model_id (int): If given, basinhopping is done on this specific model
        """
        indices_fraction = self.indices_fraction

        if model_id is None:
            parent_model = select.get_a_parent(pool)
            # make a copy
            parent = copy.deepcopy(parent_model)
            inheritance = [parent.label]
        else:
            for model in pool.good_pool:
                if model.label == model_id:
                    parent = copy.deepcopy(model)
                    inheritance = [parent.label]
                    break

        # Get the necessary variables
        parent_carts = parent.astr.cart_coords
        parent_species = parent.astr.species
        max_z_cart = parent_carts[:, 2].max()
        surface_inds = [i for i, site in enumerate(parent.astr.sites)
                        if max_z_cart - site.coords[2] < surface_thickness]

        # Get surface_fracs to perturb
        total_surface_atoms = len(surface_inds)
        # use indices_fraction; default to 1
        num_atoms_to_perturb = int(total_surface_atoms * indices_fraction)
        # Displace the coords according to these indices
        D_inds = random.sample(surface_inds, num_atoms_to_perturb)

        num_perturbed = 0
        # Make sure at least half of the atoms are actually displaced
        jumps_needed = int(0.5 * len(D_inds))
        for atom_ind_in_par in D_inds:
            one_coords = parent_carts[atom_ind_in_par]
            replaced = False
            tries = 0
            # 50 tries is enough to get a perturbed coordinate if one exists
            while not replaced and tries < 100:
                tries += 1
                # Max perturbation is in Å
                radius = self.max_perturbation
                if tries > 50:  # try to move somewhere below max_perturb
                    radius = unif(0.09, self.max_perturbation)
                perturb = self.get_point_on_sphere(radius)
                new_cart = one_coords + perturb

                # check if the new coords satisfies min & max dists
                if dc.satisfies_all_dists(new_cart, parent.astr,
                                          self.element_syms,
                                          self.min_dist_dict,
                                          max_dist_dict=self.max_dist_dict,
                                          atom_index_in_astr=atom_ind_in_par):
                    parent.astr.replace(atom_ind_in_par,
                                        parent_species[atom_ind_in_par],
                                        new_cart, coords_are_cartesian=True)
                    replaced = True
                    num_perturbed += 1

        if num_perturbed >= jumps_needed:
            return parent.astr, inheritance
        else:
            return None, None

    def perturb_comp(self, select, pool,
                     z_bounds=None, dc_astr=None, model=None):
        """
        Function to generate a child model by changing the composition of a
        parent model.
        If delta_comps list exists, adds or removes one unit comp from it.

        NOTE: The species listed in delta_comps should always be subset of the
              species present in parent
        Args:

        select (obj): Select object

        pool (obj): Pool object

        model_id (int): If given, basinhopping is done on this specific model
        """
        if model is None:
            parent_model = select.get_a_parent(pool)
            parent = copy.deepcopy(parent_model)
            # make a copy
        else:
            parent = copy.deepcopy(model)
        parent_astr = parent.astr
        parent_comp = parent_astr.composition.as_dict()
        inheritance = [parent.label]

        # if delta comps list exists
        # select a unit composition from the delta comps list
        if len(self.delta_comps) > 0:
            unit = np.random.choice(self.delta_comps)
            unit_comp = Composition(unit).as_dict()
        else:  # else select random unit compositon
            unit_comp = {}
            while True:
                for sym in self.element_syms.values():
                    unit_comp[sym] = np.random.randint(3)
                if sum(unit_comp.values()) > 0:
                    break

        # choose whether to add or remove the unit composition
        add_unit_comp = False
        if random.random() < self.add_rem_comp_frac:
            add_unit_comp = True

        # Add random sites to the parent
        if add_unit_comp:
            for sps in unit_comp.keys():
                num_added = 0
                while num_added < unit_comp[sps]:
                    fracs = [unif(0, 1), unif(0, 1), unif(0, 1)]
                    if z_bounds is None:
                        carts = parent_astr.lattice.get_cartesian_coords(fracs)
                        print(type(carts))
                    else:
                        x_cart = parent_astr.lattice.a*fracs[0]
                        y_cart = parent_astr.lattice.b*fracs[1]
                        z_cart = fracs[2]*z_bounds[1] + \
                            (1-fracs[2])*z_bounds[0]
                        carts = (x_cart, y_cart, z_cart)
                    # Check dists with rest of the atoms in parent_astr
                    if dc.satisfies_all_dists(carts,
                                              parent.astr,
                                              self.element_syms,
                                              self.min_dist_dict,
                                              new_carts_species=sps):
                        if dc_astr:
                            # check with dc_astr as well (if gb geometry)
                            if dc.satisfies_all_dists(carts,
                                                      dc_astr,
                                                      self.element_syms,
                                                      self.min_dist_dict,
                                                      new_carts_species=sps):
                                dc_astr.append(sps, carts,
                                               coords_are_cartesian=True)
                                parent_astr.append(sps, carts,
                                                   coords_are_cartesian=True)
                        else:
                            parent_astr.append(sps, carts,
                                               coords_are_cartesian=True)
                        num_added += 1
        else:  # remove random sites from the parent
            rem_inds = []
            for sps in unit_comp.keys():
                n_sps = int(parent_comp[sps])
                n_rem_sps = int(unit_comp[sps])
                inds = np.random.choice(n_sps, n_rem_sps, replace=False)
                rem_inds += list(inds)
            parent_astr.remove_sites(rem_inds)

        return parent_astr, inheritance

    def get_point_on_sphere(self, r):
        """
        Returns a random point on a sphere of radius r

        Args:

        r (float): radius of the sphere
        """

        # get random point (x, y, z) using normal distribution
        point = np.random.randn(3)
        # normalize the point
        point_mag = np.linalg.norm(point)
        point = point / point_mag
        # multiply by radius
        point = point * r

        return point


class gb_ops(object):
    """
    Functions for creating initial_population and creating child
    structures with mating operations of grain boundary models. Basinhopping
    class object (hop) is used to make basinhopping child models.
    """

    def __init__(self, hop, str_constraints):
        """
        Args:

        hop (object): basinhopping class object

        str_constraints (dict): structure constraints dict from
                                inputs.make_objects
        """
        # Save basinhopping object as an attribute because
        # hop.perturb_sites() is used for child generation
        self.hop = hop
        # basinhopping vs mating fraction : only used for gb geometry search
        self.hop_mate_frac = 0.3  # 30% hop, 70% mate
        if 'hop_mate_frac' in str_constraints:
            self.hop_mate_frac = str_constraints['hop_mate_frac']
        if 'init_gb_astr' not in str_constraints:
            print('Error: Initial grain boudanry not provided. ')
        else:
            self.init_gb_astr = str_constraints['init_gb_astr']

        if 'iface_thickness' not in str_constraints:
            print('Error: Thickness of interface region not provided. ')
        else:
            self.iface_thickness = str_constraints['iface_thickness']

        if 'iface_z_mid' not in str_constraints:
            print('Error: Mid point of grain boundary not provided. ')
        else:
            self.iface_z_mid = str_constraints['iface_z_mid']

        if 'num_slices' not in str_constraints:
            print('Error: NUmber of slices to cut for mating not provided. ')
        else:
            self.num_slices = str_constraints['num_slices']

        self.slice_axes = [0, 1]  # default slicing axes of x and y only
        if 'slice_axes' in str_constraints:
            self.slice_axes = str_constraints['slice_axes']

        print(f"Hopping mate frac is: {self.hop_mate_frac}")

        # num_species is taken from species_dict from structure_record
        self.num_species = str_constraints['num_species']

        # DU:
        # save species data, same as in initial population
        # If species are all properly labeled in order,
        # and storing them in lists for later access:
        self.sym_species = []
        self.min_num_sp = []
        self.max_num_sp = []
        for index in range(1, self.num_species + 1):
            self.sym_species.append(
                str_constraints['species' + str(index)]['name'])
            self.min_num_sp.append(
                str_constraints['species' + str(index)]['min_num'])
            self.max_num_sp.append(
                str_constraints['species' + str(index)]['max_num'])

        # variables will be labeled as species1, min_num_sp1, max_num_sp1, etc
        # If species are all properly labeled in order:
        # (and storing them as individual variables)
        # for index in range(1, self.num_species + 1):
        #     setattr(self, 'sym_species' + str(index),
        #             str_constraints['species' + str(index)]['name'])
        #     setattr(self, 'min_num_sp' + str(index),
        #             str_constraints['species' + str(index)]['min_num'])
        #     setattr(self, 'max_num_sp' + str(index),
        #             str_constraints['species' + str(index)]['max_num'])

        # If species are not all in order, find species in str_constraints
        # for key in str_constraints:
        #     if key[:7] == 'species':
        #         index = key[7:]
        #         setattr(self, 'sym_species' + index,
        #                 str_constraints[key]['name'])
        #         setattr(self, 'min_num_sp' + index,
        #                 str_constraints[key]['min_num'])
        #         setattr(self, 'max_num_sp' + index,
        #                 str_constraints[key]['max_num'])

        self.min_dist_dict = str_constraints['min_dist_dict']
        self.species_dict = str_constraints['species_dict']
        self.element_syms = str_constraints['element_syms']
        self.iface_latt = str_constraints['iface_latt']
        # Selective dynamics range from input file if provided
        self.sd_true_above = str_constraints['sd_true_above']
        self.sd_true_below = str_constraints['sd_true_below']

        # hollow gb structure
        copy_g = self.get_hollow_gb()
        copy_g.sort()
        self.hollow_init_gb = copy.deepcopy(copy_g)

        # create an astr with only atoms near gb iface from hollow gb
        # get inds of atoms near gb iface within max of min bond dists
        max_of_min_dists = max(self.min_dist_dict.values())
        half_zrange = ((self.iface_thickness + 2*max_of_min_dists)
                       / (self.init_gb_astr.lattice.c * 2))
        min_z_t = self.iface_z_mid + half_zrange
        max_z_b = self.iface_z_mid - half_zrange

        deeper_atom_inds = []
        for i, site in enumerate(copy_g.sites):
            if not max_z_b <= site.c <= min_z_t:
                deeper_atom_inds.append(i)

        copy_g.remove_sites(deeper_atom_inds)
        copy_g.sort()
        self.astr_for_dist_check = copy.deepcopy(copy_g)

    def get_hollow_gb(self):
        """
        Function to remove sites from the interface region of init_gb_astr.
        This hollow_gb will be used as the base for all models.

        No arguments needed
        """
        init_gb_astr = self.init_gb_astr
        iface_z_mid = self.iface_z_mid
        iface_thickness = self.iface_thickness

        gb_c = init_gb_astr.lattice.c
        copy_gb = init_gb_astr.copy()

        # Get coords and species
        gb_sites = copy_gb.sites
        sorted_sites = sorted(gb_sites, key=lambda x: x.coords[2])
        min_z_top = iface_z_mid + (iface_thickness / (gb_c * 2))
        max_z_bot = iface_z_mid - (iface_thickness / (gb_c * 2))
        self.hollow_botz = max_z_bot
        self.hollow_topz = min_z_top

        top_ind, bot_ind = None, None
        for i, site in enumerate(sorted_sites):
            if site.c >= max_z_bot and not bot_ind:
                bot_ind = i
            if site.c >= min_z_top and not top_ind:
                top_ind = i
                break
        mid_sites = sorted_sites[bot_ind: top_ind]
        rem_inds = []
        for i, site in enumerate(copy_gb.sites):
            if site in mid_sites:
                rem_inds.append(i)
        copy_gb.remove_sites(rem_inds)

        return copy_gb

    def get_rem_inds(self, child_astr):
        """
        Function to adjust the composition of the child structure. Removes
        atoms randomly if more than required are present. Adds atoms randomly
        to the interface region if less than required are present.

        Note that if the compositon is too far from required composition, the
        adjusted structure may lose important features due to randomization.

        Args:

        child_astr (obj): pymatgen Structure object of grain boundary
        """
        # get num species at random for each species present in element_syms,
        # in the order in which they were added to the list
        # DU
        n_sp = [np.random.randint(self.min_num_sp[sp], self.max_num_sp[sp]+1)
                for sp in range(self.num_species)]

        # get iface sites in the child_gb_astr
        child_astr_sites = child_astr.sites
        iface_inds_in_gb = []
        for i, site in enumerate(child_astr.sites):
            if self.hollow_botz <= site.c <= self.hollow_topz:
                iface_inds_in_gb.append(i)
        iface_sites = [site for i, site in enumerate(child_astr_sites)
                       if i in iface_inds_in_gb]
        site_sps = [site.specie for site in iface_sites]
        site_sps = [i.name for i in site_sps]
        iface_sps = self.sym_species
        comp_dict = {}
        for sp in iface_sps:
            comp_dict[sp] = site_sps.count(sp)
            # comp_dict is the composition dictionary of iface in child_astr
        # DU
        # get num species to be removed for each species
        diff_sp = np.zeros(self.num_species)
        for species in comp_dict.keys():
            # find species in storage
            sp_index = self.sym_species.index(species)
            diff_sp[sp_index] = n_sp[sp_index] - comp_dict[species]

        # if difference is positive, add sites and return []
        for n, diff in enumerate(diff_sp):
            sp = self.sym_species[n]
            self.add_sites_diff(diff, sp, child_astr)

        # if difference is negative, return rem_inds -> remove sites
        rem_inds = []
        rem_sp = np.zeros(self.num_species)

        random.shuffle(iface_inds_in_gb)
        for ind in iface_inds_in_gb:
            site = child_astr.sites[ind]
            # find specie name in our list
            sp_index = self.sym_species.index(site.specie.name)
            if rem_sp[sp_index] < -diff_sp[sp_index]:
                rem_inds.append(ind)
                rem_sp[sp_index] += 1

        return rem_inds

    def add_sites_diff(self, diff, sp, child_astr):
        """
        Adds given number of random sites of given species to child structure

        Args:

        diff (int): number of new sites to add

        sp (str): species name of the new sites

        child_astr (obj): pymatgen Structure object of child structure
        """
        # This is half the thickness (- 1 Å tolerance)
        half_z_thickness = (self.iface_thickness - 0.3) /   \
            (self.init_gb_astr.lattice.c * 2)
        zmin = self.iface_z_mid - half_z_thickness
        zmax = self.iface_z_mid + half_z_thickness
        ztol = max(self.min_dist_dict.values()) / \
            (self.init_gb_astr.lattice.c * 2)

        # remove non-relevant sites from the structure before sending to
        # dc.satisfies_all_dists(). This saves lot of time.
        dc_astr = copy.deepcopy(child_astr)
        rem_inds = []
        for i, site in enumerate(dc_astr.sites):
            if not (zmin-ztol) <= site.c <= (zmax+ztol):
                rem_inds.append(i)
        dc_astr.remove_sites(rem_inds)

        num_added, tries = 0, 0
        while num_added < diff:  # and tries < 1000: #(leave this structure)
            tries += 1
            # coords = child_astr.cart_coords
            new_c = [unif(0, 1), unif(0, 1), unif(zmin, zmax)]
            new_c = child_astr.lattice.get_cartesian_coords(new_c)
            if dc.satisfies_all_dists(new_c, dc_astr, self.element_syms,
                                      self.min_dist_dict,
                                      new_carts_species=sp):
                dc_astr.append(sp, new_c, coords_are_cartesian=True)
                child_astr.append(sp, new_c, coords_are_cartesian=True)
                num_added += 1
        del dc_astr

    def new_sites_coords(self, carts, old_axis_bounds, new_axis_bounds, axis):
        """
        Given a set of cartesian coordinates, converts them into fractional
        coordinates w.r.t "new_axis_bounds" along the "axis" provided.

        Args:

        carts (list/array): list of cartesian coordinates

        old_axis_bounds (list): fractional bounds of given carts in given axis

        new_axis_bounds (list): fractional bounds of new lattice in given axis

        axis (int): axis along which to make slices (0, 1, 2 for x, y and z)
        """
        old_min, old_max = old_axis_bounds
        new_min, new_max = new_axis_bounds

        # change the coordinates of given carts to the new_axis_bounds
        add_fcs, add_sps = [], []
        for site in carts:
            if axis == 0:
                x = site.a
            elif axis == 1:
                x = site.b
            elif axis == 2:
                x = site.c
            new_x = (x - old_min) / (old_max - old_min)  # normalize
            new_x = new_x * (new_max - new_min) + new_min  # transform

            if axis == 0:
                add_fcs.append([new_x, site.b, site.c])
            elif axis == 1:
                add_fcs.append([site.a, new_x, site.c])
            elif axis == 2:
                add_fcs.append([site.a, site.b, new_x])
            add_sps.append(site.species)

        return add_sps, add_fcs

    def overlap_grains(self):
        """
        Function to create a new interface region (structure) by overlapping
        the top grain and the bottom grain. Randomly removes atoms after
        overlap to maintain required composition. This new interface region
        will be used as the initial population for gb geometry searches.

        No arguments needed
        """
        init_gb_astr = self.init_gb_astr
        iface_thickness = self.iface_thickness

        # Lattice of gb interface: a, b are same,
        # c lattice vector is from input
        latt = init_gb_astr.lattice.matrix.copy()
        latt[2] = [0, 0, iface_thickness]
        latt = Lattice(latt)
        window_frac = iface_thickness / init_gb_astr.lattice.c
        # Cut the portion randomly from bottom grain and top grain
        z_fracs = init_gb_astr.frac_coords[:, 2]
        # bring all between 0, 1
        z_fracs = [i - math.floor(i) for i in z_fracs]

        bot_sites, top_sites = [], []
        while len(bot_sites) <= 1 or len(top_sites) <= 1:
            cut_bot = random.uniform(min(z_fracs), self.iface_z_mid - 0.05)
            top_cut = random.uniform(self.iface_z_mid + 0.05, max(z_fracs))

            # Add sites to the above lattice
            sorted_sites = sorted(init_gb_astr.sites,
                                  key=lambda x: x.coords[2])
            for site in sorted_sites:
                if cut_bot < site.c < cut_bot + window_frac:
                    bot_sites.append(site)
                if top_cut > site.c > top_cut - window_frac:
                    top_sites.append(site)

        new_bot_sps, new_bot_fc = \
            self.new_sites_coords(bot_sites,
                                  [cut_bot, cut_bot + window_frac], [0, 1], 2)
        new_top_sps, new_top_fc = \
            self.new_sites_coords(top_sites,
                                  [top_cut - window_frac, top_cut], [0, 1], 2)
        species = new_bot_sps + new_top_sps
        coords = new_bot_fc + new_top_fc

        child = Structure(latt, species, coords)
        # child.merge_sites(tol=1, mode='delete')
        child = child.get_sorted_structure()
        self.move_coords_inside(child)

        return child

    def overlapped_iface_implant(self, overlapped_iface):
        """
        Function chooses a composition for the random model and then places
        atoms from the given overalpped interface structure to hollow gb

        Args:

        overlapped_iface (obj): pymatgen Structure object of interface region
        """
        hollow_init_gb = self.hollow_init_gb
        gb_c = hollow_init_gb.lattice.c

        # convert the overlapped_iface coords as per the main gb lattice
        overlapped_carts = overlapped_iface.cart_coords
        z_center = (overlapped_carts[:, 2].max() -
                    overlapped_carts[:, 2].min()) / 2
        # Get the z translate vector
        # i.e., gb_z_center - overlapped_z_center
        z_translate = self.iface_z_mid * gb_c - z_center
        gb_iface_carts = overlapped_carts.copy()
        gb_iface_carts[:, 2] = gb_iface_carts[:, 2] + z_translate
        gb_iface_sps = [i.name for i in overlapped_iface.species]

        # get num species for each species present in element_syms
        all_num_sps = [np.random.randint(self.min_num_sp[sp],
                                         self.max_num_sp[sp]+1) for sp in
                       range(self.num_species)]
        all_sps = self.sym_species

        # Add atoms from overlapped iface to hollow gb
        new_gb = copy.deepcopy(hollow_init_gb)
        dc_astr = copy.deepcopy(self.astr_for_dist_check)
        added_inds = []
        for sps, n_sps in zip(all_sps, all_num_sps):
            num_added, tries = 0, 0
            while num_added < n_sps and tries < 1000:
                tries += 1
                test_ind = np.random.randint(0, len(gb_iface_sps))
                if test_ind in added_inds:
                    continue
                if not gb_iface_sps[test_ind] == sps:
                    continue
                test_carts = gb_iface_carts[test_ind]
                if dc.satisfies_all_dists(test_carts, dc_astr,
                                          self.element_syms,
                                          self.min_dist_dict,
                                          new_carts_species=sps):
                    dc_astr.append(sps, test_carts, coords_are_cartesian=True)
                    new_gb.append(sps, test_carts, coords_are_cartesian=True)
                    num_added += 1
                    added_inds.append(test_ind)
            if tries >= 1000:
                # print ('Could not find atoms satisfying distances. '
                #            'Try increasing min distances in input.')
                break
        del dc_astr
        # for sites in new_gb with no sd_flags, add [False, False, False]
        # This will prevent errors in next step
        for i in range(len(new_gb)):
            if 'selective_dynamics' not in new_gb[i].properties.keys():
                new_gb[i].properties['selective_dynamics'] = \
                    [False, False, False]

        return new_gb.get_sorted_structure()

    def grain_implant(self, iface_to_implant):
        """
        Function places the given interface structure completely in the hollow
        gb. If any atom does not satisfy distance check, tries to move that
        atom within max_perturbation.

        Args:

        iface_to_implant (obj): pymatgen structure object to be implanted
        """
        hollow_init_gb = self.hollow_init_gb
        astr_for_dist_check = self.astr_for_dist_check
        iface_z_mid = self.iface_z_mid
        iface_thickness = self.iface_thickness
        gb_c = self.init_gb_astr.lattice.c

        min_z_top = iface_z_mid + (iface_thickness / (gb_c * 2))
        max_z_bot = iface_z_mid - (iface_thickness / (gb_c * 2))

        # Add sites from iface_to_implant to hollow_init_gb
        zmin, zmax = 0, 1  # limits for fract. coordinates along z direction

        # Maintain a tolerance of 0.2 Å between implant and hollow_gb
        ztol = 0.2 / gb_c
        newz_min, newz_max = max_z_bot + ztol, min_z_top - ztol

        add_fcs, add_sps = [], []
        for site in iface_to_implant.sites:
            newc = (site.c - zmin)/(zmax - zmin)  # normalize
            newc = newc * (newz_max - newz_min) + newz_min  # transform
            add_fcs.append([site.a, site.b, newc])
            add_sps.append(site.specie.name)

        # get cart coords of all iface atoms
        add_carts = hollow_init_gb.lattice.get_cartesian_coords(add_fcs)

        # create an astr with only atoms near gb iface from hollow gb
        new_gb = hollow_init_gb.copy()
        for i in range(len(add_carts)):
            # add each site to new_gb if it satisfies distance check
            if dc.satisfies_all_dists(add_carts[i], astr_for_dist_check,
                                      self.element_syms, self.min_dist_dict,
                                      new_carts_species=add_sps[i]):
                new_gb.append(add_sps[i], add_carts[i],
                              coords_are_cartesian=True)
            else:  # try to perturb the atom coords by self.max_perturbation
                replaced = False
                tries = 0
                while not replaced and tries < 100:
                    # If more than 1000 tries automatically
                    # skips adding that atom
                    tries += 1
                    jump = self.hop.max_perturbation
                    perturb = self.hop.get_point_on_sphere(jump)
                    new_cart = add_carts[i] + perturb
                    if dc.satisfies_all_dists(new_cart,
                                              astr_for_dist_check,
                                              self.element_syms,
                                              self.min_dist_dict,
                                              new_carts_species=add_sps[i]):
                        new_gb.append(add_sps[i], new_cart,
                                      coords_are_cartesian=True)
                        replaced = True

        # for sites in new_gb with no sd_flags, add [False, False, False]
        # This will prevent errors in next step
        for i in range(len(new_gb)):
            if 'selective_dynamics' not in new_gb[i].properties.keys():
                new_gb[i].properties['selective_dynamics'] = \
                    [False, False, False]

        return new_gb.get_sorted_structure()

    def separate_gb(self, gb_astr):
        """
        Separates the interface region from grain boundary cell. A tolerance of
        0.2 Å is added along z-direction for the interface structure.

        Returns interface structure

        Args:

        gb_astr (obj): pymatgen structure object of gb
        """
        iface_z_mid = self.iface_z_mid
        iface_thickness = self.iface_thickness

        gb_latt_matrix = gb_astr.lattice.matrix
        gb_c = gb_astr.lattice.c

        # Get new lattices for all grains
        # bot_matrix = gb_latt_matrix.copy()
        # TODO: Add tolerance here
        # bot_matrix[2] = [0, 0, ((gb_c - iface_thickness)/2) + 1.5]
        # bot_latt = Lattice(bot_matrix)

        mid_matrix = gb_latt_matrix.copy()
        mid_matrix[2] = [0, 0, iface_thickness + 0.2]
        mid_latt = Lattice(mid_matrix)

        # top_matrix = gb_latt_matrix.copy()
        # TODO: Add tolerance here
        # top_matrix[2] = [0, 0, ((gb_c - iface_thickness) / 2) + 1.5]
        # top_latt = Lattice(top_matrix)

        # Get coords and species
        gb_sites = gb_astr.sites
        sorted_sites = sorted(gb_sites, key=lambda x: x.coords[2])
        min_z_top = iface_z_mid + (iface_thickness / (gb_c * 2))
        max_z_bot = iface_z_mid - (iface_thickness / (gb_c * 2))

        top_ind, bot_ind = None, None
        for i, site in enumerate(sorted_sites):
            if site.c >= max_z_bot and not bot_ind:
                bot_ind = i
            if site.c >= min_z_top and not top_ind:
                top_ind = i
                break
        # bot_sites = sorted_sites[:bot_ind]
        # top_sites = sorted_sites[top_ind:]
        mid_sites = sorted_sites[bot_ind:top_ind]

        # translate sites in bottom, middle and top grains
        # to maintain clarity
        # bot_z_max, bot_z_min = max_z_bot + 1.5/gb_c, 0
        mid_z_max, mid_z_min = min_z_top, max_z_bot
        # top_z_max, top_z_min = 1, min_z_top - 1.5/gb_c

        # new_bot_fc, new_bot_sps = [], []
        # for site in bot_sites:
        #    newc = (site.c - bot_z_min) / (bot_z_max - bot_z_min)
        #    new_bot_fc.append([site.a, site.b, newc])
        #    new_bot_sps.append(site.species)

        new_mid_fc, new_mid_sps = [], []
        for site in mid_sites:
            newc = (site.c - mid_z_min) / (mid_z_max - mid_z_min + 0.2/gb_c)
            newc = newc + 0.1/(iface_thickness + 0.2)
            new_mid_fc.append([site.a, site.b, newc])
            new_mid_sps.append(site.species)

        # new_top_fc, new_top_sps = [], []
        # for site in top_sites:
        #    newc = (site.c - top_z_min) / (top_z_max - top_z_min)
        #    new_top_fc.append([site.a, site.b, newc])
        #    new_top_sps.append(site.species)

        # bot_grain = Structure(bot_latt, new_bot_sps, new_bot_fc)
        mid_grain = Structure(mid_latt, new_mid_sps, new_mid_fc)
        # top_grain = Structure(top_latt, new_top_sps, new_top_fc)

        return mid_grain

    def fraction_slice(self, astr, axis):
        """
        Function to slice a given structure into 'num_slices' slices along the
        given axis.

        Returns a lattice object which is same for all blocks and list of
        sliced blocks where each block is list of sites

        Args:

        astr (obj):  pymatgen Structure object

        axis (int): axis along which to make slices (0, 1, 2 for x, y and z)
        """
        num_slices = self.num_slices
        # since equal cuts, all blocks would have same lattice
        latt = astr.lattice
        new_mat = astr.lattice.matrix.copy()
        if axis == 0:
            new_mat[0] = [latt.a/num_slices, 0, 0]
        elif axis == 1:
            new_mat[1] = [0, latt.b/num_slices, 0]
        elif axis == 2:
            new_mat[2] = [0, 0, latt.c/num_slices]
        new_latt = Lattice(new_mat)

        all_sites = astr.sites
        sorted_sites = sorted(all_sites, key=lambda x: x.coords[axis])

        # determine the ax - coordinates to make cuts
        # frac coords from 0 - 1
        cut_locs = [(i+1) * 1/num_slices for i in range(num_slices)]
        cut_atom_inds = []
        for n, cut_loc in enumerate(cut_locs):
            for i, site in enumerate(sorted_sites):
                if axis == 0:
                    x = site.a
                elif axis == 1:
                    x = site.b
                elif axis == 2:
                    x = site.c
                if n == len(cut_atom_inds):
                    if x >= cut_loc:
                        cut_atom_inds.append(i)
                        continue
        # Add 0 at beginning and total num of sites at the
        # end for easy indexing
        cut_atom_inds.append(len(sorted_sites))
        cut_atom_inds.reverse()
        cut_atom_inds.append(0)
        cut_atom_inds.reverse()

        # do the same for cut_locs
        cut_locs.reverse()
        cut_locs.append(0)
        cut_locs.reverse()

        # Note: If there are no sorted_sites in any of the blocks,
        # cut_atom_inds would be less than cut_locs. This results in an error
        # in next step. However, it is good to not have such structures (with
        # empty regions) as a block
        blocks = []
        for i in range(len(cut_atom_inds)-1):
            bl = sorted_sites[cut_atom_inds[i]: cut_atom_inds[i+1]]
            bl_bounds = [cut_locs[i], cut_locs[i+1]]
            block = [bl, bl_bounds]
            blocks.append(block)

        return new_latt, blocks

    def join_slices_to_mold(self, blocks_dict, axis):
        """
        Function to join the 'num_slices' number of blocks (lists of sites)
        along the direction of axis and adds them to form a interface
        structure

        Returns pymatgen structure object of the joned interface

        Args:

        blocks_dict (dict): dictionary of blocks list from each parent
        Ex: {'p1': blocks_p1, 'p2': blocks_p2}
            where blocks_p1 = [[sites], [sites], [sites]]
            pre-condition - the blocks all must be of same size (lattice)

        axis: (int) axis along which to make slices (0, 1, 2 for x, y and z)
        """
        num_slices = self.num_slices
        iface_latt = self.iface_latt

        cut_locs = [(i+1) * 1/num_slices for i in range(num_slices)]
        cut_locs.reverse()
        cut_locs.append(0)
        cut_locs.reverse()

        mold_ax_bounds = []
        for i in range(len(cut_locs)-1):
            p = [cut_locs[i], cut_locs[i+1]]
            mold_ax_bounds.append(p)

        random_blocks = []
        i = 0
        while len(random_blocks) < len(mold_ax_bounds):
            # choose blocks from parents alternatively
            # NOTE: Do not shuffle the list of blocks when choosing
            # NOTE: Raises IndexError due to less blocks when there
            # are no sites in some blocks (in fraction_slice). Let it be.
            # Do not resolve it as such child models are not good.
            if i % 2 == 0:
                block = blocks_dict['p1'][i]
                if len(block) == 0:
                    block = blocks_dict['p2'][i]
                random_blocks.append(block)
            if i % 2 == 1:
                block = blocks_dict['p2'][i]
                if len(block) == 0:
                    block = blocks_dict['p1'][i]
                random_blocks.append(block)
            i += 1

        species, coords = [], []
        for ax_bnds, block in zip(mold_ax_bounds, random_blocks):
            block_sites, block_bounds = block
            new_sps, new_fcs = self.new_sites_coords(block_sites, block_bounds,
                                                     ax_bnds, axis)
            for sp, fc in zip(new_sps, new_fcs):
                species.append(sp)
                coords.append(fc)

        new_str = Structure(iface_latt, species, coords)

        return new_str.get_sorted_structure()

    def mate(self, select, pool, same_cluster=None):
        """
        Selects two parents and performs 'mate by slicing' operation to return
        a child structure.

        Returns child gb structure, inheritance (method used as string)

        Args:

        select (obj): Select object

        pool (obj): Pool object

        same_cluster (bool): Whether to mate from same cluster or not.
                            If None, then ignored.
        """
        # get two parents
        num_parents = 2
        parents = select.get_parents(pool, num_parents,
                                     same_cluster=same_cluster)
        parent1, parent2 = parents[0], parents[1]
        inheritance = [parent1.label, parent2.label]
        # choose axis to slice
        axes = self.slice_axes
        axis = random.choice(axes)
        # get the slice blocks two lists from two parents
        _, blocks_1 = self.fraction_slice(parent1.gb_iface, axis)
        _, blocks_2 = self.fraction_slice(parent2.gb_iface, axis)
        # check if we have same number of blocks in both parents
        if len(blocks_1) != len(blocks_2):
            return None, None
        # make it to a dict
        blocks_dict = {}
        blocks_dict['p1'] = blocks_1
        blocks_dict['p2'] = blocks_2
        # the lattice for all models (child and parents) is same
        # get a child gb_iface structure
        child = self.join_slices_to_mold(blocks_dict, axis)
        # move all coords inside the lattice
        self.move_coords_inside(child)
        # maintain composition of the child structure
        if child.num_sites > 1:
            child.merge_sites(tol=1, mode='delete')
        # Add this ga_child to init_gb_astr
        child_gb_astr = self.grain_implant(child)

        return child_gb_astr, inheritance

    def random_model(self, reg_id):
        """
        Function to get a random model for the initial population. Similar to
        random_model_obj.random_model. But, the random models are not entirely
        random. They are made by combining the top grain and bottom grain.

        Args:

        reg_id (obj): structure_record.register_id() object
        """
        done = False
        while not done:
            active_iface = self.overlap_grains()
            gb_iface = self.overlapped_iface_implant(active_iface)
            done = self.gb_iface_comp_check(gb_iface)

        gb_model = structure_record.model(gb_iface, reg_id)
        gb_model.inheritance = 'random'
        gb_model.made_by = 'random'
        gb_model.sd_true_above = self.sd_true_above
        gb_model.sd_true_below = self.sd_true_below

        return gb_model

    def get_model_old(self, select, pool, reg_id):
        """
        Function to get a child model through basinhopping or mating operations

        Returns a new gb model object

        Args:

        select (obj): selection.Select object

        pool (obj): selection.Pool object

        reg_id (obj): structure_record.register_id() object
        """
        hop = self.hop

        do_hop = False
        if random.random() <= self.hop_mate_frac:
            do_hop = True

        correct_comp = False
        tries = 0
        if do_hop:
            parent_model = select.get_a_parent(pool)
            label = parent_model.label
        while correct_comp is False and tries <= 10:
            try:
                if do_hop:
                    perturbed_iface, inheritance = hop.perturb_sites(
                        select, pool, model_id=label)
                    self.move_coords_inside(perturbed_iface)
                    new_astr = self.grain_implant(perturbed_iface)
                    maker = 'perturb_sites'
                else:  # Do self.mate()
                    new_astr, inheritance = self.mate(select, pool)
                    maker = 'fraction_slice'
            except:
                if do_hop:
                    parent_model = select.get_a_parent(pool)
                    label = parent_model.label
                continue
            if new_astr is None:
                continue
            if any(np.isnan(new_astr.cart_coords.flatten())):
                continue
            new_astr.sort()
            # update tries for every new_gb created (when reaches this point)
            tries += 1
            correct_comp = self.gb_iface_comp_check(new_astr)

        # Adjust composition after 10 failed attempts
        if not correct_comp:
            rem_inds = self.get_rem_inds(new_astr)
            new_astr.remove_sites(rem_inds)

        new_model = structure_record.model(new_astr, reg_id)
        new_model.inheritance = inheritance
        new_model.made_by = maker
        new_model.sd_true_above = self.sd_true_above
        new_model.sd_true_below = self.sd_true_below

        print(
            f"New model {new_model.label} inheritance is: "
            + f"{new_model.inheritance}")

        if not correct_comp:
            print(f'Adjusted the composition of model {new_model.label}')

        # print ('New model made using {} method on parent models {}'.format(
        #                                    maker, inheritance))

        return new_model

    def get_model(self, select, pool, reg_id):
        """
        Function to get a child model through basinhopping or mating
        operations. Operators are provided from the select object.
        Available operators for gb_ops are:
        perturb_sites, fraction_slice, and fraction_slice_same_cluster and
        fraction_slice_dif_cluster. The latter two are only applicable if
        clustering is being employed.

        Returns a new gb model object

        Args:

        select (obj): selection.Select object

        pool (obj): selection.Pool object

        reg_id (obj): structure_record.register_id() object
        """

        hop = self.hop
        operator = np.random.choice(
            select.operators, p=select.operator_frequencies)

        correct_comp = False
        tries_for_correct_comp = 0
        hollow_bounds = [self.hollow_botz, self.hollow_topz]
        if operator == "perturb_sites" or operator == "perturb_comp":
            parent_model = select.get_a_parent(pool)
            label = parent_model.label
        while correct_comp is False and tries_for_correct_comp <= 10:
            try:
                if operator == "perturb_sites":
                    perturbed_iface, inheritance = hop.perturb_sites(
                        select, pool, model_id=label)
                    # Note: pertured_iface will be None when there are not
                    # enough perturbations performed in parent. Then, an error
                    # occurs in next step
                    self.move_coords_inside(perturbed_iface)
                    new_astr = self.grain_implant(perturbed_iface)
                elif operator == "perturb_comp":
                    perturbed_iface, inheritance = hop.perturb_comp(
                        select,
                        pool,
                        z_bounds=hollow_bounds,
                        dc_astr=self.astr_for_dist_check,
                        model=parent_model)
                    self.move_coords_inside(perturbed_iface)
                    new_astr = self.grain_implant(perturbed_iface)
                elif operator == "fraction_slice_same_cluster":
                    new_astr, inheritance = self.mate(
                        select, pool, same_cluster=True)
                elif operator == "fraction_slice_dif_cluster":
                    new_astr, inheritance = self.mate(
                        select, pool, same_cluster=False)
                elif operator == "fraction_slice":
                    new_astr, inheritance = self.mate(
                        select, pool)
            except:
                print(
                    "Exception! Unable to get correct composition."
                    f"Operator is: {operator}")
                traceback.print_exc()
                if operator == "perturb_sites" or operator == "perturb_comp":
                    parent_model = select.get_a_parent(pool)
                    label = parent_model.label
                continue
            if new_astr is None:
                continue
            if any(np.isnan(new_astr.cart_coords.flatten())):
                continue
            new_astr.sort()
            # update tries_for_correct_comp for every new_gb created
            tries_for_correct_comp += 1
            correct_comp = self.gb_iface_comp_check(new_astr)

        # Adjust composition after 10 failed attempts
        if not correct_comp:
            rem_inds = self.get_rem_inds(new_astr)
            new_astr.remove_sites(rem_inds)

        new_model = structure_record.model(new_astr, reg_id)
        new_model.inheritance = inheritance
        pool.update_parent_selection(inheritance)
        new_model.made_by = operator
        new_model.sd_true_above = self.sd_true_above
        new_model.sd_true_below = self.sd_true_below

        print(
            f"New model {new_model.label} inheritance is: "
            + f"{new_model.inheritance}")

        if not correct_comp:
            print(f'Adjusted the composition of model {new_model.label}')
        return new_model

    def move_coords_inside(self, astr):
        """
        For a given structure object, move all sites within the unit cell.
        Eg: [-0.1, 0.4, 1.2] --> [0.9, 0.4, 0.2]

        Returns 'astr' with all atoms inside

        Args:

        astr (obj): pymatgen Structure object
        """
        species = astr.species
        fc = astr.frac_coords
        fc = np.where((fc < 0) | (fc > 1), fc - np.floor(fc), fc)

        # replace all the coords in astr
        all_inds = [i for i in range(len(species))]
        astr.remove_sites(all_inds)
        for sp, coords in zip(species, fc):
            astr.append(sp, coords, coords_are_cartesian=False)

    def gb_iface_comp_check(self, new_gb):
        """
        Checks that the number of atoms per species of given gb structure
        satisfies the allowed range.

        Returns True if composition satifies the allowed range

        Args:

        new_gb (obj): the grain boundary structure object
        """
        new_gb.sort()
        new_comp = new_gb.composition
        hollow_comp = self.hollow_init_gb.composition

        # DU
        correct_comp = True
        for sp in range(self.num_species):
            sym = self.sym_species[sp]
            min_sp = self.min_num_sp[sp]
            max_sp = self.max_num_sp[sp]
            if not min_sp <= new_comp[sym] - hollow_comp[sym] <= max_sp:
                correct_comp = False
                break

        return correct_comp


class surface_ops(object):
    """
    Contains all the functions related to the structure manipulation of
    surface searches of a slab. Includes both for initial population and
    mating and basinhopping.
    """

    def __init__(self, hop, surface_ops_params):
        """
        Args:

        surface_ops_params: A dictionary with mandatory and other optional
                            keywords as required. The keywords are -

        init_slabs_dict: A dictionary of paths to surface slab configurations.
                         The slabs may be of different areas. New surface
                         configurations are investigated on these different
                         area substrates.


        surface_thickness (float): Thickness of the surface layer from top of
                                   the surface (in Å)

        substrate_thickness (float): Thickness of the substrate considered from
                                     the bottom of the intial slab structures

        separation (float): The separation between surface layer bottom z and
                            substrate top z

        constrain_z (bool): Whether z coordinate should be randomly chosen or
                            taken from the initial slab structures. (Used in
                            combination with XRR code)

        composition (str): The composition of the surface layer (Ex: 'Al2O3')
                           If not given, surface layer composition will satisfy
                           only min-max atoms per species

        min_dist_dict, max_dist_dict, num_slices, species_dict and element_syms
        will be added to surface_ops_params internally for conveniece
        """
        # Treat substrates differently based on their areas
        if 'init_slabs_dict' not in surface_ops_params:
            print("Error: Please provide path to initial surface "
                  "configuration slabs as init_slabs_dict")
        else:
            self.init_slabs_dict = surface_ops_params['init_slabs_dict']

        self.surface_thickness = 1  # in Å
        if 'surface_thickness' in surface_ops_params:
            self.surface_thickness = surface_ops_params['surface_thickness']

        self.substrate_thickness = 10  # in Å
        if 'substrate_thickness' in surface_ops_params:
            self.substrate_thickness = surface_ops_params[
                'substrate_thickness'
            ]

        # separation between surface layer bottom z and substrate top z
        self.separation = 2  # Å
        if 'separation' in surface_ops_params:
            self.separation = surface_ops_params['separation']

        self.constrain_z = False
        if 'constrain_z' in surface_ops_params:
            self.constrain_z = surface_ops_params['constrain_z']

        self.sd_no_z = False
        if 'sd_no_z' in surface_ops_params:
            self.sd_no_z = surface_ops_params['sd_no_z']

        self.sd_cut_off = self.substrate_thickness  # default freeze substrate
        if 'sd_cut_off' in surface_ops_params:
            self.sd_cut_off = surface_ops_params['sd_cut_off']

        # The composition of the surface layer
        self.comp_dict = None
        if 'comp_dict' in surface_ops_params:
            self.comp_dict = surface_ops_params['comp_dict']

        # Following keywords will be present in surface_ops_params
        self.min_dist_dict = surface_ops_params['min_dist_dict']
        self.max_dist_dict = surface_ops_params['max_dist_dict']
        self.num_slices = surface_ops_params['num_slices']
        self.species_dict = surface_ops_params['species_dict']
        for sp_key in self.species_dict.keys():
            if 'bonds_to' not in self.species_dict[sp_key]:
                self.species_dict[sp_key]['bonds_to'] = None
        self.element_syms = surface_ops_params['element_syms']

        # Add basinhopping object (hop) as an attribute for convenience
        self.hop = hop
        # basinhopping vs mating fraction
        self.hop_mate_frac = 0.3  # 30% hop, 70% mate
        if 'hop_mate_frac' in surface_ops_params:
            self.hop_mate_frac = surface_ops_params['hop_mate_frac']

        self.num_slices = 2
        if 'num_slices' in surface_ops_params:
            self.num_slices = surface_ops_params['num_slices']

    def init_zs_to_species_dict(self, slab_astr, species_id):
        """
        Function to add the z-coordinates of the atoms in the surface layer to
        the species dict.

        Returns True if atoms of all species are successfully added. Returns
        False if atoms of any one species are not present in the surface layer
        of the slab_astr.

        Args:

        species_id (int): It is 1 or 2 or 3 dpepending on whether it is
                          species1 or species2 or species3 and so on..
        """
        surface_thickness = self.surface_thickness
        element_syms = self.element_syms
        species_dict = self.species_dict

        slab_sites = slab_astr.sites
        z_cart_max = slab_astr.cart_coords[:, 2].max()

        surface_sites = [site for site in slab_sites if
                         site.coords[2] > (z_cart_max - surface_thickness)]
        z_carts = np.unique([site.coords[2] for site in surface_sites
                             if site.specie.name == element_syms[species_id]])

        if len(z_carts) == 0:
            print('{} atoms are not present in the surface layer '
                  'of init_slab_astr'.format(element_syms[species_id]))
        added = False
        for k in species_dict.keys():
            if species_dict[k]['name'] == element_syms[species_id]:
                self.species_dict[k]['z_carts'] = z_carts
                added = True

        return added

    def get_zs_for_species(self, species_name, atoms_per_species):
        """
        Function to get a list of cartesian z-coordinates for the given
        species. The list contains the number of atoms w.r.t atoms_per_species

        Args:

        species_name (str): The symbol of the species in the surface layer for
                            which z-coordinates are required

        atoms_per_species (dict): a dict which specifies the number of atoms
                                  for each species
        """
        # TODO: Add a tolerance parameter to slightly vary z-coords
        species_dict = self.species_dict
        num_species = atoms_per_species[species_name]

        for k in species_dict.keys():
            if species_dict[k]['name'] == species_name:
                species_z_carts = species_dict[k]['z_carts']
                break

        species_zs = []
        copy_z_carts = list(species_z_carts.copy())
        while len(species_zs) < num_species:
            if len(copy_z_carts) > 0:
                random.shuffle(copy_z_carts)
                z = copy_z_carts.pop()
                species_zs.append(z)
            else:
                copy_z_carts = list(species_z_carts.copy())
                continue

        return species_zs

    def choose_init_slab(self, ab=None, abs_tol=0.2):
        """
        Returns a slab_astr (pymatgen structure) object by choosing randomly
        from the init_slabs_dict

        Args:

        ab (array/list (2, 3)): Two lattice vectors a, b given as an array/list

        abs_tol (float): The maximum value for the sum of absolute difference
                         between the "ab" given and "slab_ab"
        """
        init_slabs_dict = self.init_slabs_dict
        keys = list(init_slabs_dict.keys())
        random.shuffle(keys)

        if ab is None:
            # return first match since keys are already shuffled
            return Structure.from_file(init_slabs_dict[keys[0]])
        else:
            for key in keys:
                astr = Structure.from_file(init_slabs_dict[key])
                slab_ab = astr.lattice.matrix[:2]
                diff = np.array(ab) - slab_ab
                # return first match since keys are already shuffled
                if np.absolute(diff).sum() < abs_tol:
                    return astr

    def get_substrate(self, slab_astr=None, ab=None, abs_tol=0.2):
        """
        Function to get the substrate (bottom) part from the given slab
        structure. If slab structure is not given, a random slab is selected
        from the init_slabs_dict. Substrate part is measured according to
        substrate thickness parameter measured from bottom of the slab.

        Args:

        slab_astr (obj): pymatgen Structure object of a slab
                         (substrate+surface)

        ab (array/list (2, 3)): Two lattice vectors a, b
                                given as an array/list

        abs_tol (float): The maximum value for the sum of absolute
                         difference between the "ab" given and "slab_ab"
        """
        if slab_astr is None:
            # Choose a init_slab from the dict
            slab_astr = self.choose_init_slab(ab=ab, abs_tol=abs_tol)

        # Get substrate from the slab_astr
        bot_z_cart = slab_astr.cart_coords[:, 2].min()
        slab_sites = slab_astr.sites
        sub_inds = [i for i, site in enumerate(slab_sites) if
                    site.coords[2] - bot_z_cart <= self.substrate_thickness]

        sub_sps = [slab_astr.species[i] for i in sub_inds]
        sub_fracs = [slab_astr.frac_coords[i] for i in sub_inds]
        substrate = Structure(slab_astr.lattice, sub_sps, sub_fracs,
                              coords_are_cartesian=False)
        return substrate

    def random_surface_layer(self, slab_astr):
        """
        Returns a random surface layer for initial population

        Args:

        slab_astr (obj): pymatgen Structure object of a slab
                         (substrate+surface)

        Algorithm:

        1. Get the no. of atoms for each species

        Update: Alternatively, if constrain_z, get the fixed z-coordinates for
        each atom of each species at this step. Later get x, y coordinates
        using the get_random_coords() and get_connected_coords() function. This
        way, the min and max distances will be satisfied together with the z-
        constraints.

        2. Decide whether fully random or semi-random

        3. For fully random:
               a. Add random coordinates for each atom to the lattice
           For semi-random:
               a. Add species 1 coords randomly (use get_coords_random())
               b. Add species 2 coords using get_connected_coords()
               c. Add species 3 coords using either of the functions

        4. If constrain_z is True, check for the set of z-coordinates

        5. Modify the atoms such that they all have the required z-coordinates

        Join the surface layer on top of same sized substrate at a given
        separation distance
        """
        lattice = slab_astr.lattice
        species_dict = self.species_dict
        element_syms = self.element_syms

        # Get the no. of atoms per species needed in the surface layer
        atoms_per_species = self.get_atoms_per_species()

        # NOTE: XRR code output is strongly influenced by changes in the atoms
        # along z-direction. So, in surface_ops(), if constrain_z is True, the
        # z-coordinates are taken from the init_slabs surface layers.
        # If constrain_z is False, the atoms in surface layer are randomly
        # populated.

        # Add z-carts from the slab_astr to species_dict for each species
        z_carts = None
        if self.constrain_z:
            for key in self.element_syms.keys():
                self.init_zs_to_species_dict(slab_astr, key)
            z_carts = self.get_zs_for_species(element_syms[1],
                                              atoms_per_species)

        # Add random coords for species 1
        num_sp1_needed = atoms_per_species[element_syms[1]]
        min_dist = self.min_dist_dict['sp1_sp1']
        max_dist = self.max_dist_dict['sp1_sp1']
        sp1_random_coords = self.get_random_coords_for_species(
            num_sp1_needed, lattice, min_dist,
            max_dist=max_dist, z_carts=z_carts)
        sps = [element_syms[1] for i in range(len(sp1_random_coords))]
        coords = sp1_random_coords

        # Fix the species 1 coords and
        # Get species 2, 3 either as bonded or random
        if 2 in element_syms.keys():
            num_sp2_needed = atoms_per_species[element_syms[2]]
            # Get bond distance limits for species 1-2 & 2-2
            min_dist_12 = self.min_dist_dict['sp1_sp2']
            max_dist_12 = self.max_dist_dict['sp1_sp2']
            min_dist_22 = self.min_dist_dict['sp2_sp2']
            sp2_z_carts = None
            if self.constrain_z:
                sp2_z_carts = self.get_zs_for_species(element_syms[2],
                                                      atoms_per_species)
                if len(sp2_z_carts) == 0:  # use random when 0
                    sp2_z_carts = None

            if species_dict['species2']['bonds_to'] is None:
                # Get random coords for species 2 as well
                sp2_frac_coords = self.get_secondary_random_coords(
                    num_sp2_needed, lattice,
                    sp1_random_coords, min_dist_12,
                    min_dist_22, z_carts=sp2_z_carts)
            elif species_dict['species2']['bonds_to'] == 1:
                # Get connected coords for species 2
                sp2_frac_coords = self.get_connected_coords(
                    num_sp2_needed, lattice,
                    sp1_random_coords, min_dist_12,
                    max_dist_12, min_dist_22,
                    connected_z_carts=sp2_z_carts)
            else:
                print("Error: Species2 'bonds_to' should be either None or 1")
            sps += [element_syms[2] for i in range(len(sp2_frac_coords))]
            coords += sp2_frac_coords

        if 3 in element_syms.keys():
            num_sp3_needed = atoms_per_species[element_syms[3]]
            # Get distances for 1-3, 2-3 && 3-3
            min_dist_13 = self.min_dist_dict['sp1_sp3']
            max_dist_13 = self.max_dist_dict['sp1_sp3']
            min_dist_23 = self.min_dist_dict['sp2_sp3']
            max_dist_23 = self.max_dist_dict['sp2_sp3']
            min_dist_33 = self.min_dist_dict['sp3_sp3']
            sp3_z_carts = None
            if self.constrain_z:
                sp3_z_carts = self.get_zs_for_species(element_syms[3],
                                                      atoms_per_species)
                if len(sp3_z_carts) == 0:  # use random when 0
                    sp3_z_carts = None

            if species_dict['species3']['bonds_to'] is None:
                # Get random coords for species 2 as well
                sp3_frac_coords = self.get_secondary_random_coords(
                    num_sp3_needed, lattice,
                    sp1_random_coords, min_dist_13,
                    min_dist_33,
                    other_min_dist=min_dist_23,
                    other_frac_coords=sp2_frac_coords,
                    z_carts=sp3_z_carts)

            elif species_dict['species2']['bonds_to'] == 1:
                # Get connected coords for species 2
                sp3_frac_coords = self.get_connected_coords(
                    num_sp3_needed, lattice,
                    sp1_random_coords, min_dist_13,
                    max_dist_13, min_dist_33,
                    connected_z_carts=sp3_z_carts)
                # TODO: Here 2, 3 bond distances are not checked

            elif species_dict['species2']['bonds_to'] == 2:
                # Get connected coords for species 2
                sp3_frac_coords = self.get_connected_coords(
                    num_sp3_needed, lattice,
                    sp2_frac_coords, min_dist_23,
                    max_dist_23, min_dist_33,
                    connected_z_carts=sp3_z_carts)
                # TODO: Here 1, 3 bond distances are not checked
            else:
                print("Error: Species3 'bonds_to' should be either None/1/2")
            sps += [element_syms[3] for i in range(len(sp3_frac_coords))]
            coords += sp3_frac_coords

        new_surf_layer = Structure(lattice, sps, coords,
                                   coords_are_cartesian=False)
        new_surf_layer.sort()

        return new_surf_layer

    def random_model(self, reg_id):
        """
        Returns a model object which is a random_surface_layer on same area
        substrate. This is for the initial population.

        Args:

        reg_id (obj): structure_record.register_id() object
        """
        # Choose a init_slab from the dict
        slab_astr = self.choose_init_slab()
        substrate = self.get_substrate(slab_astr=slab_astr)

        # Get substrate from the slab_astr
        # bot_z_cart = slab_astr.cart_coords[:, 2].min()
        # slab_sites = slab_astr.sites
        # sub_inds = [i for i, site in enumerate(slab_sites) if \
        #              site.coords[2] - bot_z_cart <= self.substrate_thickness]

        # sub_sps = [slab_astr.species[i] for i in sub_inds]
        # sub_fracs = [slab_astr.frac_coords[i] for i in sub_inds]
        # substrate = Structure(slab_astr.lattice, sub_sps, sub_fracs,
        #                                coords_are_cartesian=False)

        # Get the surface layer size of the slab
        surface_layer = self.random_surface_layer(slab_astr)

        # Combine substrate and the surface layer using separation
        surface_minz = surface_layer.cart_coords[:, 2].min()
        substrate_maxz = substrate.cart_coords[:, 2].max()
        z_diff = substrate_maxz + self.separation - surface_minz
        # Re-scale the z-coordinates to get separation correctly
        new_surface_carts = surface_layer.cart_coords.copy()
        new_surface_carts[:, 2] += z_diff
        # Add the surface atoms to substrate
        for i in range(len(new_surface_carts)):
            substrate.append(surface_layer.species[i], new_surface_carts[i],
                             coords_are_cartesian=True)

        surf_model = structure_record.model(substrate, reg_id)
        surf_model.inheritance = 'random'

        return surf_model

    def get_model(self, select, pool, reg_id):
        """
        Returns a new model created either by mating or basinhopping

        Args:

        select (obj): selection.Select object

        pool (obj): selection.Pool object

        reg_id (obj): structure_record.register_id object for bookkeeping
        """
        hop = self.hop

        correct_comp = False
        while correct_comp is False:
            try:
                if random.random() <= self.hop_mate_frac:
                    # basinhopping
                    perturbed_slab, inheritance \
                        = hop.perturb_sites(
                            select,
                            pool,
                            surface_thickness=self.surface_thickness
                        )
                    self.move_coords_inside(perturbed_slab)
                    new_astr = perturbed_slab
                    maker = 'perturb_sites'
                else:  # mating
                    new_astr, inheritance = self.mate(select, pool)
                    maker = 'fraction_slice'
            except:
                continue
            if new_astr is None:
                continue
            if any(np.isnan(new_astr.cart_coords.flatten())):
                continue
            new_astr.sort()
            # Constrain z coords to be that in the initial structure
            if self.constrain_z:
                self.set_zs_from_init_astr(new_astr)
            new_astr, correct_comp = self.surface_comp_check(new_astr)

        new_model = structure_record.model(new_astr, reg_id)
        new_model.inheritance = inheritance
        new_model.made_by = maker

        return new_model

    def get_atoms_per_species(self):
        """
        Returns a dict with number of atoms for each species made from
        species_dict to be used while making a random model for initial
        population.
        """
        species_dict = self.species_dict
        comp_dict = self.comp_dict

        atoms_per_species = {}
        # get num_species_1
        for k in species_dict.keys():
            name = species_dict[k]['name']
            num_atoms = random.randint(species_dict[k]['min_num'],
                                       species_dict[k]['max_num'])
            atoms_per_species[name] = num_atoms

        # if comp_dict is given, get the num atoms for other species
        # based on composition
        # Currently gives atoms_per_species close to required comp.
        # Ex: For Al2O3; instead of Al_3O_4.5 gives Al3O4 (nearest integer).
        # TODO: Enable exact composition option
        if comp_dict:
            # get total number of species in surface layer using comp_dict
            all_species = list(comp_dict.keys())
            # fix no. of atoms for species 1
            fixed_sps = all_species[0]
            del all_species[0]
            fixed_num = atoms_per_species[fixed_sps]

            for species in all_species:
                atoms_per_species[species] = int(fixed_num *
                                                 comp_dict[species]
                                                 / comp_dict[fixed_sps])

        return atoms_per_species

    def get_random_coords_for_species(self, num_coords_needed, lattice,
                                      min_dist, max_dist=None, z_carts=None):
        """
        Returns a list of random fractional coords as required
        (num_coords_needed), each separated by a distance within the min_dist
        and max_dist

        Args:

        num_coords_needed (int): Number of random coordinates needed for the
                                 given species

        lattice (obj): pymatgen lattice object of the surface layer. (The c
                       lattice vector should be corresponding to the fractional
                       z_carts)

        min_dist (float): the minimum distance required between any two atoms
                          of the given species

        max_dist (float): the maximum distance within which at least one atom
                          of same species should exist

        z_carts (list): The list of (fractional) z-coordinates for all the
                         atoms of the given species. The length of the list
                         should be equal to num_coords_needed parameter

        Algorithm:

        While len(random_coords) < num;
            Get a fractional coordinate.
            Calculate the distance between the two coords  wrt lattice
            If the distance satisfies min and max dist,
            add it to random_coords.
            Else, continue..
        """
        random_coords = []
        coords = [random.random(), random.random(), random.random()]
        if z_carts:
            # divide z_carts by the c lattice vector of lattice
            z_carts = [z/lattice.c for z in z_carts]
            # Replace z coordinate with given z-coord
            coords[2] = z_carts[0]
            num_added = 1
        random_coords.append(coords)

        tries = 0
        while num_added < num_coords_needed and tries < 1000:
            tries += 1
            new_fracs = [random.random(), random.random(), random.random()]
            if z_carts:
                new_fracs[2] = z_carts[num_added]
            new_carts = lattice.get_cartesian_coords(new_fracs)
            if len(lattice.get_points_in_sphere(random_coords,
                                                new_carts, min_dist)) == 0:
                random_coords.append(new_fracs)
                num_added += 1
                tries = 0

        random.shuffle(random_coords)
        return random_coords

    def get_secondary_random_coords(self, num_coords_needed, lattice,
                                    primary_frac_coords, min_dist_12,
                                    min_dist_22, other_min_dist=None,
                                    other_frac_coords=None, z_carts=None):
        """
        Returns random coords which satisfy different distance constaints -
        among same species and with species 1. The species 1 coords should be
        provided as primary_frac_coords.

        Args:

        num_coords_needed (int): Number of random coordinates needed for the
                                 given species

        lattice (obj): pymatgen lattice object of the surface layer. (The c
                       lattice vector should be corresponding to the fractional
                       z_carts)

        primary_frac_coords (list/array(nx3)): List of primary coords. Minimum
                                               distance is checked using
                                               min_dist_12

        min_dist_12 (float): the minimum distance required between
                             any two atoms of species 2 & 1

        min_dist_22 (float): the minimum distance between two atoms of
                             species 2 & 2 (secondary speices)

        z_carts (list): The list of (fractional) z-coordinates for all the
                         atoms of the given species. The length of the list
                         should be equal to the num_coords_needed parameter

        """
        # Add random skipped coords to this structure
        tries, num_added = 0, 0
        if z_carts is not None:
            z_fracs = np.array(z_carts) / lattice.c
            if len(z_fracs) != num_coords_needed:
                print("Error: Z-carts not present for all coords!")

        secondary_coords = []
        while num_added < num_coords_needed and tries < 1000:
            tries += 1
            new_fracs = [random.random(), random.random(), random.random()]
            if z_fracs is not None:
                new_fracs[2] = z_fracs[num_added]
            new_carts = lattice.get_cartesian_coords(new_fracs)

            add_new_carts = True
            if not len(lattice.get_points_in_sphere(primary_frac_coords,
                                                    new_carts,
                                                    min_dist_12)) == 0:
                add_new_carts = False
            if not len(secondary_coords) == 0:
                if not len(lattice.get_points_in_sphere(secondary_coords,
                                                        new_carts,
                                                        min_dist_22)) == 0:
                    add_new_carts = False
            if other_min_dist is not None and other_frac_coords is not None:
                if not len(lattice.get_points_in_sphere(other_frac_coords,
                                                        new_carts,
                                                        other_min_dist)) == 0:
                    add_new_carts = False
            if add_new_carts:
                secondary_coords.append(new_fracs)
                num_added += 1
                tries = 0

        return secondary_coords

    def get_connected_coords(self, lattice, fixed_coords,
                             min_dist_12, max_dist_12, min_dist_22,
                             connected_z_carts=None):
        """
        For a given list of coords (of say species_1), returns the species 2
        coords such that they satisfy -
        - minimum and maximum bond distance between atoms of species 1 & 2 resp
        - minimum bond distance between two atoms of species 2
        - number of bonds for each atom in species 1

        Args:

        lattice (obj): pymatgen lattice object of the surface layer. (The c
                       lattice vector should be corresponding to the fractional
                       z_carts)

        fixed_coords (list/array(nx3)): List of primary coords. The new
                                        coordinates will be connected/bonded to
                                        the fixed coords.

        min_dist_12 (float): the minimum distance required between any two
                             atoms of species 2 & 1

        max_dist_12 (float): the maximum allowed bond distance between species
                             1 & 2. Beyond this, the atoms are considered not
                             connected.

        min_dist_22 (float): the minimum distance between two atoms of
                             species 2 & 2 (secondary speices)

        connected_z_carts (list): The list of (fractional) z-coordinates for
                                  all the atoms of the species to be connected.
                                  The length of the list should be equal to the
                                  num_coords_needed parameter

        NOTE: For a constrain_z type random models, species_2_zs contain
              only the z-coordinate for each random atom. Now we determine only
              the x & y coordinates which satisfy the above requirements

        Algorithm:

        1. Make empty lists of species_2_coords, bonded_species_1_coords
        2. Randomly choose incomplete coord from species_2_zs
        3. Randomly choose a full coord from species_1_coords
        4. Randomly get (x, y) on the circle on the plane of x-y which satisy
           1&2 bond distance
        5. Add the species_1_coords index to bonded_species_1_coords
        6. Add the complete species_2 (x,y,z) to species_2_coords
        7. Go to step 2 and continue with species_1_coords. If all
           species_1_coords are bonded, then again choose randomly from all
        8. Return species_2_coords
        """
        fixed_carts = list(lattice.get_cartesian_coords(fixed_coords))

        connected_coords = []
        skipped_z_carts = []
        for i in range(len(connected_z_carts)):
            if len(fixed_carts) > 0:
                # Choose a fixed atom coords to get connected coords
                center_carts = fixed_carts.pop()
            else:
                fixed_carts = list(lattice.get_cartesian_coords(fixed_coords))
            # Randomly choose z-cart of the new connected coord from the list
            new_coord_z_cart = connected_z_carts[i]

            # Get the bond distance within the limits provided
            tries = 0
            found_coord = False
            while not found_coord and tries < 1000:
                tries += 1
                # Get the xy on the circle which satisfy the distance
                bond_dist = unif(min_dist_12, max_dist_12)
                new_carts = self.get_point_on_circle(center_carts, bond_dist,
                                                     new_coord_z_cart)
                # Check if the new point satisfies all minimum distance
                # constraints
                if len(lattice.get_points_in_sphere(fixed_coords, new_carts,
                                                    min_dist_12)) == 0:
                    if len(connected_coords) == 0:
                        new_fracs = lattice.get_fractional_coords(new_carts)
                        connected_coords.append(new_fracs)
                        print(tries)
                        tries = 0
                        found_coord = True
                        continue
                    if len(lattice.get_points_in_sphere(connected_coords,
                                                        new_carts,
                                                        min_dist_22)) == 0:
                        new_fracs = lattice.get_fractional_coords(new_carts)
                        connected_coords.append(new_fracs)
                        print(tries)
                        tries = 0
                        found_coord = True
            if not found_coord:
                skipped_z_carts.append(new_coord_z_cart)

        # Add random skipped coords to this structure
        tries, num_added = 0, 0
        if len(skipped_z_carts) > 0:
            skipped_z_fracs = np.array(skipped_z_carts) / lattice.c
        else:
            skipped_z_fracs = None
        if skipped_z_fracs is not None:
            while num_added < len(skipped_z_fracs) and tries < 1000:
                tries += 1
                new_fracs = [random.random(), random.random(), random.random()]
                new_fracs[2] = skipped_z_fracs[num_added]
                new_carts = lattice.get_cartesian_coords(new_fracs)
                if len(lattice.get_points_in_sphere(fixed_coords,
                                                    new_carts,
                                                    min_dist_12)) == 0:
                    if len(lattice.get_points_in_sphere(connected_coords,
                                                        new_carts,
                                                        min_dist_22)) == 0:
                        connected_coords.append(new_fracs)
                        num_added += 1
                        tries = 0

        return connected_coords

    def get_point_on_circle(self, center_carts, radius, z_cart):
        """
        Returns (x, y, z_cart) which is at a distance of radius from the
        center_carts

        Args:

        center_carts (list): the cartesian coordiantes of the center of the
                             sphere

        radius (float): radius of the sphere

        z_cart (float): cartesian z-coordinate of the point for which x, y will
                        be calculated
        """
        # Get the vertical distance r_vert
        r_vert = center_carts[2] - z_cart
        # Get the angle theta such that radius * sin (theta) = r_vert
        theta = asin(r_vert / radius)
        # Get r_horz which is radius * cos (theta)
        r_horz = cos(theta)
        # Get a random x within x_min and x_max (x1 +- r_horz)
        x = unif(center_carts[0] - r_horz, center_carts[0] + r_horz)
        # Calculate y to satisfy the equation of circle
        solution = sqrt(radius**2 - (center_carts[0] - x) ** 2 -
                        (center_carts[2] - z_cart) ** 2)

        if random.random() < 0.5:
            y = solution + center_carts[1]
        else:
            y = solution - center_carts[1]

        return [x, y, z_cart]

    def set_zs_from_init_astr(self, new_astr):
        """
        For a given a new child slab structure, the z-coordinates of the
        surface layer atoms will be replaced with those of the initial
        structure with same area.

        Args:

        new_astr (obj): pymatgen Structure object

        Algorithm:

        Get the ab of the new_astr
        Get the init_astr with same ab from the init_slabs_dict
        Get the z-coordinates of the init_astr
        Randomly replace the z-coordiantes according to the species
        """
        new_ab = new_astr.lattice.matrix[:2]
        init_astr = self.choose_init_slab(ab=new_ab)

        for key in self.element_syms.keys():
            self.init_zs_to_species_dict(init_astr, key)

        new_carts = new_astr.cart_coords
        # all_species = new_astr.species
        bot_z_cart = new_carts[:, 2].min()
        surface_inds = [i for i, site in enumerate(new_astr.sites)
                        if site.coords[2] - bot_z_cart >
                        self.substrate_thickness]
        surface_species = [new_astr.sites[i].specie.name
                           for i in surface_inds]
        surface_sites = [new_astr.sites[i] for i in surface_inds]
        new_non_surf_inds = [i for i in range(len(new_carts))
                             if i not in surface_inds]
        substrate_z_max = new_astr.cart_coords[new_non_surf_inds][:, 2].max()

        modified_surf_carts = []
        for site in surface_sites:
            # get new cart_coords
            current_carts = site.coords
            # get z from species_dict
            curr_sps = site.specie.name
            new_z = None
            for key in self.species_dict.keys():
                if self.species_dict[key]['name'] == curr_sps:
                    new_z = random.choice(self.species_dict[key]['z_carts'])
                    break
            if new_z is None:
                print('Error: Surface site species not in species dict.')

            modified_carts = [current_carts[0], current_carts[1], new_z]
            modified_surf_carts.append(modified_carts)

        # scale new_z to maintain appropriate separation
        modified_surf_carts = np.array(modified_surf_carts)
        modified_z_min = modified_surf_carts[:, 2].min()
        shift_zs = substrate_z_max + self.separation - modified_z_min
        # change modified z_carts according to separation
        modified_zs = [z + shift_zs for z in modified_surf_carts[:, 2]]
        modified_surf_carts[:, 2] = modified_zs

        # Remove all surface atoms from new_astr
        new_astr.remove_sites(surface_inds)
        for specie, coord in zip(surface_species, modified_surf_carts):
            new_astr.append(specie, coord, coords_are_cartesian=True)

    def get_slices_from_parent(self, parent):
        """
        For a given parent, slices it along a line through center. Returns
        coordinates and species from both halves. Ex: (slice_1_coords,
        slice_1_species, slice_2_coords, slice_2_species)
        Select one half from one parent and the other half from the second
        parent to make a child structure.

        Args:

        parent (obj): structure_record.model object
        """
        # Get surface carts from the parent astr
        parent_sites = parent.astr.sites
        bot_z_cart = parent.astr.cart_coords[:, 2].min()
        surface_inds = [i for i, site in enumerate(parent_sites)
                        if site.coords[2] - bot_z_cart >
                        self.substrate_thickness]
        surface_species = [parent_sites[i].specie.name for i in surface_inds]
        surface_carts = np.array(
            [parent_sites[i].coords for i in surface_inds])

        # Translate this bottom_left_xy to be (0, 0)
        surface_xys = surface_carts.T[0:2].T
        shift_xy = np.array([-surface_xys[:, 0].min(),
                             -surface_xys[:, 1].min()])
        surface_xys = surface_xys + shift_xy

        # Get the point (x_max/2, y_max_2) & slope for the slicing 2D line
        point = [surface_xys[:, 0].max()/2, surface_xys[:, 1].max()/2]
        slope = tan(unif(0, 2*pi))
        # The eq. of line is ==> y = slope * (x - point[0]) + point [1]
        # Get surface carts below line (for P1) and above line (for P2)
        below_inds, above_inds = [], []
        for i, xy in enumerate(surface_xys):
            if xy[1] <= slope * (xy[0] - point[0]) + point[1]:
                below_inds.append(i)
            elif xy[1] >= slope * (xy[0] - point[0]) + point[1]:
                above_inds.append(i)

        below_xys = np.array([surface_xys[i] for i in below_inds])
        below_zs = np.array([surface_carts[i][2] for i in below_inds])
        below_sps = [surface_species[i] for i in below_inds]
        below_carts = np.concatenate((below_xys, below_zs.reshape(-1, 1)),
                                     axis=1)

        above_xys = np.array([surface_xys[i] for i in above_inds])
        above_zs = np.array([surface_carts[i][2] for i in above_inds])
        above_sps = [surface_species[i] for i in above_inds]
        above_carts = np.concatenate((above_xys, above_zs.reshape(-1, 1)),
                                     axis=1)

        return below_carts, below_sps, above_carts, above_sps

    def mate(self, select, pool):
        """
        Performs mating by slicing for given two models and returns child
        structure. Selects two parents from the pool and gets one half from
        each parent. Join both slices and places it on top of substrate. Here
        both the parents should be of same lattice in the x-y direction.

        Returns child surface model

        Args:

        select (obj): selection.Select object

        pool (obj) : selection.Pool object
        """
        # get two parents P1, P2
        num_parents = 2
        parents = select.get_parents(pool, num_parents, same_ab=True)
        parent1, parent2 = parents[0], parents[1]
        inheritance = [parent1.label, parent2.label]
        child_ab = parent1.astr.lattice.matrix[:2]

        # Get one slice each from parent 1 and parent 2 separately
        slice1_carts, slice1_sps, _, __ = self.get_slices_from_parent(parent1)
        _, __, slice2_carts, slice2_sps = self.get_slices_from_parent(parent2)

        # Join both slices
        child_surf_carts = np.concatenate((slice1_carts, slice2_carts))
        child_surf_sps = list(slice1_sps) + list(slice2_sps)

        # Do random translation on xy plane
        random_v = np.array([unif(0, parent1.astr.lattice.a),
                             unif(0, parent1.astr.lattice.b), 0])
        child_surf_carts = child_surf_carts + random_v

        # Place coords on top of substrate
        substrate = self.get_substrate(ab=child_ab)

        # Maintain separation or if constrain_z => set_zs_from_init_astr
        surface_minz = child_surf_carts[:, 2].min()
        substrate_maxz = substrate.cart_coords[:, 2].max()
        z_diff = substrate_maxz + self.separation - surface_minz

        # Re-scale the z-coordinates to get separation correctly
        child_surf_carts[:, 2] += z_diff

        # Place the surface atoms to substrate
        for i in range(len(child_surf_carts)):
            # check if each cart satisfies distance constraints with substrate
            # and rest of the surface layer atoms
            if dc.satisfies_all_dists(child_surf_carts[i], substrate,
                                      self.element_syms, self.min_dist_dict,
                                      max_dist_dict=self.max_dist_dict,
                                      new_carts_species=child_surf_sps[i]):
                substrate.append(child_surf_sps[i], child_surf_carts[i],
                                 coords_are_cartesian=True)

        # move all coords inside the lattice
        self.move_coords_inside(substrate)
        return substrate, inheritance

    def update_atoms_composition(self, slab_astr):
        """
        Function to adjust the surface atoms by adding or removing minimum
        number of atoms to maintain the required composition

        Returns the slab_astr

        Args:

        slab_astr (obj): pymatgen structure object of the slab
        """
        astr = copy.deepcopy(slab_astr)

        # Get surface carts from the slab astr
        slab_sites = astr.sites
        bot_z_cart = astr.cart_coords[:, 2].min()
        non_surface_inds = [i for i, site in enumerate(slab_sites)
                            if site.coords[2] - bot_z_cart <
                            self.substrate_thickness]
        astr.remove_sites(non_surface_inds)

        curr_comp = astr.composition.as_dict()
        target_comp = self.comp_dict
        # Check if the composition should be corrected
        if astr.composition.reduced_formula == \
                Composition(target_comp).reduced_formula:
            return slab_astr

        curr_comp_arr = np.array(list(curr_comp.values()))
        target_comp_arr = np.array(list(target_comp.values()))
        # divide curr comp by target comp of that species
        curr_comp_per_sps = curr_comp_arr / target_comp_arr
        # get average
        curr_comp_avg = np.average(curr_comp_per_sps)
        # Find the nearest integer to the average
        nearest_int = np.rint(curr_comp_avg)
        # Get the "to be" modified composition of the surface
        new_comp = target_comp_arr * nearest_int
        # diff in atoms from current to new comp
        diff_atoms = new_comp - curr_comp_arr

        rem_inds = []
        for i, key in enumerate(curr_comp.keys()):
            if diff_atoms[i] < 0:  # delete random atoms
                removed = 0
                while removed < abs(diff_atoms[i]):
                    rem_i = random.randint(0, slab_astr.num_sites - 1)
                    if slab_sites[rem_i].specie.name == key:
                        rem_inds.append(rem_i)
                        removed += 1
            elif diff_atoms[i] > 0:  # add atoms randomly
                added = 0
                while added < diff_atoms[i]:
                    new_fracs = [random.random(), random.random(),
                                 random.random()]
                    if self.constrain_z:
                        # set z from one of the existing atoms
                        rand_i = random.randint(0, slab_astr.num_sites - 1)
                        if slab_sites[rand_i].specie.name == key:
                            new_fracs[2] = slab_sites[rand_i].frac_coords[2]
                    new_carts = astr.lattice.get_cartesian_coords(new_fracs)
                    if dc.satisfies_all_dists(new_carts, slab_astr,
                                              self.element_syms,
                                              self.min_dist_dict,
                                              max_dist_dict=self.max_dist_dict,
                                              new_carts_species=key):
                        slab_astr.append(key, new_carts,
                                         coords_are_cartesian=True)
                        added += 1
        slab_astr.remove_sites(rem_inds)
        del astr
        return slab_astr

    def surface_comp_check(self, slab_astr):
        """
        Function to check that the atoms in surface layer satisfy the min_num
        and max_num atoms for each species.

        Returns True if satisfies.

        Args:

        slab_astr (obj): pymatgen structure object of the slab
        """
        astr = copy.deepcopy(slab_astr)

        # Get surface carts from the slab astr
        slab_sites = astr.sites
        bot_z_cart = astr.cart_coords[:, 2].min()
        non_surface_inds = [i for i, site in enumerate(slab_sites)
                            if site.coords[2] - bot_z_cart <
                            self.substrate_thickness]
        astr.remove_sites(non_surface_inds)
        surf_comp = astr.composition.as_dict()
        inv_syms = {v: k for k, v in self.element_syms.items()}

        # Check if the composition should be corrected
        if self.comp_dict is not None:
            slab_astr = self.update_atoms_composition(slab_astr)

        # Check comp for the updated slab_astr
        comp_ok = True
        for sps, num in surf_comp.items():
            key = 'species{}'.format(inv_syms[sps])
            min_num = self.species_dict[key]['min_num']
            max_num = self.species_dict[key]['max_num']
            if not min_num <= num <= max_num:
                comp_ok = False

        return slab_astr, comp_ok

    def move_coords_inside(self, astr):
        """
        For a given structure object, move all sites within the unit cell.
        Eg: [-0.1, 0.4, 1.2] --> [0.9, 0.4, 0.2]

        returns 'astr' with all atoms inside

        Args:

        astr (obj): pymatgen Structure object
        """
        species = astr.species
        fc = astr.frac_coords
        fc = np.where((fc < 0) | (fc > 1), fc - np.floor(fc), fc)

        # replace all the coords in astr
        all_inds = [i for i in range(len(species))]
        astr.remove_sites(all_inds)
        for sp, coords in zip(species, fc):
            astr.append(sp, coords, coords_are_cartesian=False)
