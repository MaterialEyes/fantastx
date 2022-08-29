"""
This module contains classes to make the initial models that FANTASTX
will seed the evolutionary algorithm with. Two types of initial models
are supported:

1. Models that use user-provided structures
2. Models that use random structures. It should be noted that this
 method of random model generation is only useful for clusters and bulk
 geometries. Grain boundaries and surfaces have their own unique methods
 of creating random structures, which can be found in the
 [`structure_operations`](../structure_operations-reference/) class.

 ---
"""

from __future__ import division, unicode_literals, print_function
from hashlib import new
from pymatgen.core.structure import Structure, PeriodicSite
from pymatgen.core.lattice import Lattice
from scipy.spatial.transform import Rotation as R
from pymatgen.io.vasp.inputs import Poscar

import os
import numpy as np
from numpy.random import uniform as unif
import collections
import json
from pymatgen.analysis import local_env

from fx19 import distance_check as dc
from fx19 import structure_record
import yaml


class make_model_from_input(object):

    def __init__(self, model_files_path):
        """
        Creates models from the input structure files provided by the user.
        The input structure files must be of either `POSCAR` or `cif` format.
        And, the structure file names should either start with the string
        `POSCAR` or end with the string `.cif` to be considered.

        Arguments:

            model_files_path (str): path to the directory with all the input
             structures
        """
        self.model_files_path = model_files_path
        all_files = os.listdir(model_files_path)
        if len(all_files) == 0:
            print('Provided files path is empty')
        poscars = [i for i in all_files if i.startswith('POSCAR')]
        cifs = [i for i in all_files if i.endswith('.cif')]
        if len(poscars) + len(cifs) == 0:
            print('Provided files path does not have files in either poscar'
                  'or cif format. Other formats are not supported currently.')
        self.all_files = poscars + cifs

    def read_structure(self, reg_id):
        """
        Reads the structure object from the input structure file (cif or
        poscar).

        If there is a structure to be read in, it will return the structure
        object. Otherwise:

        - If it fails, then it will return None.
        - If there are no more structures to be read in, it will return 0.

        Arguments:

            reg_id (obj): `structure_record.register_id()` object for
             book-keeping
        """
        if len(self.all_files) > 0:
            s = self.model_files_path + '/' + self.all_files.pop()
            try:
                astr_from_file = Structure.from_file(s, sort=True)
                input_model = structure_record.model(astr_from_file, reg_id)
                input_model.inheritance = 'from_file'
                return input_model
            except Exception:
                print('Pymatgen failed to make structure from {}'.format(s))
                return None
        else:
            return 0


class make_random_model(object):

    def __init__(self, str_constraints):
        """
        Makes random models for sampling the search space. This class is used
        only for `cluster` or `bulk` geometries. (The initial population for
        `gb` and `surface` are within gb_ops and surface_ops classes.)

        Arguments:

            str_constraints (dict) - all the constraints for making
             random models
        """
        # dictionary of min_dist for different bonds
        self.min_dist_dict = str_constraints['min_dist_dict']
        self.max_dist_dict = str_constraints['max_dist_dict']
        self.species_dict = str_constraints['species_dict']
        self.element_syms = str_constraints['element_syms']
        self.inv_syms = {v: 'sp' + str(k)
                         for k, v in self.element_syms.items()}
        self.shape = str_constraints['shape']

        # defaults
        self.box_abc = [20, 20, 20]  # angstroms
        if 'box_abc' in str_constraints:
            self.box_abc = str_constraints['box_abc']

        self.box_angles = [90, 90, 90]  # degrees
        if 'box_angles' in str_constraints:
            self.box_angles = str_constraints['box_angles']

        self.max_dia = 8  # angstroms
        if 'max_dia' in str_constraints:
            self.max_dia = str_constraints['max_dia']

        self.max_bonds = {}
        for i in self.element_syms:
            self.max_bonds[i] = 2
        if 'max_bonds' in str_constraints:
            self.max_bonds = str_constraints['max_bonds']

        self.num_species = str_constraints['num_species']
        self.allow_self_bonding = False
        if 'allow_random_model_self_bonding' in str_constraints:
            self.allow_self_bonding = str_constraints[
                'allow_random_model_self_bonding']

        if self.num_species == 1:
            if not self.allow_self_bonding:
                print("'allow_random_model_self_bonding' was set to False,"
                      " but only one species is present. Setting to True.")
                self.allow_self_bonding = True
        # save species1 data
        # DU:
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

    def get_cluster_in_box(self):
        """
        Creates a new model for the initial population with a random structure
        of cluster geometry (with vacuum in all directions). The steps that it
        follows are:

        1. Makes lattice with maximum diameter cube
        2. Get the random coordinates
        3. Add vacuum in all three directions

        Returns:

            `structure`: the pymatgen structure object corresponding to the
             cluster.
        """
        max_dia = self.max_dia
        max_bond_dist = max(self.max_dist_dict.values())

        # get species and make an empty lattice box
        species, cum_sum = self.get_n_species()
        latt = Lattice.from_parameters(max_dia, max_dia, max_dia, 90, 90, 90)

        atoms_too_close = True
        while atoms_too_close is True:
            cart_coords = self.get_n_coords_linear(species, max_dia, latt)
            if cart_coords is None:
                continue
            cluster = Structure(latt, species, cart_coords,
                                coords_are_cartesian=True)

            # check distance between different pairs of species
            # atoms_too_close = dc.check_all_bonds(cluster, min_dist_dict,
            #                                      cum_sum)

            # check if atleast one nearest neighbor (nn) less
            # than max_bond_dist
            for i in range(len(cluster.sites)):
                nn = cluster.get_neighbors(cluster.sites[i], max_bond_dist)
                if len(nn) < 1:
                    continue

            atoms_too_close = False

        # put the cluster in a box
        # get thickness of cluster in all three directions
        x_thick = self.get_thickness(cluster, axis=0)
        y_thick = self.get_thickness(cluster, axis=1)
        z_thick = self.get_thickness(cluster, axis=2)

        # add vacuum in all three directions
        self.add_vac(cluster, x_thick, axis=0)
        self.add_vac(cluster, y_thick, axis=1)
        self.add_vac(cluster, z_thick, axis=2)

        return cluster

    def get_bulk_structure(self):
        """
        Creates a new model for the initial population with a random structure
        of bulk geometry. The steps that it
        follows are:

        1. Makes lattice with maximum diameter cube
        2. Get the random coordinates

        Returns:

            `structure`: the pymatgen structure object corresponding to the
             bulk.
        """
        max_bond_dist = max(self.max_dist_dict.values())

        # get species and make an empty lattice box
        species, _ = self.get_n_species()
        latt = Lattice.from_parameters(
            self.box_abc[0], self.box_abc[1], self.box_abc[2],
            self.box_angles[0], self.box_angles[1], self.box_angles[2])

        built_structure = False
        while not built_structure:
            cart_coords = self.get_n_coords_linear_bulk(
                species, latt)
            if cart_coords is None:
                continue
            bulk = Structure(latt, species, cart_coords,
                             coords_are_cartesian=True)

            # check if atleast one nearest neighbor (nn) less
            # than max_bond_dist
            for i in range(len(bulk.sites)):
                nn = bulk.get_neighbors(bulk.sites[i], max_bond_dist)
                if len(nn) < 1:
                    continue

            built_structure = True

        return bulk

    def random_model(self, reg_id):
        """
        Creates a random structure and make it into a `Model` object

        Arguments:

            reg_id: the `reg_id` object which assigns the model its unique
             label

        Returns:

            `model`: the random `model` object
        """
        if self.shape == "bulk":
            astr = self.get_bulk_structure()
        else:
            astr = self.get_cluster_in_box()
        rand_model = structure_record.model(astr, reg_id)
        rand_model.inheritance = 'random'
        rand_model.made_by = 'random'
        return rand_model

    def get_n_species(self):
        """
        Function to get a list of species which satisfy the composition based
        on min num and max num atoms for each species.

        Returns:
            (list, list): lists of the species, and the cumulative sum
             of each species count
        """
        species = []
        count = []
        # NOTE: composition is decided here randomly
        # DU
        for sp_index in range(self.num_species):
            target_species = self.sym_species[sp_index]
            if self.min_num_sp[sp_index] == self.max_num_sp[sp_index]:
                num_species = self.min_num_sp[sp_index]
            else:
                num_species = int(
                    unif(self.min_num_sp[sp_index],
                         self.max_num_sp[sp_index]+1))
            for _ in range(num_species):
                species.append(target_species)
            count.append(num_species)

        # For fixed composition, we do not change total num_atoms
        # So, min and max should be same for each species and,
        # thus supercells are not considered at this point.

        cum_sum = [sum(count[:i+1]) for i in range(len(count))]

        return species, cum_sum

    def get_thickness(self, astr, axis=2):
        """
        Function to get the thickness of the structure along one axis.
        Determines thickness based on the minimum and maximum of cartesian
        coordinates along the axis

        Arguments:

            astr (obj): pymatgen structure object

            axis (int): 0, 1, 2 for x, y, and z axes respectively

        Returns:

            float: the thickness of the structure along the given axis
        """
        cart_coords = astr.cart_coords
        axis_coords = cart_coords[:, axis]
        axis_thickness = max(axis_coords) - min(axis_coords)

        return axis_thickness

    def add_vac(self, astr, cluster_thickness, axis=0):
        """
        For a given structure, adds vacuum on the provided axis.

        Arguments:

            astr (obj): pymatgen structure object

            cluster_thickness (float): thickness of the structure in the axis

            axis (int): axis in which to add vacuum. Must be one of 0, 1, or 2

        Returns:

            `structure`: the modified pymatgen `structure` object
        """

        # get required input data
        frac_coords = astr.frac_coords
        species = astr.species
        # for convenience
        ct = cluster_thickness  # along the axis of interest
        bt = self.box_abc[axis]  # short for box thickness

        # array of axis coords
        axis_coords = frac_coords[:, axis]

        # convert axis coords according to box axis length
        axis_new_coords = (axis_coords - min(axis_coords)) * \
                          ((ct/bt)/(max(axis_coords) - min(axis_coords)))

        # translate new axis coords to center of box axis
        translate_to_center = 0.5 - 0.5 * (ct/bt)
        axis_new_coords = axis_new_coords + translate_to_center

        # create new frac coords with these changes
        new_frac_coords = frac_coords.copy()
        new_frac_coords[:, axis] = axis_new_coords

        # replace new_frac_coords with existing frac_coords
        for i, new_coords in enumerate(new_frac_coords):
            specie = species[i]
            astr.replace(i, specie, new_coords)

        # change lattice with box axis
        latt_matrix = astr.lattice.matrix
        new_latt_matrix = latt_matrix.copy()
        new_latt_matrix[axis][axis] = bt
        new_latt = Lattice(new_latt_matrix)
        astr.lattice = new_latt

        return astr

    def get_n_coords_linear(self, species, max_dia, latt):
        """
        Given maximum allowed diamter of a cluster, this function adds random
        coordinates in a chain like fashion connected to the previous added
        atom which satisfies distance constraints with other atoms present.

        Arguments:

            num_atoms (int) - number of atoms needed in the structure

            max_dia (float) - maximum diameter of the cluster

            latt (obj): pymatgen `Lattice` object which gives all lattice
             information for the atoms

        Returns:

            list: the cartesian coordinates
        """
        # shuffle the species prior to assembly
        np.random.shuffle(species)
        num_atoms = len(species)
        inv_syms = self.inv_syms

        # start with the first site placed at the origin
        old_sps = species[0]
        old_point = np.array([0, 0, 0])
        species_added = [species[0]]
        available_bonds = [self.max_bonds[species[0]]]
        attached_bonds = [[]]
        coords = [old_point]
        coords_added = 1

        # Iterate through each of the remaining species and add them in turn
        ref_atom = 0
        non_referenced_atoms = []
        flipped_species = (self.num_species == 1)
        failed_addition = False
        outside_cluster_attempts = 0
        failed_dist_attempts = 0
        while coords_added < num_atoms:
            new_sps = species[coords_added]

            # Check if same-species bonding is allowed
            bond_ok = True
            if not self.allow_self_bonding:
                bond_ok = (new_sps != old_sps)

            target_bonds_avail = available_bonds[ref_atom]

            # If all checks fail, then first attempt to add the atom to a
            # different atom. If iterated through all atoms, then swap
            # atomic species if possible and repeat.
            # If neither is possible, then return None
            if failed_addition or not bond_ok or target_bonds_avail == 0:
                if len(non_referenced_atoms) == 0 and not flipped_species:
                    # attempt to flip the species
                    for n, sp in enumerate(species[coords_added:]):
                        if sp != new_sps:
                            species[coords_added] = sp
                            species[coords_added + n] = new_sps
                            break
                        if n == len(species[coords_added:]) - 1:
                            return None
                    flipped_species = True
                    non_referenced_atoms = [i for i in range(coords_added)]
                elif len(non_referenced_atoms) == 0 and flipped_species:
                    return None

                # choose a new atom at random to add to
                ref_atom = np.random.choice(non_referenced_atoms)
                non_referenced_atoms.remove(ref_atom)
                old_sps = species[ref_atom]
                old_point = coords[ref_atom]
                failed_addition = False
                continue

            # Draw the bond length from a uniform distribution
            dist_key = inv_syms[old_sps] + '_' + inv_syms[new_sps]
            if inv_syms[old_sps] > inv_syms[new_sps]:
                dist_key = inv_syms[new_sps] + '_' + inv_syms[old_sps]
            min_bond_dist = self.min_dist_dict[dist_key]
            max_bond_dist = self.max_dist_dict[dist_key]
            radius = unif(min_bond_dist, max_bond_dist)

            # Make sure that the coordinate is
            if coords_added > 1:
                min_distance = 3*np.pi/8
                new_point, failed_addition = self.get_max_sep_point_on_sphere(
                    radius,
                    min_distance,
                    attached_bonds[ref_atom],
                    100)

                if failed_addition:
                    continue
            else:
                new_point = self.get_point_on_sphere(radius)

            # translate the point near the old_point
            new_point = new_point + old_point

            # check if the translated point is within cluster diamter box
            # prevent too many repetitive attempts if this check keeps failing
            if not np.linalg.norm(new_point) < max_dia/2:
                outside_cluster_attempts += 1
                if outside_cluster_attempts > 10:
                    failed_addition = True
                    outside_cluster_attempts = 0
                continue

            # check distances with all previous points, accounting for the
            # presence of multiple species. This is important to account for
            # neighbors of neighbors.
            # prevent too many repetitive attempts if this check keeps failing
            if not dc.satisfies_all_dists_quick(new_point, coords, new_sps,
                                                species_added, inv_syms,
                                                self.min_dist_dict,
                                                self.max_dist_dict):
                failed_dist_attempts += 1
                if failed_dist_attempts > 10:
                    failed_addition = True
                    failed_dist_attempts = 0
                continue

            # add the new_point
            coords.append(new_point)
            coords_added += 1
            species_added.append(new_sps)

            # update all bond information
            available_bonds[ref_atom] -= 1
            bond_vector = (new_point - old_point)
            bond_vector = bond_vector/np.linalg.norm(bond_vector)
            attached_bonds[ref_atom].append(bond_vector)
            attached_bonds.append([-bond_vector])
            available_bonds.append(self.max_bonds[new_sps] - 1)

            # newest atom will be first atom to try to add other new atoms to
            old_point = new_point
            old_sps = new_sps
            ref_atom = coords_added - 1
            non_referenced_atoms = [i for i in range(coords_added - 1)]
            flipped_species = (self.num_species == 1)
            outside_cluster_attempts = 0
            failed_dist_attempts = 0
            # failed_addition = False

        # move coords relative to center of cube
        coords = np.array(coords)
        coords = np.full((3,), max_dia/2) + coords

        return coords

    def get_n_coords_linear_bulk(self, species, latt):
        """
        Given lattice information, this function adds random coordinates to
        the box in a chain like fashion. Each new atom is connected to the
        previously added atoms, automatically satisfying (periodic) distance
        constraints with other atoms present.

        Arguments:

            num_atoms (int) - number of atoms needed in the structure

            latt (obj): pymatgen `Lattice` object which gives all lattice
             information for the atoms

        Returns:

            list: the cartesian coordinates
        """
        # shuffle the species prior to assembly
        np.random.shuffle(species)
        num_atoms = len(species)
        inv_syms = self.inv_syms

        # start with the first site placed at the origin
        old_sps = species[0]
        old_point = np.array([0, 0, 0])
        species_added = [species[0]]
        available_bonds = [self.max_bonds[species[0]]]
        attached_bonds = [[]]
        coords = [old_point]
        coords_added = 1

        # Iterate through each of the remaining species and add them in turn
        ref_atom = 0
        non_referenced_atoms = []
        flipped_species = (self.num_species == 1)
        failed_addition = False
        failed_dist_attempts = 0
        while coords_added < num_atoms:
            new_sps = species[coords_added]

            # Check if same-species bonding is allowed
            bond_ok = True
            if not self.allow_self_bonding:
                bond_ok = (new_sps != old_sps)

            target_bonds_avail = available_bonds[ref_atom]

            # If all checks fail, then first attempt to add the atom to a
            # different atom. If iterated through all atoms, then swap
            # atomic species if possible and repeat.
            # If neither is possible, then return None
            if failed_addition or not bond_ok or target_bonds_avail == 0:
                if len(non_referenced_atoms) == 0 and not flipped_species:
                    # attempt to flip the species
                    for n, sp in enumerate(species[coords_added:]):
                        if sp != new_sps:
                            species[coords_added] = sp
                            species[coords_added + n] = new_sps
                            break
                        if n == len(species[coords_added:]) - 1:
                            return None
                    flipped_species = True
                    non_referenced_atoms = [i for i in range(coords_added)]
                elif len(non_referenced_atoms) == 0 and flipped_species:
                    return None

                # choose a new atom at random to add to
                ref_atom = np.random.choice(non_referenced_atoms)
                non_referenced_atoms.remove(ref_atom)
                old_sps = species[ref_atom]
                old_point = coords[ref_atom]
                failed_addition = False
                continue

            # Draw the bond length from a uniform distribution
            dist_key = inv_syms[old_sps] + '_' + inv_syms[new_sps]
            if inv_syms[old_sps] > inv_syms[new_sps]:
                dist_key = inv_syms[new_sps] + '_' + inv_syms[old_sps]
            min_bond_dist = self.min_dist_dict[dist_key]
            max_bond_dist = self.max_dist_dict[dist_key]
            radius = unif(min_bond_dist, max_bond_dist)

            # Make sure that the coordinate is
            if coords_added > 1:
                min_distance = 3 * np.pi / 8
                new_point, failed_addition = self.get_max_sep_point_on_sphere(
                    radius,
                    min_distance,
                    attached_bonds[ref_atom],
                    100)

                if failed_addition:
                    continue
            else:
                new_point = self.get_point_on_sphere(radius)

            # translate the point near the old_point
            new_point = new_point + old_point

            # check distances with all previous points, accounting for the
            # presence of multiple species. This is important to account for
            # neighbors of neighbors.
            # prevent too many repetitive attempts if this check keeps failing
            if not dc.satisfies_all_dists_quick(new_point, coords, new_sps,
                                                species_added, inv_syms,
                                                self.min_dist_dict,
                                                self.max_dist_dict,
                                                latt):
                failed_dist_attempts += 1
                if failed_dist_attempts > 25:
                    failed_addition = True
                    failed_dist_attempts = 0
                continue

            # add the new_point
            coords.append(new_point)
            coords_added += 1
            species_added.append(new_sps)

            # update all bond information
            available_bonds[ref_atom] -= 1
            bond_vector = (new_point - old_point)
            bond_vector = bond_vector/np.linalg.norm(bond_vector)
            attached_bonds[ref_atom].append(bond_vector)
            attached_bonds.append([-bond_vector])
            available_bonds.append(self.max_bonds[new_sps] - 1)

            # newest atom will be first atom to try to add other new atoms to
            old_point = new_point
            old_sps = new_sps
            ref_atom = coords_added - 1
            non_referenced_atoms = [i for i in range(coords_added - 1)]
            flipped_species = (self.num_species == 1)
            failed_dist_attempts = 0

        # wrap coords into periodic box
        for i in coords:
            for j in range(3):
                if i[j] < 0:
                    i[j] += latt.abc[j]

        return np.array(coords)

    def get_point_on_sphere(self, r):
        """
        Get a random point on a sphere of radius *r*

        Arguments:

            r (float) - radius of the sphere

        Returns:

            (array): the random cartesian coordinates on the sphere
        """

        # get random point (x, y, z) using normal distribution
        point = np.random.randn(3)
        # normalize the point
        point_mag = np.linalg.norm(point)
        point = point / point_mag
        # multiply by radius
        point = point * r

        return point

    def get_max_sep_point_on_sphere(self, r, min_ang_distance,
                                    other_points, n_attempts):
        """
        Get a random point on a sphere of radius *r* which is as far
        away from a collection of other points on the sphere as possible

        Arguments:

            r (float) - radius of the sphere

            min_ang_distance (float) - the minimum great circle angular
             distance which must separate the new point and any other
             point already on the sphere

            other_points (array) - the unit vectors of the other points on
             the sphere

            n_attempts (int) - the number of attempts to try and add the point
             to the sphere

        Returns:

            (array, bool):
            - the random cartesian coordinates on the sphere
            - whether or not the point achieved the desired ang. separation
        """

        farthest_distance = -1
        new_point_attempt = 0
        failed_addition = False
        while new_point_attempt <= n_attempts and\
                farthest_distance < min_ang_distance:
            # Grab proposed vector
            new_point = self.get_point_on_sphere(r)
            current_vector = new_point / np.linalg.norm(new_point)

            # determine great circle distance to every other bonded atom
            dotp = np.dot(
                other_points,
                current_vector)
            crossp = np.cross(
                other_points,
                current_vector)
            distances = np.arctan2(
                np.linalg.norm(crossp, axis=1), dotp)

            # if this distance is greater than before, store optimal new bond
            current_distance = np.min(distances)
            if current_distance > farthest_distance:
                optimal_point = np.copy(new_point)
                farthest_distance = current_distance

            new_point_attempt += 1

        if farthest_distance < min_ang_distance:
            failed_addition = True

        return optimal_point, failed_addition


class make_random_molecule_model(object):

    def __init__(self, str_constraints):
        """
        Makes random models for sampling the search space. This class is used
        only for `molecule` geometries. See `make_random_model` for methods
        used for `cluster` and `bulk` geometries, and see the
        `gb_ops` and `surface_ops` classes in `structure_operations` for
        methods used for grain boundaries and surfaces.

        Arguments:

            str_constraints (dict) - all the constraints for making
             random models
        """

        if 'fragments_yaml' not in str_constraints:
            print('Error! Must provide yaml for molecular fragments when'
                  ' constructing random molecules!')
        else:
            self.fragments_yaml = str_constraints['fragments_yaml']

        # Read in fragment yaml
        self.fragments_dict = self._load_fragment_data()

        if 'fragments_directory' not in str_constraints:
            print('Error! Must provide directory which stores fragment xyzs.')
        else:
            self.fragments_directory = str_constraints['fragments_directory']

        # dictionary of min_dist for different bonds
        self.min_dist_dict = str_constraints['min_dist_dict']
        self.element_syms = str_constraints['element_syms']
        self.shape = str_constraints['shape']

        if 'box_abc' in str_constraints:
            self.box_abc = str_constraints['box_abc']

        # defaults
        self.max_dia = 8
        self.max_bond_dist = 3

        if 'max_dia' in str_constraints:
            self.max_dia = str_constraints['max_dia']
        if 'max_bond_dist' in str_constraints:
            self.max_bond_dist = str_constraints['max_bond_dist']

        self.num_species = str_constraints['num_species']
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

        self.assembly_attempts = 100
        self.attachment_attempts = 10
        self.fragment_rotation_attempts = 200
        self.number_of_fragments = 6
        self.add_H = False
        self.bond_lengths = self._load_bond_length_data()

        self.attached_fragments = 0
        # self.visualization_dir = \
        #     "/mnt/c/Users/dunru/Research/fantastx/FeBPy3/
        # attachment_visualizations"

    def _initialize_molecule(self, sf):
        """
        Initializes a molecule by creating an empty box and placing
        the starting atom at the center.

        Arguments:
            starting_atom (str): the chemical symbol corresponding to
             the atom which will be the seed of the molecule.
        """
        # create lattice
        max_dia = self.max_dia
        latt = Lattice.from_parameters(max_dia, max_dia, max_dia, 90, 90, 90)
        box_center = [max_dia/2., max_dia/2., max_dia/2.]

        # create the molecule dictionary structure, and assign it the
        # attachment sites of the initial fragment
        attachment_sites = sf["attachment_sites"].copy()
        available_attachments = sf["available_attachments"].copy()
        molecule = {"attachment_sites": attachment_sites,
                    "available_attachments": available_attachments,
                    "total_avail_attachments": available_attachments.copy(),
                    "fragments": None,
                    "fragment_vectors": {}
                    }

        # Create the starting pymatgen structure
        if sf["type"] == "atom":
            starting_coords = [box_center]
            starting_atom = [sf["name"]]
            astr = Structure(latt, starting_atom, starting_coords,
                             coords_are_cartesian=True)
            molecule["fragment_vectors"][0] = []
            molecule["fixed_atoms"] = [0]
        else:
            if "geometric_center_coords" in sf:
                geom_cent_coords = sf["geometric_center_coords"]
                coord_offset = np.array(
                    [box_center[i] - geom_cent_coords[i] for i in range(3)]
                )
            else:
                coord_offset = np.array(box_center)
            fragment_POSCAR_file = self.fragments_directory + "/" +\
                sf["poscar"]
            fragment_POSCAR = Poscar.from_file(fragment_POSCAR_file)
            fragment_astr = fragment_POSCAR.structure
            species = fragment_astr.atomic_numbers
            new_coords = fragment_astr.cart_coords + coord_offset
            astr = Structure(latt, species, new_coords,
                             coords_are_cartesian=True)

            # To make sure that fragments which will attach don't ovelap
            # with the central fragment, create fragment vectors pointing
            # toward the box center
            for i in attachment_sites:
                molecule["fragment_vectors"][i] = np.array(box_center) -\
                    astr.sites[i].coords
                molecule["fragment_vectors"][i] =\
                    molecule["fragment_vectors"][i] /\
                    np.linalg.norm(molecule["fragment_vectors"][i])

        # add oxidation states to molecule atoms if provided
        if 'oxidation_states' in sf:
            self._oxidize_structure(astr, sf['oxidation_states'])

        # molecule["charge"] += sf["charge"]

        self.attached_fragments = 1
        # poscar_filename = self.visualization_dir + \
        #     "/" + "POSCAR" + str(self.attached_fragments)
        # poscar = Poscar(astr, true_names=True)
        # poscar.write_file(filename=poscar_filename,
        #                   direct=False, vasp4_compatible=False)
        # self.attached_fragments += 1

        return molecule, astr

    def _oxidize_structure(self, astr, oxidation_states):
        """
        Adds oxidation states to a structure

        Arguments:
            astr (obj): pymatgen `structure` object which will have oxidation
             states added
            oxidation_states (iterable): the oxidation states which
             will be assigned. Each state can be provided as a iterable itself,
             in which case one of the items in the iterable will be chosen at
             random as the assigned oxidation state
        """
        if not hasattr(oxidation_states, '__iter__'):
            print("Error! Oxidation state not specified in iterable format")
        else:
            if len(oxidation_states) != astr.num_sites:
                print("Error! Length of oxidation states iterable must be the"
                      " same as the number of sites in the structure.")
            else:
                oxi_states = []
                for i in oxidation_states:
                    if hasattr(i, '__iter__'):
                        oxi_state = float(np.random.choice(i))
                    else:
                        oxi_state = float(i)
                    oxi_states.append(oxi_state)
                astr.add_oxidation_state_by_site(oxi_states)

    def attach_fragment(self, fragment, current_molecule,
                        current_molecule_astr):
        """
        Attaches a molecular fragment to an atom. The steps that it follows
        are:

        1. Choose random fragment from list of fragments
        2. Retrieve fragment POSCAR
        3. Add fragment with random orientation to atom
        4. Check that fragment does not intersect with any other fragments.
        5. If intersects, repeat steps 3-4 until it does not (or until
         a maximum limit is reached).

        Arguments:
            fragment (str): the string id corresponding to the fragment which
            will be added.
            current_molecule (dict): dictionary representation of the model's
             attachment sites, fragments, fragment vectors and charge.
            current_molecule_astr (obj): pymatgen `Structure` object
             corresponding to the molecule

        Returns:
            bool: True if fragment was successfully attached, False otherwise.

        """
        # Grab fragment structure
        fragment_info = self.fragments_dict[fragment]
        fragment_POSCAR_file = self.fragments_directory + "/" +\
            fragment_info["poscar"]
        fragment_POSCAR = Poscar.from_file(fragment_POSCAR_file)

        if "geometric_center_coords" in fragment_info:
            geom_cent_coords = fragment_info["geometric_center_coords"]
            geom_cent_offset = np.linalg.norm(geom_cent_coords)
        else:
            geom_cent_coords = None
            geom_cent_offset = 0.0

        # hydrogenate fragment and update attachment sites and
        # attachment site availability (how many att each site can have)
        # Modify charge of fragment if H are added
        if self.add_H:
            fragment_structure, frag_attach_avail =\
                self._hydrogenate_fragment(
                    fragment_POSCAR.structure,
                    fragment_info)
            frag_attach_sites = fragment_info["attachment_sites"].copy()
            frag_attach_total = frag_attach_avail.copy()

            # num_H = fragment_structure.composition.as_dict()["H"]
            # frag_charge += num_H
        else:
            fragment_structure = fragment_POSCAR.structure.copy()
            frag_attach_sites = fragment_info["attachment_sites"].copy()
            frag_attach_avail = fragment_info["available_attachments"].copy()
            frag_attach_total = frag_attach_avail.copy()

        if 'oxidation_states' in fragment_info:
            oxidation_states = fragment_info["oxidation_states"]
            if self.add_H:
                num_H = fragment_structure.composition.as_dict()["H"]
                oxidation_states.extend([1]*num_H)
            self._oxidize_structure(
                fragment_structure, fragment_info['oxidation_states'])

        # Grab current molecule fragment vectors
        cur_mol_fragment_vectors = current_molecule["fragment_vectors"]

        # initialize variables
        frag_att_list_index = 0
        aa = 0
        embedded_site_ids = []
        fragment_bond_dist = 2.0
        bond_translation = np.array([0, 0, fragment_bond_dist])

        attached = False
        while not attached and aa < self.attachment_attempts:
            # Copy the pymatgen structures to avoid altering them in the loop
            molecule_astr = current_molecule_astr.copy()
            fragment_astr = fragment_structure.copy()

            # Choose the fragment attachment site at random
            frag_attach_probs = np.array(
                frag_attach_avail) / sum(frag_attach_avail)
            num_avail_att_sites = len(frag_attach_sites)
            frag_att_list_index = np.random.choice(range(num_avail_att_sites),
                                                   p=frag_attach_probs)
            frag_attach_site = fragment_astr.sites[
                frag_attach_sites[frag_att_list_index]]

            # Next, if this attachment site is not already on the z-axis,
            # rotate the fragment around the y-axis until it is, and shift
            # until the attachment site is at (0,0,0)
            self._align_fragment_site(fragment_astr, frag_attach_site,
                                      geom_cent_coords)

            # Grab the molecule attachment site at random, with
            # probability determined by availability. Better would
            # be an energetic determination
            m_att_avail = current_molecule["available_attachments"]
            m_att_sites = current_molecule["attachment_sites"]
            m_att_probs = np.array(m_att_avail) / sum(m_att_avail)
            m_att_index = np.random.choice(range(len(m_att_sites)),
                                           p=m_att_probs)
            m_att_site_id = m_att_sites[m_att_index]
            m_att_total = current_molecule["total_avail_attachments"]
            molecule_attach_site = molecule_astr.sites[m_att_site_id]
            molecule_attach_coords = molecule_attach_site.coords

            # grab the bond length for this attachment if a bond
            # dict exists
            if self.bond_lengths is not None:
                frag_atom = frag_attach_site.specie.name
                attach_atom = molecule_attach_site.specie.name
                bond_syms = tuple(sorted([frag_atom, attach_atom]))
                fragment_bond_dist = self.bond_lengths[bond_syms][1.0]
                bond_translation = np.array([0.0, 0.0, fragment_bond_dist])

            # Rotate fragment to lie in appropriate angular region
            mol_site_fragment_vectors = cur_mol_fragment_vectors[m_att_site_id]
            rotation_angles, optimal_vector = self._optimally_rotate_fragment(
                mol_site_fragment_vectors,
                m_att_total[m_att_index],
                m_att_avail[m_att_index],
            )

            # Attach the fragment
            attached, embedded_site_ids = self._merge_structures(
                fragment_astr,
                molecule_astr,
                molecule_attach_coords,
                rotation_angles,
                bond_translation)

            aa += 1

        # If successfully attached (e.g. no sites break distance constraints)
        # then update molecule and set of fragments
        if attached:

            # Update available attachments
            m_att_avail[m_att_index] -= 1

            frag_att_site_avail = frag_attach_avail[frag_att_list_index]
            if frag_att_site_avail == 1:
                frag_attach_sites.pop(frag_att_list_index)
                frag_attach_avail.pop(frag_att_list_index)
                frag_attach_total.pop(frag_att_list_index)
            else:
                frag_att_site_avail -= 1

            # Update molecule fragment vectors
            if m_att_site_id in cur_mol_fragment_vectors:
                cur_mol_fragment_vectors[m_att_site_id].append(
                    optimal_vector)
            else:
                cur_mol_fragment_vectors[m_att_site_id] = \
                    [optimal_vector]

            # Extend molecule with new (non-passivated) addition sites, and
            # fragment info: sites which belong to the fragment,
            # and the fragment axis
            new_mol_attach_sites = [i + current_molecule_astr.num_sites
                                    for i in frag_attach_sites]
            current_molecule["attachment_sites"].extend(new_mol_attach_sites)
            current_molecule["available_attachments"].extend(frag_attach_avail)
            current_molecule["total_avail_attachments"].extend(
                frag_attach_total)

            # Update fragment information stored by the molecule
            stored_fragment_info = {
                "fragment_vector": optimal_vector,
                "molecule_attach_site": m_att_site_id,
                "site_ids": embedded_site_ids
            }
            if current_molecule["fragments"] is not None:
                current_molecule["fragments"].append(stored_fragment_info)
            else:
                current_molecule["fragments"] = [stored_fragment_info]

            # Add fragment vectors which correspond to the fragment itself
            frag_center = optimal_vector * (fragment_bond_dist +
                                            geom_cent_offset)
            for n, a_site in enumerate(frag_attach_sites):
                if frag_attach_avail[n] != 0:
                    mol_site = a_site + current_molecule_astr.num_sites
                    center_point = molecule_attach_coords + frag_center
                    a_site_vector = center_point -\
                        molecule_astr.sites[mol_site].coords
                    a_site_vector = a_site_vector /\
                        np.linalg.norm(a_site_vector)
                    cur_mol_fragment_vectors[mol_site] = \
                        [a_site_vector]

            current_molecule_astr = molecule_astr

        return attached, current_molecule, current_molecule_astr

    def _align_fragment_site(self, fragment_astr, frag_attach_site,
                             geom_cent_coords=None):
        """
        Rotate fragment around geometric center until the fragment
        attaching site is aligned with the negative z axis, then
        shift fragment until fragment attaching site is at the origin.

        This rotation is conducted by rotating around the y-axis, as the
        fragment is assumed to be 2-dimensional and live on the x-z plane.

        Arguments:
            fragment_astr (obj): pymatgen `structure` object corresponding
             to the molecular fragment
            frag_attach_site (obj): pymatgen `PeriodicSite` object
             corresponding to the atomic site which will be bonded to the
             molecule
            geom_cent_coords (array): coordinates of the geometric center
             of the fragment
        """
        asite_coords = frag_attach_site.coords
        if not (np.isclose(asite_coords[0], 0.0) and
                np.isclose(asite_coords[2], 0.0)):
            # align geometric center with the origin
            center_shift = np.zeros(3)
            if not geom_cent_coords is None:
                if not np.allclose(np.zeros(3), geom_cent_coords):
                    center_shift = np.zeros(3) - geom_cent_coords

            asite_coords = asite_coords + center_shift
            angle = np.arctan2(np.linalg.norm(
                np.cross(asite_coords, [0, 0, -1])
            ), np.dot(asite_coords, [0, 0, -1]))
            if asite_coords[0] < 0:
                angle = 2 * np.pi - angle
            rotation = R.from_euler(
                'y', angle, degrees=False)
            z_offset = np.linalg.norm(asite_coords)
            asite_coords = asite_coords - center_shift

            for site in fragment_astr.sites:
                site.coords = site.coords + center_shift
                site.coords = rotation.apply(site.coords)
                site.coords = site.coords + np.array([0, 0, z_offset])
                site.coords = site.coords - center_shift

    def _merge_structures(self, a, b, attach_coords=None,
                          rotation_angles=None, translation=None):
        """
        Append the sites of one pymatgen structure onto another. Appends in
        place, so nothing is returned.

        Arguments:
            a (obj): pymatgen `structure` object that will be appended
            b (obj): base pymatgen `structure` object
            attach_coords (vector): if provided, site coordinates of structure
             a will be taken as being relative to this coordinate
            rotation_angles (iterable): Euler rotation angles around the z-axis,
             x-axis and then z-axis again.
            translation (vector): numpy vector which will be added to all
             sites in a before rotating and appending

        Returns:
            (bool, list):
             - `True` if attachment was successful
             - indices of sites which were appended
        """
        # Attach the fragment
        attached_sites = 0
        attached = False
        embedded_site_ids = []
        for site in a.sites:
            if translation is not None:
                site.coords = site.coords + translation
            if rotation_angles is not None:
                rotation = R.from_euler(
                    'zxz', rotation_angles, degrees=False)
                site.coords = rotation.apply(site.coords)
            if attach_coords is not None:
                site.coords = site.coords + attach_coords

            # attempt to add site to molecule
            atom_satisfies_dists = dc.satisfies_all_dists(
                site.coords,
                b,
                self.element_syms,
                self.min_dist_dict,
                max_dist_dict=None,
                new_carts_species=site.specie.name)
            if not atom_satisfies_dists:
                print("Did not satisfy dists. Need to re-rotate")
                break
            else:
                b.append(site.species,
                         site.coords,
                         coords_are_cartesian=True)
                attached_sites += 1
                embedded_site_ids.append(b.num_sites - 1)

        if attached_sites == a.num_sites:
            attached = True

        return attached, embedded_site_ids

    def _optimally_rotate_fragment(self,
                                   mol_site_fragment_vectors,
                                   m_att_total,
                                   m_att_avail,):
        """
        Rotate fragment around attachment site until it both lies as far
        away from the other attached fragments as possible, and also
        will be approximately the expected angular distance away from
        the other other fragments if the attachment site is fully occupied.

        This rotation will occur by rotations around the z-axis, x-axis, then
        the z-axis again, a ZXZ proper Euler rotation.

        Arguments:
            mol_site_fragment_vectors (list): other fragment vectors attached
             to the molecule attachment site
            m_att_total (int): the total number of fragments which can be
             attached to the molecule attachment site
            m_att_avail (int): the remaining number of fragments which can be
             attached to the molecule attachment site.
        """

        expected_bond_angles = [180, 180, 120, 109, 105, 90]
        sigma_dist = .1*np.pi  # rotation tolerance
        (optimal_vector, zθ_one, xθ, zθ_two) =\
            self._generate_vector_and_angles()
        optimal_angles = [xθ, zθ_two]

        current_vector = np.copy(optimal_vector)
        fra = 0
        smallest_difference = np.inf

        # adjust min_distance based on number of fragment vectors attached currently
        n_preattached_frags = len(mol_site_fragment_vectors)
        if n_preattached_frags != 0:
            # determine approx. expected angular distances for geometry
            if n_preattached_frags != m_att_total - m_att_avail:
                expec_dist = np.pi * \
                    expected_bond_angles[m_att_total - 1] / 180
            else:
                expec_dist = np.pi * \
                    expected_bond_angles[m_att_total - 2] / 180

            # rotate fragment accordingly
            while fra < self.fragment_rotation_attempts and\
                    smallest_difference > sigma_dist:
                # determine great circle distance to every other fragment
                dotp = np.dot(
                    mol_site_fragment_vectors,
                    current_vector)
                crossp = np.cross(
                    mol_site_fragment_vectors,
                    current_vector)
                distances = np.arctan2(
                    np.linalg.norm(crossp, axis=1), dotp)

                # if this distance is greater than before, store optimal
                # fragment vector
                current_dist = np.min(distances)
                expec_difference = abs(current_dist - expec_dist)
                if expec_difference < smallest_difference:
                    optimal_angles = [xθ, zθ_two]
                    optimal_vector = np.copy(current_vector)
                    smallest_difference = expec_difference
                    print(current_dist)

                (current_vector, zθ_one, xθ,
                    zθ_two) = self._generate_vector_and_angles()
                fra += 1

        rotation_angles = [zθ_one, optimal_angles[0], optimal_angles[1]]

        return rotation_angles, optimal_vector

    def _generate_vector_and_angles(self):
        """
        Create random unit vector and return its spherical angles.
        The first angle is only relevant if the vector is aligned
        with the z-axis before rotation, in which case it is the angle
        by which the vector will be twisted before rotation.
        The angles xθ and zθ_two correspond to the latitude angle and
        longitude angle respectively.
        """
        zθ_one = np.random.random_sample()*2*np.pi
        xθ = np.random.random_sample()*np.pi
        zθ_two = np.random.random_sample()*2*np.pi
        vector = np.array(
            [np.sin(xθ) * np.sin(zθ_two),
             -np.sin(xθ) * np.cos(zθ_two),
             np.cos(xθ)]
        )
        return (vector, zθ_one, xθ, zθ_two)

    def _load_bond_length_data(self):
        """
        Loads bond length data from json file. This file can be
        assembled by the user, or inherited from the FANTASTX default.
        The FANTASTX default is comprised of pymatgen data, and data
        added by D.U.
        This bond length data is comprised of keys of sorted element
        pairs, corresponding to dictionary of bond orders and bond lengths.
        """
        cwd = os.getcwd()
        with open(cwd + "/bond_lengths.json") as f:
            data = collections.defaultdict(dict)
            for row in json.load(f):
                els = sorted(row["elements"])
                data[tuple(els)][row["bond_order"]] = row["length"]
            return data

    def _load_fragment_data(self):
        """
        Loads fragment information into a dictionary from a yaml file
        """
        with open(self.fragments_yaml) as ifile:
            fragments_dict = yaml.load(ifile, Loader=yaml.FullLoader)

        return fragments_dict

    def _hydrogenate_fragment(self, fragment, fragment_info):
        """
        Passivates a fragment with hydrogen. The information needed
        to passivate the fragment is provided in the fragment YAML.

        Arguments:
            fragment (obj): pymatgen `Structure` corresponding to
             the fragment
            fragment_info (dict): dictionary containing the information
             for each fragment.
        """
        site_passivation_probabilities =\
            fragment_info["H_site_addition_probs"]
        attachment_sites = fragment_info["attachment_sites"]
        attachment_availability =\
            fragment_info["available_attachments"].copy()

        # Copy the fragment sites into what will be the new fragment
        new_sites = np.copy(fragment.sites)
        new_sites = new_sites.tolist()

        # Read in cutoff distances for assigning bonds
        alt_max_bond_dists = {("N", "C"): 2.0,
                              ("C", "C"): 2.0,
                              ("N", "N"): 2.0}

        # Grab neighbors using these cutoff distances
        NN_object = local_env.CutOffDictNN(alt_max_bond_dists)
        neighbors = NN_object.get_all_nn_info(fragment)

        # Iterate through sites and add a H based on passivation probability
        for index, site in enumerate(fragment.sites):
            prob = site_passivation_probabilities[index]
            r = np.random.random_sample()
            if r <= prob:
                # add H to site!
                neighbor_vectors = [neigh['site'].coords -
                                    site.coords for neigh in neighbors[index]]
                if len(neighbor_vectors) == 2:
                    H_vector = -neighbor_vectors[0] - neighbor_vectors[1]
                    if np.allclose(H_vector, np.zeros(len(H_vector))):
                        rand_vector = np.random.random_sample(len(H_vector))
                        perp_vector = np.cross(H_vector, rand_vector)
                        H_vector = perp_vector / np.linalg.norm(perp_vector)

                    H_vector = H_vector / np.linalg.norm(H_vector)
                elif len(neighbor_vectors) == 3:
                    a1 = neighbor_vectors[0]
                    a2 = neighbor_vectors[1]
                    a3 = neighbor_vectors[2]
                    H_vector = np.cross(a1, a2) +\
                        np.cross(a2, a3) +\
                        np.cross(a3, a1)
                    H_vector = H_vector / np.linalg.norm(H_vector)

                    # check angles
                    i = np.array(neighbor_vectors)
                    neighbor_norm = np.linalg.norm(i)
                    H_norm = np.linalg.norm(H_vector)
                    norm_product = neighbor_norm*H_norm
                    angles = np.arccos(
                        np.dot(i, H_vector) / norm_product * 180 / np.pi
                    )
                    tot_angle = sum(angles)

                    # flip around if put on the wrong side
                    if tot_angle < 270:
                        H_vector = -H_vector
                else:
                    # try to place far away from the other bonded neighbors
                    # precision is not necessary as relaxation will occur
                    nv_mags = np.linalg.norm(neighbor_vectors, axis=1)
                    nv_mags = np.reshape(nv_mags, (-1, 1))
                    normalized_neighbor_vectors = np.divide(
                        neighbor_vectors, nv_mags)
                    (unit_vector, _, _, _) = self._generate_vector_and_angles()
                    optimal_vector = np.copy(unit_vector)
                    r = 0
                    farthest_distance = 0
                    while r < 100:
                        # determine great circle distance to every other
                        # neighbor
                        dotp = np.dot(
                            normalized_neighbor_vectors, unit_vector)
                        crossp = np.cross(
                            normalized_neighbor_vectors, unit_vector)
                        distances = np.arctan2(
                            np.linalg.norm(crossp, axis=1), dotp)

                        current_distance = np.min(distances)
                        if current_distance > farthest_distance:
                            optimal_vector = np.copy(unit_vector)
                            farthest_distance = current_distance

                        (unit_vector, _, _, _) =\
                            self._generate_vector_and_angles()
                        r += 1

                    H_vector = np.copy(optimal_vector)

                H_bond_syms = tuple(sorted(["H", site.specie.name]))
                H_bond_length = self.bond_lengths[H_bond_syms][1.0]
                H_bond = H_bond_length*H_vector
                H_coord = site.coords + H_bond
                H_site = PeriodicSite(
                    species="H",
                    coords=H_coord,
                    lattice=fragment.lattice,
                    coords_are_cartesian=True)
                new_sites.append(H_site)

                # update attachment information for the molecule
                if index in attachment_sites:
                    attach_index = attachment_sites.index(index)
                    attachment_availability[attach_index] -= 1
        hydrogenated_fragment_structure = Structure.from_sites(new_sites)
        return hydrogenated_fragment_structure, attachment_availability

    def _initialize_fragments(self):
        """
        From the fragment dictionary constructed from the fragment YAML,
        assign the starting fragment and randomly choose the set of fragments
        which will be used to assemble the remainder of the molecule.
        """
        starting_fragment = self.fragments_dict[0]

        addable_fragments = list(self.fragments_dict.keys())[1:]

        frag_counts = [self.fragments_dict[i]["count"]
                       for i in addable_fragments]
        assembly_probabilities = np.array(frag_counts)/np.sum(frag_counts)

        # Choose the fragments which will comprise this molecule at random
        chosen_fragments = np.random.choice(addable_fragments,
                                            size=self.number_of_fragments,
                                            replace=True,
                                            p=assembly_probabilities)

        return starting_fragment, chosen_fragments

    def build_molecule(self):
        """
        Constructs a molecule from a set of fragments. Will attempt to add
        fragments until all fragments have been added. If it ever fails to
        add a fragment, it will restart the process. If failure occurs a
        pre-specified number of times, an error is thrown and the random
        model construction process fails.

        !!! note
            The central fragment will always the first fragment in the
            fragment YAML file. This fragment can be either an atom, or a
            fragment itself.
        """
        starting_fragment, chosen_fragments = self._initialize_fragments()

        # Initialize the molecule with only a single seed atom
        molecule, molecule_astr = self._initialize_molecule(
            starting_fragment)
        print("Initialized molecule!")

        # Add fragments
        assembled = False
        assembly_attempts = 0
        while not assembled and assembly_attempts < self.assembly_attempts:
            added_fragments = 0
            for fragment in chosen_fragments:
                attached, molecule, molecule_astr =\
                    self.attach_fragment(
                        fragment, molecule, molecule_astr)
                if not attached:
                    assembly_attempts += 1
                    molecule, molecule_astr =\
                        self._initialize_molecule(starting_fragment)
                    print("Re initialized molecule")
                    break
                else:
                    # print(f"Now molecule is: {molecule}")
                    added_fragments += 1
                    self.attached_fragments += 1
            if added_fragments == self.number_of_fragments:
                assembled = True

        if assembled:
            # Now, sort molecule_astr and molecule representation
            s_indices = np.argsort(molecule_astr)
            s_map = {s_indices[i]: i for i in range(len(s_indices))}
            molecule["attachment_sites"] =\
                [s_map[i] for i in molecule["attachment_sites"]]
            molecule["fixed_atoms"] =\
                [s_map[i] for i in molecule["fixed_atoms"]]

            for fragment in molecule["fragments"]:
                fragment["molecule_attach_site"] =\
                    s_map[fragment["molecule_attach_site"]]
                fragment["site_ids"] =\
                    [s_map[i] for i in fragment["site_ids"]]

            mol_frag_vector_keys = list(molecule["fragment_vectors"].keys())
            for key in mol_frag_vector_keys:
                val = molecule["fragment_vectors"][key]
                molecule["fragment_vectors"].pop(key)
                molecule["fragment_vectors"][s_map[key]] = val

            site_array = np.array(molecule_astr.sites)
            sorted_sites = site_array[s_indices]
            molecule_astr = Structure.from_sites(
                sorted_sites,
                charge=molecule_astr._charge)
            print("Assembled and sorted molecule!")
        else:
            print("Failed to assemble molecule within "
                  f"{self.assembly_attempts} attempts.")

        return molecule, molecule_astr.get_sorted_structure()

    def random_model(self, reg_id):
        """
        Creates a random molecule and make it into a `Model` object. Stores
        the molecule dictionary in the model object for later reference.

        Arguments:

            reg_id: the `reg_id` object which assigns the model its unique
             label

        Returns:

            `model`: the random `model` object
        """
        mol, astr = self.build_molecule()
        rand_model = structure_record.model(astr, reg_id)
        rand_model.molecule_representation = mol
        rand_model.inheritance = 'random'
        rand_model.made_by = 'random'
        return rand_model
