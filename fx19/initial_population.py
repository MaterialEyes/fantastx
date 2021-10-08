
from __future__ import division, unicode_literals, print_function

"""
This module contains classes to make models from user provided structures (if
any) and then remaining as random models for initial population.
"""
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

import os
import numpy as np
from numpy.random import uniform as unif

from fx19 import distance_check as dc
from fx19 import structure_record


class make_model_from_input(object):

    def __init__(self, model_files_path):
        """
        Creates models from the input structure files provided by the user. The
        input structuure files must be of wither 'POSCAR' or 'cif' format. And,
        the structure file names should either start with the string 'POSCAR'
        or end with the string '.cif' to be considered.

        Args:

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
            print('Provied files path do not have files in either poscar or'
                  ' cif format. Other formats are not supported currently.')
        self.all_files = poscars + cifs

    def read_structure(self, reg_id):
        """
        Returns structure object from the input structure file (cif or poscar)

        Returns None if pymatgen structure object failed to create from file

        Returns 0 if all input files are completed

        Args:

        reg_id (obj): structure_record.register_id() object for bookkeeping
        """
        if len(self.all_files) > 0:
            s = self.model_files_path + '/' + self.all_files.pop()
            try:
                astr_from_file = Structure.from_file(s, sort=True)
                input_model = structure_record.model(astr_from_file, reg_id)
                input_model.inheritance = 'from_file'
                return input_model
            except:
                print('Pymatgen failed to make structure from {}'.format(s))
                return None
        else:
            return 0


class make_random_model(object):

    def __init__(self, str_constraints):
        """
        Makes random models for sampling the search space. This class is used
        only for 'cluster' or 'bulk' geometries. (The initial population for
        'gb' and 'surface' are within gb_ops and surface_ops classes.)

        Args:

        str_constraints (dict) - dictionary of all the constraints for making
                                 random models
        """
        # dictionary of min_dist for different bonds
        self.min_dist_dict = str_constraints['min_dist_dict']
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
        of cluster geometry (with vacuum in all directions)

        - Makes lattice with maximum diameter cube
        - Get the random cooridnates
        - Add vacuum in all three directions
        """
        max_dia = self.max_dia
        min_dist_dict = self.min_dist_dict
        max_bond_dist = self.max_bond_dist
        # get species
        species, cum_sum = self.get_n_species()
        num_atoms = len(species)
        latt = Lattice.from_parameters(max_dia, max_dia, max_dia, 90, 90, 90)

        atoms_too_close = True
        while atoms_too_close is True:
            cart_coords = self.get_n_coords_linear(num_atoms, max_dia)
            if cart_coords is None:
                continue
            cluster = Structure(latt, species, cart_coords,
                                coords_are_cartesian=True)

            # check distance between different pairs of species
            atoms_too_close = dc.check_all_bonds(cluster, min_dist_dict,
                                                 cum_sum)

            # check if atleast one nearest neighbor (nn) less
            # than max_bond_dist
            for i in range(len(cluster.sites)):
                nn = cluster.get_neighbors(cluster.sites[i], max_bond_dist)
                if len(nn) < 1:
                    continue

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

    def random_model(self, reg_id):
        """
        Use the random structure created and make it into a Model object

        Args:

        reg_id: the reg_id object which assigns the model its unique label.
        """
        astr = self.get_cluster_in_box()
        rand_model = structure_record.model(astr, reg_id)
        rand_model.inheritance = 'random'
        rand_model.made_by = 'random'
        return rand_model

    def get_n_species(self):
        """
        Function to get a list of species which satisfy the compositon based on
        min num and max num atoms for each species.

        Returns list of species, cumulative sum of each species count
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

        Args:

        astr (obj): pymatgen structure object

        axis (int): 0, 1, 2 for x, y, and z axes respectively
        """
        cart_coords = astr.cart_coords
        axis_coords = cart_coords[:, axis]
        axis_thickness = max(axis_coords) - min(axis_coords)

        return axis_thickness

    def add_vac(self, astr, cluster_thickness, axis=0):
        """
        For a given structure, adds vacuum in the provided axis

        Returns the modified structure object

        Args:

        astr (obj) - pymatgen structure object

        cluster_thickness (float) - thickness of the structure in the axis

        axis (int 0/1/2) - axis in which to add vacuum
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

    def get_n_coords_linear(self, num_atoms, max_dia):
        """
        Given maximum allowed diamter of a cluster, this function adds random
        coordinates in a chain like fashion connected to the previous added
        atom which satisfies distance constraints with other atoms present.

        Returns a list of cartesian coordinates

        Args:

        num_atoms (int) - number of atoms needed in the structure

        max_dia (float) - maximum diameter of the cluster
        """
        # start from origin
        old_point = np.array([0, 0, 0])
        coords = []
        coords_added = 0
        new_point_attempt = 0
        while coords_added < num_atoms:
            min_bond_dist = min(self.min_dist_dict.values())
            max_bond_dist = self.max_bond_dist
            radius = unif(min_bond_dist, max_bond_dist)
            new_point = self.get_point_on_sphere(radius)

            # returns None if the algo cannot add a new point in 500 attempts
            # if the cluster_diameter is too small, this algo hangs trying to
            # add new point
            new_point_attempt += 1
            if new_point_attempt > 1000:
                return None

            # translate the point near the old_point
            new_point = new_point + old_point

            # check if the translated point is within cluster diamter box
            if not np.linalg.norm(new_point) < max_dia/2:
                continue

            # check distances with all previous points
            # using max of min_dists for initial population
            max_of_min_dists = max(self.min_dist_dict.values())
            if not dc.one_to_many_distances(new_point, coords,
                                            max_of_min_dists):
                continue

            # add the new_point and reset the no. of attempts
            coords.append(new_point)
            new_point_attempt = 0
            old_point = new_point
            coords_added += 1

        # move coords relative to center of cube
        coords = np.array(coords)
        coords = np.full((3,), max_dia/2) + coords

        # shuffle the coords
        np.random.shuffle(coords)

        # coords are cartesian
        return coords

    def get_point_on_sphere(self, r):
        """
        Get a random point on a sphere of radius r

        Args:

        r (float) - radius of the sphere
        """

        # get random point (x, y, z) using normal distribution
        point = np.random.randn(3)
        # normalize the point
        point_mag = np.linalg.norm(point)
        point = point / point_mag
        # multiply by radius
        point = point * r

        return point
