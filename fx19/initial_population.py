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
from matplotlib.pyplot import thetagrids
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
import traceback


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
        of cluster geometry (with vacuum in all directions). The steps that it
        follows are:

        1. Makes lattice with maximum diameter cube
        2. Get the random cooridnates
        3. Add vacuum in all three directions

        Returns:

            `structure`: the pymatgen structure object corresponding to the
             cluster.
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
        Creates a random structure and make it into a `Model` object

        Arguments:

            reg_id: the `reg_id` object which assigns the model its unique
             label

        Returns:

            `model`: the random `model` object
        """
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

    def get_n_coords_linear(self, num_atoms, max_dia):
        """
        Given maximum allowed diamter of a cluster, this function adds random
        coordinates in a chain like fashion connected to the previous added
        atom which satisfies distance constraints with other atoms present.

        Arguments:

            num_atoms (int) - number of atoms needed in the structure

            max_dia (float) - maximum diameter of the cluster

        Returns:

            list: the cartesian coordinates
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
            print('Error! Must provide yaml for molecular fragments when constructing'
                  'random molecules!')
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
        self.add_H = True
        if self.add_H:
            self.bond_lengths = self._load_bond_length_data()
        else:
            self.bond_lengths = None

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
                    "fragments": None,
                    "fragment_vectors": {}}

        # Create the starting pymatgen structure
        if sf["type"] == "atom":
            starting_coords = [box_center]
            starting_atom = [sf["name"]]
            astr = Structure(latt, starting_atom, starting_coords,
                             coords_are_cartesian=True)
            molecule["fragment_vectors"][0] = []
        else:
            if "com_coords" in sf:
                com_coords = sf["com_coords"]
                coord_offset = np.array(
                    [box_center[i] - com_coords[i] for i in range(3)]
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

            for i in attachment_sites:
                molecule["fragment_vectors"][i] = np.array(box_center) -\
                    astr.sites[i].coords
                molecule["fragment_vectors"][i] =\
                    molecule["fragment_vectors"][i] /\
                    np.linalg.norm(molecule["fragment_vectors"][i])

        return molecule, astr

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
             attachment sites, fragments and fragment vectors.
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
        num_preexisting_sites = current_molecule_astr.num_sites
        cur_mol_fragment_vectors = current_molecule["fragment_vectors"]

        # hydrogenate fragment and update attachment sites and
        # attachment site availability (how many att each site can have)
        if self.add_H:
            fragment_structure, frag_attach_avail =\
                self._hydrogenate_fragment(
                    fragment_POSCAR.structure,
                    fragment_info)
            frag_attach_sites = fragment_info["attachment_sites"].copy()
        else:
            fragment_structure = fragment_POSCAR.structure.copy()
            frag_attach_sites = fragment_info["attachment_sites"].copy()
            frag_attach_avail = fragment_info["available_attachments"].copy()

        attached = False
        frag_attach_site_id = frag_attach_sites[0]
        frag_att_list_index = 0
        aa = 0
        fragment_bond_dist = 2.0
        bond_translation = np.array([0, 0, fragment_bond_dist])
        while not attached and aa < self.attachment_attempts:
            # Copy the pymatgen structures to avoid altering them in the loop
            molecule_astr = current_molecule_astr.copy()
            fragment_astr = fragment_structure.copy()

            # Choose the fragment attachment site at random
            frag_attach_probs = np.array(
                frag_attach_avail) / sum(frag_attach_avail)

            print(
                f"Available frag attach sites and probs: {frag_attach_sites}"
                f" and {frag_attach_avail}")
            num_avail_att_sites = len(frag_attach_sites)
            frag_att_list_index = np.random.choice(range(num_avail_att_sites),
                                                   p=frag_attach_probs)
            frag_attach_site_id = frag_attach_sites[frag_att_list_index]
            frag_attach_site = fragment_astr.sites[frag_attach_site_id]
            print(f"Chosen fragment attachment site: {frag_attach_site_id}")

            # Next, if this attachment site is not already on the z-axis,
            # rotate the fragment around the y-axis until it is, and shift
            # until the attachment site is at (0,0,0)
            if not (np.isclose(frag_attach_site.coords[0], 0.0) and
                    np.isclose(frag_attach_site.coords[2], 0.0)):
                asite_coords = frag_attach_site.coords
                angle = np.arctan2(np.linalg.norm(
                    np.cross(asite_coords, [0, 0, -1])
                ), np.dot(asite_coords, [0, 0, -1]))
                if asite_coords[0] < 0:
                    angle = 2 * np.pi - angle
                rotation = R.from_euler(
                    'y', angle, degrees=False)
                z_offset = np.linalg.norm(asite_coords)
                for site in fragment_astr.sites:
                    site.coords = rotation.apply(site.coords)
                    site.coords = site.coords + np.array([0, 0, z_offset])

            # Grab the molecule attachment site at random, with
            # probability determined by availability. Better would
            # be an energetic determination
            m_att_avail = current_molecule["available_attachments"]
            m_att_sites = current_molecule["attachment_sites"]

            print(
                f"Available attach sites and probs: {m_att_sites} "
                f"and {m_att_avail}")

            m_att_probs = np.array(m_att_avail) / sum(m_att_avail)
            m_att_index = np.random.choice(range(len(m_att_sites)),
                                           p=m_att_probs)
            m_att_site_id = m_att_sites[m_att_index]

            molecule_attach_site =\
                current_molecule_astr.sites[m_att_site_id]
            molecule_attach_coords = molecule_attach_site.coords

            # All fragments lie on the X-Z plane, centered on the z-axis
            # We will translate the fragment by the bonding distance, then
            # rotate the fragment by random angles around the z, then x, then
            # z axis again (this ZXZ rotation is a proper Euler rotation)
            (optimal_vector, zθ_one, xθ, zθ_two) =\
                self._generate_vector_and_angles()
            current_vector = np.copy(optimal_vector)
            optimal_angles = [xθ, zθ_two]
            fra = 0
            farthest_distance = 0

            min_distance = np.pi/2
            if len(cur_mol_fragment_vectors[m_att_site_id]) != 0:
                while fra < self.fragment_rotation_attempts and\
                        farthest_distance < min_distance:
                    # determine great circle distance to every other fragment
                    dotp = np.dot(
                        cur_mol_fragment_vectors[m_att_site_id],
                        current_vector)
                    crossp = np.cross(
                        cur_mol_fragment_vectors[m_att_site_id],
                        current_vector)
                    distances = np.arctan2(
                        np.linalg.norm(crossp, axis=1), dotp)

                    # if this distance is greater than before, store optimal
                    # fragment vector
                    current_distance = np.min(distances)
                    if current_distance > farthest_distance:
                        optimal_angles = [xθ, zθ_two]
                        optimal_vector = np.copy(current_vector)
                        farthest_distance = current_distance

                    (current_vector, zθ_one, xθ,
                     zθ_two) = self._generate_vector_and_angles()
                    fra += 1

            xθ = optimal_angles[0]
            zθ_two = optimal_angles[1]

            rotation = R.from_euler(
                'zxz', [zθ_one, xθ, zθ_two], degrees=False)

            # Attach the fragment
            attached_sites = 0
            for site in fragment_astr.sites:
                site.coords = site.coords + bond_translation
                site.coords = rotation.apply(site.coords)
                site.coords = site.coords + molecule_attach_coords

                # attempt to add site to molecule
                atom_satisfies_dists = dc.satisfies_all_dists(
                    site.coords,
                    molecule_astr,
                    self.element_syms,
                    self.min_dist_dict,
                    max_dist_dict=None,
                    new_carts_species=site.specie.name)
                if not atom_satisfies_dists:
                    print("Did not satisfy dists. Need to re-rotate")
                    break
                else:
                    molecule_astr.append(site.species,
                                         site.coords,
                                         coords_are_cartesian=True)
                    attached_sites += 1

            if attached_sites == len(fragment_astr.sites):
                attached = True
            aa += 1

        # If successfully attached (e.g. no sites break distance constraints)
        # then update molecule and set of fragments
        if attached:
            current_molecule_astr = molecule_astr
            m_att_avail[m_att_index] -= 1

            frag_att_site_avail = frag_attach_avail[frag_att_list_index]
            if frag_att_site_avail == 1:
                frag_attach_sites.pop(frag_att_list_index)
                frag_attach_avail.pop(frag_att_list_index)
            else:
                frag_att_site_avail -= 1

            # gather

            if m_att_site_id in cur_mol_fragment_vectors:
                cur_mol_fragment_vectors[m_att_site_id].append(
                    optimal_vector)
            else:
                cur_mol_fragment_vectors[m_att_site_id] = \
                    [optimal_vector]

            # Extend molecule with new (non-passivated) addition sites, and
            # fragment info: sites which belong to the fragment,
            # and the fragment axis
            frag_attach_sites =\
                [i + num_preexisting_sites for i in frag_attach_sites]
            current_molecule["attachment_sites"].extend(frag_attach_sites)
            current_molecule["available_attachments"].extend(frag_attach_avail)
            fragment_info = {
                "fragment_vector": current_vector,
                "fragment_sites": fragment_astr.sites
            }
            if current_molecule["fragments"] is not None:
                current_molecule["fragments"].append(fragment_info)
            else:
                current_molecule["fragments"] = [fragment_info]

            # also update fragment vectors for available fragment sites which
            # have not had other fragments attach yet
            if "com_coords" in fragment_info:
                com_offset = np.linalg.norm(fragment_info["com_coords"])
            else:
                com_offset = 0.0
            frag_center = current_vector * (np.linalg.norm(current_vector) +
                                            com_offset)
            for n, a_site in enumerate(frag_attach_sites):
                if frag_attach_avail[n] != 0:
                    a_site_vector = frag_center -\
                        current_molecule_astr.sites[a_site].coords
                    a_site_vector = a_site_vector /\
                        np.linalg.norm(a_site_vector)
                    cur_mol_fragment_vectors[a_site] = \
                        [a_site_vector]

        return attached, current_molecule, current_molecule_astr

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
             np.sin(xθ) * np.cos(zθ_two),
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
            if added_fragments == self.number_of_fragments:
                assembled = True

        if assembled:
            print("Assembled molecule!")
        else:
            print("Failed to assemble molecule within "
                  f"{self.assembly_attempts} attempts.")

        return molecule, molecule_astr

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
