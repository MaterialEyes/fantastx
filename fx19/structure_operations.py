from __future__ import division, unicode_literals, print_function

"""
This module only contains the different funcitons to mutate the structure,
mate two given structures, change composition from a given structure.

Structure constraints
Geometry of search
Mating_probability and no. of parents
Mutation probability, mutation fractions (% atoms and magnitude)
"""
from pymatgen.core.structure import Structure, Lattice
from pymatgen.transformations.standard_transformations import \
                                            RotationTransformation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from numpy.random import uniform as unif
import numpy as np
import random, copy
import math
from math import asin, cos, sqrt

from fx19 import structure_record
from fx19 import distance_check as dc


class Evolve(object):
    """
    A wrapper around classes mating and basinhopping. Calls these classes to
    generate a single model. Will decide whether to do mating (GA) or
    mutation (basinhopping) to generate a new structure.
    """
    def __init__(self, mate, hop, evolve_params):
        """
        mate (obj): a mating object
        hop (obj): a basinhopping object
        evolve_params (dict): The keys are the following:
                              'probabilities': {1:0.1, 2:0.2, 3:0.5, 4:0.2}
                              'num_species': (int) number of species
                              'species1'/'species2'/.. : 'Al'/'O'/'H'/...
        """
        self.mate = mate
        self.hop = hop

        # probability given as fraction to choose evolve method
        self.basinhopping_fraction = 0.34
        self.mate_by_slice_fraction = 0.33
        self.mate_by_swap_fraction = 0.33

        if 'basinhopping_fraction' in evolve_params:
            self.basinhopping_fraction = evolve_params['basinhopping_fraction']

        if 'mate_by_slice_fraction' in evolve_params:
            self.mate_by_slice_fraction = \
                                    evolve_params['mate_by_slice_fraction']

        if 'mate_by_swap_fraction' in evolve_params:
            self.mate_by_swap_fraction = evolve_params['mate_by_swap_fraction']

        if not self.basinhopping_fraction + self.mate_by_slice_fraction + \
                                    self.mate_by_swap_fraction == 1:
            print ('Sum of the provided evolve method fractions is not equal'
            ' to 1. So, using default values of 0.33, 0.33, 0.34 for hop, mate'
            '_by_swap and mate_by_slice respectively.')
            self.basinhopping_fraction = 0.34
            self.mate_by_slice_fraction = 0.33
            self.mate_by_swap_fraction = 0.33

        self.num_species = evolve_params['num_species']
        # Make species dicts as attributes
        self.species1 = evolve_params['species1']
        # save species2 data if exists
        if self.num_species > 1:
            self.species2 = evolve_params['species2']
        # save species2 data if exists
        if self.num_species > 2:
            self.species3 = evolve_params['species3']
        # save species2 data if exists
        if self.num_species > 3:
            self.species4 = evolve_params['species4']
        # save species2 data if exists
        if self.num_species > 4:
            self.species5 = evolve_params['species5']


    def get_model(self, select, pool, reg_id):
        """
        get new model using mating or basinhopping

        Args:
        select (obj): Select object
        pool (obj): Pool object
        reg_id(obj): register_id object

        """
        hop = self.hop
        mate = self.mate
        method = np.random.choice([1, 2, 3], p=[self.basinhopping_fraction,
                                                self.mate_by_slice_fraction,
                                                self.mate_by_swap_fraction])
        methods_dict = {1: 'basinhopping',
                        2: 'mate_by_slice',
                        3: 'mate_by_swap'}

        correct_comp = False
        while correct_comp is False:
            try:
                if method == 1:
                    new_astr, inheritance = hop.perturb_sites(select, pool)
                elif method == 2:
                    new_astr, inheritance = mate.mate_by_slicing(select, pool)
                    mate.move_atoms_to_within_cluster(new_astr)
                elif method == 3:
                    new_astr, inheritance = mate.mate_by_random_swap(select,
                                                                     pool)
                    mate.move_atoms_to_within_cluster(new_astr)
            except:
                continue
            if new_astr is None:
                continue
            if any(np.isnan(new_astr.cart_coords.flatten())):
                continue
            new_astr.sort()
            new_comp = new_astr.composition

            all_ok = []
            sp1 = self.species1
            sp1_ok = False
            sym_sp1, min_sp1, max_sp1 = sp1['name'], sp1['min_num'], \
                                                    sp1['max_num']
            if min_sp1 <= new_comp[sym_sp1] <= max_sp1:
                sp1_ok = True
            all_ok.append(sp1_ok)
            if self.num_species > 1:
                sp2 = self.species2
                sp2_ok = False
                sym_sp2, min_sp2, max_sp2 = sp2['name'], sp2['min_num'], \
                                                         sp2['max_num']
                if min_sp2 <= new_comp[sym_sp2] <= max_sp2:
                    sp2_ok = True
                all_ok.append(sp2_ok)
            if self.num_species > 2:
                sp3 = self.species3
                sp3_ok = False
                sym_sp3, min_sp3, max_sp3 = sp3['name'], sp3['min_num'], \
                                                         sp3['max_num']
                if min_sp3 <= new_comp[sym_sp3] <= max_sp3:
                    sp3_ok = True
                all_ok.append(sp3_ok)
            if self.num_species > 3:
                sp4 = self.species4
                sp4_ok = False
                sym_sp4, min_sp4, max_sp4 = sp4['name'], sp4['min_num'], \
                                                         sp4['max_num']
                if min_sp4 <= new_comp[sym_sp4] <= max_sp4:
                    sp4_ok = True
                all_ok.append(sp4_ok)
            if self.num_species > 4:
                sp5 = self.species5
                sp5_ok = False
                sym_sp5, min_sp5, max_sp5 = sp5['name'], sp5['min_num'], \
                                                         sp5['max_num']
                if min_sp5 <= new_comp[sym_sp5] <= max_sp5:
                    sp5_ok = True
                all_ok.append(sp5_ok)
            if False not in all_ok:
                correct_comp = True

        new_model = structure_record.model(new_astr, reg_id)
        new_model.inheritance = inheritance
        new_model.made_by = methods_dict[method]

        return new_model


class mating(object):

    def __init__(self, mating_params):
        """
        Creates a child structure by mating 2 or 3 parents

        Args:

        mating_params (dict): a dictionary of all the parameters required for
                              performing mating on parents
        Eg: {
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
           'mu': -6.76069604253}},
        """
        # Defaults for the parameters
        self.mirror_slice_before_join = True

        if 'mirror_slice_before_join' in mating_params:
            if isinstance(mating_params['mirror_slice_before_join'], bool):
                self.mirror_slice_before_join = \
                                    mating_params['mirror_slice_before_join']
            else:
                print ('mirror_slice_before_join parameter should be a boolean.'
                        ' Setting to defaults True')

        if mating_params['shape'] == 'cluster':
            self.max_dia = mating_params['max_dia']
            self.box_abc = mating_params['box_abc']

        self.num_species = mating_params['num_species']
        self.species_dict = mating_params['species_dict']
        self.min_dist_dict = mating_params['min_dist_dict']

        # Make species dicts as attributes
        self.species1 = mating_params['species1']
        # save species2 data if exists
        if self.num_species > 1:
            self.species2 = mating_params['species2']
        # save species2 data if exists
        if self.num_species > 2:
            self.species3 = mating_params['species3']
        # save species2 data if exists
        if self.num_species > 3:
            self.species4 = mating_params['species4']
        # save species2 data if exists
        if self.num_species > 4:
            self.species5 = mating_params['species5']

    def get_attach_type(self):
        """
        Function to get the attach type - mirror and attach, or direct attach

        """
        attach_type = 'direct'

        # if True, return either mirror and direct attach type
        if self.mirror_slice_before_join:
            if random.random() > 0.5:
                attach_type = 'mirror'

        return attach_type

    def mate_by_slicing(self, select, pool):
        """
        Mates parents by for a cluster of a gb by rotating and slicing at the
        fraction required
        Uses num_parents, change_num_atoms from the class attributes.

        Returns the child Atoms object

        Args:

        select (obj): Select object
        pool (obj): Pool object

        Returns:
            childAtoms (ASE Atoms): Returns the offspring ASE atoms object
        """
        # Get num_parents and select them parents
        num_parents = 2
        # NOTE: deepcopy already done in get_a_parent()
        parents = select.get_parents(pool, num_parents)
        parent1, parent2 = parents[0], parents[1]
        inheritance = [parent1.label, parent2.label]

        # rotate all parents randomly and slice them
        temp1 = self.rotate_astr(parent1.astr)
        slice1 = self.fraction_slice(temp1)
        temp2 = self.rotate_astr(parent2.astr)
        slice2 = self.fraction_slice(temp2)

        # Attach two slices at a time
        if random.randint(0, 2) == 0:
            attach_type = 'direct'
        else:
            attach_type = 'mirror'
        child = self.attach_slices(slice1, slice2, attach_type=attach_type)

        return child, inheritance

    def rotate_astr(self, astr, rotate_type='random'):
        """
        Given a structure, returns same structure with all "atoms" rotated at
        random angle (0:360) along random vector([0, 0, 0]:[3, 3, 3])

        Args:
        astr (obj): pymatgen structure object
        rotate_type (str): 'random' or 'mirror' for rotation of structure
        """
        species = astr.species
        first_coords = astr.cart_coords
        if rotate_type=='random':
            # perfrom random rotation transformation
            hkl = [random.randint(0, 3), random.randint(0, 3), random.randint(0, 3)]
            rotate = RotationTransformation(hkl, unif(0, 360))
            temp_astr = rotate.apply_transformation(astr)
        elif rotate_type=='mirror':
            hkl = [random.randint(0, 3), random.randint(0, 3), 0]
            rotate = RotationTransformation(hkl, 180)
            temp_astr = rotate.apply_transformation(astr)

        # NOTE: The lattice is rotated, but coords are still same
        # Place old cart_coords in temp_parent lattice
        all_inds = [i for i in range(len(astr.cart_coords))]
        temp_astr.remove_sites(all_inds)
        for specie, coord in zip(species, first_coords):
            temp_astr.append(specie, coord, coords_are_cartesian=True)

        # Translate the cluster to center of box
        fc = temp_astr.frac_coords
        range_x, range_y, range_z = fc[:, 0], fc[:, 1], fc[:, 2]
        cent_x, cent_y, cent_z = (max(range_x) + min(range_x))/2, \
                                 (max(range_y) + min(range_y))/2, \
                                 (max(range_z) + min(range_z))/2
        cent = np.array([cent_x, cent_y, cent_z])
        trans_vector = np.array([0.5, 0.5, 0.5]) - cent
        temp_astr.translate_sites(all_inds, trans_vector)

        # Get conventional structure (2 ways)
        # 1. modify_lattice from parent1 (straight forward)
        # 2. use SpacegroupAnalyzer
        temp_astr.lattice = astr.lattice # Approach 1
        # If the above causes any issues, use this approach 2
        # sp = SpacegroupAnalyzer(temp_parent)
        # prepped_parent = sp.get_conventional_standard_structure()

        return temp_astr

    def fraction_slice(self, astr):
        """
        For a given astr, this function slices at (1/num_parents) from bottom
        and returns the bottom part

        Args:
        astr (obj): pymatgen structure object
        """
        # Fixed num_parents to 2.
        num_parents = 2
        z_mids = astr.cart_coords[:, 2]
        # Determine z_cut to get ~ equal fractions from all parents
        z_cut = (max(z_mids) + min(z_mids)) / num_parents

        # NOTE: Due to random rotation, the fraction we slice is different
        # for all parents
        # Get indices of slicing atoms, i.e., above z_cut
        rm_inds = []
        for i, z in enumerate(z_mids):
            if z >= z_cut:
                rm_inds.append(i)
        # Remove these atoms from parent1
        astr.remove_sites(rm_inds)

        return astr

    def attach_slices(self, slice1, slice2, attach_type='mirror'):
        """
        Given two slices, rotates and attaches slices and returns attached
        structure

        Args:
        slice1 (obj): pymatgen structure object of one slice
        slice2 (obj): pymatgen structure object of second slice
        attach_type: 'mirror' or 'direct'

        (Pseudocode)
        If attach_type=='mirror':
            rotate slice2 along (x, y, 0) plane to 180 degrees
            You should get mirrored image

        find max_z of slice1 (cartesian)
        find min_z of slice2
        z_trans = each_z2 + (max_z1 - min_z2) + 1

        find cent_x1, cent_y1 for slice1
        find cent_x2, cent_y2 for slice2
        align both cent_x1, cent_x2 ; cent_y1, cent_y2

        add the transformed slice2 coords to slice1

        return slice1
        """
        astr = copy.deepcopy(slice1)
        new_slice2 = copy.deepcopy(slice2)
        # If mirror one slice before attachment,
        if attach_type=='mirror':
            new_slice2 = self.rotate_astr(slice2, rotate_type='mirror')
        slice1_coords = astr.cart_coords
        slice2_coords = new_slice2.cart_coords

        max_z_1 = max(slice1_coords[:, 2])
        min_z_2 = min(slice2_coords[:, 2])
        # Get translate vector along z i.e., x, y are 0
        z_trans = np.array([0, 0, (max_z_1 - min_z_2) + 1])   # tolerance of z+1
        add_coords = slice2_coords + z_trans
        # center add_coords on slice1 i.e., align centers along x and y
        range_x1, range_y1 = slice1_coords[:, 0], slice1_coords[:, 1]
        range_x2, range_y2 = slice2_coords[:, 0], slice2_coords[:, 1]
        cent_x1, cent_y1 = (max(range_x1) + min(range_x1))/2, \
                                 (max(range_y1) + min(range_y1))/2
        cent_x2, cent_y2 = (max(range_x2) + min(range_x2))/2, \
                                 (max(range_y2) + min(range_y2))/2
        xy_trans = np.array([(cent_x1 - cent_x2), (cent_y1 - cent_y2), 0])
        add_coords = add_coords + xy_trans

        # species of add_coords
        add_species = new_slice2.species
        # add each of add_coords to slice1
        for specie, coord in zip(add_species, add_coords):
            slice1.append(specie, coord, coords_are_cartesian=True)
        slice1.sort()

        return slice1

    def mate_by_random_swap(self, select, pool):
        """
        NOTE: Works only for fixed composition searches.

        Atoms are randomly added to a new cell from all parents based on index.
        Checks distance after every added atomic coordinate. For all indices,
        that failed distance check, the coords are moved within radius "1"
        sphere and tried.

        The trick is to add an index of atom from only one parent

        Args:
        select (obj): Select object
        pool (obj): Pool object
        """
        # Get num_parents and select them parents
        num_parents = 2
        parents = select.get_parents(pool, num_parents)
        parent1, parent2 = parents[0], parents[1]
        inheritance = [parent1.label, parent2.label]
        p1_sites = parent1.astr.sites
        p2_sites = parent2.astr.sites
        list_of_p_sites = [p1_sites, p2_sites]

        child = copy.deepcopy(parent1.astr)
        all_inds = [i for i in range(len(child.cart_coords))]
        child.remove_sites(all_inds)

        # add atoms from both parents in to one structure
        child_sites = parent1.astr.sites + parent2.astr.sites
        species = [i.species for i in child_sites]
        coords = [i.coords for i in child_sites]
        latt = parent1.astr.lattice
        child = Structure(latt, species, coords)

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

    def add_random_sites(self,child, list_of_parent_sites, indices, move=False):
        """
        child structure to which sites to be added randomly from different set
        of parent_sites.

        Args:
        child (obj): pymatgen structure object of the child
        list_of_parent_sites (list): list of sites from parent structures
        indices (list): Indices that are shuffled randomly for num_atoms_child
        move (bool): Whether to move and try to add a site coords when adding
        """
        p1_sites, p2_sites = list_of_parent_sites[0], list_of_parent_sites[1]

        rem_inds = []

        # If 2 parents
        num_parents = 2
        inds1, inds2 = self.divide_index_list(indices)
        # Add sites from each parent sites in circular fashion
        for i in range(len(inds1)):
            if not self.add_site(child, p1_sites, inds1[i], move=move):
                rem_inds.append(inds1[i])
            if i < len(inds2):
                if not self.add_site(child, p2_sites, inds2[i], move=move):
                    rem_inds.append(inds2[i])

        return rem_inds

    def add_site(self, child, parent_sites, index, move=False):
        """
        Adds the parent_site to child according to the index provided
        Checks distance for new site and adds only if satisfies.
        The coords are moved and then tested if move is set to True.

        Returns 'True' if site is added to child

        Args:
        child (obj): child structure object
        parent_sites (list/array): sites of parent structure
        index (int): index of the parent site to add
        move (bool): Whether to move and try to add a site coords when adding
        """
        child_coords =  child.cart_coords
        specie_to_add = parent_sites[index].specie.name
        coords_to_add = parent_sites[index].coords

        if move:
            radius = random.random() * 1.5
            translate = self.get_point_on_sphere(radius)
            coords_to_add = coords_to_add + translate

        if dc.satisfies_all_dists(coords_to_add, specie_to_add, child,
                                  self.min_dist_dict,self.species_dict):
            child.append(specie_to_add, coords_to_add,
                                  coords_are_cartesian=True)
            return True
        else:
            return False

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

    def divide_index_list(self, indices):
        """
        Returns 2 or 3 lists of indices for 2 or 3 parents
        Function separated for clarity

        Args:
        indices (list): Indices that are shuffled randomly for num_atoms_child
        """
        A = indices[ : len(indices)//2 ]
        B = indices[ len(indices)//2 : ]
        if len(A) > len(B):
            list_1, list_2 = A, B
        else:
            list_1, list_2 = B, A
        return list_1, list_2

    def check_atoms_for_all_species(self, astr):
        """
        For any (newly created) structure, checks if the number of atoms for
        each species lie within minimum and maximum number allowed for that
        species.

        Args:
        astr (obj): pymatgen structure object
        """
        num_species = self.num_species

        all_ok = []
        # Check for species1
        sp1_ok = self.check_atoms_for_a_specie(self.species1, astr)
        all_ok.append(sp1_ok)
        # If species2 exists, chekc species2  and so on..
        if num_species > 1:
            sp2_ok = self.check_atoms_for_a_specie(self.species2, astr)
            all_ok.append(sp2_ok)
        if num_species > 2:
            sp3_ok = self.check_atoms_for_a_specie(self.species3, astr)
            all_ok.append(sp3_ok)
        if num_species > 3:
            sp4_ok = self.check_atoms_for_a_specie(self.species4, astr)
            all_ok.append(sp4_ok)
        if num_species > 4:
            sp5_ok = self.check_atoms_for_a_specie(self.species5, astr)
            all_ok.append(sp5_ok)

        if False in all_ok:
            return False
        else:
            return True

    def check_atoms_for_a_specie(self, specie, astr):
        """
        For any (newly created) structure, checks if the number of atoms for
        given single species lie within minimum and maximum number allowed for
        that species.

        Args:
        specie (dict): containing 'name', 'min_num' and 'max_num'
        astr (obj): pymatgen structure object
        """
        # Get symbol set of species in structure
        astr_species_symbols = astr.symbol_set
        composition = astr.composition
        symbol = specie['name']
        num_atoms_in_astr = composition[symbol]

        # It is possible mating could remove some species
        # Check if given specie exists in the structure
        if specie['min_num'] == 0:
            if symbol not in astr_species_symbols:
                # num atoms for the species in structure is 0
                # Eliminate check_1 and check_2
                return True

        check_1 = False
        if symbol in astr_species_symbols:
            check1 = True

        # check if its atoms are within the min-max
        check_2 = False
        if specie['min_num'] <= num_atoms_in_astr <= specie['max_num']:
            check_2 = True

        satisfies = False
        if check_1 is True and check_2 is True:
            satisfies = True

        return satisfies

    def move_atoms_to_within_cluster(self, child):
        """
        In cluster geometry, after a child is generated by mating,
        check if any of the atoms are outside the maximum radius of the cluster.
        If so, move them randomly to somewhere within the cluster.

        Args:

        child (obj): pymatgen structure object
        """
        radius, abc = self.max_dia/2, self.box_abc
        origin = [abc[0]/2, abc[1]/2, abc[2]/2]

        # get atom indices that needs to be moved
        child_sites = child.sites
        species = child.species
        move_inds = [i for i, site in enumerate(child_sites) \
                                if dc.dist(origin, site.coords) > radius]

        # move those atoms in same way as in perturb_sites
        remove_inds = []
        for i in move_inds:
            tries = 0
            replaced = False
            while not replaced and tries < 1000:
                tries += 1
                new_cart = [unif(origin[0] - radius, origin[0] + radius),
                            unif(origin[1] - radius, origin[1] + radius),
                            unif(origin[2] - radius, origin[2] + radius)]
                if dc.dist(origin, new_cart) > radius:
                    continue
                # Check distance and replace with new coords
                if dc.satisfies_all_dists(new_cart, species[i].name,
                                          child, self.min_dist_dict,
                                          self.species_dict, remove_index=i):
                    child.replace(i, species[i], new_cart,
                                        coords_are_cartesian=True)
                    replaced = True
            # remove those atoms that cannot be replaced
            if not replaced:
                remove_inds.append(i)
        child.remove_sites(remove_inds)


class basinhopping(object):
    """
    Class that handles making child models using basinhopping methods
    """
    def __init__(self, basinhopping_params):
        """
        Args:

        basinhopping_params (dict): dictionary with all the required
        basinhopping parameters


        Eg: {
            # list of range of frac_coords in each direction to perturb
            'indices_fraction': 0.6 # fraction of total atoms to perturb

            'max_perturbation': 0.5, # maximum perturbation distance in Å
            'min_dist_dict': # dictionary of minimum bond distances
             {'sp1_sp1': 2.3, 'sp1_sp2': 1.5, 'sp2_sp2': 1.2},
            'species_dict': # dictionary of species information
             {'species1': {'name': 'Al',
               'min_num': 36,
               'max_num': 36,
               'mu': -3.35958515625},
              'species2': {'name': 'O',
               'min_num': 30,
               'max_num': 30,
               'mu': -6.76069604253}}}
        """
        # default indices_fraction is 1 ; perturb all atoms (indices)
        self.indices_fraction = 1
        self.max_perturbation = 0.15
        self.min_dist_dict = basinhopping_params['min_dist_dict']
        self.species_dict = basinhopping_params['species_dict']
        shape = basinhopping_params['shape']

        if 'indices_fraction' in basinhopping_params:
            if 0 < basinhopping_params['indices_fraction'] <= 1:
                self.indices_fraction = basinhopping_params['indices_fraction']
            else:
                print ('Provided indices_fraction out of range (0,1]. '
                        'Using default..')

        if 'max_perturbation' in basinhopping_params:
            if not 0 < basinhopping_params['max_perturbation'] <= 0.5:
                print ('max_perturbation should be between (0, 0.5]. '
                    'More than 0.5 would be throw the atoms too far. Check the'
                    ' jump distance by lattice vectors * max_perturbation. '
                    'Using default value of 0.15')
            else:
                self.max_perturbation = basinhopping_params['max_perturbation']

        # maximum diameter and box lattice parameters of the cluster (geometry)
        if shape == 'cluster':
            self.max_dia = basinhopping_params['max_dia'] # default is 8 Å
            self.box_abc = basinhopping_params['box_abc'] # default a=b=c=20 Å

    def perturb_sites(self, select, pool, model_id=None, gb=False):
        """
        Displaces atoms in a parent (cluster or gb_iface) using uniform
        distribution

        Args:

        select (obj): Select object
        pool (obj): Pool object
        model_id (int): If given, basinhopping is done on this specific model
        gb (bool): True if the search is 'gb'
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
        frac_coords = parent.astr.frac_coords
        species = parent.astr.species

        if gb:
            frac_coords = parent.gb_iface.frac_coords
            species = parent.gb_iface.species

        # Get frac_coords to perturb
        total_num_atoms = len(frac_coords)
        # use indices_fraction; default to 1
        num_atoms_to_perturb = int(total_num_atoms * indices_fraction)
        D_inds = random.sample(range(0, total_num_atoms), num_atoms_to_perturb)
        D_coords = [frac_coords[i] for i in D_inds]

        num_perturbed = 0
        jumps_needed = int(0.5 * len(D_coords))
        for i, one_coords in zip(D_inds, D_coords):
            # Get updated cart_coords
            all_cart_coords = parent.astr.cart_coords
            if gb:
                all_cart_coords = parent.gb_iface.cart_coords
            # remove current index from cart_coords
            # rem_cart_coords = np.delete(all_cart_coords, i, 0)
            # Randomly perturb within sphere of radius = max_perturbation
            replaced = False
            tries = 0
            while not replaced and tries < 1000:
                tries += 1
                jump = self.max_perturbation #unif(0, self.max_perturbation)
                perturb = self.get_point_on_sphere(jump)
                new_frac = one_coords + perturb
                if not gb: # cluster
                    new_cart = parent.astr.lattice.get_cartesian_coords(
                                                                    new_frac)
                    # check if new cart is inside the cluster radius
                    origin = (self.box_abc[0]/2,
                              self.box_abc[1]/2,
                              self.box_abc[2]/2)
                    if dc.dist(origin, new_cart) > self.max_dia/2:
                        continue
                else:
                    new_cart = parent.gb_iface.lattice.get_cartesian_coords(
                                                                    new_frac)
                # Check distance and replace with new coords
                if not gb and dc.satisfies_all_dists(new_cart,
                                          species[i].name,
                                          parent.astr,
                                          self.min_dist_dict,
                                          self.species_dict,
                                          remove_index=i):
                    parent.astr.replace(i, species[i], new_cart,
                                        coords_are_cartesian=True)
                    replaced = True
                    num_perturbed += 1
                if gb and dc.satisfies_all_dists(new_cart,
                                          species[i].name,
                                          parent.gb_iface,
                                          self.min_dist_dict,
                                          self.species_dict,
                                          remove_index=i):
                    parent.gb_iface.replace(i, species[i], new_cart,
                                        coords_are_cartesian=True)
                    replaced = True
                    num_perturbed += 1

        if not num_perturbed < jumps_needed:
            if gb:
                return parent.gb_iface, inheritance
            else:
                return parent.astr, inheritance
        else:
            return None, None

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
    Includes functions for creating initial_population and creating child
    structures with mating operations of grain boundary models. However,
    basinhopping class object (hop) is used to make basinhopping child models.

    Here the initial population is not random. Grain 1 and grain 2 are
    made to overlap and remove extra atoms.
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
        self.hop_mate_frac = 0.3 # 30% hop, 70% mate
        if 'hop_mate_frac' in str_constraints:
            self.hop_mate_frac = str_constraints['hop_mate_frac']
        if 'init_gb_astr' not in str_constraints:
            print ('Error: Initial grain boudanry not provided. ')
        else:
            self.init_gb_astr = str_constraints['init_gb_astr']

        if 'iface_thickness' not in str_constraints:
            print ('Error: Thickness of interface region not provided. ')
        else:
            self.iface_thickness = str_constraints['iface_thickness']

        if 'iface_z_mid' not in str_constraints:
            print ('Error: Mid point of grain boundary not provided. ')
        else:
            self.iface_z_mid = str_constraints['iface_z_mid']

        if 'num_slices' not in str_constraints:
            print ('Error: NUmber of slices to cut for mating not provided. ')
        else:
            self.num_slices = str_constraints['num_slices']

        self.slice_axes = [0, 1]  # default slicing axes of x and y only
        if 'slice_axes' in str_constraints:
            self.slice_axes = str_constraints['slice_axes']

        # num_species is taken from species_dict from structure_record
        self.num_species = str_constraints['num_species']
        # save species1 data, same as in initial population
        # species1 should always exist
        self.sym_species1 = str_constraints['species1']['name']
        self.min_num_sp1 = str_constraints['species1']['min_num']
        self.max_num_sp1 = str_constraints['species1']['max_num']
        # save species2 data if exists
        if self.num_species > 1:
            if 'species2' in str_constraints and str_constraints['species2']:
                self.sym_species2 = str_constraints['species2']['name']
                self.min_num_sp2 = str_constraints['species2']['min_num']
                self.max_num_sp2 = str_constraints['species2']['max_num']
        # save species3 data if exists
        if self.num_species > 2:
            if 'species3' in str_constraints and str_constraints['species3']:
                self.sym_species3 = str_constraints['species3']['name']
                self.min_num_sp3 = str_constraints['species3']['min_num']
                self.max_num_sp3 = str_constraints['species3']['max_num']
        # save species4 data if exists
        if self.num_species > 3:
            if 'species4' in str_constraints and str_constraints['species4']:
                self.sym_species4 = str_constraints['species4']['name']
                self.min_num_sp4 = str_constraints['species4']['min_num']
                self.max_num_sp4 = str_constraints['species4']['max_num']
        # save species5 data if exists
        if self.num_species > 4:
            if 'species5' in str_constraints and str_constraints['species5']:
                self.sym_species5 = str_constraints['species5']['name']
                self.min_num_sp5 = str_constraints['species5']['min_num']
                self.max_num_sp5 = str_constraints['species5']['max_num']

        self.min_dist_dict = str_constraints['min_dist_dict']
        self.species_dict = str_constraints['species_dict']
        self.element_syms = str_constraints['element_syms']
        self.iface_latt = str_constraints['iface_latt']

        # hollow gb structure
        copy_g = self.init_gb_astr.copy()

        # Get coords and species
        g_sites = copy_g.sites
        sor_sites = sorted(g_sites, key=lambda x: x.coords[2])

        half_zrange = (self.iface_thickness / (self.init_gb_astr.lattice.c * 2))
        min_z_t = self.iface_z_mid + half_zrange
        max_z_b = self.iface_z_mid - half_zrange
        self.hollow_botz = max_z_b
        self.hollow_topz = min_z_t
        top_i, bot_i = None, None
        for i, site in enumerate(sor_sites):
            if site.c >= max_z_b and not bot_i:
                bot_i = i
            if site.c >= min_z_t and not top_i:
                top_i = i
                break
        mids = sor_sites[bot_i : top_i]
        rem_i = []
        for i, site in enumerate(copy_g.sites):
            if site in mids:
                rem_i.append(i)
        copy_g.remove_sites(rem_i)

        self.hollow_init_gb = copy_g.get_sorted_structure()

    def overlap_grains(self):
        """
        cut bottom grain the size of iface_thickness
        add top grain sites to the cut grain
        remove half the atoms randomly from the interface

        No arguments needed
        """
        init_gb_astr = self.init_gb_astr
        iface_thickness = self.iface_thickness

        # Lattice of gb interface: a, b are same, c lattice vector is from input
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

        new_bot_sps, new_bot_fc = self.new_sites_coords(bot_sites,
                                [cut_bot, cut_bot + window_frac], [0, 1], 2)
        new_top_sps, new_top_fc = self.new_sites_coords(top_sites,
                                [top_cut - window_frac, top_cut], [0, 1], 2)
        species = new_bot_sps + new_top_sps
        coords = new_bot_fc + new_top_fc

        child = Structure(latt, species, coords)
        child.merge_sites(tol=1, mode='delete')
        child = child.get_sorted_structure()
        self.move_coords_inside(child)

        return child

    def get_rem_inds(self, child_astr):
        """
        remove random indices from total number of atoms in child.
        Such that it satisfies each species min_num and max_num

        Args:
        child_astr (obj): pymatgen structure object og grain boundary
        """
        # get num species for each species present in element_syms
        n_sp1 = round(unif(self.min_num_sp1, self.max_num_sp1))
        n_sp2, n_sp3, n_sp4, n_sp5 = 0, 0, 0, 0
        if self.num_species > 1:
            n_sp2 = round(unif(self.min_num_sp2, self.max_num_sp2))
        if self.num_species > 2:
            n_sp3 = round(unif(self.min_num_sp3, self.max_num_sp3))
        if self.num_species > 3:
            n_sp4 = round(unif(self.min_num_sp4, self.max_num_sp4))
        if self.num_species > 4:
            n_sp5 = round(unif(self.min_num_sp5, self.max_num_sp5))

        # get iface sites in the child_gb_astr
        child_astr_sites = child_astr.sites
        iface_inds_in_gb = []
        for i, site in enumerate(child_astr.sites):
            if self.hollow_botz <= site.c <= self.hollow_topz:
                 iface_inds_in_gb.append(i)
        iface_sites = [site for i, site in enumerate(child_astr_sites) \
                                    if i in iface_inds_in_gb]
        site_sps = [site.specie for site in iface_sites]
        site_sps = [i.name for i in site_sps]
        iface_sps = set(site_sps)
        comp_dict = {}
        for sp in iface_sps:
            comp_dict[sp] = site_sps.count(sp)
            # comp_dict is the composition dictionary of iface in child_astr
        # get num species to be removed for each species
        diff_sp1, diff_sp2, diff_sp3, diff_sp4, diff_sp5 = 0, 0, 0, 0, 0
        for specie in comp_dict.keys():
            if self.sym_species1 == specie:
                diff_sp1 = n_sp1 - comp_dict[specie]
            if self.num_species > 1:
                if self.sym_species2 == specie:
                    diff_sp2 = n_sp2 - comp_dict[specie]
            if self.num_species > 2:
                if self.sym_species3 == specie:
                    diff_sp3 = n_sp3 - comp_dict[specie]
            if self.num_species > 3:
                if self.sym_species4 == specie:
                    diff_sp4 = n_sp4 - comp_dict[specie]
            if self.num_species > 4:
                if self.sym_species5 == specie:
                    diff_sp5 = n_sp5 - comp_dict[specie]

        # if difference is positive, add sites and return []
        if diff_sp1 > 0:
            sp_1 = self.sym_species1
            self.add_sites_diff(diff_sp1, sp_1, child_astr)
        if diff_sp2 > 0:
            sp_2 = self.sym_species2
            self.add_sites_diff(diff_sp2, sp_2, child_astr)
        if diff_sp3 > 0:
            sp_3 = self.sym_species3
            self.add_sites_diff(diff_sp3, sp_3, child_astr)
        if diff_sp4 > 0:
            sp_4 = self.sym_species4
            self.add_sites_diff(diff_sp4, sp_4, child_astr)
        if diff_sp5 > 0:
            sp_5 = self.sym_species5
            self.add_sites_diff(diff_sp5, sp_5, child_astr)

        # if difference is negative, return rem_inds -> remove sites
        rem_inds = []
        rem_sp1, rem_sp2, rem_sp3, rem_sp4, rem_sp5 = 0, 0, 0, 0, 0

        random.shuffle(iface_inds_in_gb)
        for ind in iface_inds_in_gb:
            site = child_astr.sites[ind]
            if site.specie.name == self.sym_species1:
                if rem_sp1 < -diff_sp1:
                    rem_inds.append(ind)
                    rem_sp1 += 1
            if self.num_species > 1:
                if site.specie.name == self.sym_species2:
                    if rem_sp2 < -diff_sp2:
                        rem_inds.append(ind)
                        rem_sp2 += 1
            if self.num_species > 2:
                if site.specie.name == self.sym_species3:
                    if rem_sp3 < -diff_sp3:
                        rem_inds.append(ind)
                        rem_sp3 += 1
            if self.num_species > 3:
                if site.specie.name == self.sym_species4:
                    if rem_sp4 < -diff_sp4:
                        rem_inds.append(ind)
                        rem_sp4 += 1
            if self.num_species > 4:
                if site.specie.name == self.sym_species5:
                    if rem_sp5 < -diff_sp5:
                        rem_inds.append(ind)
                        rem_sp5 += 1

        return rem_inds

    def add_sites_diff(self, diff, sp, child_astr):
        """
        Add random sites to the child iface to get correct composition

        Args:
        diff (int): number of new sites to add
        sp (str): species name
        child_astr (Structure): pymatgen structure object of child_astr
        """
        # This is half the thickness (- 1 Å tolerance)
        half_z_thickness = (self.iface_thickness - 0.3) /   \
                                (self.init_gb_astr.lattice.c * 2)
        zmin = self.iface_z_mid - half_z_thickness
        zmax = self.iface_z_mid + half_z_thickness

        # remove non-relevant sites from the structure before sending to
        # dc.satisfies_all_dists(). This saves lot of time.
        dc_astr = copy.deepcopy(child_astr)
        rem_inds = []
        for i, site in enumerate(dc_astr.sites):
            if not (zmin-0.02) <= site.c <= (zmax+0.02):
                rem_inds.append(i)
        dc_astr.remove_sites(rem_inds)

        num_added, tries = 0, 0
        while num_added < diff: #and tries < 1000: #(leave this structure)
            tries += 1
            coords = child_astr.cart_coords
            new_c = [unif(0, 1), unif(0, 1), unif(zmin, zmax)]
            new_c = child_astr.lattice.get_cartesian_coords(new_c)
            if dc.satisfies_all_dists(new_c, sp, dc_astr,
                                      self.min_dist_dict, self.species_dict):
                child_astr.append(sp, new_c, coords_are_cartesian=True)
                num_added += 1
        del dc_astr

    def new_sites_coords(self, sites, old_axis_bounds, new_axis_bounds, axis):
        """
        Given set of cartesian coords as "sites"
        converts them into fractional coordinates within the "new_axis_bounds"
        along the "axis" provided.

        Args:
        sites (list/array): list of cartesian coordinates
        old_axis_bounds (list): fractional coords of bounds of block in z
                                direction
        new_axis_bounds (list): fractional coords of bounds of new lattice in z
                                direction
        axis: (int) axis along which to make slices (0, 1, 2 for x, y and z)
        """

        sorted_sites = sorted(sites, key=lambda x: x.coords[axis])
        axis_min, axis_max = sites[0].frac_coords[axis], \
                                sites[-1].frac_coords[axis]
        old_min, old_max = old_axis_bounds
        new_min, new_max = new_axis_bounds

        # change the coordinates of given sites to the new_axis_bounds
        add_fcs, add_sps = [], []
        for site in sites:
            if axis == 0:
                x = site.a
            elif axis == 1:
                x = site.b
            elif axis == 2:
                x = site.c
            new_x = (x - old_min) / (old_max - old_min) # normalize
            new_x = new_x * (new_max - new_min) + new_min # transform

            if axis ==0:
                add_fcs.append([new_x, site.b, site.c])
            elif axis == 1:
                add_fcs.append([site.a, new_x, site.c])
            elif axis == 2:
                add_fcs.append([site.a, site.b, new_x])
            add_sps.append(site.species)

        return add_sps, add_fcs

    def get_hollow_gb(self):
        """
        Function to remove sites from the interface region of init_gb_astr.
        This hollow_gb will be used for all models in a search.

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

        top_ind, bot_ind = None, None
        for i, site in enumerate(sorted_sites):
            if site.c >= max_z_bot and not bot_ind:
                bot_ind = i
            if site.c >= min_z_top and not top_ind:
                top_ind = i
                break
        mid_sites = sorted_sites[bot_ind : top_ind]
        rem_inds = []
        for i, site in enumerate(copy_gb.sites):
            if site in mid_sites:
                rem_inds.append(i)
        copy_gb.remove_sites(rem_inds)

        return copy_gb

    def grain_implant(self, iface_to_implant):
        """
        Take init_gb_astr, remove all atoms in the middle.
        Add each atom in iface_to_implant to the hollow space

        Args:
        iface_to_implant (obj): pymatgen structure object to be implanted
        """
        hollow_init_gb = self.hollow_init_gb
        iface_z_mid = self.iface_z_mid
        iface_thickness = self.iface_thickness
        gb_c = self.init_gb_astr.lattice.c

        min_z_top = iface_z_mid + (iface_thickness / (gb_c * 2))
        max_z_bot = iface_z_mid - (iface_thickness / (gb_c * 2))

        # Add sites from iface_to_implant to hollow_init_gb
        zmin, zmax = 0, 1 # limits for fractional coordinates along z direction

        # Maintain a tolerance of 0.2 Å between implant and hollow_gb
        ztol = 0.2 / gb_c
        newz_min, newz_max = max_z_bot + ztol, min_z_top - ztol

        add_fcs, add_sps = [], []
        for site in iface_to_implant.sites:
            newc = (site.c - zmin)/(zmax - zmin) # normalize
            newc = newc * (newz_max - newz_min) + newz_min # transform
            add_fcs.append([site.a, site.b, newc])
            add_sps.append(site.species)

        new_gb = hollow_init_gb.copy()
        for i in range(len(add_fcs)):
            new_gb.append(add_sps[i], add_fcs[i])

        # for sites in new_gb with no sd_flags, add [False, False, False]
        # This will prevent errors in next step
        for i in range(len(new_gb)):
            if 'selective_dynamics' not in new_gb[i].properties.keys():
                new_gb[i].properties['selective_dynamics'] = \
                                                [False, False, False]

        new_gb.merge_sites(tol=1, mode='delete')
        rem_inds = self.get_rem_inds(new_gb)
        new_gb.remove_sites(rem_inds)

        return new_gb.get_sorted_structure()

    def separate_gb(self, gb_astr):
        """
        Separates the interface region from grain boundary and
        returns interface structure
        Args:
            gb_astr (obj): pymatgen structure object of gb

        NOTE: buffer will be considered when joining the grains to iface_box
            buffer: the tolerance when joining box to top and bottom grains
        """
        iface_z_mid = self.iface_z_mid
        iface_thickness = self.iface_thickness

        gb_latt_matrix = gb_astr.lattice.matrix
        gb_c = gb_astr.lattice.c

        # Get new lattices for all grains
        #bot_matrix = gb_latt_matrix.copy()
        # TODO: Add tolerance here
        #bot_matrix[2] = [0, 0, ((gb_c - iface_thickness)/2) + 1.5]
        #bot_latt = Lattice(bot_matrix)

        mid_matrix = gb_latt_matrix.copy()
        mid_matrix[2] = [0, 0, iface_thickness + 0.2]
        mid_latt = Lattice(mid_matrix)

        #top_matrix = gb_latt_matrix.copy()
        # TODO: Add tolerance here
        #top_matrix[2] = [0, 0, ((gb_c - iface_thickness) / 2) + 1.5]
        #top_latt = Lattice(top_matrix)

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
        #bot_sites = sorted_sites[:bot_ind]
        #top_sites = sorted_sites[top_ind:]
        mid_sites = sorted_sites[bot_ind:top_ind]

        # translate sites in bottom, middle and top grains
        # to maintain clarity
        bot_z_max, bot_z_min = max_z_bot + 1.5/gb_c, 0
        mid_z_max, mid_z_min = min_z_top, max_z_bot
        top_z_max, top_z_min = 1, min_z_top - 1.5/gb_c

        #new_bot_fc, new_bot_sps = [], []
        #for site in bot_sites:
        #    newc = (site.c - bot_z_min) / (bot_z_max - bot_z_min)
        #    new_bot_fc.append([site.a, site.b, newc])
        #    new_bot_sps.append(site.species)

        new_mid_fc, new_mid_sps = [], []
        for site in mid_sites:
            newc = (site.c - mid_z_min ) / (mid_z_max - mid_z_min + 0.2/gb_c)
            newc = newc + 0.1/(iface_thickness + 0.2)
            new_mid_fc.append([site.a, site.b, newc])
            new_mid_sps.append(site.species)

        #new_top_fc, new_top_sps = [], []
        #for site in top_sites:
        #    newc = (site.c - top_z_min) / (top_z_max - top_z_min)
        #    new_top_fc.append([site.a, site.b, newc])
        #    new_top_sps.append(site.species)

        #bot_grain = Structure(bot_latt, new_bot_sps, new_bot_fc)
        mid_grain = Structure(mid_latt, new_mid_sps, new_mid_fc)
        #top_grain = Structure(top_latt, new_top_sps, new_top_fc)

        return mid_grain

    def fraction_slice(self, astr, axis):
        """
        For a given astr, this function slices it into required number of
        blocks along the provided axis

        Args:
        astr:  structure object
        num_slices: (int) number of blocks
        axis: (int) axis along which to make slices (0, 1, 2 for x, y and z)

        Returns a lattice object (same for all blocks) and
                list of blocks (sites lists)
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
        # Add 0 at beginning and total num of sites at the end for easy indexing
        cut_atom_inds.append(len(sorted_sites))
        cut_atom_inds.reverse()
        cut_atom_inds.append(0)
        cut_atom_inds.reverse()

        # do the same for cut_locs
        cut_locs.reverse()
        cut_locs.append(0)
        cut_locs.reverse()

        blocks = []
        for i in range(len(cut_atom_inds)-1):
            bl = sorted_sites[cut_atom_inds[i] : cut_atom_inds[i+1]]
            bl_bounds = [cut_locs[i], cut_locs[i+1]]
            block = [bl, bl_bounds]
            blocks.append(block)

        return new_latt, blocks

    def join_slices_to_mold(self, blocks_dict, axis):
        """
        divides iface_latt into equal number of blocks in the direction of axis.
        change the ax-coordinate of each site and append to respective mold
        Ex: if axis = 1, y - coordinate needs to be changed

        Args:
        blocks_dict: dictionary of blocks list from each parent
            Ex: {'p1': blocks_p1, 'p2': blocks_p2}
            blocks_p1 = [[sites], [sites], [sites]]
            pre-condition - the blocks all must be of same size (lattice)
        axis: (int) axis along which to make slices (0, 1, 2 for x, y and z)

        returns pymatgen structure object of the joned interface
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
        while len(random_blocks) < len(mold_ax_bounds):
            # choose blocks from parents alternatively
            block = random.choice(blocks_dict['p1'])
            if block not in random_blocks:
                random_blocks.append(block)
            block = random.choice(blocks_dict['p2'])
            if block not in random_blocks:
                random_blocks.append(block)

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

    def mate(self, select, pool):
        """
        Takes 2 parents and mates them

        Args:
        select (obj): Select object
        pool (obj): Pool object

        Returns child gb structure, method used as string
        """
        # get two parents
        num_parents = 2
        parents = select.get_parents(pool, num_parents)
        parent1, parent2 = parents[0], parents[1]
        inheritance = [parent1.label, parent2.label]
        # choose axis to slice
        axes = self.slice_axes
        axis = random.choice(axes)
        # get the slice blocks two lists from two parents
        _, blocks_1 = self.fraction_slice(parent1.gb_iface, axis)
        _, blocks_2 = self.fraction_slice(parent2.gb_iface, axis)
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
        child.merge_sites(tol=1, mode='delete')
        # Add this ga_child to init_gb_astr
        child_gb_astr = self.grain_implant(child)

        return child_gb_astr, inheritance

    def random_model(self, reg_id):
        """
        Similar to random_model_obj.random_model
        Wrapper around overlap_grains() and grain_implant()

        Args:
        reg_id (obj): register_id object
        """
        done = False
        while not done:
            active_iface = self.overlap_grains()
            gb_iface = self.grain_implant(active_iface)
            done = self.gb_iface_comp_check(gb_iface)

        gb_model = structure_record.model(gb_iface, reg_id)
        gb_model.inheritance = 'random'

        return gb_model

    def get_model(self, select, pool, reg_id):
        """
        The name of this function is set to get_model() to match with the
        similar model in 'evolve' class.

        A wrapper around the fraction_slice() and
        join_slices_to_mold() functions

        Args:
        select (obj): Select object
        pool (obj): Pool object
        reg_id (obj): register_id object

        Returns a new gb model object
        """
        hop = self.hop

        correct_comp = False
        while correct_comp is False:
            try:
                if random.random() <= self.hop_mate_frac:
                    # Do hop.perturb_sites()
                    perturbed_iface, inheritance = hop.perturb_sites(
                                                        select, pool, gb=True)
                    self.move_coords_inside(perturbed_iface)
                    new_astr = self.grain_implant(perturbed_iface)
                    maker = 'perturb_sites'
                else: # Do gb_ops_obj.mate()
                    new_astr, inheritance = self.mate(select, pool)
                    maker = 'fraction_slice'
            except:
                continue
            if new_astr is None:
                continue
            if any(np.isnan(new_astr.cart_coords.flatten())):
                continue
            new_astr.sort()
            correct_comp = self.gb_iface_comp_check(new_astr)

        new_model = structure_record.model(new_astr, reg_id)
        new_model.inheritance = inheritance
        new_model.made_by = maker

        #print ('New model made using {} method on parent models {}'.format(
        #                                    maker, inheritance))

        return new_model

    def move_coords_inside(self, astr):
        """
        For a given structure object, move all sites within the unit cell.
        Eg: [-0.1, 0.4, 1.2] --> [0.9, 0.4, 0.2]

        returns 'astr' with all atoms inside

        Args:

        astr: pymatgen Structure object
        """
        species = astr.species
        fc = astr.frac_coords
        fc = np.where((fc<0) | (fc>1), fc - np.floor(fc), fc)

        # replace all the coords in astr
        all_inds = [i for i in range(len(species))]
        astr.remove_sites(all_inds)
        for sp, coords in zip(species, fc):
            astr.append(sp, coords, coords_are_cartesian=False)

    def gb_iface_comp_check(self, new_gb):
        """
        Checks that the number of atoms per species of given gb structure is
        within the allowed range

        Args:

        new_gb: the grain boundary structure object
        """
        new_gb.sort()
        new_comp = new_gb.composition
        hollow_comp = self.hollow_init_gb.composition

        all_ok = []
        sp1_ok = False
        sym1 = self.sym_species1
        if self.min_num_sp1 <= new_comp[sym1] - hollow_comp[sym1] <= \
                                                        self.max_num_sp1:
            sp1_ok = True
        all_ok.append(sp1_ok)
        if self.num_species > 1:
            sp2_ok = False
            sym2 = self.sym_species2
            if self.min_num_sp2 <= new_comp[sym2] - hollow_comp[sym2] <= \
                                                        self.max_num_sp2:
                sp2_ok = True
            all_ok.append(sp2_ok)
        if self.num_species > 2:
            sp3_ok = False
            sym3 = self.sym_species3
            if self.min_num_sp3 <= new_comp[sym3] - hollow_comp[sym3] <= \
                                                        self.max_num_sp3:
                sp3_ok = True
            all_ok.append(sp3_ok)
        if self.num_species > 3:
            sp4_ok = False
            sym4 = self.sym_species4
            if self.min_num_sp4 <= new_comp[sym4] - hollow_comp[sym4] <= \
                                                        self.max_num_sp4:
                sp4_ok = True
            all_ok.append(sp4_ok)
        if self.num_species > 4:
            sp5_ok = False
            sym5 = self.sym_species5
            if self.min_num_sp5 <= new_comp[sym5] - hollow_comp[sym5] <= \
                                                        self.max_num_sp5:
                sp5_ok = True
            all_ok.append(sp5_ok)

        correct_comp = False
        if False not in all_ok:
            correct_comp = True

        return correct_comp

         #####

class surface_ops(object):
    """
    Contains all the functions related to the structure manipulation of
    surface searches of a slab. Inclues both for initial population and
    mating and basinhopping.
    """
    def __init__(self, surface_ops_params):
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
            print ("Error: Please provide path to initial surface "
                   "configuration slabs as init_slabs_dict")
        else:
            self.init_slabs_dict = surface_ops_params['init_slabs_dict']

        self.surface_thickness = 1 # in Å
        if 'surface_thickness' in surface_ops_params:
            self.surface_thickness = surface_ops_params['surface_thickness']

        substrate_thickness = 10 # in Å
        if 'substrate_thickness' in surface_ops_params:
            self.substrate_thickness = surface_ops_params['substrate_thickness']

        # separation between surface layer bottom z and substrate top z
        self.separation = 2 # Å
        if 'separation' in surface_ops_params:
            self.separation = surface_ops_params['separation']

        constrain_z = False
        if 'constrain_z' in surface_ops_params:
            self.constrain_z = surface_ops_params['constrain_z']

        # The composition of the surface layer
        self.comp_dict = None
        if 'comp_dict' in surface_ops_params:
            self.comp_dict = surface_ops_params['comp_dict']

        # Following keywords will be present in surface_ops_params
        self.min_dist_dict = surface_ops_params['min_dist_dict']
        self.max_dist_dict = surface_ops_params['max_dist_dict']
        self.num_slices = surface_ops_params['num_slices']
        self.species_dict = surface_ops_params['species_dict']
        self.element_syms = surface_ops_params['element_syms']

    def default_zs_to_species_dict(self, slab_astr, species_id):
        """
        Returns boolean whether default cartesian z-coordinates for the surface
        species were added to the species_dict attribute of that species

        Args:

        species_id (int): It is 1 or 2 or 3 dpepending on whether it is
                          species1 or species2 or species3 and so on..
        """
        surface_thickness = self.surface_thickness
        element_syms = self.element_syms
        species_dict = self.species_dict

        slab_sites = slab_astr.sites
        z_cart_max = slab_astr.cart_coords[:, 2].max()

        surface_sites = [site for site in slab_sites if \
                        site.coords[2] > (z_cart_max - surface_thickness)]
        z_carts = np.unique([site.coords[2] for site in surface_sites \
                        if site.specie.name == element_syms[species_id]])

        if len(z_carts) == 0:
            print ('{} atoms are not present in the surface layer '
                    'of init_slab_astr'.format(element_syms[species_id]))
        added = False
        for k in species_dict.keys():
            if species_dict[k]['name'] == element_syms[species_id]:
                self.species_dict[k]['z_carts'] = z_carts
                added = True

        return added

    def get_zs_for_species(self, species_name, atoms_per_species):
        """
        Returns a list of cartesian z-coordinates for the species

        Args:

        species_name (str): The symbol of the species in the surface layer for
                            which z-coordinates are required

        num_species (int): The number of atoms for the species (or z-
                           coordinates required)
        """
        #TODO: Add a tolerance parameter to slightly vary z-coords
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

        if not ab:
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

    def random_surface_layer(self, slab_astr):
        """
        Returns a random surface layer for initial population

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
        Return the slab+surface structure
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
                self.default_zs_to_species_dict(slab_astr, key)
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
                if len(sp2_z_carts) == 0: # use random when 0
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
                print ("Error: Species2 'bonds_to' should be either None or 1")
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
                if len(sp3_z_carts) == 0: # use random when 0
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
                print ("Error: Species3 'bonds_to' should be either None/1/2")
            sps += [element_syms[3] for i in range(len(sp3_frac_coords))]
            coords += sp3_frac_coords

        new_surf_layer = Structure(lattice, sps, coords,
                                   coords_are_cartesian=False)

        return new_surf_layer

    def random_model(self, reg_id):
        """
        Returns a model object which is a random_surface_layer on same area
        substrate.

        Args:

        reg_id (obj): register_id object
        """
        # Choose a init_slab from the dict
        slab_astr = self.choose_init_slab()

        # Get substrate from the slab_astr
        bot_z_cart = slab_astr.cart_coords[:, 2].min()
        slab_sites = slab_astr.sites
        sub_inds = [i for i, site in enumerate(slab_sites) if \
                        site.coords[2] - bot_z_cart <= self.substrate_thickness]

        sub_sps = [slab_astr.species[i] for i in sub_inds]
        sub_fracs = [slab_astr.frac_coords[i] for i in sub_inds]
        substrate = Structure(slab_astr.lattice, sub_sps, sub_fracs,
                                        coords_are_cartesian=False)

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

    def get_model():
        """
        returns a new model created either by mating or basinhopping
        """
        pass

    def get_atoms_per_species(self):
        """
        Returns a dict with number of atoms for each species.
        To be used when making a random surface layer for initial population
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
                atoms_per_species[species] = int(fixed_num * \
                                    comp_dict[species] / comp_dict[fixed_sps])

        return atoms_per_species

    def get_random_coords_for_species(self, num_coords_needed, lattice,
                                      min_dist, max_dist=None, z_carts=None):
        """
        Returns a (num) list of random fractional coords, each separated by a
        distance within the min_dist and max_dist

        Algorithm:

        While len(random_coords) < num;
            Get a fractional coordinate.
            Calculate the distance between the two coords  wrt lattice
            If the distance satisfies min and max dist, add it to random_coords.
            Else, continue..

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
                         should be equal to the num_coords_needed parameter
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

    def get_secondary_random_coords(self, num_coords_needed, lattice, primary_frac_coords, min_dist_12, min_dist_22, other_min_dist=None, other_frac_coords=None, z_carts=None):
        """
        Returns random coords which satisfy distance constaints among same
        species and with species 1

        Args:

        num_coords_needed (int): Number of random coordinates needed for the
                                 given species

        lattice (obj): pymatgen lattice object of the surface layer. (The c
                       lattice vector should be corresponding to the fractional
                       z_carts)

        primary_frac_coords (list/array(nx3)): List of primary coords. Minimum
                                               distance is checked using
                                               min_dist_12

        min_dist_12 (float): the minimum distance required between any two atoms
                          of species 2 & 1

        min_dist_22 (float): the minimum distance between two atoms of
                             species 2 & 2 (secondary speices)

        z_carts (list): The list of (fractional) z-coordinates for all the
                         atoms of the given species. The length of the list
                         should be equal to the num_coords_needed parameter

        """
        # Add random skipped coords to this structure
        tries, num_added = 0, 0
        if z_carts:
            z_fracs = np.array(z_carts) / lattice.c
            if len(z_fracs) != num_coords_needed:
                print ("Error: Z-carts not present for all coords!")

        secondary_coords = []
        while num_added < num_coords_needed and tries < 1000:
            tries += 1
            new_fracs = [random.random(), random.random(), random.random()]
            if z_fracs:
                new_fracs[2] = z_fracs[num_added]
            new_carts = lattice.get_cartesian_coords(new_fracs)
            if len(lattice.get_points_in_sphere(primary_frac_coords,
                                                new_carts, min_dist_12)) == 0:
                if len(lattice.get_points_in_sphere(secondary_coords,
                                                new_carts, min_dist_22)) == 0:
                    if other_min_dist and other_frac_coords:
                        if len(lattice.get_points_in_sphere(other_frac_coords,
                                            new_carts, other_min_dist)) == 0:
                            secondary_coords.append(new_fracs)
                            num_added += 1
                            tries = 0
                    else:
                        secondary_coords.append(new_fracs)
                        num_added += 1
                        tries = 0

        return secondary_coords

    def get_connected_coords(self, num_coords_needed, lattice, fixed_coords, min_dist_12, max_dist_12, min_dist_22, connected_z_carts=None):
        """
        For a given list of coords (of say species_1), returns the species 2
        coords such that they satisfy -
        - minimum and maximum bond distance between atoms of species 1 & 2 resp
        - minimum bond distance between two atoms of species 2
        - number of bonds for each atom in species 1

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
                # Check if the new point satisfies all minimum distance constraints
                if len(lattice.get_points_in_sphere(fixed_coords, new_carts,
                                                            min_dist_12)) == 0:
                    if len(connected_coords) == 0:
                        new_fracs = lattice.get_fractional_coords(new_carts)
                        connected_coords.append(new_fracs)
                        print (tries)
                        tries = 0
                        found_coord = True
                        continue
                    if len(lattice.get_points_in_sphere(connected_coords, new_carts,
                                                            min_dist_22)) == 0:
                        new_fracs = lattice.get_fractional_coords(new_carts)
                        connected_coords.append(new_fracs)
                        print (tries)
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
                                                new_carts, min_dist_12)) == 0:
                    if len(lattice.get_points_in_sphere(connected_coords,
                                                new_carts, min_dist_22)) == 0:
                        connected_coords.append(new_fracs)
                        num_added += 1
                        tries = 0

        return connected_coords

    def get_point_on_circle(self, center_carts, radius, z_cart):
        """
        Returns (x, y, z_cart) which is at a distance of radius from the center_carts

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
        solution = sqrt(radius**2 - (center_carts[0] - x) ** 2 - \
                        (center_carts[2] - z_cart) ** 2)

        if random.random() < 0.5:
            y = solution + center_carts[1]
        else:
            y = solution - center_carts[1]

        return [x, y, z_cart]

    # functions for mating operations
    def mate(self, parent_1, parent_2):
        """
        Performs mating by slicing for given two models and returns child
        structure. Here both the parents should be of same lattice in the x-y
        direction.


        Algorithm:
        1. Select a line that passes through the center of the lattice
        2. Cut both parents surface layers into two halves
        3. Select one half from one parent and the other half from the second
           parent
        4. Join them
        5. Attach the child surface layer on top of the substrate slab
        """

    def surface_comp_check(self, surface_astr):
        """
        For terminology use these variables. All are pymatgen structure objects.
        slab_astr - substrate (or bulk) slab with a surface layer on top
        surface_layer - only the top layer which corresponds to surface
        substrate - only the bottom slab which corresponds to bulk or substrate

        """
        pass
