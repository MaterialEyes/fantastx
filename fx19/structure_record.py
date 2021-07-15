
from __future__ import division, unicode_literals, print_function

"""
This module contains classes to manage the structures, their corresponding
attributes and evaluated data

Also contains general functions (if any required)
"""
from pymatgen.core.structure import Structure, Lattice
from pymatgen.core.composition import Composition

class register_id(object):
    def __init__(self):
        self.label = 0
    def create_id(self):
        self.label += 1
        return self.label


class model(object):
    """
    Information related to each strucutre object including the object itself are
    stored as attributes to the model object
    """
    def __init__(self, astr, reg_id):
        """
        Given a structure object, makes model object assuming defaults for all
        attrubites

        Args:

        astr (obj): pymatgen.core.structure.Structure object
        reg_id (obj): register_id object
        """

        # id of the model
        # age is also known by id number
        self.label = reg_id.create_id()
        # pymatgen structure object
        self.astr = astr
        # To save middle interface grain gor gb model
        self.gb_iface = None
        # where the structure came from (list of ints)
        self.inheritance = None
        self.made_by = None
        # set exp sim functions as variables
        self.Xsim1 = None
        self.Xsim2 = None
        self.Xsim3 = None
        self.Xsim4 = None
        # total energy after relaxation
        self.tot_en = None
        # Used if vasp is energy_code to resubmit calculation
        self.converged = None # used
        # energy objective value (like gamma or adsorption energy)
        self.obj0_val = None
        # exp_sim objective values
        # NOTE: 1, 2, 3, 4 are decided internally at the beginning
        self.obj1_val = 0
        self.obj2_val = 0
        self.obj3_val = 0
        self.obj4_val = 0
        # (single or multi objective function) overall value
        self.overall_val = None
        # selection probability based on overall_val
        # gets updated after every new added strucutre
        self.selection_prob = None
        # How many times this structure is selected from get_parent()
        self.times_chosen_as_parent = None

    def check_atom_density(self, box_astr, num_reg, axis=2):
        """
        For a box filled with atoms, checks if following condition satisfies along
        the axis,
            divides the box into equal number of regions (n_R) and checks that
            (total_atoms/n_R) * 0.5 < atoms in region R < (total_atoms/n_R) * 1.5

        NOTE: should not be used for cluster in a box

        Args:
        box_astr (obj): structure object of the box (gb or total box)
        num_reg (int): number of equal regions
        axis (int): the direction along which regions should be divided
                0, 1, 2 for x, y, and z axes respectively

        return True if satisfies
        """

        all_coords = box_astr.cart_coords
        num_atoms = len(all_coords)
        sorted_coords = sorted(all_coords, key=lambda x: x[axis])
        sums = []
        bnds = [(i+1) * box_astr.lattice.abc[axis]/num_reg for i in range(num_reg)]
        n = 0
        for bound in bnds:
            for i, p in enumerate(sorted_coords):
                if p[axis] > bound:
                    sums.append(i+1)
                    break
        sums.append(100)
        counts = [sums[0]]
        for i in range(len(sums)-1):
            counts.append(sums[i+1] - sum(counts))
        bools = []
        X = num_atoms/num_reg
        for i in range(len(counts)):
            bools.append(X * 0.5 < counts[i] < X * 1.5)
        if all(bools):
            return True
        else:
            return False

    def set_overall_val(self, weights):
        """
        (Deprecated)
        Overall_val is set everytime new model is added to pool in
        selection.Pool

        Sets overall_val for a model. Basically, this is weights times each val
        This does not change until weights are changed.

        Args:

        weights (list): list of weights for each objective function.
        """
        if all(weights) > 0:
            w0, w1, w2, w3, w4 = weights
            val_0 = self.obj0_val
            val_1, val_2, val_3, val_4 = 0, 0, 0, 0
            if self.obj1_val is not None:
                val_1 = self.obj1_val
            if self.obj2_val is not None:
                val_2 = self.obj2_val
            if self.obj3_val is not None:
                val_3 = self.obj3_val
            if self.obj4_val is not None:
                val_4 = self.obj4_val

            # Overall_val is the sum of weights * obj_val
            overall_val = 1/w0 * val_0 + 1/w1 * val_1 + 1/w2 * val_2 \
                                                + 1/w3 * val_3+ 1/w4 * val_4
            self.overall_val = overall_val


class structure_constraints(object):
    """
    Reads all the inputs provided by user and assumes defaults for some
    parameters
    """

    def __init__(self, str_record):
        """
        Args:

        str_record (dict): dictionary of all the parameters from input file
        under structure_record
        """

        self.def_min_dist = 2 # minimum distance between atoms in angstroms
        self.def_max_dist = 5
        self.min_num_atoms = 30
        self.max_num_atoms = 101
        self.max_bond_dist = 4

        if 'max_bond_dist' in str_record:
            self.max_bond_dist = str_record['max_bond_dist']

        # save details of all species as a dict
        if 'species' not in str_record:
            print ('species - names, atoms and min_dist are not mentioned'
                   ' in input file. These are mandatory!')
        else:
            species_dict = str_record['species']
            self.num_species = len(species_dict)

        # make a min_dist dictionary with default min_dist for all bonds
        min_dist, max_dist = {}, {}
        keys = ['sp1_sp1', 'sp1_sp2', 'sp1_sp3', 'sp1_sp4', 'sp1_sp5',
                'sp2_sp2', 'sp2_sp3', 'sp2_sp4', 'sp2_sp5',
                'sp3_sp3', 'sp3_sp4', 'sp3_sp5',
                'sp4_sp4', 'sp4_sp5',
                'sp5_sp5']
        # trick to remove bonds with more species
        sps = ['sp1', 'sp2', 'sp3', 'sp4', 'sp5']
        sps = sps[:self.num_species]
        min_dist_keys = []
        for key in keys:
            a,b = key.split('_')
            if a in sps and b in sps:
                min_dist_keys.append(key)
        for key in min_dist_keys:
            min_dist[key] = self.def_min_dist
            max_dist[key] = self.def_max_dist

        # check if min_dist is given for any bonds and replace
        if 'min_dist' in str_record:
            given_dist_dict = str_record['min_dist']
            given_keys = given_dist_dict.keys()
            for key in given_keys:
                min_dist[key] = str_record['min_dist'][key]
        self.min_dist_dict = min_dist

        if 'max_dist' in str_record:
            given_max_dists = str_record['max_dist']
            given_keys = given_max_dists.keys()
            for key in given_keys:
                max_dist[key] = str_record['max_dist'][key]
        self.max_dist_dict = max_dist

        # see that all attributes for all species are present by placing
        # defaults for species1 and that of species1 for the rest
        element_syms = {}
        if 'species1' not in species_dict:
            print ('Please provide species1 under species in input file.')
        else:
            self.species1 = species_dict['species1']
            if 'name' not in self.species1:
                print ('Please specify element name (Ex: \'Al\') of specie 1.')
            element_syms[1] = self.species1['name']
            if 'min_num' not in self.species1:
                self.species1['min_num'] = self.min_num_atoms
            else:
                self.min_num_atoms = self.species1['min_num']
            if 'max_num' not in self.species1:
                self.species1['max_num'] = self.max_num_atoms
            else:
                self.max_num_atoms = self.species1['max_num']

        if self.num_species > 1:
            if 'species2' in species_dict:
                self.species2 = species_dict['species2']
                if 'name' not in self.species2:
                    print ('Please specify element name (Ex: \'Al\') of specie 2.')
                element_syms[2] = self.species2['name']
                if 'min_num' not in self.species2:
                    self.species2['min_num'] = self.min_num_atoms
                if 'max_num' not in self.species2:
                    self.species2['max_num'] = self.max_num_atoms
            else:
                print ('Error: Please check the input format for species')

        if self.num_species > 2:
            if 'species3' in species_dict:
                self.species3 = species_dict['species3']
                if 'name' not in self.species3:
                    print ('Please specify element name (Ex: \'Al\') of specie 3.')
                element_syms[3] = self.species3['name']
                if 'min_num' not in self.species3:
                    self.species3['min_num'] = self.min_num_atoms
                if 'max_num' not in self.species3:
                    self.species3['max_num'] = self.max_num_atoms
            else:
                print ('Error: Please check the input format for species')

        if self.num_species > 3:
            if 'species4' in species_dict:
                self.species4 = species_dict['species4']
                if 'name' not in self.species4:
                    print ('Please specify element name (Ex: \'Al\') of specie 4.')
                element_syms[4] = self.species4['name']
                if 'min_num' not in self.species4:
                    self.species4['min_num'] = self.min_num_atoms
                if 'max_num' not in self.species3:
                    self.species4['max_num'] = self.max_num_atoms
            else:
                print ('Error: Please check the input format for species')

        if self.num_species > 4:
            if 'species5' in species_dict:
                self.species5 = species_dict['species5']
                if 'name' not in self.species5:
                    print ('Please specify element name (Ex: \'Al\') of specie 5.')
                element_syms[5] = self.species5['name']
                if 'min_num' not in self.species5:
                    self.species5['min_num'] = self.min_num_atoms
                if 'max_num' not in self.species5:
                    self.species5['max_num'] = self.max_num_atoms

        self.element_syms = element_syms

        #########################cluster parameters############################
        # shape and related
        if 'cluster' in str_record:
            self.shape = 'cluster'
        elif 'gb' in str_record:
            self.shape = 'gb'
        # TODO: add other shapes here

        if self.shape == 'cluster':
            if 'box_abc' in str_record['cluster']:
                self.box_abc = str_record['cluster']['box_abc']
            else:
                print ('The lattice lengths of the box are not specified.'
                        ' Using default orthogonal box of a=b=c=20Å')
                self.box_abc = [20, 20, 20]

            if 'max_dia' in str_record['cluster']:
                self.max_dia = str_record['cluster']['max_dia']
            else:
                print ('The maximum diameter of the cluster is not specified. '
                        'Using default diameter of 8Å')
                self.max_dia = 8

        ####################cluster parameters ends###########################
        #########################gb parameters begins#########################
        if self.shape == 'gb':
            init_gb_astr_path = str_record['gb']['init_gb_astr']
            self.init_gb_astr = Structure.from_file(init_gb_astr_path)
            if 'iface_thickness' in str_record['gb']:
                self.iface_thickness = str_record['gb']['iface_thickness']
            else:
                self.iface_thickness = 10 # angstroms
            if 'iface_z_mid' in str_record['gb']:
                self.iface_z_mid = str_record['gb']['iface_z_mid']
            else:
                self.iface_z_mid = 0.5 # assuming default mid as 0.5

            gb_latt_matrix = self.init_gb_astr.lattice.matrix
            self.iface_latt = Lattice([gb_latt_matrix[0],
                                      gb_latt_matrix[1],
                                      [0, 0, self.iface_thickness]])
            self.num_slices = 2
            if 'num_slices' in str_record['gb']:
                self.num_slices = str_record['gb']['num_slices']

            self.hop_mate_frac = 0.5
            if 'hop_mate_frac' in str_record['gb']:
                self.hop_mate_frac = str_record['gb']['hop_mate_frac']

            """
            We get best matched gb interface structure from ingrained.
            For ingrained, this grain data needs to be provided.
            Grain data is not required for Fantastx.

            self.grain1_data = None # {'orientation': (1,1,1), 'tilt': 15, 'name': Al}
            #self.grain1_orientation =
            #self.grain1_tilt =
            self.grain2_data = None # {'orientation': (1,1,0), 'tilt': -10, 'name': Ge}
            #self.grain2_orientation =
            #self.grain2.tilt =
            """
        ######################### gb parameters ends #########################
        ##################### surface parameters begins ######################
        if self.shape == 'surface':
            surface_params = str_record['surface']
            init_slabs_dir = surface_params['init_slabs_dir']
            poscars = [init_slabs_dir + '/' + i for i in \
                      os.listdir(init_slabs_dir) if i.startswith('POSCAR_slab')]
            init_slabs_dict = {}
            for p in range(len(poscars)):
                init_slabs_dict[p+1] = poscars[p]

            self.init_slabs_dict = init_slabs_dict

            if 'surface_thickness' in surface_params:
                self.surface_thickness = surface_params['surface_thickness']

            if 'substrate_thickness' in surface_params:
                self.substrate_thickness = \
                                    surface_params['substrate_thickness']

            if 'separation' in surface_params:
                self.separation = surface_params['separation']

            if 'constrain_z' in surface_params:
                self.constrain_z = surface_params['constrain_z']

            if 'composition' in surface_params:
                composition = Composition(surface_params['composition'])
                self.comp_dict = composition.as_dict()

            self.num_slices = 2
            if 'num_slices' in surface_params:
                self.num_slices = surface_params['num_slices']

            self.hop_mate_frac = 0.5
            if 'hop_mate_frac' in surface_params:
                self.hop_mate_frac = surface_params['hop_mate_frac']


    def get_constraints(self):
        """
        Returns a dictionary of all the constraints listed above
        To be used in structure_operations
        """

        return self.__dict__


















##
