from __future__ import division, unicode_literals, print_function

"""
This module performs does relaxations and then populates the following data for
each model:

1. evaluate_energy : does structure relaxation, and gives energy

2. energy_obj : The first objective function (energy) is calculated

Functions are written assuming we use independent compilations of energy codes

NOTE: Use Malsky version's gulp.py, lammps.py and vasp.py
"""


from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.lammps.data import LammpsData
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.inputs import Poscar

import os
import shutil
# import math
import numpy as np
import subprocess as sp


class lammps_code(object):

    def __init__(self, energy_params):
        """
        energy_params: dictionary of all the parameters

            Eg: {'main_path': <path to directory in which fantastx is ran>,
                 'shape': 'gb',
                 'energy_files_path': <path_to_input_files>,
                 'energy_exec_cmd': 'lmp_mpi -in in.min',
                 'sym_mu_dict': {'Al': -3.35958515625, 'O': -6.76069604253},
                 'atom_style': 'charge'}
        """
        self.main_path = energy_params['main_path']
        self.shape = energy_params['shape']
        # path to make new folder to do calculations
        self.relax_path = None
        # path to copy input files for each calculation
        self.energy_files_path = energy_params['files_path']
        # lammps execution command as a string
        # Ex: 'lmp_mpi -in in.min'
        self.energy_exec_cmd = energy_params['energy_exec_cmd']
        # Make a new folder to store all lammps.label (log files) for convenience
        # save hollow_botz and hollow_topz for use in sd_flags
        self.hollow_botz = None
        self.hollow_topz = None

        # DU
        # Save species names and their chemical potentials for identification
        self.sym_mu_dict = {}
        for key, value in energy_params['element_syms'].items():
            self.sym_mu_dict[value] = energy_params['mu'][key]

        # chemical potentials of each species

        # default atom_style
        def_atom_style = 'charge'
        self.atom_style = def_atom_style
        if 'atom_style' in energy_params:
            self.atom_style = energy_params['atom_style']

    def prep_job_folder(self, model, reg_id):
        """
        Function to
        - sanity check the provided input files (if any)
        - copy the input files to the org calc directory (relax_path)

        input files for lammps: in.min and potential file
        """
        main_path = self.main_path
        # create folders for model and relaxation
        # ProcessPoolExecutor has multiple parallel processes make models
        # at same time. So, reg_id is not up to date always.
        while True:
            if not model.inheritance == 'from_file':
                try:
                    model_path = main_path + '/calcs/' + str(model.label)
                    os.mkdir(model_path)
                except FileExistsError:
                    model.label = reg_id.create_id()
                    continue
                else:
                    break
            else:
                model_path = main_path + '/calcs/' + str(model.label)
                os.mkdir(model_path)
                break

        relax_path = main_path + '/calcs/' + str(model.label) + '/relax'
        os.mkdir(relax_path)
        self.relax_path = relax_path
        astr = model.astr
        files_path = self.energy_files_path
        atom_style = self.atom_style
        # write model structure to POSCAR and store it in /relax
        new_poscar = relax_path + '/POSCAR_unrelaxed'
        sd_flags = [[0, 0, 0] for i in range(len(astr))]
        gb_poscar = Poscar(astr, selective_dynamics=sd_flags)
        gb_poscar.write_file(new_poscar)
        # check if both files exist.
        file_list = os.listdir(files_path)
        # For LAMMPS: check for in.min file in the files_path
        if 'in.min' in file_list:
            input_file = files_path + '/in.min'
        else:
            print('in.min file not present in energy_inputs_path. This file '
                  'is mandatory.')
        # copy files from files_path to relax_path
        shutil.copy(input_file, relax_path)
        # NOTE: The path to lammps potential file should be specified in in.min
        # get data_file and copy to calc path
        lammps_data = LammpsData.from_structure(astr, ff_elements=None,
                                                atom_style=atom_style)
        # write data file for lammps in relax_path
        data_file_path = relax_path + '/in.data'
        lammps_data.write_file(data_file_path)

        # (optional) make any changes to the in.min if necessary

        # returns nothing

    def relax(self, model, reg_id):
        """
        copy input files and structure (model),
        relax

        args:

        model: model object of the new model which is to be realxed
        reg_id: reg_id object
        """
        # prepare the folder to start energy calc
        self.prep_job_folder(model, reg_id)
        # start the lammps calculation
        relax_path = self.relax_path
        os.chdir(relax_path)
        lammps_exec = self.energy_exec_cmd.split()
        with open('log_lammps.{}'.format(model.label), 'w') as log_file:
            lammps_job = sp.Popen(
                lammps_exec, stdout=sp.PIPE, stderr=sp.STDOUT)
            for each_line in lammps_job.stdout:
                line = each_line.decode('utf-8')
                log_file.write(line)
        # wait for the calculation to finish
        lammps_job.wait()

        # save total energy to model attributes
        total_energy = None
        with open('log_lammps.{}'.format(model.label), 'r') as log:
            lines = log.readlines()
            string = 'Step Temp E_pair E_mol TotEng Press'
            for i, line in enumerate(lines):
                if string in line:
                    total_energy = float(lines[i+2].split()[4])
        if not total_energy:
            print ('Model {} energy not found in log_lammps.{} file'.format(
                                                model.label, model.label))
            print ('LAMMPS relaxation on model {} NOT successful'.format(
                                                model.label))
            # quit()
        else:
            model.tot_en = total_energy
            # For lammps, assume always converged after relaxation
            model.converged = True
            # get objective value from energy and save to attributes
            # NOTE: objective function for cluster is assumed to be epa
            # NOTE: objective function for gb - ?????

            # Make sure to 'dump' relaxed structure to 'rlx.str' file in in.min
            # get symbols of elements as a list
            astr = model.astr
            symbols = []
            for i in astr.species:
                if i.symbol not in symbols:
                    symbols.append(i.symbol)
            relaxed_astr = self.get_relaxed_cell('rlx.str', 'in.data', symbols)
            # save relaxed structure in model.astr and to poscar
            POSCAR_relaxed = relax_path + '/POSCAR_relaxed'
            relaxed_astr.sort()
            self.move_atoms_inside(relaxed_astr)
            relaxed_astr.to(filename=POSCAR_relaxed, fmt='poscar')
            model.astr = relaxed_astr
            # NOTE: Do not overwrite model.astr as it could be used in Xsim(s)
            # Save the grain boundary as a model attribute
            comp_dict = relaxed_astr.composition.as_dict()
            astr_elems = [i.name for i in relaxed_astr.composition.elements]

            # DU
            free_en = total_energy
            for elem in astr_elems:
                if elem in self.sym_mu_dict.keys():
                    free_en -= comp_dict[elem]*self.sym_mu_dict[elem]
                else:
                    print("Error. LAMMPS species " + elem +
                          " not contained in input yaml file.")
            model.obj0_val = float(free_en)

        # Following are done in relax:
        # save relaxed_structure - done in do_relaxation
        # relaxed_structure is now the structure of the model
        # save other attributes of the model after relaxation (energy, gamma etc)
        # checks if relaxation is successful; gives error message and do not go
        # ahead with the structure (goes back and creates new strucutre)

    def get_relaxed_cell(self, rlx_astr, data_in_path, element_symbols):
        """
        (written by Benjamin Revard)

        Parses the relaxed cell from the dump.atom file.
        Returns the relaxed cell as a Cell object.
        Args:

        rlx_astr: the path (as a string) to the dump.atom file
        in_data_path: the path (as a string) to the in.data file
        element_symbols: a tuple containing the set of chemical symbols of
                         all the elements in the compositions space
        """

        # read the dump.atom file as a list of strings
        with open(rlx_astr, 'r') as atom_dump:
            lines = atom_dump.readlines()

        # get the lattice vectors
        a_data = lines[5].split()
        b_data = lines[6].split()
        c_data = lines[7].split()

        # default assume tilt is 0
        xy, xz, yz = 0, 0, 0
        # parse the tilt factors if thye exist
        if len(a_data) > 2:
            xy = float(a_data[2])
        if len(b_data) > 2:
            xz = float(b_data[2])
        if len(c_data) > 2:
            yz = float(c_data[2])

        # parse the bounds
        xlo_bound = float(a_data[0])
        xhi_bound = float(a_data[1])
        ylo_bound = float(b_data[0])
        yhi_bound = float(b_data[1])
        zlo_bound = float(c_data[0])
        zhi_bound = float(c_data[1])

        # compute xlo, xhi, ylo, yhi, zlo and zhi according to the conversion
        # given by LAMMPS
        # http://lammps.sandia.gov/doc/Section_howto.html#howto-12
        xlo = xlo_bound - min([0.0, xy, xz, xy + xz])
        xhi = xhi_bound - max([0.0, xy, xz, xy + xz])
        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max([0.0, yz])
        zlo = zlo_bound
        zhi = zhi_bound

        # construct a Lattice object from the lo's and hi's and tilts
        a = [xhi - xlo, 0.0, 0.0]
        b = [xy, yhi - ylo, 0.0]
        c = [xz, yz, zhi - zlo]
        relaxed_lattice = Lattice([a, b, c])

        # get the number of atoms
        num_atoms = int(lines[3])

        # get the atom types and their Cartesian coordinates
        types = []
        relaxed_cart_coords = []
        for i in range(num_atoms):
            atom_info = lines[9 + i].split()
            types.append(int(atom_info[1]))
            relaxed_cart_coords.append([float(atom_info[2]) - xlo,
                                        float(atom_info[3]) - ylo,
                                        float(atom_info[4]) - zlo])

        # read the atom types and corresponding atomic masses from in.data
        with open(data_in_path, 'r') as data_in:
            lines = data_in.readlines()
        types_masses = {}
        for i in range(len(lines)):
            if 'Masses' in lines[i]:
                for j in range(len(element_symbols)):
                    types_masses[int(lines[i + j + 2].split()[0])] = float(
                        lines[i + j + 2].split()[1])

        # map the atom types to chemical symbols
        types_symbols = {}
        for symbol in element_symbols:
            for atom_type in types_masses:
                # round the atomic masses to one decimal point for comparison
                if format(float(Element(symbol).atomic_mass), '.1f') == format(
                        types_masses[atom_type], '.1f'):
                    types_symbols[atom_type] = symbol

        # make a list of chemical symbols (one for each site)
        relaxed_symbols = []
        for atom_type in types:
            relaxed_symbols.append(types_symbols[atom_type])

        return Structure(relaxed_lattice, relaxed_symbols, relaxed_cart_coords,
                         coords_are_cartesian=True)

    def move_atoms_inside(self, astr):
        """
        For a given structure object, move all sites within the unit cell.
        Eg: [-0.1, 0.4, 1.2] --> [0.9, 0.4, 0.2]

        returns 'astr' with all atoms inside

        Args:

        astr: pymatgen Structure object
        """
        species = astr.species
        fc = astr.frac_coords
        fc = np.where((fc < 0) | (fc > 1), fc - np.floor(fc), fc)

        # replace all the coords in astr
        all_inds = [i for i in range(len(species))]
        astr.remove_sites(all_inds)
        for sps, coords in zip(species, fc):
            astr.append(sps, coords, coords_are_cartesian=False)


class gulp_code(object):
    # TODO: create all functions for running gulp energy code
    """
    funcitons to create a folder,
    copy input files and structure (model),
    relax,
    get_energy
    (get_other_data from each relaxation)
    """

    def __init__(self, path_to_input_files, model):
        """
        path_to_input_files : path to folder containing all general input files
        model: structure / atoms object
        """

    def get_structure_file(self, model):
        """
        function to make required structure file (cif?) from the structure/atoms
        object
        """

    def relax(self, path_to_input_files, model):
        """
        copy input files and structure (model),
        relax
        model_id = model._id
        model_energy = self.get_energy()
        model_en_obj_val = self.energy_obj()

        returns model_id, model_energy, model_en_obj_val

        ** This is to keep track of each organism directly from output of this
        function. Helps in parallelization, to gather information from the
        working nodes to master node.
        """

    def get_energy(self, output_file):
        """
        get final energy
        also get other data from output file (if required)
        """

    def energy_obj(self, relaxed_en):
        """
        Describes a function of relaxed energy (objective function) and returns
        the value

        In direct cases where energy is directly used, this will pass
        If surface energy or adsorption energy or other forms of derived energy
        functions are described here.
        """


class vasp_code(object):

    def __init__(self, energy_params):
        """
        energy_params: dictionary of all the parameters

            Eg: {'main_path': <path to direcctory in which fantastx is ran>,
                 'shape': 'gb',
                 'energy_files_path': <path_to_input_files>,
                 'energy_exec_cmd': 'srun <path_to_vasp_binary>',
                 'sym_mu_dict': {'Al': -3.35958515625, 'O': -6.76069604253},
                 'atom_style': 'charge'}

        # TODO: Change how objective function is calculated.
        """
        self.main_path = energy_params['main_path']
        self.shape = energy_params['shape']
        # path to make new folder to do calculations
        self.relax_path = None
        # path to copy input files for each calculation
        self.energy_files_path = energy_params['files_path']
        # vasp execution command as a string
        # Ex: 'mpirun <path_to_vasp_binary>'
        self.energy_exec_cmd = energy_params['energy_exec_cmd']
        # how many times to resubmit job if not converged
        self.resubmit = energy_params['resubmit']

        #  All these parameters for use in sd_flags
        # These are stored in inputs after object creation
        self.hollow_botz = None
        self.hollow_topz = None
        self.substrate_thickness = None
        self.sd_cut_off = None
        self.sd_no_z = None

        # This will be used to make potcars
        all_pots = [i for i in os.listdir(self.energy_files_path) if
                    i.startswith('POTCAR')]
        all_pots = [self.energy_files_path + '/' + i for i in all_pots]
        pdict = {}
        for a_pot in all_pots:
            with open(a_pot) as f:
                lines = f.readlines()
                for line in lines:
                    if 'TITEL' in line:  # assuming only PBE TODO: LDA and others
                        x = line.split('PBE')[1].split()[0]
                        if '_' in x:
                            x = x.split('_')[0]
                        pdict[x] = a_pot
        self.pot_dict = pdict

        # DU
        # Save species names and their chemical potentials for identification
        self.sym_mu_dict = {}
        for key, value in energy_params['element_syms'].items():
            self.sym_mu_dict[value] = energy_params['mu'][key]

        # default parameters for INCAR (only if necessary)
        # Or directly use the input files the user provided.

    def prep_job_folder(self, model, reg_id):
        """
        # returns nothing
        Function to
        - sanity check that all input files exist (and any flags if needed)
        - copy the input files to the org calc directory (relax_path)

        input files for vasp: INCAR, KPOINTS, POTCAR & POSCAR from model
        """
        main_path = self.main_path
        # create folders for model and relaxation
        # ProcessPoolExecutor has multiple parallel processes make models
        # at same time. So, reg_id is not up to date always.
        while True:
            if not model.inheritance == 'from_file':
                try:
                    model_path = main_path + '/calcs/' + str(model.label)
                    os.mkdir(model_path)
                except FileExistsError:
                    model.label = reg_id.create_id()
                    continue
                else:
                    break
            else:
                model_path = main_path + '/calcs/' + str(model.label)
                os.mkdir(model_path)
                break

        relax_path = main_path + '/calcs/' + str(model.label) + '/relax'
        os.mkdir(relax_path)
        self.relax_path = relax_path
        astr = model.astr
        files_path = self.energy_files_path

        # sort the structure according to electronegativities
        astr.sort()
        # get the sorted species in the structure
        sorted_elems = astr.composition.elements
        sorted_syms = [i.name for i in sorted_elems]
        # get potcar by concatenating the potcars in the same order
        all_lines = []
        for sps in sorted_syms:
            with open(self.pot_dict[sps]) as p:
                lines = p.readlines()
                all_lines = all_lines + lines

        # pot_path = self.relax_path + '/POTCAR'

        # write model structure to POSCAR and store it in /relax
        new_poscar = relax_path + '/POSCAR_unrelaxed'
        poscar = relax_path + '/POSCAR'
        potcar = relax_path + '/POTCAR'
        with open(potcar, 'w') as pot:
            pot.writelines(all_lines)

        if self.shape == 'gb':
            self.write_gb_poscar(model, new_poscar)

        if self.shape == 'surface':
            self.write_surface_poscar(model, new_poscar,
                                      sd_cut_off=self.sd_cut_off,
                                      sd_no_z=self.sd_no_z)
        # TODO: implement selective dynamics for cluster & other geometries

        shutil.copy(new_poscar, poscar)
        # copy INCAR, KPOINTS to the relax path
        shutil.copy(files_path + '/INCAR', relax_path + '/INCAR')
        shutil.copy(files_path + '/KPOINTS', relax_path + '/KPOINTS')

        print('Job prep finished. Submitting...')

    def relax(self, model, reg_id):
        """
        # Following are done in relax:
        # save relaxed_structure - done in do_relaxation
        # relaxed_structure is now the structure of the model
        # save other attributes of model after relaxation (energy, gamma etc)
        # checks if relaxation is successful; gives error message and do not go
        # ahead with the structure (goes back and creates new strucutre)
        # returns nothing

        copy input files and structure (model),
        calls run_vasp to relax

        args:

        model: model object of the new model which is to be realxed
        reg_id: reg_id object
        """
        # prepare the folder to start energy calc
        self.prep_job_folder(model, reg_id)
        # start the lammps calculation
        relax_path = self.relax_path
        # Go to job directory
        os.chdir(relax_path)
        self.run_vasp(model)

    def re_relax(self, model):
        """
        Deprecated

        checcks if converged, resubmits if resubmit > 0
        save output files fo previous run with _resubmited_number
        """
        if not model.converged and self.resubmit !=0:
            relax_path = self.main_path + '/calcs/' + str(model.label) + '/relax'
            os.chdir(relax_path)
            shutil.copy('OUTCAR', 'OUTCAR_{}'.format(self.resubmit-1))
            shutil.copy('CONTCAR', 'CONTCAR_{}'.format(self.resubmit-1))
            shutil.copy('OSZICAR', 'OSZICAR_{}'.format(self.resubmit-1))
            shutil.copy('POSCAR', 'POSCAR_{}'.format(self.resubmit-1))
            shutil.copy('CONTCAR', 'POSCAR')
            self.run_vasp(model)
        self.resubmit = self.resubmit - 1

    def run_vasp(self, model):
        """
        run vasp in the job directory (relax_path)
        check if converged and resubmit if necessary
        save energy and objective function to model object

        Args:

        model: model object
        """
        # returns nothing

        vasp_exec = self.energy_exec_cmd.split()
        log_file = open('job.log', 'w')
        err_file = open('job.err', 'w')
        sp.call(vasp_exec, stdout=log_file, stderr=err_file)
        # sp.call will wait for the calculation to finish

        # TODO: get energy
        # check if calculation is converged
        converged = False
        outcar = self.relax_path + '/OUTCAR'
        with open(outcar) as out:
            lines = out.readlines()
            for line in lines:
                if 'reached required accuracy' in line:
                    converged = True
                    break
        if not converged:
            print('Energy calculation of model {} not'
                  ' converged'.format(model.label))

        # if converged, get energy
        if converged:
            model.converged = converged

            # get total energy from output files
            oszicar = self.relax_path + '/OSZICAR'
            with open(oszicar) as oz:
                lines = oz.readlines()
            if lines[-1].split()[3] == 'E0=':
                total_energy = float(lines[-1].split()[4])
                model.tot_en = total_energy

            # get relaxed structure
            try:
                contcar = self.relax_path + '/CONTCAR'
                shutil.copy(contcar, self.relax_path + '/POSCAR_relaxed')
                relaxed_astr = Structure.from_file(contcar)
                relaxed_astr.sort()
                self.move_atoms_inside(relaxed_astr)
                model.astr = relaxed_astr
            except:
                print('Relaxed structure not available in CONTCAR')

            # evaluate objective function and save as model attribute
            comp_dict = relaxed_astr.composition.as_dict()
            astr_elems = [i.name for i in
                          relaxed_astr.composition.elements]

            # DU
            # Evaluate free energy by calculating chemical potential contribution
            free_en = total_energy
            for elem in astr_elems:
                if elem in self.sym_mu_dict.keys():
                    free_en -= comp_dict[elem]*self.sym_mu_dict[elem]
                else:
                    print("Error. VASP species " + elem +
                          " not contained in input yaml file.")

            model.obj0_val = float(free_en)
            # other objective functions should be evaluated here.

    def move_atoms_inside(self, astr):
        """
        For a given structure object, move all sites within the unit cell.
        Eg: [-0.1, 0.4, 1.2] --> [0.9, 0.4, 0.2]

        returns 'astr' with all atoms inside

        Args:

        astr: pymatgen Structure object
        """
        species = astr.species
        fc = astr.frac_coords
        fc = np.where((fc < 0) | (fc > 1), fc - np.floor(fc), fc)

        # replace all the coords in astr
        all_inds = [i for i in range(len(species))]
        astr.remove_sites(all_inds)
        for sps, coords in zip(species, fc):
            astr.append(sps, coords, coords_are_cartesian=False)

    def write_gb_poscar(self, model, file_name):
        """
        For a newly created model, set sd_flags to each site according to its
        z-coordinate. All interface region atoms would have [T,T,T] and others
        would have [F,F,F]. Then, write the POSCAR file in relax_path.
        """
        frac_zmin, frac_zmax = self.hollow_botz, self.hollow_topz
        frac_zs = model.astr.frac_coords[:, 2]
        bs = []
        for z in frac_zs:
            b = 0
            if frac_zmin < z < frac_zmax:
                b = 1
            bs.append(b)
        sd_flags = [[bool(i), bool(i), bool(i)] for i in bs]
        gb_poscar = Poscar(model.astr, selective_dynamics=sd_flags)
        gb_poscar.write_file(file_name)

    def write_surface_poscar(self, model, file_name, sd_cut_off=None,
                                                 sd_no_z=False):
        """
        For a newly created model, set sd_flags to each site according to its
        z-coordinate. All interface region atoms would have [T,T,T] and others
        would have [F,F,F]. Then, write the POSCAR file in relax_path.
        """
        if not sd_cut_off: # automatically freeze substrate
            sd_cut_off = self.substrate_thickness

        slab_sites = model.astr.sites
        bot_z_cart = model.astr.cart_coords[:, 2].min()
        surface_inds = [i for i, site in enumerate(slab_sites) if \
                            site.coords[2] - bot_z_cart > sd_cut_off]

        sd_flags = []
        for i in range(len(slab_sites)):
            sd_flag = [0, 0, 0]
            if i in surface_inds:
                if sd_no_z is True:
                    sd_flag = [1, 1, 0]
                else:
                    sd_flag = [1, 1, 1]
            sd_flags.append(sd_flag)
        sd_flags = [[bool(flag[0]), bool(flag[1]), bool(flag[2])] \
                                    for flag in sd_flags]

        gb_poscar = Poscar(model.astr, selective_dynamics=sd_flags)
        gb_poscar.write_file(file_name)
