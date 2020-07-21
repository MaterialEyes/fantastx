
from __future__ import division, unicode_literals, print_function

"""
Reads all the input files

Saves the experimental data in required format, making it easy to compare with
simulated experimental data later.
"""

"""
Types of experimental data files:
Ex: file extensions

Based on the file extension found, call specific function to parse the file and
save data for subsequent steps
"""

from fx19 import structure_record
from fx19 import initial_population
from fx19 import energy
from fx19 import experimental_simulation
from fx19 import selection
from fx19 import structure_operations
from fx19.structure_operations import gb_ops


def make_objects(i_dict):
    """
    Takes all the user provided input parameters as a dictionary and makes
    objects for the fantastx run. Assumes defaults for some mandatory
    parameters if needed.

    Args:
    i_dict - (dict) dictionary of all the user-provided input parameters read
             from yaml file
    """
    all_objects = {}
    # register_id object to id the models
    reg_id = structure_record.register_id()
    all_objects['reg_id'] = reg_id

    # make structure_constraints object
    str_record = i_dict['structure_record']
    constraints_obj = structure_record.structure_constraints(str_record)
    str_constraints = constraints_obj.get_constraints()
    # add species_dict to str_constraints as well (used in gb_ops)
    str_constraints['species_dict'] = i_dict['structure_record']['species']
    all_objects['constraints_obj'] = constraints_obj

    # make input models object
    input_model_obj = None
    if 'model_files_path' in i_dict['inputs']:
        model_files_path = i_dict['inputs']['model_files_path']
        input_model_obj = initial_population.make_model_from_input(
                                                            model_files_path)
    all_objects['input_model_obj'] = input_model_obj

    # For cluster, initial population module is used for random models
    if str_constraints['shape'] == 'cluster':
        # make_random_model object from initial_population
        random_model_obj = initial_population.make_random_model(str_constraints)
        all_objects['random_model_obj'] = random_model_obj

    # make energy_code object
    energy_params = get_energy_params(i_dict)
    energy_params['shape'] = str_constraints['shape']
    energy_params['element_syms'] = str_constraints['element_syms']
    energy_pkg = i_dict['energy_code'] # 'vasp' or 'lammps' or 'gulp'
    if energy_pkg == 'gulp':
        energy_code = energy.gulp_code(energy_params)
    elif energy_pkg == 'vasp':
        energy_code = energy.vasp_code(energy_params)
    elif energy_pkg == 'lammps':
        energy_code = energy.lammps_code(energy_params)
    else:
        print ('Please set energy_code in inputs as one of vasp'
                                'or lammps or gulp')
    all_objects['energy_code'] = energy_code

    # make experimental_simulation object(s)
    exp_sim_methods = ['PDF', 'GB_STEM', 'PRISM', 'GSASII', 'FEFF']
    if 'exp_sim_1' in i_dict:
        if i_dict['exp_sim_1'] in exp_sim_methods:
            method_1 = i_dict['exp_sim_1']
            if not 'exp_sim_1_params' in i_dict:
                print ('exp_sim_1_params not provided. They are mandatory.')
            if method_1 == 'PDF':
                Xsim1_params = get_pdf_params(i_dict, 'exp_sim_1_params')
                Xsim_1 = experimental_simulation.pdf_of_model(Xsim1_params)
            if method_1 == 'GB_STEM':
                Xsim1_params = get_ingrained_params(i_dict, 'exp_sim_1_params')
                Xsim1_params['init_gb_path'] = str_record['gb']['init_gb_astr']
                Xsim_1 = experimental_simulation.gb_ingrained(Xsim1_params)
            all_objects['Xsim_1'] = Xsim_1


    # Pool object (contains good_pool and bad_pool)
    pool_params = {}
    pool_params['capacity'] = i_dict['pool_capacity']
    pool_params['energy_pkg'] = energy_pkg
    pool = selection.Pool(pool_params)
    all_objects['pool'] = pool

    # selection type of objective function
    if not 'select_objective' in i_dict:
        print ('Error: Please provide single or multi objective function')
    elif i_dict['select_objective'] == 'multi':
        select_params = {}
        select_params['type'] = 'multi'
        if 'weights' in i_dict: # defaults assumed in selection module
            select_params['weights'] = i_dict['weights']
        if 'temp' in i_dict:
            select_params['temp'] = i_dict['temp']
        select = selection.Select(select_params)
    elif i_dict['select_objective'] == 'single':
        select_params = {}
        select_params['type'] = 'single'
        select_params['weights'] = [1, 1, 1, 1, 1]
        select = selection.Select(select_params)
    all_objects['select'] = select

    # Mating object from structure_operations
    mating_params = get_mating_params(i_dict, str_constraints)
    mate = structure_operations.mating(mating_params)
    # all_objects['mate'] = mate

    # Basinhopping object from structure_operations
    basinhopping_params = {}
    if 'basinhopping_constraints' in i_dict:
        basinhopping_params = i_dict['basinhopping_constraints']
    # NOTE: contains 'perturb_box', 'indices_fraction', 'scale_fraction',
    # 'jump_fraction' and 'scale_direction'
    basinhopping_params['min_dist_dict'] = str_constraints['min_dist_dict']
    basinhopping_params['species_dict'] = i_dict['structure_record']['species']
    hop = structure_operations.basinhopping(basinhopping_params)
    # all_objects['hop'] = hop

    # For gb, overlap and remove sites is used for random models
    gb_ops_obj = None
    if str_constraints['shape'] == 'gb':
        str_constraints['hop_mate_frac'] = mating_params['hop_mate_frac']
        gb_ops_obj = structure_operations.gb_ops(hop, str_constraints)
        all_objects['gb_ops_obj'] = gb_ops_obj

    # Evolve object - wrapper on mating and basinhopping
    evolve_params = get_evolve_params(i_dict, str_constraints)
    evolve = structure_operations.Evolve(mate, hop, evolve_params)
    all_objects['evolve'] = evolve


    ################### Develop the below objects

    #Stopper
    #stopper = structure_record.Stopper(str_record)

    return all_objects

def get_energy_params(i_dict):
    """
    Returns a dictionary with all the parameters, mandatory and optional, to be
    used to make energy object for each calculation.

    Args:
    i_dict - (dict) dictionary of all the user-provided input parameters read
             from yaml file
    """
    energy_params = {}
    # add main_path, i.e., where the search started to energy_params
    energy_params['main_path'] = i_dict['main_path']

    # energy_code
    if 'energy_code' not in i_dict:
        print ('Please provide energy code details. This is mandatory')
    else:
        energy_params['energy_code'] = i_dict['energy_code']

    # energy obj_fn
    if 'energy_obj_fn' not in i_dict:
        print ('Please provide objective function as string. This is mandatory')
    else:
        energy_params['energy_obj_fn'] = i_dict['energy_obj_fn']

    # energy code execution command (Mandatory)
    if 'energy_exec_cmd' not in i_dict:
        print ('Please provide the execution command for the energy '
                    'code. This is mandatory. Ex: \"lmp_mpi -in in.min\"')
    else:
        energy_params['energy_exec_cmd'] = i_dict['energy_exec_cmd']

    # chemical potentials
    mu = {1:0, 2:0, 3:0, 4:0, 5:0}
    if i_dict['energy_obj_fn'] == 'mu_based':
        species_dict = i_dict['structure_record']['species']
        try:
            mu[1] = species_dict['specie1']['mu']
            if 'specie2' in species_dict:
                mu[2] = species_dict['specie2']['mu']
            if 'specie3' in species_dict:
                mu[3] = species_dict['specie3']['mu']
            if 'specie4' in species_dict:
                mu[4] = species_dict['specie4']['mu']
            if 'specie5' in species_dict:
                mu[5] = species_dict['specie5']['mu']
        except:
            print ('Error: For \'mu_based\' objective function, '
                   'chemical potentials must be provided for every species!')
    energy_params['mu'] = mu

    # Number of times to resubmit if not converged (for vasp)
    energy_params['resubmit'] = 0
    if energy_params['energy_code'] == 'vasp':
        if 'resubmit' in i_dict:
            energy_params['resubmit'] = i_dict['resubmit']

    # any specific energy code parameters like atom_style etc
    if 'energy_code_params' in i_dict:
        energy_code_params = i_dict['energy_code_params']
        energy_code_keys = list(energy_code_params.keys())
        for key in energy_code_keys:
            energy_params[key] = energy_code_params[key]

    # files_path
    if 'energy_files_path' not in i_dict['inputs']:
        print ('Please provide path to folder with input energy files for'
               ' relaxation.')
    else:
        energy_params['files_path'] = i_dict['inputs']['energy_files_path']

    return energy_params

def get_pdf_params(i_dict, exp_sim_params_id):
    """
    Reads the i_dict and returns pdf_params for experimental simulation method
    that is used in search (if provided)
    Does not mention defaults if not provided in input file. That happens in
    experimental_simulation module

    Args:
    i_dict - (dict) dictionary of all the user-provided input parameters read
             from yaml file
    exp_sim_params_id - (str) 'exp_sim_1_params' if only one experimetnal
                        simulation method.

    #TODO: add 'exp_sim_2_params' if 2 sim methods are used
    """
    pdf_params = i_dict[exp_sim_params_id]
    # add main_path, i.e., where the search started to energy_params
    pdf_params['main_path'] = i_dict['main_path']

    return pdf_params

def get_ingrained_params(i_dict, exp_sim_params_id):
    """
    Similar to pdf params. Except, defaults are not provided for all params.
    If some important params are not provided, prints error message and exits.

    Args:
    i_dict - (dict) dictionary of all the user-provided input parameters read
             from yaml file
    exp_sim_params_id - (str) 'exp_sim_1_params' if only one experimetnal
                        simulation method.

    #TODO: add 'exp_sim_2_params' if 2 sim methods are used
    """
    gb_ingrained_params = i_dict[exp_sim_params_id]
    gb_ingrained_params['main_path'] = i_dict['main_path']

    #if 'init_gb_path' not in gb_ingrained_params:
    #    gb_ingrained_params['init_gb_path'] = None
    if 'progress_file' not in gb_ingrained_params:
        gb_ingrained_params['progress_file'] = None
    if 'ing_opt_params' not in gb_ingrained_params:
        gb_ingrained_params['ing_opt_params'] = None
    if not 'dm3_path' in gb_ingrained_params:
        gb_ingrained_params['dm3_path'] = None

    return gb_ingrained_params

def get_mating_params(i_dict, str_constraints):
    """
    Collects all user provided parameters for mating, uses defaults if necessary

    Includes other parameters as required, like num_species from str_constraints

    Args:
    i_dict - (dict) dictionary of all the user-provided input parameters read
             from yaml file
    str_constraints - (dict) dictionary of all the constraints for making
                      random models
    """
    # NOTE: There aren't any mandatory params for mating.
    # If there are not any in input_file.yaml, assume all defaults and proceed
    mating_params = {}
    if 'mating_constraints' in i_dict:
        mating_params = i_dict['mating_constraints']

    # NOTE: mating_constraints contain 'num_parents_fraction' and
    # 'attach_type_fraction'

    # Add other necessary constraints from before
    mating_params['min_dist_dict'] = str_constraints['min_dist_dict']
    mating_params['species_dict'] = i_dict['structure_record']['species']
    mating_params['num_species'] = str_constraints['num_species']

    # species dicts
    keys = ['specie1', 'specie2', 'specie3', 'specie4', 'specie5']
    for specie in keys:
        if specie in str_constraints:
            mating_params[specie] = str_constraints[specie]

    return mating_params

def get_evolve_params(i_dict, str_constraints):
    """
    Returns parameters for the 'evolve' object

    Args:
    i_dict - (dict) dictionary of all the user-provided input parameters read
             from yaml file
    str_constraints - (dict) dictionary of all the constraints for making
                      random models
    """

    evolve_params = {}
    if 'evolve_probabilities' in i_dict:
        probs_dict = i_dict['evolve_probabilities']
        for key in probs_dict.keys():
            if key not in [1, 2, 3, 4]:
                print ('Specified probability key to method does not exist. '
                        'Keys should be only 1, 2, 3 or 4.')
        if sum(probs_dict.values()) != 1:
            print ('Error: Sum of probabilities should be equal to 1')
    evolve_params['probabilities'] = probs_dict
    evolve_params['num_species'] = str_constraints['num_species']
    # species dicts
    keys = ['specie1', 'specie2', 'specie3', 'specie4', 'specie5']
    for specie in keys:
        if specie in str_constraints:
            evolve_params[specie] = str_constraints[specie]
    return evolve_params

# assume experimental pdf is given
def read_input_exp_files(filename):
    """
    Depending on type of experimental data, call corresponding functions

    filename: path to exp data file
    filename should be of form '{experiment_type}_{composition}_exp.{extension}'
    experiment_type: {'PDF', 'XRD', 'TEM' ..}
    composition: {'Al2O3' or 'O3Al2' ..}
    Ex: 'PDF_IrO2_exp.txt'

    Make folder(s) and save data as required by experimental_simulation(s)
    return list of path_to_exp_data_folder(s)
    """

    pass
