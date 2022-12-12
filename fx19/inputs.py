from __future__ import division, unicode_literals, print_function
from fx19 import structure_record
from fx19 import initial_population
from fx19 import energy
from fx19 import experimental_simulation
from fx19 import selection, epsilonSelection, clusteredSelection
from fx19 import structure_operations
from fx19.clustering import HierarchicalClusterer, CompositionalClusterer
from fx19.fingerprinting import Comparator

import os


def make_objects(i_dict):
    """
    Function to make objects of different classes using the input parameters
    provided in the input file. Assumes defaults for optional parameters that
    are not provided.

    Arguments:

        i_dict (dict): dictionary of all the user-provided input parameters
         read from yaml file

    Returns:

        dict: dictionary storing all created objects
    """
    all_objects = {}
    # register_id object to id the models
    reg_id = structure_record.register_id()
    all_objects['reg_id'] = reg_id

    # Make MongoDB database object
    if 'database' in i_dict:
        from pymongo import MongoClient
        all_objects['database'] = connect_to_mongodb(**i_dict["database"])
    else:
        all_objects['database'] = None

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
    if str_constraints['shape'] == 'bulk' or\
            str_constraints['shape'] == 'cluster':
        # make_random_model object from initial_population
        random_model_obj = initial_population.make_random_model(
            str_constraints)
        all_objects['random_model_obj'] = random_model_obj

    # make energy_code object
    energy_params = get_energy_params(i_dict)
    energy_params['shape'] = str_constraints['shape']
    energy_params['element_syms'] = str_constraints['element_syms']
    energy_pkg = i_dict['energy_code']  # 'vasp' or 'lammps' or 'gulp'
    if energy_pkg == 'gulp':
        energy_code = energy.gulp_code(energy_params)
    elif energy_pkg == 'vasp':
        energy_code = energy.vasp_code(energy_params)
    elif energy_pkg == 'lammps':
        energy_code = energy.lammps_code(energy_params)
    else:
        print('Please set energy_code in inputs as one of vasp'
              'or lammps or gulp')
    all_objects['energy_code'] = energy_code
    print(f"Energy code: {energy_pkg}")

    # make experimental_simulation object(s)
    exp_sim_methods = ['PDF', 'GB_STEM', 'PRISM', 'GSASII', 'XANES', 'XRD']
    if 'exp_sim_1' in i_dict:
        if i_dict['exp_sim_1'] in exp_sim_methods:
            method_1 = i_dict['exp_sim_1']
            if 'exp_sim_1_params' not in i_dict:
                print('exp_sim_1_params not provided. They are mandatory.')
            if method_1 == 'PDF':
                Xsim1_params = get_pdf_params(i_dict, 'exp_sim_1_params')
                Xsim_1 = experimental_simulation.pdf_of_model(Xsim1_params)
            if method_1 == 'GB_STEM':
                Xsim1_params = get_ingrained_params(i_dict, 'exp_sim_1_params')
                Xsim1_params['init_gb_path'] = str_record['gb']['init_gb_astr']
                Xsim_1 = experimental_simulation.gb_ingrained(Xsim1_params)
            if method_1 == "XANES":
                Xsim1_params = get_xanes_params(i_dict, 'exp_sim_1_params')
                Xsim_1 = experimental_simulation.xanes_of_model(Xsim1_params)
            if method_1 == "XRD":
                Xsim1_params = get_xrd_params(i_dict, 'exp_sim_1_params')
                Xsim_1 = experimental_simulation.xrd_of_model(Xsim1_params)
            all_objects['Xsim_1'] = Xsim_1

    # Get the MOEA and search mode based on provided inputs
    selection_mod = selection  # uses distance from pareto MOEA
    mod_str = 'selection.py'
    ob_fn = 'Distance from pareto'
    cl_bool = False

    # selection type of objective function
    if 'select_params' not in i_dict:
        print('Error: Please provide select_params keyword and objective'
              ' keyword specifying single or multiobjective optimization.')
    else:
        select_params = i_dict['select_params']
        if "selection_algorithm" in select_params:
            algorithm = select_params["selection_algorithm"]
            if algorithm not in \
                ["distance_from_pareto",
                 "epsilon_moea",
                 "clustered_selection"]:
                print('Error: Please provide valid selection algorithm.')
            else:
                if algorithm == "epsilon_moea":
                    if 'epsilons' in i_dict['select_params']:
                        selection_mod = epsilonSelection
                        mod_str = 'epsilonSelection.py'
                        ob_fn = 'Epsilon-MOEA'
                    else:
                        print('Error. Chose epsilon_moea, but did not '
                              'provide epsilon values. Using '
                              'distance_from_pareto.')
                if algorithm == "clustered_selection":
                    if 'cluster_params' in i_dict:
                        selection_mod = clusteredSelection
                        mod_str = 'clusteredSelection.py'
                        ob_fn = 'Clustered Selection'
                        cl_bool = True
                    else:
                        print('Error. Chose clustered_selection, but either '
                              'did not provide clustering parameters. '
                              'Using distance_from_pareto')
        else:
            print("Selection algorithm not provided."
                  "Using distance_from_pareto.")

    print('Objective function: {}\nClustering: {}'.format(ob_fn, cl_bool))
    print('Using Pool & Select classes from {} module'.format(mod_str))

    if 'objective_fn_type' not in select_params.keys():
        print('Error: Please provide select_params keyword and objective'
              ' keyword specifying single or multiobjective optimization.')

    if select_params['objective_fn_type'] not in ['multi', 'single']:
        print('Error: Select objective should be a string of either'
              ' single or multi.')
    if select_params['objective_fn_type'] == 'multi':
        select = selection_mod.Select(select_params)
        # weights, num_required_above_50 & num_models_before_pareto are in
        # select_params if provided
    else:
        select_params['objective_fn_type'] = 'single'
        select_params['weights'] = [1, 1, 1, 1, 1]
        select = selection_mod.Select(select_params)
    all_objects['select'] = select

    # Pool object (contains good_pool and bad_pool)
    pool_params = {}
    pool_params['capacity'] = i_dict['population_limits']['pool']
    pool_params['energy_pkg'] = energy_pkg
    if 'epsilons' in i_dict['select_params']:
        pool_params['epsilons'] = i_dict['select_params']['epsilons']
        if cl_bool:
            if 'dominance_algorithm' in i_dict['select_params']:
                da = i_dict['select_params']['dominance_algorithm']
                if da not in ['pareto_dominance', 'epsilon_dominance']:
                    print("Error! ClusteredSelection dominance_algorithm not"
                          "a valid choice. Please choose either"
                          "pareto_dominance or epsilon_dominance. By default,"
                          " pareto_dominance has been chosen.")
                else:
                    pool_params['dominance_algorithm'] =\
                        i_dict['select_params']['dominance_algorithm']

    if 'fingerprint_params' in i_dict:
        fp_params = i_dict['fingerprint_params']
        fp_label = fp_params['label']
        tolerance = {fp_label: fp_params['tolerance']}
        comparator_obj = Comparator(label=fp_label, tolerances=tolerance)
        print("Created comparator_obj.")
        if fp_label == "valle-oganov":
            if 'comp_values' in fp_params:
                comparator_obj.set_valle_oganov_comparator(
                    fp_params['comp_values'])
            else:
                comparator_obj.set_valle_oganov_comparator()
        if fp_label == "ewald-sum-matrix" or \
                fp_label == "sine-matrix" or \
                fp_label == "mbtr":
            if fp_label == "mbtr":
                species = []
                for _, value in str_constraints["species_dict"].items():
                    species.append(value["name"])
                if 'mbtr_values' in fp_params:
                    comparator_obj.set_mbtr_descriptor(
                        _species=species,
                        mbtr_values=fp_params["mbtr_values"]
                    )
                else:
                    comparator_obj.set_mbtr_descriptor(_species=species)
            if fp_label == "sine-matrix":
                if 'sine-matrix_values' in fp_params:
                    comparator_obj.set_sine_matrix_descriptor(
                        sm_values=fp_params["sine-matrix_values"])
                else:
                    comparator_obj.set_sine_matrix_descriptor()
            if fp_label == "ewald-sum-matrix":
                if 'ewald-sum-matrix_values' in fp_params:
                    comparator_obj.set_ewald_sum_matrix_descriptor(
                        esm_values=fp_params["ewald-sum-matrix_values"])
                else:
                    comparator_obj.set_ewald_sum_matrix_descriptor()
            if 'distance_metric' in fp_params:
                comparator_obj.set_distance_calculator(
                    fp_params['distance_metric'])
            else:
                comparator_obj.set_distance_calculator()
        elif fp_label == "rematch-soap" or fp_label == "average-soap":
            species = []
            for _, value in str_constraints["species_dict"].items():
                species.append(value["name"])
            if 'soap_values' in fp_params:
                comparator_obj.set_soap_descriptor(
                    _species=species,
                    soap_values=fp_params['soap_values']
                )
            else:
                comparator_obj.set_soap_descriptor(
                    _species=fp_params['species'])

            if 'kernel_generator_values' in fp_params:
                comparator_obj.set_kernel_generator(
                    fp_params['kernel_generator_values'])
            else:
                comparator_obj.set_kernel_generator()
        if 'zbounds' in fp_params:
            comparator_obj.zbounds = fp_params['zbounds']
        if 'rem_vac' in fp_params:
            comparator_obj.rem_vac = fp_params['rem_vac']
        fingerprint_params = i_dict["fingerprint_params"]
        # If the fingerprint is a soap descriptor, then the
        # species names need to be passed in.
        if fingerprint_params["label"] == "rematch-soap":
            species = []
            for _, value in str_constraints["species_dict"].items():
                species.append(value["name"])
            fingerprint_params["species"] = species
        pool_params['comparator_obj'] = comparator_obj
        all_objects['comparator_obj'] = comparator_obj
    else:
        print("Could not find fingerprint params")
    # Also create cluster object if cluster_params in i_dict
    if 'cluster_params' in i_dict:
        print("Creating pool cluster object.")
        if i_dict['cluster_params']['type'] == "hierarchical":
            if 'distance_calculation' in i_dict['cluster_params']:
                if i_dict['cluster_params']['distance_calculation'] == \
                        'xsim' or 'fingerprint_params' not in i_dict:
                    if 'exp_sim_1' in i_dict:
                        cluster_obj = HierarchicalClusterer(
                            i_dict['cluster_params'], xsim=Xsim_1)
                        if 'fingerprint_params' not in i_dict:
                            print("Tried to use fingerprinting for "
                                  "distance_calculator. However, "
                                  "fingerprint_params "
                                  "not found, using xsim as "
                                  "distance_calculator "
                                  "for clustering instead.")
                    else:
                        if 'fingerprint_params' not in i_dict:
                            print("Neither Xsim or fingerprinting provided."
                                  " Cannot perform clustering.")
                        else:
                            cluster_obj = HierarchicalClusterer(
                                i_dict['cluster_params'],
                                comparator_obj=all_objects['comparator_obj'])
                            print("Tried to use Xsim as distance"
                                  "calculation, but Xsim not provided. "
                                  "Using fingerprinting instead.")
                else:
                    cluster_obj = HierarchicalClusterer(
                        i_dict['cluster_params'],
                        comparator_obj=all_objects['comparator_obj'])
            else:
                print("Error. distance_calculation keyword not "
                      "found in cluster_params.")
                if 'fingerprint_params' not in i_dict:
                    i_dict["cluster_params"]['distance_calculation'] =\
                        'xsim'
                    cluster_obj = HierarchicalClusterer(
                        i_dict['cluster_params'], xsim=Xsim_1)
                    print("Fingerprint_params not found, using "
                          "xsim as distance_calculator for clustering.")
                else:
                    i_dict["cluster_params"]['distance_calculation'] =\
                        'fingerprint'
                    cluster_obj = HierarchicalClusterer(
                        i_dict['cluster_params'],
                        comparator_obj=all_objects['comparator_obj'])
                    print("Fingerprint_params found, using "
                          "fingerprinting as distance_calculator "
                          "for clustering.")
        elif i_dict['cluster_params']['type'] == 'compositional':
            cluster_obj = CompositionalClusterer()
        else:
            print("Please provide a valid type of cluster object.")
            cluster_obj = None
        pool_params['cluster_obj'] = cluster_obj
        all_objects['cluster_obj'] = cluster_obj
        print("Created pool cluster_obj of type {}".format(cluster_obj.type))
    pool = selection_mod.Pool(pool_params)
    all_objects['pool'] = pool

    # Mating object from structure_operations
    mating_params = get_mating_params(i_dict, str_constraints)
    mate = structure_operations.mating(mating_params)
    # all_objects['mate'] = mate

    # Basinhopping object from structure_operations
    basinhopping_params = {}
    if 'basinhopping_constraints' in i_dict:
        basinhopping_params = i_dict['basinhopping_constraints']
    # NOTE: contains  'indices_fraction' and 'max_perturbation'
    basinhopping_params['min_dist_dict'] = str_constraints['min_dist_dict']
    basinhopping_params['max_dist_dict'] = str_constraints['max_dist_dict']
    basinhopping_params['element_syms'] = str_constraints['element_syms']
    basinhopping_params['species_dict'] = i_dict['structure_record']['species']
    basinhopping_params['shape'] = str_constraints['shape']
    if str_constraints['shape'] == 'cluster' or\
            str_constraints['shape'] == 'molecule':
        basinhopping_params['max_dia'] = str_constraints['max_dia']
        basinhopping_params['box_abc'] = str_constraints['box_abc']
        basinhopping_params['origin'] = str_constraints['origin']
    if str_constraints['shape'] == "molecule":
        basinhopping_params['fixed_species'] = str_constraints['fixed_species']
    hop = structure_operations.basinhopping(basinhopping_params)
    # all_objects['hop'] = hop

    if str_constraints['shape'] == 'molecule':
        random_model_obj = structure_operations.mol_ops(hop, str_constraints)
        all_objects['random_model_obj'] = random_model_obj
        all_objects['evolve'] = random_model_obj

    # For gb, overlap and remove sites is used for random models
    gb_ops_obj = None
    if str_constraints['shape'] == 'gb':
        gb_ops_obj = structure_operations.gb_ops(hop, str_constraints)
        all_objects['gb_ops_obj'] = gb_ops_obj

        energy_code.hollow_botz = gb_ops_obj.hollow_botz
        energy_code.hollow_topz = gb_ops_obj.hollow_topz
        all_objects['energy_code'] = energy_code

    # Evolve object - wrapper on mating and basinhopping
    evolve_params = get_evolve_params(str_constraints)
    if str_constraints['shape'] == 'cluster' or\
            str_constraints['shape'] == 'bulk':
        evolve = structure_operations.Evolve(mate, hop, evolve_params)
        all_objects['evolve'] = evolve

    # For surface layer searches, create surface_ops object
    surface_ops_obj = None
    if str_constraints['shape'] == 'surface':
        init_slabs_path = str_record['surface']['init_slabs_dir']
        init_slabs_dict = {i: init_slabs_path + '/' + slab_file
                           for i, slab_file in
                           enumerate(os.listdir(init_slabs_path))}
        str_constraints['init_slabs_dict'] = init_slabs_dict

        surface_ops_obj = structure_operations.surface_ops(
            hop, str_constraints)
        all_objects['surface_ops_obj'] = surface_ops_obj

        energy_code.substrate_thickness = surface_ops_obj.substrate_thickness
        energy_code.sd_cut_off = surface_ops_obj.sd_cut_off
        energy_code.sd_no_z = surface_ops_obj.sd_no_z
        all_objects['energy_code'] = energy_code

    # Develop any other below objects

    return all_objects


def get_energy_params(i_dict):
    """
    Determines all the parameters, mandatory and optional, to be
    used to make energy object for each calculation.

    Arguments:

        i_dict (dict): dictionary of all the user-provided input parameters
         read from yaml file

    Returns:

        dict: all the determined parameters
    """
    energy_params = {}
    # add main_path, i.e., where the search started to energy_params
    energy_params['main_path'] = i_dict['main_path']

    # energy_code
    if 'energy_code' not in i_dict:
        print('Please provide energy code details. This is mandatory')
    else:
        energy_params['energy_code'] = i_dict['energy_code']

    # energy code execution command (Mandatory)
    if 'energy_exec_cmd' not in i_dict:
        print('Please provide the execution command for the energy '
              'code. This is mandatory. Ex: \"lmp_mpi -in in.min\"')
    else:
        energy_params['energy_exec_cmd'] = i_dict['energy_exec_cmd']

    # DU
    # chemical potentials
    species_dict = i_dict['structure_record']['species']
    mu = {}
    for species in species_dict.keys():
        index = int(species[7:])
        if 'mu' in species_dict[species].keys():
            mu[index] = species_dict[species]['mu']
        else:
            print('Error encountered with species ' + str(index) + ': '
                  'Chemical potentials must be provided in the dictionary'
                  + ' for each species!')
            mu[index] = 0

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
        print('Please provide path to folder with input energy files for'
              ' relaxation.')
    else:
        energy_params['files_path'] = i_dict['inputs']['energy_files_path']

    return energy_params


def get_pdf_params(i_dict, exp_sim_params_id):
    """
    Reads the i_dict and returns pdf_params for experimental simulation method
    that is used in search (if provided). Does not mention defaults if not
    provided in input file. That happens in
    experimental_simulation module

    Arguments:

        i_dict (dict): dictionary of all the user-provided input parameters
         read from yaml file

        exp_sim_params_id (str): `'exp_sim_1_params'` if only one experimetnal
         simulation method.

    Returns:

        dict: all the determined parameters

    !!! TODO

        Add `'exp_sim_2_params'` if 2 sim methods are used
    """
    pdf_params = i_dict[exp_sim_params_id]
    # add main_path, i.e., where the search started to energy_params
    pdf_params['main_path'] = i_dict['main_path']

    return pdf_params


def get_xanes_params(i_dict, exp_sim_params_id):
    """
    Reads the i_dict and returns xanes_params for experimental simulation
    method that is used in search (if provided).
    Does not mention defaults if not provided in input file. That happens in
    experimental_simulation module.

    Arguments:

        i_dict (dict): dictionary of all the user-provided input parameters
         read from yaml file

        exp_sim_params_id (str): `'exp_sim_1_params'` if only one experimetnal
         simulation method.

    Returns:

        dict: all the determined parameters

    !!! TODO

        Add `'exp_sim_2_params'` if 2 sim methods are used
    """
    xanes_params = i_dict[exp_sim_params_id]
    xanes_params['main_path'] = i_dict['main_path']

    return xanes_params


def get_ingrained_params(i_dict, exp_sim_params_id):
    """
    Function to conveniently combine different parameters provided by user and
    other defualts (if not user-provided) to be used by mating class. Throws
    error when mandatory parameters are not provided by the user.

    Arguments:

        i_dict (dict): dictionary of all the user-provided input parameters
         read from yaml file

        exp_sim_params_id (str): `'exp_sim_1_params'` if only one experimetnal
         simulation method.

    Returns:

        dict: all the determined parameters

    !!! TODO

        Add `'exp_sim_2_params'` if 2 sim methods are used
    """
    gb_ingrained_params = i_dict[exp_sim_params_id]
    gb_ingrained_params['main_path'] = i_dict['main_path']

    # if 'init_gb_path' not in gb_ingrained_params:
    #    gb_ingrained_params['init_gb_path'] = None
    if 'progress_file' not in gb_ingrained_params:
        gb_ingrained_params['progress_file'] = None
    if 'ing_opt_params' not in gb_ingrained_params:
        gb_ingrained_params['ing_opt_params'] = None
    if 'dm3_path' not in gb_ingrained_params:
        gb_ingrained_params['dm3_path'] = None

    return gb_ingrained_params


def get_xrd_params(i_dict, exp_sim_params_id):
    """
    """
    xrd_params = i_dict[exp_sim_params_id]
    xrd_params['main_path'] = i_dict['main_path']
    return xrd_params


def get_mating_params(i_dict, str_constraints):
    """
    Function to conveniently combine different parameters provided by user and
    other defualts (if not user-provided) to be used by mating class.

    Arguments:

        i_dict (dict): dictionary of all the user-provided input parameters
         read from yaml file

        str_constraints (dict): dictionary of all the constraints for making
         random models

    Returns:

        dict: all the determined parameters
    """
    # NOTE: There aren't any mandatory params for mating.
    # If there are not any in input_file.yaml, assume all defaults and proceed
    mating_params = {}
    if 'mating_constraints' in i_dict:
        mating_params = i_dict['mating_constraints']
        # contains 'mirror_slice_before_join' boolean parameter if provided

    # Add other necessary constraints from before
    mating_params['min_dist_dict'] = str_constraints['min_dist_dict']
    mating_params['species_dict'] = i_dict['structure_record']['species']
    mating_params['num_species'] = str_constraints['num_species']
    mating_params['shape'] = str_constraints['shape']
    mating_params['element_syms'] = str_constraints['element_syms']
    if mating_params['shape'] == 'cluster'\
            or mating_params['shape'] == 'molecule':
        mating_params['box_abc'] = str_constraints['box_abc']
        mating_params['max_dia'] = str_constraints['max_dia']
        mating_params['origin'] = str_constraints['origin']
    if mating_params['shape'] == 'molecule':
        mating_params['fixed_species'] = str_constraints['fixed_species']

    # species dicts
    # DU
    for i in range(1, str_constraints['num_species']+1):
        species = 'species' + str(i)
        if species in str_constraints:
            mating_params[species] = str_constraints[species]

    return mating_params


def get_evolve_params(str_constraints):
    """
    Determines parameters to be used for the `'evolve'` class

    Args:

        str_constraints (dict): dictionary of all the constraints for making
         random models

    Returns:

        dict: all the determined parameters
    """
    evolve_params = {}
    evolve_params['shape'] = str_constraints['shape']
    evolve_params['num_species'] = str_constraints['num_species']
    # species dicts
    # DU
    for i in range(1, str_constraints['num_species']+1):
        species = 'species' + str(i)
        if species in str_constraints:
            evolve_params[species] = str_constraints[species]
    return evolve_params


def connect_to_mongodb(host='localhost', port=27017, username=None,
                       password=None, database='science'):
    client = MongoClient(f'mongodb://{username}:{password}@{host}:{port}')
    return client[database]
