
"""
This module contains functions which are used in run_fx.py
"""

import traceback


def initialize_remote_database(CRI):
    launchpad = LaunchPad(
        host=CRI,
        port=None,
        name=None,
        username=None,
        password=None,
        authsource='admin',
        uri_mode=True,
        user_indices=[],
        wf_user_indices=[],
    )
    launchpad.reset("", require_password=False)
    return launchpad


def get_working_jobs(futures):
    """
    Checks if any jobs in futures is still running and returns number of
    running jobs

    Arguments:
        futures (list): list of future objects (`concurrent.futures`)
    """
    if len(futures) == 0:
        return 0
    else:
        running = 0
        for future in futures:
            if not future.done():
                running += 1

        return running


def write_data(model, data_file):
    """
    Writes the model data to data_file

    Arguments:
        model (obj): `structure_record.model()` object
        data_file (str): path to the `data_file`
    """
    try:
        with open(data_file, 'a') as f:
            if model.obj1_val:
                line = '{0}\t{1:<14}\t{2:.6f}\t{3:.6f}\t{4:.6e}\t{5}\n'.format(
                    model.label, str(model.inheritance), model.tot_en,
                    model.obj0_val, model.obj1_val, model.made_by)
            else:
                line = '{0}\t{1:<14}\t{2:.6f}\t{3:.6f}\t{4}\n'.format(
                    model.label, str(model.inheritance), model.tot_en,
                    model.obj0_val, model.made_by)
            f.write(line)
    except:
        print(f"Couldn't find data_file {data_file}")

# Temporary selection probs based on overall_value


def temp_selection_probs(pool):
    """
    (Deprecated)
    Always the minimum overall value gets selevtion_prob of 1.
    """
    ov = [model.overall_val for model in pool.good_pool]
    norm_vals = []
    if not len(ov) < 3:
        norm_vals = [(i - max(ov))/(min(ov) - max(ov)) for i in ov]
        norm_probs = norm_vals/sum(norm_vals)
        for model in pool.good_pool:
            model.selection_prob = model.selection_prob / norm_probs


def relax(model, reg_id, energy_code):
    """
    Performs energy relaxation of the given model.

    Arguments:
        model (obj): `structure_record.model()` object
        reg_id (obj): `structure_record.register_id()` object
        energy_code (obj): energy code object (lammps_code or vasp_code)
    """
    try:
        energy_code.relax(model, reg_id)
    except:
        traceback.print_exc()
        print('Duplicate label in parallel processes. Skipping..')
        return None
    resubmitted = 2
    if not model.converged:
        for i in range(energy_code.resubmit):
            if resubmitted < energy_code.resubmit and not model.converged:
                resubmitted += 1
                try:
                    energy_code.re_relax(model)
                except:
                    print("Model cannot be relaxed.")
                    continue
    return model


def make_model(random_model_obj, evolve, select, pool, reg_id,
               model_type='random', model=None):
    """
    Makes a model from input, at random, or inherited from parents.

    Arguments:
        random_model_obj (obj): object which makes random structural
         models for the given structural geometry.
        evolve (obj): `structure_operations.evolve()` object
        select (obj): `select` object from one of the selection algorithms
        pool (obj): `pool` object from one of the selection algorithms
        reg_id (obj): `structure_record.register_id()` object
        model_type (str): `random` or `evolved`.
         - `random` - make random model for initial population
         - `evolved` - make child model by evolution
        model (obj): `structure_record.model()` object. If provided,
          it is directly taken to energy evaluation step.

    Returns:
        (obj): new (and fully evaluated) `structure_record.model()` object
    """
    # read models from input files (if any)
    if model_type == 'inputs':
        if model is not None:
            new_model = model
        else:
            return 0
    # make new random model
    if model_type == 'random':
        new_model = random_model_obj.random_model(reg_id)

    # make new model from parents
    if model_type == 'evolved':
        model_is_unique = False
        while not model_is_unique:
            new_model = evolve.get_model(select, pool, reg_id)
            if new_model is None:
                continue
            # check redundancy of the new model with all previous models
            if pool.comparator is not None:
                model_is_unique = pool.comparator.check_model_uniqueness(
                    new_model,
                    pool.all_models,
                    exact=True)
            else:
                model_is_unique = True
        # add the new_model inheritance to select.all_parent_labels
        # select.all_parent_labels += new_model.inheritance

    return new_model


def separate_gb(energy_code, gb_ops_obj, model):
    """
    For gb search, separate the gb_iface from the relaxed gb
    Does nothing if not gb search
    Arguments:
        energy_code - energy code object (lammps_code or vasp_code)
        gb_ops_obj - gb_ops_obj from structure_operations.py
        model (obj): structure_record.model() object
    """
    # For grain boundary search, assign grain_interface as model attribute
    if energy_code.shape == 'gb':
        model.gb_iface = gb_ops_obj.separate_gb(model.astr)
    else:
        pass


def do_Xsim(model, Xsim_1):
    """
    Do Xsim if needed and assign the corresponding objective function value.
    If no experimental simulation needed or Xsim_1 is None, does nothing.
    Arguments:
        model (obj): structure_record.model() object
        Xsim_1 (obj): experimental simulation object (such as pdf_of_model or
         gb_ingrained)
    """
    if Xsim_1:
        if not model.converged:
            print('Energy calculation of model {} is not'
                  ' converged'.format(model.label))
            return None
        # get the relaxed structure
        relaxed_str = model.astr
        if relaxed_str is None:
            print('Relaxed structure not available. Skipping Xsim..')

            return None
        else:
            # if relaxed structure exists
            model.Xsim1 = Xsim_1.name
            model, Xsim_val = Xsim_1.evaluate_obj(model)
            model.num_of_obj += 1
            return model
    else:
        return model


def update_pool(evald_futures, models_evald, pool, select,
                data_file, sim_ids):
    """
    Calculates the objective values for all models and updates pool with
    best models

    Arguments:
        evald_futures (list): submitted energy evaluation futures objects
        models_evald (int): count of number of fully evaluated models
        pool (obj): `pool` object from `selection.py`
        select (obj): `select` object from `selection.py`
        data_file (str): path to `data_file` to write model data
        sim_ids (bool): True if experimental simulation is used

    Returns:
        (list, int, obj, obj):
        - Updated evald_futures
        - Updated models_evald
        - Updated `pool` object
        - Updated `select` object
    """
    # remove all futures with an exception
    rem_inds, process_inds = [], []
    for i, future in enumerate(evald_futures):
        if future.done():
            rem_inds.append(i)
            if not future.exception():
                process_inds.append(i)

    # get all futures which should be processed
    futures_to_process = [evald_futures[i] for i in process_inds]
    # remove all done futures from evald_futures
    evald_futures = [evald_futures[i] for i in range(len(evald_futures))
                     if i not in rem_inds]
    for future in futures_to_process:
        model = future.result()
        # Add to either good_pool or bad_pool
        # Selection_probs are also updated
        if model is not None:
            select = pool.add_to_pool(model, select, sim_ids=sim_ids)
            if model.made_by is not None and model.made_by != "random":
                select.all_parent_labels += model.inheritance
            # write data to file
            write_data(model, data_file)
            models_evald += 1

    return evald_futures, models_evald, pool, select


def update_nonparallel_pool(models_evald, model_evaled, pool, select,
                            data_file, sim_ids):
    """
    Calculates the obejctive values for all models and updates pool with
    best models

    Arguments:
        evald_futures (list): list of submitted energy evaluation futures
         objects
        models_evald (int): count of number of fully evaluated models
        pool (obj): `pool` object from `selection.py`
        select (obj): `select` object from `selection.py`
        data_file (str): path to `data_file` to write model data
        sim_ids (bool): True if experimental simulation is used

    Returns:
        (int, obj, obj):
        - Updated models_evald
        - Updated `pool` object
        - Updated `select` object
    """
    model = model_evaled
    # Add to either good_pool or bad_pool
    # Selection_probs are also updated
    select = pool.add_to_pool(model, select, sim_ids=sim_ids)
    # write data to file
    write_data(model, data_file)
    models_evald += 1

    return models_evald, pool, select


def cluster_models(pool, data_file, xsim, cluster_obj, visualize=False):
    '''
    Clusters models using hierarchical clustering of experimental 
    objective scores. Can output images of the cluster dendrogram,
    and the clustering in objective function space.

    Arguments:
        pool (obj): `pool` object from `selection.py`
        data_file (str):  file containing the objective function values for all
         evaluated models.
        xsim (obj): the experimental_simulation object which will calculate the
         similarity scores for each model pair.
        cluster_obj (obj): the clustering object which will perform all clustering
         operations.
    '''
    models = pool.all_models
    obj_fncs = cluster_obj.read_in_objective_functions(data_file)
    distance_matrix, sorted_labels, sorted_models =\
        cluster_obj.create_distance_matrix(
            models, xsim)
    cluster_obj.calculate_clustering(
        sorted_models, sorted_labels, obj_fncs, distance_matrix)

    if visualize:
        cluster_obj.visualize_clusters()
