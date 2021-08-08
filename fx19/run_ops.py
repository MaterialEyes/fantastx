###################### Functions for run_fx : Begin ####################################
import time


def get_working_jobs(futures):
    """
    futures: dictionary of job output future objects labelled w.r.t model
    labels as keys

    Checks if any jobs in futures is still running and returns number of
    running jobs

    Args:
    futures - (list) list of futures objects (concurrent_futures)
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
    write the model data to data_file

    Args:
    model - model object
    data_file - path to the data_file
    """
    with open(data_file, 'a') as f:
        if model.obj1_val:
            line = '{0}\t{1}\t\t{2:.6f}\t{3:.6f}\t{4:.6}\n'.format(model.label,
                                                                   model.inheritance, model.tot_en, model.obj0_val, model.obj1_val)
        else:
            line = '{0}\t{1}\t\t{2:.6f}\t{3:.6f}\n'.format(model.label,
                                                           model.inheritance, model.tot_en, model.obj0_val)
        f.write(line)

# Temporary selection probs based on overall_value


def temp_selection_probs(pool):
    """
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
    Do energy relaxation of the given model

    Args:
    model - model object
    reg_id  - register_id object
    energy_code - energy_code object (lammps_code or vasp_code)
    """
    try:
        energy_code.relax(model, reg_id)
    except FileExistsError:
        print('Duplicate label in parallel processes. Skipping..')
        return None
    resubmitted = 2
    if model.converged == False:
        for i in range(len(energy_code.resubmit)):
            if resubmitted < energy_code.resubmit and model.converged == False:
                resubmitted += 1
                try:
                    energy_code.re_relax(model)
                except:
                    continue
    return model


def make_model(random_model_obj, evolve, select, pool, reg_id,
               model_type='random', model=None):
    """
    [random_model_obj, reg_id, evolve, select, pool,]
    Make a random model or a child model

    Args:
    random_model_obj - make_random_model object or gb_ops_obj
    evolve - evolve object
    select - select object from selection.py
    pool - pool object from selection.py
    reg_id - register_id object
    model_type (str): 'random' or 'evolved'
                      'random' - make random model for initial population
                      'evolved' - make child model by evolution
    model (model obj): if a model object is provided as inputs model_type
                        it is directly taken to energy evaluation step.
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
        new_model = evolve.get_model(select, pool, reg_id)
        # add the new_model inheritance to select.all_parent_labels
        select.all_parent_labels += new_model.inheritance

    return new_model, select


def separate_gb(energy_code, gb_ops_obj, model):
    """
    For gb search, separate the gb_iface from the relaxed gb
    Pass if not gb search

    Args:
    energy_code - energy_code object (lammps_code or vasp_code)
    gb_ops_obj - gb_ops_obj from structure_operations.py
    model - model obj
    """
    # For grain boundary search, assign grain_interface as model attribute
    if energy_code.shape == 'gb':
        model.gb_iface = gb_ops_obj.separate_gb(model.astr)
    else:
        pass


def do_Xsim(model, Xsim_1):
    """
    Do Xsim if needed and assign the corresponding objective function value
    Pass if no Xsim

    model - model object
    Xsim_1 - experimental simulation object (pdf_of_model or gb_ingrained)
    """
    if Xsim_1:
        # get the relaxed structure
        relaxed_str = model.astr
        if relaxed_str is None:
            print('Relaxed structure not available. Skipping Xsim..')

            return None
        else:
            # if relaxed structure exists
            model.Xsim1 = Xsim_1.name
            model, Xsim_val = Xsim_1.evaluate_obj(model)

            return model


def update_pool(evald_futures, models_evald, pool, select,
                data_file, sim_ids):
    """
    Calculates the obejctive values for all models and updates pool with
    best models

    Returns updated (evald_futures, pool, models_evald)

    Args:
    evald_futures - (list) list of submitted energy evaluation futures objects
    models_evald - (int) count of number of fully evaluated models
    pool - pool object from selection.py
    select - select object from selection.py
    data_file - path to data_file to write model data
    sim_ids - (bool) True if experimental simulation is used
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
        select = pool.add_to_pool(model, select, sim_ids=sim_ids)
        # write data to file
        write_data(model, data_file)
        models_evald += 1

    return evald_futures, models_evald, pool, select
