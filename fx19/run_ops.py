###################### Functions for run_fx : Begin ####################################
import time

def input_models_done(futures):
    """
    checks the future.result() for each future.
    If 0 is found, returns True
    Else returns False

    Args:
    futures - (list) list of futures objects (concurrent_futures)
    """
    if len(futures) == 0:
        return False
    for future in futures:
        if future.done():
            res = future.result()
            if isinstance(res, int):
                return True
    return False

def get_working_jobs(futures):
    """
    futures: dictionary of job output future objects labelled w.r.t model
    labels as keys

    Checks if any jobs in futures is still running and returns number of
    running jobs

    Args:
    futures - (list) list of futures objects (concurrent_futures)
    """
    s1 = time.time()
    if len(futures) == 0:
        return 0
    else:
        running = 0
        for future in futures:
            if not future.done():
                running +=1
        total_time = time.time() - s1
        with open('/ufrc/hennig/kvs.chaitanya/relaxation/Fantastx/Apr_1_gb/speed_1/working_jobs_time.txt', 'a') as f:
            if total_time > 0.5:
                f.write('{0}\t\t{1:.8f}\n'.format(len(futures),total_time))

        return running

def update_futures(futures, processed_labels):
    """
    Remove the processed futures from the list of futures

    Args:
    futures - (list) list of futures objects (concurrent_futures)
    processed_labels - (list) list of model labels that are already processed
    """
    inds = []
    for i, future in enumerate(futures):
        if future.done():
            if not future.exception():
                model = future.result()
                if model != 0:
                    if model.label in processed_labels:
                        inds.append(i)
    futures = [futures[i] for i in range(len(futures)) if i not in inds]

    return futures

def update_pool(master_pool, evald_futures, simd_futures, models_evald,
                weights, pool, select, energy_code, gb_ops_obj, Xsim_1,
                data_file, sims):
    """
    Calculates the obejctive values for all models and updates pool with
    best models

    Returns updated (evald_futures, simd_futures, pool)

    Args:
    master_pool - (obj) ProcessPoolExecutor to paralellize Xsim within the node
    evald_futures - (list) list of submitted energy evaluation futures objects
    simd_futures - (list) list of submitted Xsim futures
    weights - (list) weights for each objecive function
    pool - pool object from selection.py
    select - select object from selection.py
    energy_code - energy_code object (lammps_code or vasp_code)
    gb_ops_obj - gb_ops_obj from structure_operations.py
    Xsim_1 - Xsim_1 object (pdf_of_model or gb_ingrained)
    data_file - path to data_file to write model data
    sims - (bool) True if experimental simulation is used
    """
    # TODO: remove time stamps
    init_evald_len, init_simd_len = len(evald_futures), len(simd_futures)
    s1 = time.time()

    # remove all futures with an exception
    rem_inds, inds_to_be_simd = [], []
    for i, future in enumerate(evald_futures):
        if future.done():
            rem_inds.append(i)
            if not future.exception():
                inds_to_be_simd.append(i)

    # get all futures which should be submitted to simulation
    futures_to_be_simd = [evald_futures[i] for i in inds_to_be_simd]
    # remove all done futures from evald_futures
    evald_futures = [evald_futures[i] for i in range(len(evald_futures)) \
                                            if i not in rem_inds]

    s2 = time.time()
    #submit all futures_to_be_simd for simulation
    for future in futures_to_be_simd:
        model = future.result()
        try:
            # separate gb_iface for the energy evaluated futures
            separate_gb(energy_code, gb_ops_obj, model)
            # Do Xsim if required
            if Xsim_1:
                out_sim = master_pool.submit(do_Xsim, model, Xsim_1)
                simd_futures.append(out_sim)
        except:
            # remove unsuccessful models with missing data
            print ('Model {} failed at exp sim..'.format(model.label))
            continue

    s3 = time.time()
    # Process simd_futures
    # remove all futures with an exception
    rem_inds, inds_of_done_simd_futures = [], []
    for i, future in enumerate(simd_futures):
        if future.done():
            rem_inds.append(i)
            if not future.exception():
                inds_of_done_simd_futures.append(i)

    futures_to_be_added_to_pool = [simd_futures[i] for i in \
                                                inds_of_done_simd_futures]
    # remove all done futures from evald_futures
    simd_futures = [simd_futures[i] for i in range(len(simd_futures)) \
                                            if i not in rem_inds]

    s4 = time.time()
    for future in futures_to_be_added_to_pool:
        model = future.result()
        # Add to either good_pool or bad_pool
        # Selection_probs are also updated
        pool.add_to_pool(model, select, sims=sims)
        # write data to file
        write_data(model, data_file)
        models_evald += 1

    s5 = time.time()
    end_evald_len, end_simd_len = len(evald_futures), len(simd_futures)
    tf = '/ufrc/hennig/kvs.chaitanya/relaxation/Fantastx/Apr_1_gb/master_pool/speed_2/update_pool_time.txt'
    with open('tf', 'a') as f:
        l='{0}\t\t{1}\t\t{2:.6f}\t\t{3:.6f}\t{4:.6f}\t\t{5:.6f}\t{6:.6f}\n'.format(
                                        (init_evald_len, init_simd_len),
                                        (end_evald_len, end_simd_len),
                                        s2-s1, s3-s2, s4-s3, s5-s4, s5-s1)
        f.write(l)

    return evald_futures, simd_futures, pool, models_evald

def write_time_update_pool(xsim_time, add_to_pool_time, loop_time,
                                    total_time, num_futures):
    with open('/ufrc/hennig/kvs.chaitanya/relaxation/Fantastx/Apr_1_gb/' + \
                        'speed_1/update_pool_time.txt', 'a') as f:
        f.write('{0:.8f}\t{1:.8f}\t{2:.8f}\t{3:.8f}\t{4}\n'.format(xsim_time,
                    add_to_pool_time, loop_time, total_time, num_futures))

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

# full_eval deprecated
def deprecated_full_eval(random_model_obj, reg_id, evolve, select,
              pool, energy_code, model_type='random',
              gb_ops_obj=None, Xsim_1=None, model=None):
    """
    A wrapper function around energy_eval and Xsim_eval.
    Both these are done one after the other as one job by executor

    Args:
    random_model_obj - make_random_model object or gb_ops_obj
    reg_id - register_id object
    evolve - evolve object
    pool - pool object from selection.py
    select - select object from selection.py
    energy_code - energy_code object (lammps_code or vasp_code)
    gb_ops_obj - gb_ops_obj from structure_operations.py
    Xsim_1 - Xsim_1 object (pdf_of_model or gb_ingrai
    model_type (str): 'random' or 'evolved'
                      'random' - make random model for initial population
                      'evolved' - make child model by evolution
    model (model obj): if a model object is provided as inputs model_type
                        it is directly taken to energy evaluation step.
    """
    eval_done = False
    while not eval_done:
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

        # evaluate energy of the new_model
        # epa or energy difference stored in obj0_val
        try:
            energy_code.relax(new_model, reg_id)
        except FileExistsError:
            print ('Duplicate label in parallel processes. Skipping..')
            continue
        resubmitted = 2
        if new_model.converged == False:
            for i in range(len(energy_code.resubmit)):
                if resubmitted < energy_code.resubmit and new_model.converged == False:
                    resubmitted += 1
                    try:
                        energy_code.re_relax(new_model)
                    except:
                        continue
        # For grain boundary search, assign grain_interface as model attribute
        if energy_code.shape == 'gb':
            new_model.gb_iface = gb_ops_obj.separate_gb(new_model.astr)

        # get the relaxed structure
        relaxed_str = new_model.astr
        if relaxed_str is None:
            print ('Relaxed structure not available. Making new model...')
            continue
        # if relaxed structure exists
        if Xsim_1:
            new_model.Xsim1 = Xsim_1.name
            new_model, Xsim_val = Xsim_1.evaluate_obj(new_model)
        eval_done = True

    return new_model

# Temporary selection probs based on overall_value
def temp_selection_probs(pool):
    """
    Always the minimum overall value gets selevtion_prob of 1.
    """
    ov = [i.overall_val for i in pool.good_pool]
    norm_vals = []
    if not len(ov) < 3:
        for model in pool.good_pool:
            i = model.overall_val
            norm_prob = (i - max(ov))/(min(ov)-max(ov))
            norm_vals.append(norm_prob)
            model.selection_prob = norm_prob
        norm_sum = sum(norm_vals)
        for model in pool.good_pool:
            model.selection_prob = model.selection_prob / norm_sum


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
        print ('Duplicate label in parallel processes. Skipping..')
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

    return new_model

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
            print ('Relaxed structure not available. Skipping Xsim..')

            return None
        else:
            # if relaxed structure exists
            model.Xsim1 = Xsim_1.name
            model, Xsim_val = Xsim_1.evaluate_obj(model)

            return model

def full_eval(model):
    """
    A wrapper function around energy_eval and Xsim_eval.
    Both these are done one after the other as one job by worker

    Args:
    model - (obj) Newly created model object which shall be evaluated

    Note:
    Uses reg_id, Xsim_1, energy_code objects which were stored as global
    parameters in all workers and master
    """
    # submit model to energy relaxation
    try:
        energy_code.relax(model, reg_id)
    except FileExistsError:
        print ('Duplicate label in parallel processes. Skipping..')
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

    # separate gb_iface for the energy evaluated futures
    separate_gb(energy_code, gb_ops_obj, model)
    # Do Xsim if required
    if Xsim_1:
        # get the relaxed structure
        relaxed_str = model.astr
        if relaxed_str is None:
            print ('Relaxed structure not available. Skipping Xsim..')
            return None
        else:
            # if relaxed structure exists
            model.Xsim1 = Xsim_1.name
            model, Xsim_val = Xsim_1.evaluate_obj(model)
            return model

def new_update_pool(evald_futures, models_evald, pool, select, data_file, sims):
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
    sims - (bool) True if experimental simulation is used
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
    evald_futures = [evald_futures[i] for i in range(len(evald_futures)) \
                                            if i not in rem_inds]
    for future in futures_to_process:
        model = future.result()
        # Add to either good_pool or bad_pool
        # Selection_probs are also updated
        pool.add_to_pool(model, select, sims=sims)
        # write data to file
        write_data(model, data_file)
        models_evald += 1

    return evald_futures, pool, models_evald
