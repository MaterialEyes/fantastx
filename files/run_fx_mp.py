#!/usr/bin/env python
# coding: utf-8
import os
import yaml
import datetime
from fx19 import inputs
from fx19.run_ops import *
import time
import multiprocessing as mp
import scipy as sp
import numpy as np
import random
import datetime

main_path = os.getcwd()
# read input file and make input dictionary
with open('input.yaml') as ifile:
    i_dict = yaml.load(ifile, Loader=yaml.FullLoader)
    i_dict['main_path'] = main_path

# make objects
all_objects = inputs.make_objects(i_dict)

# Assign objects from all_objects to local variables
reg_id = all_objects['reg_id']
input_model_obj = all_objects['input_model_obj']
db = all_objects['database']
gb_ops_obj = None
if 'gb_ops_obj' in all_objects:
    gb_ops_obj = all_objects['gb_ops_obj']
# kwargs for full_eval() function
if gb_ops_obj is not None:
    random_model_obj = gb_ops_obj
    evolve = gb_ops_obj
else:
    random_model_obj = all_objects['random_model_obj']
    evolve = all_objects['evolve']

energy_code = all_objects['energy_code']
sim_ids = None
if 'Xsim_1' in all_objects.keys():
    Xsim_1 = all_objects['Xsim_1']
    sim_ids = [1]
else:
    Xsim_1 = None
# TODO: add Xsim2 and Xsim3 etc.. and count in sim_ids

pool = all_objects['pool']
select = all_objects['select']

# Create a folder 'Calcs' where all calculations take place
if 'calcs' in os.listdir(main_path):
    now = datetime.datetime.now()
    new_name = 'old_{}_{}_{}_{}_{}_{}'.format(now.year, now.month, now.day,
                                              now.hour, now.minute, now.second)
    os.rename('calcs', new_name)

calcs = i_dict['main_path'] + '/calcs'
os.mkdir(calcs)

# Create data file where objective function values and inheritance
# information will be written.
data_file = main_path + '/data_file'
with open(data_file, 'w') as f:
    first_line = 'Label   Inheritance     Total Energy    Obj_0' + \
        '           Obj_1           Operator\n\n'
    if not Xsim_1:
        first_line = 'Label   Inheritance     Total Energy    Obj_0' + \
            '           Operator\n\n'
    f.write(first_line)

# Initialize job log file
job_file = main_path + "/job_log.txt"

##############################
# Set up evaluation function #
##############################


def create_and_eval(seed, model_obj, evolve, select, pool, reg_id, energy_code,
                    Xsim, model_type, model, m_list):
    """
    A wrapper function around energy_eval and Xsim_eval.
    Both these are done one after the other as one job by worker

    Args:
    model - (obj) Newly created model object which shall be evaluated

    Note:
    Uses reg_id, Xsim_1, energy_code objects which were stored as global
    parameters in all workers and master
    """
    # First, set all possible random number generators with the provided
    # random number seed (important with multiprocessing!)
    np.random.seed(seed)
    # sp.random.seed(seed)
    random.seed(seed)
    # create model
    new_model = make_model(model_obj, evolve, select,
                           pool, reg_id, model_type, model)

    # submit model to energy relaxation
    relaxed_model = relax(new_model, reg_id, energy_code)
    if relaxed_model is None:
        m_list.append(None)
    else:
        print(f"Model converged: {relaxed_model.converged}")

    if energy_code.shape == "gb":
        relaxed_model.gb_iface = model_obj.separate_gb(model.astr)

    # do the experimental evaluation
    exp_eval_model = do_Xsim(relaxed_model, Xsim)

    m_list.append(exp_eval_model)


####################################
#  SET UP MULTIPROCESSING OBJECTS  #
####################################
manager = mp.Manager()
model_list = manager.list()
jobs = []
workers = i_dict['workers']
max_running_jobs = workers['max_workers']
num_initial_pop = i_dict['population_limits']['initial_population']
total_models_needed = i_dict['population_limits']['total_population']
processed_models = 0
models_evald = 0
evald_futures, simd_futures = [], []
pool_status_update = 2
seed_index = 0
seed_multiplier = 102573
try:
    seed_multiplier = int(datetime.datetime.now().strftime(
        '%m%d%H%M%S'))
    print(f"Random seed multiplier: {seed_multiplier}")
except:
    print("Tried and failed to set random seed using the date and time.\n")

###########################
#  RUN INPUT CALCULATIONS #
###########################
input_models = []
if input_model_obj is not None:
    for i in range(len(input_model_obj.all_files)):
        new_model = input_model_obj.read_structure(reg_id)
        if new_model is not None:
            # read_structure() returns 0 when all files are done
            if not isinstance(new_model, int):
                input_models.append(new_model)

    # evaluate the input models
    for input_model in input_models:
        time.sleep(1)
        while get_working_mp_jobs(jobs) >= max_running_jobs:
            continue
        seed_index += 1
        seed = seed_index * seed_multiplier + 1
        out = mp.Process(target=create_and_eval, args=(seed,
                                                       input_model_obj,
                                                       evolve,
                                                       select,
                                                       pool,
                                                       reg_id,
                                                       energy_code,
                                                       Xsim_1,
                                                       'inputs',
                                                       input_model,
                                                       model_list))
        jobs.append(out)
        out.start()
        print(f"Successfully submitted input model {input_model.label}\n")

processed_models, models_evald, pool, select = update_pool_mp(model_list,
                                                              processed_models,
                                                              models_evald,
                                                              pool, select,
                                                              data_file,
                                                              db,
                                                              sim_ids)
working_jobs = get_working_mp_jobs(jobs)

# For molecules, which have no random generation, wait for
# at least a few workers to finish before proceeding
if all_objects['constraints_obj'].shape == 'molecule':
    processed_models, models_evald, pool, select = update_pool_mp(model_list,
                                                                  processed_models,
                                                                  models_evald,
                                                                  pool, select,
                                                                  data_file,
                                                                  db,
                                                                  sim_ids)
    while models_evald < min(len(input_models), num_initial_pop):
        processed_models, models_evald, pool, select = update_pool_mp(model_list,
                                                                      processed_models,
                                                                      models_evald,
                                                                      pool, select,
                                                                      data_file,
                                                                      db,
                                                                      sim_ids)

print('Input models are finished. Making random models..\n')
start_time = time.time()
# Make random models & evolved models
while models_evald < total_models_needed:
    time.sleep(1)
    working_jobs = get_working_mp_jobs(jobs)
    # In some cases (lammps based), working_jobs always < max_workers
    # Ensure some structures are always in the queue for each worker so that
    # workers wont be idle if some step on master becomes bottle neck
    while working_jobs < max_running_jobs and models_evald < total_models_needed:
        # make model
        if models_evald < num_initial_pop:
            print("Submitting random job\n")
            model_mech = "random"
        else:
            print("Submitting evolved job\n")
            model_mech = "evolved"

        seed_index += 1
        seed = seed_index * seed_multiplier + 1
        # Create the job and send it to multiprocessing for evaluation
        out = mp.Process(target=create_and_eval, args=(seed,
                                                       random_model_obj,
                                                       evolve,
                                                       select,
                                                       pool,
                                                       reg_id,
                                                       energy_code,
                                                       Xsim_1,
                                                       model_mech,
                                                       None,
                                                       model_list))
        jobs.append(out)
        out.start()

        # Update the pool
        processed_models, models_evald, pool, select = update_pool_mp(model_list,
                                                                      processed_models,
                                                                      models_evald,
                                                                      pool, select,
                                                                      data_file,
                                                                      db,
                                                                      sim_ids)
        working_jobs = get_working_mp_jobs(jobs)

        if models_evald % pool_status_update == 0 and\
                models_evald >= i_dict['population_limits']['pool']:
            job_log = open(job_file, "a+")
            job_log.write(
                f"Update due to models_evald reaching: {models_evald}\n")
            # print statements which output visualization information
            if "selection_algorithm" in i_dict["select_params"]:
                if i_dict["select_params"]["selection_algorithm"] ==\
                        "distance_from_pareto":
                    good_pool = pool.good_pool
                    good_pool_labels = [model.label for model in good_pool]
                    job_log.write("Current good_pool population models: "
                                  f"{good_pool_labels}.\n")
                    job_log.close()
                    print(
                        "Current good_pool population models: "
                        f"{good_pool_labels}.\n")
                elif i_dict["select_params"]["selection_algorithm"] ==\
                        "epsilon_moea":
                    pop_labels = [
                        model.label for model in pool.population.models]
                    archive_labels = [
                        model.label for model in pool.archive.models]
                    job_log.write("Current pool population models: "
                                  f"{pop_labels}\n")
                    job_log.write("Current pool archive models:"
                                  f"{archive_labels}\n")
                    job_log.close()
                    print("Current pool population models: "
                          f"{pop_labels}\n")
                    print("Current pool archive models:"
                          f"{archive_labels}\n")
                elif i_dict["select_params"]["selection_algorithm"] ==\
                        "clustered_selection":
                    nd_pop_labels = [
                        model.label for
                        model in pool.population.non_dominated_models]
                    job_log.write(
                        "Current pool population non-dominated models: "
                        f"{nd_pop_labels}\n")
                    job_log.close()
                    print(
                        "Current pool population non-dominated models: "
                        f"{nd_pop_labels}\n")
            else:
                good_pool = pool.good_pool
                good_pool_labels = [model.label for model in good_pool]
                job_log.write(
                    "Current good_pool population models: "
                    f"{good_pool_labels}.\n")
                job_log.close()
                print(
                    "Current good_pool population models: "
                    f"{good_pool_labels}.\n")

# process extra calculations running in last batch
while get_working_mp_jobs(jobs) > 0:
    processed_jobs, models_evald, pool, select = update_pool_mp(model_list,
                                                                processed_jobs,
                                                                models_evald,
                                                                pool, select,
                                                                data_file,
                                                                db,
                                                                sim_ids)

# print statements which output visualization information
job_log = open(job_file, "a+")
job_log.write(
    f"Update due to models_evald reaching: {models_evald}")
# print statements which output visualization information
if "selection_algorithm" in i_dict["select_params"]:
    if i_dict["select_params"]["selection_algorithm"] ==\
            "distance_from_pareto":
        good_pool = pool.good_pool
        good_pool_labels = [model.label for model in good_pool]
        job_log.write("Current good_pool population models: "
                      f"{good_pool_labels}.\n")
        job_log.close()
        print(
            "Current good_pool population models: "
            f"{good_pool_labels}.\n")
    elif i_dict["select_params"]["selection_algorithm"] ==\
            "epsilon_moea":
        pop_labels = [
            model.label for model in pool.population.models]
        archive_labels = [
            model.label for model in pool.archive.models]
        job_log.write("Current pool population models: "
                      f"{pop_labels}\n")
        job_log.write("Current pool archive models:"
                      f"{archive_labels}\n")
        job_log.close()
        print("Current pool population models: "
              f"{pop_labels}\n")
        print("Current pool archive models:"
              f"{archive_labels}\n")
    elif i_dict["select_params"]["selection_algorithm"] ==\
            "clustered_selection":
        nd_pop_labels = [
            model.label for
            model in pool.population.non_dominated_models]
        job_log.write(
            "Current pool population non-dominated models: "
            f"{nd_pop_labels}\n")
        job_log.close()
        print(
            "Current pool population non-dominated models: "
            f"{nd_pop_labels}\n")
else:
    good_pool = pool.good_pool
    good_pool_labels = [model.label for model in good_pool]
    job_log.write(
        "Current good_pool population models: "
        f"{good_pool_labels}.\n")
    job_log.close()
    print(
        "Current good_pool population models: "
        f"{good_pool_labels}.\n")

print(f"Current operator probabilities: {select.operator_frequencies}\n")
print("Done!\n")
print(f"Total time: {time.time() - start_time}\n")
