#!/usr/bin/env python
# coding: utf-8
import os
import yaml
import sys
import datetime
import random
import copy
from fx19 import inputs
from fx19 import distance_check as dc
from fx19.structure_operations import gb_ops
from fx19.run_ops import *
from fx19.clustering import hierarchical_clusterer, compositional_clusterer
import multiprocessing as mp
import traceback

import numpy as np
from time import sleep

# dask import
from dask_jobqueue import SLURMCluster, PBSCluster
from dask.distributed import Client, LocalCluster

# change worker unresponsive time to 3h (Assuming max elapsed time for one calc)
import dask
import dask.distributed
dask.config.set({'distributed.comm.timeouts.tcp': '3h'})

main_path = os.getcwd()
# read input file and make input dictionary
with open('epsilon_selection.yaml') as ifile:
    i_dict = yaml.load(ifile, Loader=yaml.FullLoader)
    i_dict['main_path'] = main_path

# make objects
all_objects = inputs.make_objects(i_dict)

# Assign objects from all_objects to local variables
reg_id = all_objects['reg_id']

input_model_obj = all_objects['input_model_obj']

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
weights = select.weights  # If single_objective, weights should be
# [1, 0, 0, 0, 0, 0] - see inputs.py

# Create a folder 'Calcs' where all calculations take place
if 'calcs' in os.listdir(main_path):
    now = datetime.datetime.now()
    new_name = 'old_{}_{}_{}_{}_{}_{}'.format(now.year, now.month, now.day,
                                              now.hour, now.minute, now.second)
    os.rename('calcs', new_name)

calcs = i_dict['main_path'] + '/calcs'
os.mkdir(calcs)

# write data to a file
data_file = main_path + '/data_file'
with open(data_file, 'w') as f:
    first_line = 'Label   Inheritance     Total Energy    Obj_0' + \
                        '           Obj_1           Operator\n\n'
    if not Xsim_1:
        first_line = 'Label   Inheritance     Total Energy    Obj_0' + \
                        '           Operator\n\n'
    f.write(first_line)

# set up everything for calculations
models_evald = 0
# the output of energy evaluation for models is stored in this dict
evald_futures, simd_futures = [], []

workers = i_dict['workers']
max_workers = workers['max_workers']
#job_script = '/home/dunruh/sample_job_script.txt'
#jobfile = open(job_script, "w+")
if workers['cluster'] == 'SLURM':
    cluster_job = SLURMCluster(cores=workers['num_cores'],
                               memory=workers['total_mem'],
                               processes=workers['processes'],
                               project=workers['project_name'],
                               queue=workers['submit_queue'],
                               interface=workers['node_type'],
                               walltime=workers['walltime'],
                               job_extra=workers['job_extra'],
                               header_skip=workers['header_skip'])
    print ("Job script for dask-worker: \n", cluster_job.job_script())
    client = Client(cluster_job)
    jobfile.write(cluster_job.job_script())
    jobfile.close()
elif workers['cluster'] == 'PBS':
    cluster_job = PBSCluster(cores=workers['num_cores'],
                             memory=workers['total_mem'],
                             project=workers['project_name'],
                             interface=workers['node_type'],
                             walltime=workers['walltime'],
                             job_extra=workers['job_extra'],
                             header_skip=workers['header_skip'])
    print ("Job script for dask-worker: \n", cluster_job.job_script())
    client = Client(cluster_job)
elif workers['cluster'] == 'local':
    client = Client('tcp://127.0.0.1:8786')
else:
    print('FANTASTX currently supports SLURM, PBS and local. Provided '
          'scheduler type not identified.')

if workers['cluster'] == 'SLURM' or workers['cluster'] == 'PBS':
    cluster_job.scale(max_workers)
    cluster_job.scale(jobs=max_workers)  # number of parallel jobs
    client = Client(cluster_job)

# full_eval function which uses global variables


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

    # separate gb_iface for the energy evaluated futures
    separate_gb(energy_code, gb_ops_obj, model)

    # Do Xsim if required
    if Xsim_1:
        if not model.converged:
            print ('Energy calculation of model {} is not'
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
            return model
    else:
        return model


# wait for workers to start on cluster
client.wait_for_workers(1)

# make new model from all input files provided, then random, then evolve
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
        new_model, select = make_model(random_model_obj, evolve, select, pool,
                                       reg_id, model_type='inputs',
                                       model=input_model)
        # relax the model in dask-workers
        out = client.submit(full_eval, new_model)
        evald_futures.append(out)
    print('Input models are finished. Making random models..')
    # Post-processing & Xsim are done along with random models for input models

num_initial_pop = i_dict['population_limits']['initial_population']
total_models_needed = i_dict['population_limits']['total_population']
working_jobs = get_working_jobs(evald_futures)

start_time = time.time()
# Make random models & evolved models
pool_status_update = 10 # number of models before current pool status is printed
while models_evald < total_models_needed:
    working_jobs = get_working_jobs(evald_futures)
    # In some cases (lammps based), working_jobs always < max_workers
    # Ensure some structures are always in the queue for each worker so that
    # workers wont be idle if some step on master becomes bottle neck
    while working_jobs < 2*max_workers and models_evald < total_models_needed:
        # make model
        if models_evald < num_initial_pop:
            new_model, select = make_model(random_model_obj, evolve, select,
                                           pool, reg_id, model_type='random')
        else:
            new_model, select = make_model(random_model_obj, evolve, select,
                                           pool, reg_id, model_type='evolved')


        # relax the model in dask-workers
        out = client.submit(full_eval, new_model)
        evald_futures.append(out)
        evald_futures, models_evald, pool, select = update_pool(evald_futures,
                                                                models_evald,
                                                                pool, select,
                                                                data_file,
                                                                sim_ids)
        working_jobs = get_working_jobs(evald_futures)

        if models_evald % pool_status_update == 0 and models_evald >= i_dict['population_limits']['pool']:
            # print statements which output visualization information
            if "selection_algorithm" in i_dict["select_params"]:
                if i_dict["select_params"]["selection_algorithm"] == "distance_from_pareto":
                    good_pool = pool.good_pool
                    good_pool_labels = [model.label for model in good_pool]
                    print(f"Current good_pool population models: {good_pool_labels}.")
                elif i_dict["select_params"]["selection_algorithm"] == "epsilon_moea":
                    pop_labels = [model.label for model in pool.population.models]
                    archive_labels = [model.label for model in pool.archive.models]
                    print(f"Current pool population models: {pop_labels}")
                    print(f"Current pool archive models: {archive_labels}")
                elif i_dict["select_params"]["selection_algorithm"] == "clustered_selection":
                    nd_pop_labels = [model.label for model in pool.population.non_dominated_models]
                    print(f"Current pool population non-dominated models: {nd_pop_labels}")
            else:
                good_pool = pool.good_pool
                good_pool_labels = [model.label for model in good_pool]
                print(f"Current good_pool population models: {good_pool_labels}.")

# process extra calculations running in last batch
while len(evald_futures) > 0:
    evald_futures, models_evald, pool, select = update_pool(evald_futures,
                                                            models_evald,
                                                            pool, select,
                                                            data_file, sim_ids)

# print statements which output visualization information
if "selection_algorithm" in i_dict["select_params"]:
    if i_dict["select_params"]["selection_algorithm"] == "distance_from_pareto":
        good_pool = pool.good_pool
        good_pool_labels = [model.label for model in good_pool]
        print(f"Current good_pool population models: {good_pool_labels}.")
    elif i_dict["select_params"]["selection_algorithm"] == "epsilon_moea":
        pop_labels = [model.label for model in pool.population.models]
        archive_labels = [model.label for model in pool.archive.models]
        print(f"Current pool population models: {pop_labels}")
        print(f"Current pool archive models: {archive_labels}")
    elif i_dict["select_params"]["selection_algorithm"] == "clustered_selection":
        nd_pop_labels = [model.label for model in pool.population.non_dominated_models]
        print(f"Current pool population non-dominated models: {nd_pop_labels}")
else:
    good_pool = pool.good_pool
    good_pool_labels = [model.label for model in good_pool]
    print(f"Current good_pool population models: {good_pool_labels}.")

print(f"Current operator probabilities: {select.operator_frequencies}")
print('Done!')
print('Total time: ', time.time() - start_time)

client.shutdown()
