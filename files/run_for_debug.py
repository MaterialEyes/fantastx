# from run_fx.py
import os, yaml, sys
import datetime
import random, copy
from fx19 import inputs
from fx19 import distance_check as dc
from fx19.structure_operations import gb_ops
from fx19.run_ops import *

import time, gc
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from time import sleep

# dask import
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

main_path = os.getcwd()
# read input file and make input dictionary
with open('gb_input.yaml') as ifile:
    i_dict = yaml.load(ifile, Loader=yaml.FullLoader)
    i_dict['main_path'] = main_path

# make objects
all_objects = inputs.make_objects(i_dict)

# Assign objects from all_objects to local variables
reg_id = all_objects['reg_id']

input_model_obj = all_objects['input_model_obj']
gb_ops_obj = all_objects['gb_ops_obj']

# kwargs for full_eval() function
if gb_ops_obj is not None:
    random_model_obj = gb_ops_obj
    evolve = gb_ops_obj
else:
    random_model_obj = all_objects['random_model_obj']
    evolve = all_objects['evolve']

energy_code = all_objects['energy_code']
if 'Xsim_1' in all_objects.keys():
    Xsim_1 = all_objects['Xsim_1']
    sims = [1]
else:
    Xsim_1 = None
# TODO: add Xsim2 and Xsim3 etc.. and count in sims

pool = all_objects['pool']
select = all_objects['select']
weights = select.weights # If single_objective, weights should be
                         # [1, 0, 0, 0, 0, 0] - see inputs.py

# Create a folder 'Calcs' where all calculations take place
if 'calcs' in os.listdir(main_path):
    now = datetime.datetime.now()
    new_name = 'old_{}_{}_{}_{}_{}_{}'.format(now.year, now.month, now.day,
                                            now.hour, now.minute, now.second)
    os.rename('calcs', new_name)

calcs = i_dict['main_path'] + '/calcs'
os.mkdir(calcs)

####### write data to a file
data_file = main_path + '/data_file'
with open(data_file, 'w') as f:
    first_line = 'id\tinheritance\t\ttotal energy\tObj_0\t\tObj_1\n\n'
    if not Xsim_1:
        first_line = 'id\t\tinheritance\t\ttotal energy\tObj_0\n\n'
    f.write(first_line)

# set up everything for calculations
models_evald = 0
# the output of energy evaluation for models is stored in this dict
evald_futures = []
processed_labels = []
# Start the ProcessPoolExecutor with num_parallel as max_workers
# NOTE: ~ total_cores/max_workers is the num cores used to do one calculation
max_workers = 2 # TODO: make an option for max_workers in the input file
###############
cluster_job = SLURMCluster(cores=1,
                           memory="2GB",
                           project='hennig',
                           queue='hpg2-compute',
                           interface='ib0',
                           walltime='2:00:00',
                           job_extra=['--ntasks 4', '--nodes=1'])
cluster_job.scale(jobs=max_workers) # number of parallel jobs
client  = Client(cluster_job)

master_pool = ProcessPoolExecutor(max_workers=16)


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
        new_model = make_model(random_model_obj, evolve, select, pool,
                                reg_id, model_type='inputs', model=input_model)
        # relax the model in dask-workers
        out = client.submit(relax, new_model, reg_id, energy_code)
        evald_futures.append(out)
    print ('Input models are finished. Making random models..')
    # Post-processing & Xsim are done along with random models for input models

num_initial_pop =  5#i_dict['initial_population']['total']
total_models_needed = 10#i_dict['structure_record']['stopper']['num_calcs']
working_jobs = get_working_jobs(evald_futures)

start_time = time.time()
# Make random models & evolved models
while models_evald < total_models_needed:
    working_jobs = get_working_jobs(evald_futures)
    # In some cases (lammps based), working_jobs always < max_workers
    while working_jobs < max_workers and models_evald < total_models_needed:
        # make model
        s=0
        if models_evald < num_initial_pop:
            s = time.time()
            new_model = make_model(random_model_obj, evolve, select, pool,
                                        reg_id, model_type='random')
        else:
            s = time.time()
            new_model = make_model(random_model_obj, evolve, select, pool,
                                        reg_id, model_type='evolved')
        if s!= 0:
            make_model_time = time.time() - s
            print ('It took {} secs to make model {}'.format(
                                    make_model_time, new_model.label))
        # relax the model in dask-workers
        out = client.submit(relax, new_model, reg_id, energy_code)
        evald_futures.append(out)
        working_jobs = get_working_jobs(evald_futures)
        # make sure 50 % of workers are working before processing futures
        if working_jobs > mar_workers * 0.5:
            processed_labels, pool, evald_futures = update_pool(
                                                        master_pool,
                                                        evald_futures,
                                                        processed_labels,
                                                        weights, pool, select,
                                                        energy_code, gb_ops_obj,
                                                        Xsim_1, data_file, sims)
        models_evald = len(processed_labels)
        if models_evald % 100 < 5:
            run_time = time.time() - start_time
            with open('/ufrc/hennig/kvs.chaitanya/relaxation/Fantastx/' + \
                        'Apr_1_gb/speed_1/models_time.txt', 'a') as f:
                f.write('{}\t{}\n'.format(run_time, models_evald))
        #temp_selection_probs(pool)

# process extra calculations running in last batch
while len(evald_futures) > 0:
    processed_labels, pool, evald_futures = update_pool(
                                                       evald_futures,
                                                       processed_labels,
                                                       weights, pool, select,
                                                       energy_code, gb_ops_obj,
                                                       Xsim_1, data_file, sims)

sorted_pool = pool.get_sorted_pool()
with open('sorted_data', 'a') as f:
    for model in sorted_pool:
        f.write('{0}\t {1}\n'.format(model.label, model.obj0_val))

#client.shutdown()

print ('Done!')
print ('Total time: ', time.time() - start_time)
