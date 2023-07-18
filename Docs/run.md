# Running FANTASTX

***

Once you have prepared the FANTASTX input YAML file and have cleaned and prepared the experimental data that you will be inverting, all that is left is to actually run FANTASTX. 

!!! warning "Don't forget!"
    If you are performing structural relaxation, then you also need to [prepare the input files](energy.md) for the structural relaxation code and include them in subdirectory of your job directory which is referenced in the FANTASTX input YAML. 
    
    If you are running XAS inversion, you need to [prepare the YAML input file](xanes.md) for the XAS simulation code you will be using (FEFF or FDMNES).

    For some structural geometries, namely grain boundaries, surfaces, and molecules, some additional inputs need to be provided so that FANTASTX can randomly construct initial atomic configurations. Don't forget to provide these additional inputs!

FANTASTX is designed to be run either directly with a python script or with a jupyter notebook with a python kernel. Example scripts and an example notebook can be found in the [FANTASTX GitHub](https://github.com/MaterialEyes/fantastx/tree/master/files), and are also shown below. One of these scripts (run_fx.py) uses Dask for parallelization, while another (run_fx_mp.py) uses python's multiprocessing for parallelization. These scripts and notebook are designed to be used for any application of FANTASTX, and can thus be used directly, or can alternatively be modified to fit special cases or add additional functionality. Each of these example scripts utilizes a set of FANTASTX helper functions which can be found in fx19/run_ops.py. 

!!! info
    These examples assume that the FANTASTX input YAML file is called "input.yaml".

??? tip "Choosing parallelization schemes"
    When choosing between Dask and multiprocessing for your parallelization, it is important to recognize the differences between how each type of parallelization functions.

    Dask runs by creating a nanny process which monitors the progress and status of worker jobs which conduct all required computations. It can be run either on your local machine or natively on high-performance computing (HPC) systems. On HPC systems, the nanny process will occupy your initial job allocation, while each worker job will be submitted to the queue as independent jobs. Notably, this means that workers automatically never have to fight with each other for job resources. However, worker jobs are only capable of running on single nodes, and the nanny needs to wait for the worker jobs to start before they can start working on computations. If queue times are long and job time limits are restrictive, this can mean that the nanny job times out before all worker jobs have even started, in which case the job will fail (or waste a significant number of cpu hours).

    In contrast, multiprocessing is designed to work only a single node, utilizing local cpu threads similar to OpenMP. While at first glance this may seem to be an issue for using this methodology with HPC systems where you want to utilize multiple nodes, it is important to recognize that the vast majority of FANTASTX's computational demands come from running external programs, such as VASP, which are run by FANTASTX using their own srun or mpirun commands. In the majority of HPC systems, these layered srun commands can run on independent nodes within the larger allocation, such as with the command "srun -N1 -n36 vasp_std", which will run VASP on it's own node with 36 threads. This allows the submission of FANTASTX jobs which request 11 nodes, and run 5 independent workers which each utilize 2 nodes in addition to the single node that FANTASTX itself is running on. This bypasses the single node restriction of the Dask worker jobs, and also makes sure that when the FANTASTX job starts that it will not waste any computation time. 
    
    While the multiprocessing scheme may seem superior to Dask, it does not always work with every simulation program that FANTASTX interfaces with. Notably, some programs such as Diffpy are not called externally, and thus the single FANTASTX node is left to simultaneously perform every PDF simulation under this scheme. Additionally, any FANTASTX tasks such as model generation are left to the single FANTASTX node, which in rare cases can cause bottlenecks. 
    
    In general, for simplicity we recommend the use of Dask for local systems, as well as HPC systems where no external program will be using more than 1 node of resources and queue times are short. In all other cases, or when you feel it is most appropriate, multiprocessing support is available. 

??? example "FANTASTX scripts"

    === "run_fx.py"
        ```python
        import traceback
        import os
        import yaml
        import datetime
        from fx19 import inputs
        from fx19.run_ops import *
        import time

        # dask import
        from dask_jobqueue import SLURMCluster, PBSCluster
        from dask.distributed import Client

        # change worker unresponsive time to 3h
        # (Assuming max elapsed time for one calc)
        import dask
        import dask.distributed
        dask.config.set({'distributed.comm.timeouts.tcp': '3h'})

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

        # Set up everything for calculations
        models_evald = 0
        evald_futures, simd_futures = [], []
        pool_status_update = 2
        workers = i_dict['workers']
        max_workers = workers['max_workers']

        # Start Dask client
        if workers['cluster'] == 'SLURM':
            cluster_job = SLURMCluster(cores=workers['num_cores'],
                                    memory=workers['total_mem'],
                                    processes=workers['processes'],
                                    project=workers['project_name'],
                                    queue=workers['submit_queue'],
                                    interface=workers['node_type'],
                                    walltime=workers['walltime'],
                                    job_extra=workers['job_extra'],
                                    # env_extra=workers['env_extra'],
                                    header_skip=workers['header_skip'])
            print("Job script for dask-worker: \n", cluster_job.job_script())
            client = Client(cluster_job)
        elif workers['cluster'] == 'PBS':
            cluster_job = PBSCluster(cores=workers['num_cores'],
                                    memory=workers['total_mem'],
                                    project=workers['project_name'],
                                    interface=workers['node_type'],
                                    walltime=workers['walltime'],
                                    job_extra=workers['job_extra'],
                                    header_skip=workers['header_skip'])
            print("Job script for dask-worker: \n", cluster_job.job_script())
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


        def full_eval(model, energy_obj, Xsim):
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
                energy_obj.relax(model, reg_id)
            except:
                traceback.print_exc()
                print('Duplicate label in parallel processes. Skipping..')
                return None

            resubmitted = 2
            if not model.converged:
                for i in range(energy_obj.resubmit):
                    if resubmitted < energy_obj.resubmit and not model.converged:
                        resubmitted += 1
                        try:
                            energy_obj.re_relax(model)
                        except:
                            continue

            # print(f"Model converged: {model.converged}")

            # separate gb_iface for the energy evaluated futures
            # separate_gb(energy_obj, gb_ops_obj, model)

            # Do Xsim if required
            if Xsim:
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
                    # print("Doing experimental simulation!!")
                    # if relaxed structure exists
                    model.Xsim1 = Xsim.name
                    model, Xsim_val = Xsim.evaluate_obj(model)
                    model.num_of_obj += 1
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
                new_model = make_model(random_model_obj, evolve, select, pool,
                                    reg_id, model_type='inputs',
                                    model=input_model)
                # relax the model in dask-workers
                out = client.submit(full_eval, new_model, energy_code, Xsim_1)
                evald_futures.append(out)
                print(
                    f"Successfully submitted input model {new_model.label}")

        num_initial_pop = i_dict['population_limits']['initial_population']
        total_models_needed = i_dict['population_limits']['total_population']
        evald_futures, models_evald, pool, select = update_pool(evald_futures,
                                                                models_evald,
                                                                pool, select,
                                                                data_file,
                                                                db,
                                                                sim_ids)
        working_jobs = get_working_jobs(evald_futures)

        # For molecules, which have no random generation, wait for
        # at least a few workers to finish before proceeding
        if all_objects['constraints_obj'].shape == 'molecule':
            evald_futures, models_evald, pool, select = update_pool(evald_futures,
                                                                    models_evald,
                                                                    pool, select,
                                                                    data_file,
                                                                    db,
                                                                    sim_ids)
            while models_evald < min(len(input_models), num_initial_pop):
                evald_futures, models_evald, pool, select = update_pool(evald_futures,
                                                                        models_evald,
                                                                        pool, select,
                                                                        data_file,
                                                                        db,
                                                                        sim_ids)

        print('Input models are finished. Making random models..')
        start_time = time.time()
        # Make random models & evolved models
        while models_evald < total_models_needed:
            working_jobs = get_working_jobs(evald_futures)
            # In some cases (lammps based), working_jobs always < max_workers
            # Ensure some structures are always in the queue for each worker so that
            # workers wont be idle if some step on master becomes bottle neck
            while working_jobs < max_workers and models_evald < total_models_needed:
                # make model
                if models_evald < num_initial_pop:
                    print("Submitting random job")
                    model_mech = "random"
                else:
                    print("Submitting evolved job")
                    model_mech = "evolved"

                # create the model then send it to the dask-workers for evaluation
                new_model = make_model(random_model_obj, evolve, select,
                                    pool, reg_id, model_type=model_mech)
                out = client.submit(full_eval, new_model, energy_code, Xsim_1)
                evald_futures.append(out)
                evald_futures, models_evald, pool, select = update_pool(evald_futures,
                                                                        models_evald,
                                                                        pool, select,
                                                                        data_file,
                                                                        db,
                                                                        sim_ids)
                working_jobs = get_working_jobs(evald_futures)

                if models_evald % pool_status_update == 0 and\
                        models_evald >= i_dict['population_limits']['pool']:
                    # print statements which output visualization information
                    if "selection_algorithm" in i_dict["select_params"]:
                        if i_dict["select_params"]["selection_algorithm"] ==\
                                "distance_from_pareto":
                            good_pool = pool.good_pool
                            good_pool_labels = [model.label for model in good_pool]
                            print(
                                "Current good_pool population models: "
                                f"{good_pool_labels}.")
                        elif i_dict["select_params"]["selection_algorithm"] ==\
                                "epsilon_moea":
                            pop_labels = [
                                model.label for model in pool.population.models]
                            archive_labels = [
                                model.label for model in pool.archive.models]
                            print("Current pool population models: "
                                f"{pop_labels}")
                            print("Current pool archive models:"
                                f"{archive_labels}")
                        elif i_dict["select_params"]["selection_algorithm"] ==\
                                "clustered_selection":
                            nd_pop_labels = [
                                model.label for
                                model in pool.population.non_dominated_models]
                            print(
                                "Current pool population non-dominated models: "
                                f"{nd_pop_labels}")
                    else:
                        good_pool = pool.good_pool
                        good_pool_labels = [model.label for model in good_pool]
                        print(
                            "Current good_pool population models: "
                            f"{good_pool_labels}.")

        # process extra calculations running in last batch
        while len(evald_futures) > 0:
            evald_futures, models_evald, pool, select = update_pool(evald_futures,
                                                                    models_evald,
                                                                    pool, select,
                                                                    data_file,
                                                                    db,
                                                                    sim_ids)

        # print statements which output visualization information
        if "selection_algorithm" in i_dict["select_params"]:
            if i_dict["select_params"]["selection_algorithm"] ==\
                    "distance_from_pareto":
                good_pool = pool.good_pool
                good_pool_labels = [model.label for model in good_pool]
                print(f"Current good_pool population models: {good_pool_labels}.")
            elif i_dict["select_params"]["selection_algorithm"] == "epsilon_moea":
                pop_labels = [model.label for model in pool.population.models]
                archive_labels = [model.label for model in pool.archive.models]
                print(f"Current pool population models: {pop_labels}")
                print(f"Current pool archive models: {archive_labels}")
            elif i_dict["select_params"]["selection_algorithm"] ==\
                    "clustered_selection":
                nd_pop_labels = [
                    model.label for model in pool.population.non_dominated_models]
                print(f"Current pool population non-dominated models: {nd_pop_labels}")
        else:
            good_pool = pool.good_pool
            good_pool_labels = [model.label for model in good_pool]
            print(f"Current good_pool population models: {good_pool_labels}.")

        print(f"Current operator probabilities: {select.operator_frequencies}")
        print('Done!')
        print('Total time: ', time.time() - start_time)

        client.shutdown()
        ```

    === "run_fx_mp.py"
        ```python
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
            sp.random.seed(seed)
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
                '%s%f')) % (2**32 - total_models_needed)
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
        ```

When you have either chosen or written your own FANTASTX python script, the last step is to run it:

#### Running FANTASTX locally

If you installed FANTASTX in a virtual environment, then first load your virtual environment. 

Then, if running FANTASTX Dask, start the Dask nanny and worker jobs in separate terminal windows:
```bash
dask-scheduler
dask-worker tcp://127.0.0.1:8786 --nprocs 6 --memory-limit 5GB
```
where the value for **nprocs** is the number of workers which will run, and value for the **memory-limit** is how much RAM is assigned to each worker.

Once the Dask nanny and workers have started, or if not running Dask, proceed by running the FANTASTX input file with python:
```python
python run_fx.py
```
if all goes well, then FANTASTX should be humming along!

#### Running FANTASTX on PBS or SLURM

When running on PBS or SLURM systems, you need to submit job script files to the job schedulers which will both allocate resources to FANTASTX and run the FANTASTX job! Here are sample job scripts for PBS and SLURM systems respectively, tested on Argonne's Carbon and LCRC (bdw) systems:

!!! example
    === "SLURM"
        ```bash
        #!/bin/bash

        #SBATCH --job-name=bulk_pdf_fantastx
        #SBATCH --account=ACCOUNT_NAME
        #SBATCH --partition=bdwall # partition we are running on
        #SBATCH --nodes=1 # here we are running a Dask job, so only need 1 node for our initial allocation
        #SBATCH --ntasks=1 # since our python script is not parallelized itself set number of tasks to 1
        #SBATCH --cpus-per-task=36 # our single task will use all 36 cpus on the node
        #SBATCH --mem=120gb # memory per node will be 120 gb
        #SBATCH -t 3-00:00 # requesting a time limit of three days
        #SBATCH --output=fantastx_xanes.out # stdout
        #SBATCH --error=fantastx_xanes.error # stderr

        export I_MPI_FABRICS=shm:tmi
        ulimit -s unlimited

        # load all the modules that you needed to install FANTASTX
        module purge
        module add StdEnv
        module delete intel intel-mkl intel-mpi
        module add gcc/9.2.0-pkmzczt
        module add intel-parallel-studio/cluster.2020.2-y7ijupg

        # load VASP
        export PATH=/soft/vasp/6.2.1/bdw_vaspsol:$PATH

        # load and activate our conda environment
        module add anaconda3/2021.05
        eval "$(conda shell.bash hook)"
        conda activate miniconda3
        conda list >> module.log

        # run FANTASTX!
        srun python run_fx.py > job.log
        ```
    === "PBS"
        ```bash
        #!/bin/bash
        #PBS -l nodes=1:ppn=4:gen6
        #PBS -l walltime=6:00:00 # total of 6 cpu hours
        #PBS -N al_gb_example
        #PBS -A ACCOUNT_NAME
        #PBS -o job.out # stdout
        #PBS -e job.err # stderr
        #PBS -j oe
        #PBS -m a

        # First change into the working directory on the nodes
        cd $PBS_O_WORKDIR

        # load our necessary modules
        module purge
        module load fftw3/3.3/impi impi intel lammps

        # activate our conda environment
        source activate ~/miniconda/envs/fx_521

        # run FANTASTX!
        PYTHONUNBUFFERED=1 run_fx.py > job.log
        ```


