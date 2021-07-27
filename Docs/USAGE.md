## FANTASTX usage:

### Grain boundary search

FANTASTX grain boundary (gb) geometry search investigates multiple grain boundary structures to find the low energy interface structure.
<br>

An experimental TEM image captures the orientation of both the grains correctly. However, the interface structure with strain induced defects are difficult to capture. FANTASTX uses genetic algorithm approach along with basin hopping to create new interface regions in a grain boundary structure in an attempt to find low energy structures that also match with the experimental grain boundary TEM image.

Single objective (only energy) or multi-objective optimization (energy and image matching) searches can be performed.

#### Preparation

* [YAML input file](#inputs)

* [Energy calculation](#energycode)

* [Raw TEM image](#tem_image)

* [Ingrained optimized parameters](#ingrained_params)

* [Inital grain boundary structure](#init_gb_poscar)

Single objective search with only aim to find low energy structures needs an input structure. This can be constructed manually by combining slabs of two grains or automatically generated using Ingrained package. See [here](#ingrained_examples) for examples.
See [here](#structure_file) for an example of POSCAR file of grain boundary created using ingrained.
<br>
Note that only the interface region will be manipulated using mating operations or basinhopping in search of low energy interface grain boundary structure.

The TEM simulation of this optimized structure has the best matching score with respect to experimental TEM image.
<br>
The details on how to use [Ingrained](https://github.com/MaterialEyes/ingrained) package are shown [here](#ingrained_examples) with some examples.
<br>
The vales of optimized parameters are also needed for FANTASTX

Input parameters for the algorithm are provided in a 'yaml' file. Some parameters are mandatory while others are optional. And based on cluster or grain boundary geometry, different parameters are needed. (Parameters not relevant to a search are voided by the algorithm).

The parameters are divided based on the modules that use them. Main keywords correspond to modules and important parameters. Within each module, there might be multiple sub-keywords. The structure of the keywords shall be maintained in the input file. For example, to specify the algorithm to stop after 10000 models, keywords structure_record, stopper and num_calcs should be specified with necessary indentations.

```YAML
structure_record:
    stopper:
        num_calcs: 10000
```

Refer to example [gb_input.yaml](#example_input_file) file for short description of all possible parameters with default values (if applicable) to keywords.

#### <a id='inputs'></a>inputs
* inputs
  * energy_files_path

#### <a id='structure_record'></a>structure_record

* structure_record
  * gb
    * init_gb_astr
    * iface_thickness
    * iface_z_mid
    * num_slices

  * max_bond_dist

  * min_dist
    * sp1_sp1
    * sp1_sp2
    * sp1_sp3
    * sp2_sp2
    * sp2_sp3
    * sp3_sp3

  * species
    * specie1
      * name
      * min_num
      * max_num
      * mu

  * stopper
    * num_calcs



  * pool_capacity

  * workers
  * energy_code
  *

#### <a id='initial_population'></a>initial_population

  * initial_population
    * total

#### <a id='pool_capacity'></a>pool_capacity

#### <a id='workers'></a>workers

#### <a id='energy_code'></a>energy_code

#### <a id='energy_obj_fn'></a>energy_obj_fn

#### <a id='energy_exec_cmd'></a>energy_exec_cmd

#### <a id='exp_sim_1'></a>exp_sim_1

#### <a id='exp_sim_1_params'></a>exp_sim_1_params

#### <a id='select_objective'></a>select_objective

#### <a id='weights'></a>weights

#### <a id='mating_constraints'></a>mating_constraints

#### <a id='basinhopping_constraints'></a>basinhopping_constraints

#### <a id='evolve_probabilities'></a>evolve_probabilities

#### <a id=''></a>

#### <a id='example_input_file'></a>gb_input.yaml file example with description

```YAML
# parameters pertaining to inputs.py module
inputs:
    energy_files_path:
    (path to energy files)

# parameters pertaining to structure_record.py module
structure_record:
    gb: # specifies shape as grain boundary & contains gb specific parameters
        init_gb_astr: /home/fx_run/POSCAR_gb # path to the initial grain boundary poscar
        iface_thickness: 5 # thickness of the interface region in Å
        iface_z_mid: 0.44 # fractional z-coordinate of mid point of two grains
        # NOTE: mating operations and random population generation for gb shape are done
        # directly in gb_ops class. So, mating parameters are provided here
        num_slices: 4 # number of slices to do while mating operation
        # fraction to use basinhopping; reminder uses mating operations
        hop_mate_frac: 0.4

    max_bond_dist: 4 # maximum allowed bond distance to a neighbor atom
    min_dist: # minimum bond distances between two atoms, species wise
      sp1_sp1: 2.3 # between two species1 & species1 atoms
      sp1_sp2: 1.5 # between two species1 & species2 atoms
      sp2_sp2: 1.2 # between two species2 & species2 atoms
      # NOTE: sp2_sp1 is an invalid keyword. Always lower index species first

    species: # Species information in the "interface region only"
        specie1:
            name: Al      # symbol of first species
            min_num: 26   # minimum number of atoms needed in interface
            max_num: 42   # maximum number of atoms allowed in interface
            mu: -3.35958525 # reference chemical potential of the species
        specie2:
            name: O
            min_num: 16
            max_num: 30
            mu: -6.7606965944

    stopper: # parameters on when to stop the algorithm
        num_calcs: 10000 # stop when 10000 at least calculations are done

# parameters pertaining to initial_population.py module
initial_population:
    total: 30  # Makes structures from input files and/or by random model generation

pool_capacity: 250 # Number of structures in good_pool (all parents)

# One objective function ; calculate energy & minimize objective function value
energy_code: lammps # either lammps or vasp
energy_obj_fn: mu_based # either mu_based or epa
# command to run lammps or vasp on worker. Example SLURM execution command -
energy_exec_cmd: srun --mpi=pmi2 lmp_mpi -in in.min

# Second objective ; optional
# For TEM image matching - 'GB_STEM'
# For PDF of a structure - 'PDF'
# keyword exp_sim_1 for first experimental simulation. Use exp_sim_2 if exp_sim_1 already exists.
exp_sim_1: 'GB_STEM'
exp_sim_1_params: # GB STEM simulation parameters
    dm3_path: # path to raw TEM .dm3 file (or raw image data file)
    progress_file: # path to progress_file from Ingrained optimization

select_objective: multi # multi for more than one objective function
weights: # weightage to each objective function in order
    - 7 # energy objective function
    - 5 # exp_sim_1 (if second objective function)
    - 5 # exp_sim_2 (if third objective function)

basinhopping_constraints:
    # maximum perturbation size for an atom from its initial position during basinhopping
    max_perturbation: 0.4 # in Å

# Following are specific to computing cluster of user (SLURM/PBS)
workers: # contains specifications for each dask worker job
    cluster: "SLURM" # Specify type of scheduler
    max_workers: 16 # Number of parallel calculations of models
    num_cores: 1 # cpus-per-task option
    total_mem: "2GB" # total memory for the job (--mem option)
    project_name: "AA-1234" # project under which job to be requested
    submit_to_queue: "general" # queue in which to submit job
    node_type: "ib0" # infiniband or haswell etc
    walltime: "10:00:00" # estimated entire walltime a worker job shall run
    job_extra: # any other PBS/SLURM submit options
        - "--ntasks=8"  # SBATCH options directly provided
        - "--nodes=1"
```
<br>
Following parameters are not needed for gb search.

```YAML
mating_constraints: # parameters for mating parents
    # fraction to use three parents for mating; reminder uses two parents
    num_parents_fraction: 0.1
    # fraction to use mirror and attach; reminder uses attach without mirroing
    attach_type_fraction: 0.3

# fraction to use different methods for generating child structures
evolve_probabilities:
    1: 0.35  # fraction to use basinhopping method - perturb atoms
    2: 0     # fraction to use basinhopping method - perturb atoms
    3: 0.45  # fraction to use mating method - mate by slice and attach
    4: 0.2   # fraction to use mating method - mate by swap atoms

```

### Nano clusters search
