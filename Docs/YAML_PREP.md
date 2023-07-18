## Preparing the YAML input file

The YAML file is essential to editing all FANTASTX settings. Here, we will discuss
all possible options that can be included in the YAML file. Each section will start
with the essential settings that must be included in order to run FANTASTX successfully,
before introducing non-essential settings that can be used if additional functionality
is required.

### General inputs (inputs)

***

**Mandatory** section which provides necessary file paths for read-in. 

#### Mandatory inputs

- **energy_files_path**: the path to the folder that contains all necessary inputs to perform either the VASP or LAMMPS simulations. In the case of VASP, this folder should contain the INCAR, POTCAR, KPOINTS, etc. In the case of LAMMPS, this folder should contain the interatomic potential files and the input file. 

#### Optional inputs

- **model_files_path**: the path to the folder containing any pre-existing structures that FANTASTX should use as its starting structures in the evolutionary process.

```YAML
inputs:
    model_files_path: # path to pre-existing structure folder
    energy_files_path:  # path to folder containing relaxation inputs
```

### Structure parameters and constrains (structure_record)

***

**Mandatory** section which tells FANTASTX what type of structure is being examined (such as a grain boundary or a cluster), provides necessary information such as the elements contained in the structure and their free energies, and also assigns distance constraints to structure manipulation operations.

#### Mandatory inputs

- **shape**: sub dictionary with a header corresponding to the type of structure being examined, and providing additional necessary information appropriate to that type of structure. Options are:
['gb', 'cluster', 'surface', 'molecule']
If looking at a 'gb' structure, the YAML information should look like:
```YAML
gb:
    init_gb_astr: # full path to the initial grain boundary structure
    iface_thickness: 2 # thickness in Å of the grain boundary interface
    iface_z_mid: 0.475 # mid-point of the grain boundary interface in lattice units.
    num_slices: 2 # how many slices to make when mating structures by slicing.
```

If looking at a 'cluster' structure, the YAML information should look like:
```YAML
cluster:
    box_abc: [20, 20, 20] # the size of the box which should contain the cluster
    max_dia: 8 # the maximum diameter of the cluster
```

If looking at a 'molecule' structure, the YAML information should look like:
```YAML
molecule:
    box_abc: [20, 20, 20] # the size of the box which should contain the molecule
    max_dia: 8 # the maximum diameter of the molecule
    fixed_species: ["Fe"] # optionaly. List of any species whose coordinates should be held fixed when basinhopping
```

If looking at a 'surface' structure, the YAML information should look like:
```YAML
surface:
    init_slabs_dir: # the path to the folder containing the initial surface slabs.
    surface_thickness: 1 # in Å
    substrate_thickness: 10 # in Å
    separation: 2 # separation in Å between the substrate top z and the surface bottom z
    constrain_z: True # boolean determining if the z coordinates of the surface should be kept fixed
    sd_cut_off: 10 # how far to freeze the substrate
    sd_no_z: False
    composition: ["Fe": 10, "Al": 2] # composition of the surface layer in a format parseable by pymatgen.
    num_slices: 2 # how many slices to make when mating structures by slicing
    hop_mate_frac: 0.3 # ratio between basinhopping and mating by slicing
```


- **species**: sub dictionary which sets the name of the species, the minimum and maximum number of atoms of that species in the structure, and the free energy *mu* of that species. The sub dictionaries should be labeled as *species1*, *species2*, ... *speciesN*, where *N* is the total number of species in the structure. The sub cards should be labeled: *name*, *min\_num*, *max\_num*, and *mu*.

#### Optional inputs

- **max_bond_dist**: minimum radius around any one atom which should contain at least one bond.

- **min_dist**: sub dictionary which sets minimum distance threshold for each pair of species. Sub cards are labeled as *spA\_spB*, where *A* and *B* correspond to *speciesA* and *speciesB*. The cards should be labeled in rising order, iterating over species *B* before incrementing species *A*, iterating over species *B* again, and so on until all possible combinations have been included. 

```YAML
structure_record:
    gb:
        init_gb_astr: # path to initial gb structure
        iface_thickness: 2
        iface_z_mid: 0.475
        num_slices: 2

    max_bond_dist: 4
    min_dist:
      sp1_sp1: 2.3
      sp1_sp2: 2.0
      sp2_sp2: 1.6
    species:
        species1:
            name: Al
            min_num: 5
            max_num: 12
            mu: -3.74328225
        species2:
            name: O
            min_num: 8
            max_num: 16
            mu: -9.973691833333334
```

### Energy Calculations (no card)

***

**Mandatory** section, as single-objective structure search uses
energy values as the objective function.

#### Mandatory inputs:
- **energy_code**: which energy calculator to use, 'LAMMPS' or 'VASP'.

- **energy_exec_cmd**: command which will run the energy executable.

#### Optional inputs:
- **energy_code_params**: Extra parameters which can be sent to either LAMMPS or VASP. Each code has one such optional parameter. These extra parameters are:
    - LAMMPS: **atom_style**: default is 'charge'. 
    - VASP: **resubmit**: number of times to resubmit a VASP job if it did not converge the first time.

```YAML
energy_code: 'lammps'
energy_exec_cmd: 'mpirun -np 2 lmp_mpi -in in.min'
energy_code_params:
    atom_style: 'charge'
```

### Experimental Simulations (no card)

***

**Optional** section which only needs to be included if performing single-objective structure search. **Mandatory** section if performing multi-objective structure search. 

#### Mandatory inputs:
- **exp\_sim\_#**: The type of experimental simulation which will be performed. Options are:
['PDF', 'GB_STEM', 'PRISM', 'GSASII', 'XANES']
The \# character should be replaced with the ordering of the experimental simulation objectives. The first experimental simulation objective will be inputted as exp\_sim\_1, the second as exp\_sim\_2, etc.

- **exp\_sim\_#\_params**: The experimental simulation parameters. These are different for each experimental simulation method.

Optional inputs: None

Below are sample YAML preparations for each possible experimental simulation, including all optional inputs to exp\_sim\_params.

```YAML
exp_sim_1: 'GB_STEM'
exp_sim_1_params: # GB STEM simulation parameters
    dm3_path: # path to raw TEM .dm3 file (or raw image data file)
    progress_file: # path to progress_file from Ingrained optimization
```

```YAML
exp_sim_1: 'XANES'
exp_sim_1_params: # XANES simulation parameters
    exp_base_ref_filepath: # path to the experimental reference spectra
    exp_exc_ref_filepath: # path to the experimental excited reference spectra. Only needed for difference calculations.
    comp_base_ref_filepath: # path to the computational reference spectra. Only needed for difference calculations.
    simulation_code: 'FDMNES' # currently only FDMNES is supported
    input_yaml_filepath: # path to yaml file containing input parameters for running the XANES simulation
    comparison_spectra_type: 'difference' # 'difference' or 'direct'
    exec_cmd: 'mpirun -np 4 fdmnes' # command to run the simulation
    spectra_distance_metric: 'rmse' # distance metric with which to compare spectra. For full options, see the 'distance_metric' item in the fingerprinting_params section
    spline_mesh_params: [7110, 7160, 0.5] # [min, max, step size] of the smoothing spline
    convolution_params: ['lorentzian',[1.33, 15., 23.5, 23.5, -8], True]
        # These items correspond to:
        #   convolution_type: gaussian or lorentzian
        #   convolution_params: parameters governing the convolution, for lorentizan params description see FDMNES paper
        #   extract_cutting_energy: whether or not the final lorentzian convolution parameter should be extracted from the calculation fermi energy
```

### Clustering (cluster_params)

***

**Optional** section that should only be included if performing clustering. Only the *epsilon_moea* and *clustered_selection* selection algorithms currently utilize clustering.

#### Mandatory inputs:

- **type**: whether to perform *compositional* or *hierarchical* clustering. 

#### Mandatory hierarchical clustering inputs:

If performing hierarchical clustering, then many more inputs are required. These are:

- **distance_calculation**: which distance metric to use for clustering, *fingerprint* or *xsim*. If you use *xsim*, then two models will be compared to each other, and the similarity of the two modeled experimental spectra will be used to calculate the distance between the models as 1-*similarity*.

- **linkage_method**: which method is used to create links in the dendrogram. Options are: 
['single', 'complete', 'average', 'weighted', 'centroid','median', 'ward']
For a detailed description of each option, refer to: [scipy.cluster.hierarchy.linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage)

- **cutoff_type**: which method to use to assign flat clusters from dendrogram. Options are:
['distance', 'inconsistent', 'maxclust_monocrit']
For a detailed description of each option, refer to: [scipy.cluster.hierarchy.fcluster](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster)

- **max_cluster**: maximum number of flat clusters created.

- **min_cluster_occupancy**: Minimum number of structures that should reside in each flat cluster.

- **min_clusters**: minimum number of flat clusters created. This is a hard limit.

- **max_incons_cutoff**: Maximum inconsistency threshold that will be used to assign flat clusters. Will overwrite the *max_clusters* and *min_cluster_occupancy* options.

- **max_dist_cutoff**: Maximum distance threshold that will be used to assign flat clusters. Will overwrite the *max_clusters* and *min_cluster_occupancy* options.

```YAML
cluster_params:
    type: 'hierarchical'
    distance_calculation: 'fingerprint'
    linkage_method: 'ward'
    cutoff_type: 'inconsistent'
    max_clusters: 10
    min_clusters: 2
    min_cluster_occupancy: 4
    max_incons_cutoff: 5.0
    max_dist_cutoff: 2.0
```

### Fingerprinting (fingerprint_params)

***

**Optional** section that should only be included if making fingerprinting-based structural comparisons, for either uniqueness or clustering purposes. All selection algorithms can use fingerprinting for uniqueness quantification.

#### Mandatory inputs:

- **label**: which fingerprinting method to use. Fully implemented options are:
['bag-of-bonds', 'valle-oganov', 'rematch-soap', 'average-soap', 'mbtr', 'ewald-sum-matrix', 'sine-matrix']

- **distance_metric**: if using the 'mbtr', 'ewald-sum-matrix', or 'sine-matrix' fingerprints, determines which distance metric to use to make fingerprinting comparisons. Options are:
["manhattan", "euclidean", "cosine", "laplacian", "gaussian", "mae", "mse", "rmse", "r2_score"]

#### Optional inputs:

- **tolerances**: a float (or a 2-item list in the case of the *bag-of-bonds* fingerprint) which determines the distance threshhold within which two structures should be considered nearly identical. For an explanation of the *bag-of-bonds* threshhold, refer to the [original paper]().

- **zbounds**: if performing grain boundary or surface structure search, then it can be helpful to limit the active fingerprinting region to a more narrowly defined region, which contains only the active structure space and a few layers of atoms on either side. 

- **rem_vac**: if performing cluster or surface structure search, then it can be helpful to remove the vacuum surrounding the cluster or surface before performing fingerprinting. Will leave a 2Å thick layer of padding on all sides.

- **fingerprint\_values**: additional inputs for each fingerprint. The labeling follows:
    - *valle-oganov*: **comp\_values**
    ```YAML
    comp_values:
        n_top: None
        dE: None
        cos_dist_max: 1e-3
        rcut: 10.
        binwidth: 0.05
        pbc: [True, True, True]
        sigma: 0.05
        nsigma: 4
        recalculate: False
    ```

    - *mbtr*: **mbtr\_values**
    ```YAML
    mbtr_values:
        k2:
            geometry: {"function": "inverse_distance"}
            grid: {"min": 0, "max": 2, "n": 100, "sigma": 0.08}
            weighting: {"function": "exp", "scale": 0.5, "threshold": 0.001}
        k3:
            geometry: {"function": "cosine"}
            grid: {"min": -1, "max": 1, "n": 100, "sigma": 0.03}
            weighting: {"function": "exp", "scale": 0.5, "threshold": 0.001}
    ```

    - *sine-matrix*: **sine-matrix\_values**
    ```YAML
    sine-matrix_values:
        permutation: 'sorted_l2'
        n_atoms_max: 24
    ```
    - *ewald-sum-matrix*: **ewald-sum-matrix\_values**
    ```YAML
    ewald-sum-matrix_values:
        permutation: 'sorted_l2'
        n_atoms_max: 24
    ```
    - *average-soap* or *rematch-soap*: **soap\_values** and **kernel\_generator\_values**
    ```YAML
    soap_values:
        rcut: 5.0
        nmax: 9
        lmax: 6
        sigma: 0.5

    kernel_generator_values:
        metric: 'linear' # sklearn distance metric used for local similarity calculation
        alpha: 1 # used only for the rematch kernel. alpha << 1 approaches best-match, alpha >> 1 approaches average
        threshold: 1e-6 # used only for the rematch kernel. Converge threshold for the sinkhorn algorithm
    ```
    The only fingerprint which does not have the any additional inputs is the *bag-of-bonds* fingerprint. 

A full **fingerprint\_params** card would look like: 
```YAML
fingerprint_params:
    label: 'mbtr' # "bag-of-bonds", "valle-oganov", "rematch-soap", "mbtr"
    mbtr_values:
        k2:
            geometry: {"function": "inverse_distance"}
            grid: {"min": 0, "max": 2, "n": 100, "sigma": 0.08}
            weighting: {"function": "exp", "scale": 0.5, "threshold": 0.001}
        k3:
            geometry: {"function": "cosine"}
            grid: {"min": -1, "max": 1, "n": 100, "sigma": 0.03}
            weighting: {"function": "exp", "scale": 0.5, "threshold": 0.001}
    distance_metric: laplacian
    tolerance: .0075 # [0.02, 0.7] # (length 2 array) for bag-of-bonds, (float) for valle-oganov or rematch soap
    zbounds: [6.877 , 8.877] # [z_low, z_high]. optional parameter which will trim the fingerprinting region of the supercell to be bounded by the provided values.
    rem_vac: False
```

### Selection population (population_limits)

***

**Mandatory** section defining the parameters which govern the active group of models, as well as the maximum
number of models which will be evaluated. The relevant card is population_limits.

#### Mandatory inputs:
- **initial_population**: number of models to create before performing any mating operations.

- **total_population**: total number of models which will be evaluated.

- **pool**: number of models contained in the active mating pool of models. For distance_from_pareto, this is the number of models in the good_pool. For epsilon_moea and clustered_selection, this is the number of models in the population.

```YAML
population_limits:
    initial_population: 30
    total_population: 100
    pool: 40
```

### Selection Algorithm (select_params)

***

**Mandatory** section which sets all selection algorithm parameters. This includes whether single- or multi-objective search will be conducted, which selection algorithm to use, possible mating operations and their relative frequencies, and specific selection algorithm options. 
Three selection algorithms have been included: *distance_from_pareto*, *epsilon_moea*, and *clustered_selection*.

#### Mandatory inputs:

- **objective_fn_type**: the type of structure search being performed: *single* or *multi* objective.

#### Optional inputs:

- **operators**: which mating operators are available. Options are:
    ['perturb_sites', 'fraction_slice', 'perturb_comp', 'fraction_slice_same_cluster', 'fraction_slice_dif_cluster']
The later two options are only available if clustering is included via the cluster_params card. By default, the operators are set to ['perturb_sites', 'fraction_slice'].
- **operator_frequencies**: the relative frequency at which FANTASTX should select each mating operator. Should be in the same order as the operators. A mating operator choice of [0.4, 0.6] would select operator 1 40% of the time, and operator 2 60% of the time. By default, the operator frequencies are set to be equal for all operators.
- **operator_assignment**: can choose to have the operator frequencies evolve over time, or can keep them fixed. Choices are: 'auto-adaptive' or 'fixed'.

#### Selection algorithm specific inputs:

- *distance_from_pareto*:
    - **weights**: parameters which scale the objective function values in the distance from pareto calculations. A weight of one corresponds to the raw objective function value.
    - **num_required_above_50**: Number of models in the good pool which are required to have a selection probability of 50% or higher. 
    - **num_models_before_pareto**: Number of models that are required to be in the good pool before the *distance_from_pareto* algorithm is used. Until then, selection probabilities are determined purely from normalized objective function values.
- *epsilon_moea*:
    - **epsilons**: parameters for binning the objective function space into the 'epsilon grid'. Should be in the same order as the objectives. Used for both `epsilon_dominance` and (optionally) `pareto_dominance`, binning the latter case for structural comparisons.
- *clustered_selection*:
    - **epsilons**: parameters for binning the objective function space into the 'epsilon grid'. Not necessary if using simple pareto_dominance for clustered selection. Should be in the same order as the objectives. Currently only supports two objectives.
    - **cluster_params**: not a *select_params* input, but is mandatory to include elsewhere in the input YAML file if performing *clustered_selection*.
    - **dominance_algorithm**: `pareto_dominance` or `epsilon_dominance`. Either algorithm will perform structural comparison if a comparator is provided, and
    will perform the comparison only within an epsilon box if epsilons are provided.

```YAML
select_params:
    selection_algorithm: 'epsilon_moea' # distance_from_pareto, epsilon_moea, clustered_selection
    objective_fn_type: 'multi' # single or multi
    epsilons: [1, 1]
    operators: ['perturb_sites', 'fraction_slice_same_cluster', 'fraction_slice_dif_cluster'] # also fraction_slice, and perturb_comp
    operator_assignment: 'auto-adaptive' # auto-adaptive or fixed
    operator_frequencies: [0.2, 0.25, 0.55] # must add up to 1
```

### Basinhopping Operations (basinhopping_constraints)

***

**Optional** section which defines additional basinhopping behavior of FANTASTX.

#### Mandatory inputs

None

#### Optional inputs

- **indices_fraction**: fraction of total atoms to perturb when performing basinhopping.

- **max_perturbation**: maximum distance any single atom can be perturbed before relaxation.

```YAML
basinhopping_constraints:
    indices_fraction: 0.5
    max_perturbation: 0.5
```


### Parallelization

***

**Mandatory** section which sets the parallel behavior of FANTASTX. Currently two parallel architectures are supported: using native python multiprocessing, or Dask. Both multiprocessing and Dask can be used on high-performance computing systems (*SLURM* or *PBS*) or local computing systems.

!!! note
    If you are running DASK on a single computer, you need to open two separate terminal windows. In the first terminal window, run the command:

        dask-scheduler

    In the second terminal window, run the command:

        dask-worker tcp://127.0.0.1:8786 --nprocs 6 --memory-limit 5GB

    where the value for nprocs is the number of workers which will run, and value for the memory-limit is how much RAM is assigned to each worker.

    If using multiprocessing, you only need to run FANTASTX as usual.

#### Mandatory inputs

Multiprocessing and Dask:

- **max_workers**: number of parallel calculations of models which will be performed.

Dask:

- **cluster**: tell FANTASTX which DASK architecture to use. Options are:
["SLURM", "PBS", "local"]

#### Optional inputs (only relevant to Dask)

See the YAML example below. 

!!! example "workers"
    ``` yaml
    workers: # contains specifications for each dask worker job
        cluster: "SLURM" # Specify type of scheduler. Can be "PBS", "SLURM", or "local".
        submit_queue: "bdwall"
        max_workers: 4 # Number of parallel calculations of models
        num_cores: 36 # cpus-per-task option
        total_mem: "80GB" # total memory for the job (--mem option)
        project_name: "XRS_FANTASTX" # project under which job to be requested
        node_type: "ib0" # infiniband or haswell etc
        walltime: "24:00:00" # estimated entire walltime a worker job shall run
        job_extra: # any other PBS/SLURM submit options
            - "--ntasks-per-node=1"
            - "--cpus-per-task=36"
            - "--nodes=1"
        env_extra: # any extra environment lines which need to be included
            - "export MPI_FABRICS=shm:tmi"
        header_skip:
            - "-n "
            - "--cpus-per-task"
    ```