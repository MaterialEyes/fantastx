# Bulk structure search: a PDF example

## Bulk search

FANTASTX bulk search investigates bulk structures to find the configuration which is both low in energy and has minimal mismatch with the experimental data. This bulk search can utilize either "position only" optimization or complete optimization of the cell lattice vectors and atomic positions. This increase in flexibility is frequently crucial due to the mismatch between DFT/MD bond lengths and experimental bond lengths.

To account for variable unit cells, bulk search in FANTASTX implements a flexible protocol which can mate parents that have different unit cells. This protocol for mating two parents is as follows:

1. Rearrange the lattice vectors of parent 1 so that their lattice vectors (**a**,**b**, **c**) have the same ordering in norm as the lattice vectors of parent 2
2. Orient both parent 1 and parent 2 such that their lattice vectors are roughly aligned with the principal axes: lattice vector **a** is aligned with the $\hat{x}$-axis, lattice vector **b** lies in the $\hat{x}-\hat{y}$ plane, and lattice vector **c** has a positive z component.
3. Create a child lattice which is either the direct average of the two parent lattices, or a random linear interpolation of them.
4. Randomly determine a lattice vector (**a**, **b** or **c**) along which both parents will be cut.
5. Randomly translate each parent along their lattice vectors. The lattice vectors which will be randomly translated along are themselves randomly chosen. While the probabilities of selecting each lattice vector can be chosen by the user, currently these probabilities are set to be 90% for the lattice vector along which cutting will occur, and 5% for each of the other two lattice vectors.
6. Determine the fractional cutting point at which to slice both parents. This point is determined to be the average of the two structure's average fractional coordinates for the chosen lattice vector.
7. Cut the two structures into two slices at the cutting point.
8. Splice two opposing slices from the two structures (either slice 1 from parent 1 and slice 2 from parent 2, or slice 2 from parent 1 and slice 1 from parent 2), randomly mirroring each slice before splicing.
   1. When splicing, if any atoms will overlap with each other then remove one of them at random. 
   2. If the composition does not satisfy constraints, then attempt to correct the composition. If removing atoms, then remove them at random. If adding atoms, then add them from either the discarded slice from parent 1 or the discarded slice from parent 2.
9. Repeat steps 4-8 until either success is reached or a threshold number of attempts is crossed.

Beyond mating two bulk parents, FANTASTX also currently supports the following genetic operations for bulk geometries:

1. Basin hopping (the *perturb_sites* operator)
2. Compositional mutation (the *perturb_comp* operator)

Bulk search works with any of the experimental methods that FANTASTX supports, though it has been especially designed to work with XRD, PDF and XANES experimental data.

## PDF simulations

PDF is an excellent experimental method for non-destructively probing the atomistic structure of materials, recording both real-space and reciprocal-space information. for directly observing the atomic structure of materials, with atomic resolution. FANTASTX utilizes the [Diffpy-CMI](www.diffpy.org) Complexing Modeling Framework to conduct PDF simulations. For technical details beyond the YAML input parameters which are needed to conduct PDF simulations using FANTASTX, which will be covered [later](#pdf_yaml_inputs) in this example, please refer to the diffpy documentation. 

The metric which is used to quantify the similarity between the simulated and experimental pdf data is by default the $\Chi^2$ value. Other metrics can be implemented, including distances derived from Laplacian or Gaussian similarity kernels.

## Example

In this example, we will use FANTASTX to invert experimental PDF data which corresponds to amorphous blue-layer IrO2. While the original reference (see [this]() publication) looked at representative IrO2 clusters, here we will be improving on this by looking at bulk amorphous IrO2. 

### Preparation

Unlike some of the more complex geometries, preparing a bulk search is relatively simple. No grain data needs to be extracted, or molecular fragments determined. Instead, the only steps are as follows. 

#### Obtaining initial bulk atomic configurations

If accelerating the search by starting from reference bulk structures, then the first step is to extract reasonable reference atomic configurations. Typical targets are structures from the [Materials Project]() or structures which arose from other simulations or FANTASTX runs. Each of these initial configurations must be saved in either the POSCAR or .cif format for input to FANTASTX. 

#### Creating the energy optimization input files

The second step is to decide what method will be used to relax the candidate structures and measure their formation energies. As mentioned [elsewhere](), FANTASTX currently supports both LAMMPS and VASP. 

If using LAMMPS, then the LAMMPS input file must be provided in the directory linked to in the YAML file (under **energy_files_path**), and must be named **in.min**. The user must also provide file(s) for the interatomic potential or force field, but these are not copied into the calculation's relax directory, and can be located anywhere accessible in the filesystem. Instead, the path to the potential inside the input file must be complete and not a relative path.

If using VASP, then the VASP INCAR, KPOINTS, and POTCAR files must be provided. The POTCAR files must be named POTCAR_(element name 1), POTCAR_(element name 2), etc. For instance, when running a calculation with Cd and Te, two POTCARS must be provided, POTCAR_Ir and POTCAR_Te. 

!!! note

​	If you wish to perform variable unit cell bulk search in VASP, make sure that ISIF=3 is set in the INCAR file. Also ensure that ENCUT is set to be equal to 1.3*ENMAX. 

#### Preparing the YAML input file

The final step is the preparation of the YAML input file. A complete guide to the preparation of FANTASTX YAML input files can be found [here](). 

##### Bulk YAML inputs

In this case, as we are running a bulk structure search, our YAML must include distinct information pertinent to bulk geometries. Under the **structure_record** keyword, a subsection titled **bulk** must be included which provides all bulk specific structural information:

```YAML
structure_record:
    bulk:
        box_abc: [6.245, 6.245, 6.245]
        box_angles: [90, 90, 90]
```

The **box_abc** keyword specifies the lengths of the lattice vectors (a, b, c) which are used by FANTASTX to construct random bulk structures if necessary.

The **box_angles** keyword specifies the lattice angles ($\alpha$, $\beta$, $\gamma$) which are used by FANTASTX to construct random bulk structures if necessary.

##### <a id='pdf_yaml_inputs'></a>PDF YAML inputs

In addition to grain boundary specific inputs, our YAML file must also include inputs specific to (S)TEM simulations:

```YAML
exp_sim_1: "PDF"
exp_sim_1_params:
    exp_pdf_file: /PATH/TO/PDF/DATA/FILE
    Qmin: 1.0
    Qmax: 50.0
    Biso_val: 0.9 # 8*pi**2*0.009
    scale: 1.2
    delta2: 2.3
    qdamp: 0.3
    xmin: 0.5
    xmax: 8.0
    minimize_method: L-BFGS-B
    coord_tol: 0.05
    var_bounds:
        Biso_val: [.01, 2.1]
        scale: [0.7, 2.0]
        delta2: [1.8, 5.0]
        qdamp: [.001, 1.0]
        a: 0.02
        b: 0.02
        c: 0.02
```

The **exp_sim_1** keyword tells FANTASTX what kind of experimental simulation is being employed. Here of course we use the "PDF" keyword since we are running (S)TEM simulations to compare to experimental TEM data. 

The **Qmin** and **Qmax** keywords set the lower and upper bounds of the q-grid for the calculated F(Q).

The **Biso_val** keyword sets the default crystallographic Debye-Waller B-factor, equal to 8$/pi^2$*$/langle u^2 /rangle$ where $/langle u^2 /rangle$ is the average mean-squared atomic displacement.

The **scale** keyword sets the default structure scale factor.

The **delta2** keyword is the coefficient for (1/r**2) contribution to peak sharpening.

The **qdamp** keyword sets the PDF Gaussian dampening factor due to limited q-resolution.

The **xmin** and **xmax** keywords set the real-space range across which the experimental and simulated data will be compared.

The **minimize_method** keyword sets the scipy optimization method that will be used to adjust the above parameters (within set bounds) to their optimal settings.

The **coord_tol** keyword

The **var_bounds** keyword and its sub-keywords sets the bounds within which any of the optimizable parameters can be adjusted.  All of these bounds except those for the lattice vectors (see the below note) must be provided as 2-item lists or tuples.

!!! note
    The **var_bounds** keyword can include sub-keywords *a*, *b* and *c* corresponding to each of the lattice vectors. These bounds are a unique case in that they can be provided as either 2-item iterables or single floats. If 2-item iterables (lists or tuples), then they set the lower and upper bounds respectively within which scaling of the lattice vector can occur. If a float instead, then this float sets these limits by adding or subtracting the float-set percentage of the current lattice vector length. 

!!! note

​	Here all of these inputs are provided using the exp_sim_1 keywords (**exp_sim_1** and **exp_sim_1_params**) since we are only employing one experimental simulation method. If we were employing an additional experimental reference, then either those simulation params would be provided under the exp_sim_2 keywords or the STEM inputs would be moved to exp_sim_2.

#### <a id='example_input_file'></a>Full YAML file

This full YAML file for this example contains other information relevant to the specific atoms (Ir and O), population information for the structure search, basinhopping constraints, fingerprint and clustering parameters, and inputs to the selection algorithm. Additionally, it includes the inputs to Dask for parallelizing the FANTASTX run, here set up to run on LCRC here at Argonne National Laboratory. For information on preparing each of these sections, refer to the [YAML preparation](../YAML_PREP) section of the documentation. This example has also been run on NERSC using the NERSC-specific set-up comprised of dask-mpi, jupyter and Docker. For information on running on NERSC, refer to [this]() example notebook. 

```YAML
inputs:
    energy_files_path: PATH/TO/VASP/FILES
    model_files_path: PATH/TO/USER/PROVIDED/STRUCTURES

structure_record:
    bulk:
        box_abc: [6.245, 6.245, 6.245]
        box_angles: [90, 90, 90]

    max_bond_dist: 3.5
    min_dist:
        sp1_sp1: 2.4 # Ir crystal bond length is 2.74A using PBE
        sp1_sp2: 1.7 # Ir-O bond length from IrO2 is 2.02A using PBE
        sp2_sp2: 1.4 # O2 bond length is 1.23A using PBE, we don't want O2
    max_dist:
        sp1_sp1: 3.1
        sp1_sp2: 2.3
        sp2_sp2: 1.9
    species:
        species1:
            name: Ir
            min_num: 7
            max_num: 7
            mu: -11.2471 # O-rich
            max_bonds: 8
        species2:
            name: O
            min_num: 14
            max_num: 22
            mu: -4.9480 # O-rich
            max_bonds: 2

population_limits:
    initial_population: 30
    total_population: 1000
    pool: 30

energy_code: vasp
energy_exec_cmd: mpirun -np 36 vasp_std > vasp_job.log

exp_sim_1: "PDF"
exp_sim_1_params:
    exp_pdf_file: PATH/TO/EXPERIMENTAL/PDF/DATA/FILE
    Qmin: 1.0
    Qmax: 50.0
    stretch: 1.0
    Biso_val: 0.9
    scale: 1.2
    delta2: 2.3
    qdamp: 0.3
    xmin: 0.5
    xmax: 8.0
    minimize_method: L-BFGS-B
    coord_tol: 0.05
    var_bounds:
        Biso_val: [.01, 2.1]
        scale: [0.7, 2.0]
        delta2: [1.8, 5.0]
        qdamp: [.001, 1.0]
        a: 0.02
        b: 0.02
        c: 0.02

basinhopping_constraints:
    max_perturbation: 0.5

fingerprint_params:
    label: "bag-of-bonds"
    tolerance: [0.02, 0.7]

select_params:
    selection_algorithm: distance_from_pareto
    objective_fn_type: multi
    num_required_above_50: 15
    num_models_before_pareto: 30
    operators: [
            "perturb_sites",
            "fraction_slice"
        ]
    operator_assignment: "fixed"
    operator_frequencies: [0.45, 0.55] # make sure add up to 1, and has same length as 'operators'. Should be [0.87, 0.13] for original gb implementation.

workers:
    cluster: "SLURM"
    submit_queue: "bdwall"
    max_workers: 10
    num_cores: 1
    total_mem: "120GB"
    project_name: PROJECT_NAME
    node_type: "ib0"
    walltime: "24:00:00"
    processes: 1
    env_extra: ["export I_MPI_FABRICS=shm:tmi"]
    job_extra:
        - "--nodes=1"
    header_skip:
        - "-c "
```

