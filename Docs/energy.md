# Structural Relaxation via MD or DFT

An important consideration for any FANTASTX run is the choice of the method which will be used to structurally relax each proposed atomic configuration. FANTASTX supports the use of both classical molecular dynamics (MD) via the LAMMPS code, and the use of density functional theory (DFT) via the VASP code. Interfacing these codes with FANTASTX requires the user to both install the code (or load the appropriate high-performance computing module) and provide FANTASTX with input files specific to the particular code of choice.

### VASP

***

!!! note
    VASP is not an open-source code. If you want to use VASP, make sure that you obtain a license beforehand.

After you have installed VASP on your system, all that you need to provide to FANTASTX is a few necessary VASP input files. At the bare minimum, these are the VASP **POTCAR**(s) and **INCAR**. If you do not include the *KSPACING* tag in your INCAR, then you must also include a **KPOINTS** file. This is recommended regardless for production runs, as the *KSPACING* tag offers inferior functionality to the use of full **KPOINTS** files.

The **POTCAR** files contain the VASP pseudopotentials, essential to any VASP calculation. FANTASTX follows a strict naming convention for POTCARS: each POTCAR must be named POTCAR_X, where X is the atomic species that the POTCAR corresponds to. For instance, the POTCAR of iron must be named POTCAR_Fe.

To assemble your **INCAR**, refer to the VASP [INCAR](https://www.vasp.at/wiki/index.php/INCAR) summary page.

!!! note
    Make sure that you choose INCAR threshholds which are strict enough that they will be sufficient for a high throughput run. If you are debating between two cutoff values, typically you should choose the higher of the two values.

!!! example "INCAR"
    ```bash
    # General Setup
    System = FE(CN)
    ALGO = Fast
    ENCUT = 420.0 eV

    # Ionic relaxation
    IBRION = 2
    ISIF = 2
    NSW = 200

    # Electronic Relaxation (SCF)
    ISMEAR = 1
    SIGMA = 0.1
    ISPIN = 2
    EDIFF = 1.0E-4
    NELM = 200
    NELMIN = 5
    LREAL = A
    LASPH = .TRUE.

    # Parallelization
    LPLANE = .TRUE.
    NCORE = 6
    ```

Finally, below is an example **KPOINTS** file.

!!! example "KPOINTS"
    ```bash
    Automatic kpoint scheme
    0
    Gamma
    1.0 1.0 1.0
    ```

When you have assembled these files, place them in a sub-directory within the folder you are running the FANTASTX calculation in. The relevant lines that you must include within the FANTASTX input YAML file are:

```yaml
inputs:
    energy_files_path: PATH_TO_VASP_FILE_SUB_DIRECTORY
```

and

```yaml
energy_code: vasp
energy_exec_cmd: "srun -N1 -n36 vasp_gam" # example VASP run command for the vasp_gam compilation
```

### LAMMPS

***

!!! note
    Unlike VASP, LAMMPS is an open source code. Any user should be able to download and install LAMMPS themselves on both their local machine as well as high-performance computing systems.

After you have installed LAMMPS on your system, the only remaining step is providing FANTASTX with a few necessary LAMMPS input files. Unlike VASP, only a single LAMMPS specific input file is needed. While in a typical LAMMPS simulation the input file can be freely named, FANTASTX enforces the strict convention that this file should be called **in.min**. In addition to this LAMMPS specific input file, the user must also provide the files for the interatomic potential (or force field) that they will be employing. For instance, in the below example **in.min** file, the target atomic species are both Al and O, and the interatomic potential file is called *ffield.comb3.NiAlO*.

For the **in.min** file, make sure that you are outputting the potential energy per atom of the configuration. An easy way to do this is to add a *compute* to the file for *pe/atom*, such as in the example below. In this example, which is being applied to a grain boundary structure search, certain regions are also being held fixed, with a force vector of [0.0, 0.0, 0.0].

!!! example "in.min"
    ```bash
    units metal
    dimension 3
    boundary p p p
    atom_style  charge
    read_data in.data


    mass 1 26.9815386
    mass 2 15.9994

    pair_style comb3 polar_off
    pair_coeff * * /home/vkolluru/Examples/fx_lammps/ffield.comb3.NiAlO Al O

    neighbor 4.0 bin
    neigh_modify one 2000
    neigh_modify delay 0 every 10 check yes

    region top_grain block INF INF INF INF 23.8 45
    region bottom_grain block INF INF INF INF 1 19.1
    group top region top_grain
    group bot region bottom_grain
    group both_grains union top bot
    fix ff both_grains setforce 0.0 0.0 0.0

    minimize 0.0 1.0e-6 1 1

    compute peatom all pe/atom
    dump myDump all custom 1000 rlx.str id type x y z c_peatom

    minimize 1.0e-6 1.0e-6 5000 10000
    ```


