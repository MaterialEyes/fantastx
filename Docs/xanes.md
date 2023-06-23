FANTASTX supports multiple XANES simulation codes. Currently those are **FDMNES** and **FEFF**, and **VASP** and **ORCA** will be added in the future. All of these simulation codes are (or will be) supported via the package **xtk**, another MaterialEyes project. When you have decided which XANES code to use, a key ingredient to the FANTASTX run is the yaml input file which assigns all XANES simulation parameters. Below are sample yaml input files for each of the supported simulation codes.

!!! important
    When performing XANES-driven FANTASTX structure search, a few general inputs are required. These must be included in exp_sim_X_params section of the general FANTASTX yaml. These parameters are:

    - `exp_base_ref_filepath`: The path to the experimental reference spectra

    - `exp_exc_ref_filepath`: # path to the experimental excited reference spectra. Only needed for difference calculations.

    - `comp_base_ref_filepath`: # path to the computational reference spectra. Only needed for difference calculations.

    - `comparison_spectra_type`: `"difference"` or `"direct"`

    - `exec_cmd`: the SLURM or MPI command which will run the actual simulation code

    - `spectra_distance_metric`: the distance metric with which to compare spectra

    - `spline_mesh_params`: the [min, max, step size] of the smoothing spline

    - `convolution_params`: the [`convolution_type`, `convolution_params`, `extract_cutting_energy`] inputs. These tell FANTASTX whether to use `"gaussian"` or `"lorentzian"` convolution, the various broadening parameters of the convolution (float for `"gaussian"` convolution, length 5 list for `"lorentzian"` convolution), and whether or not the final `"lorentzian"` convolution parameter should be extracted from the calculation fermi energy.

### FDMNES yaml file

***

Below is a sample **FDMNES** yaml file. General information which must be included in all XANES yaml input files, regardless of the simulation method, is at the top. FDMNES specific inputs are provided in the sub-section **fdmnes_cards**. As noted in the YAML itself, two conventions need to be followed:

- To include a card in the yaml file, but not in the final input file, assign it a value of `null`.

- To only include a card, with no associated value, such as including the line **Green**, assign it a value of `"include"`.

Additionally, there are two special inputs: **Atom** and **Atom_conf** that have slightly different handling:

- **Atom**: use a dictionary as the input.

- **Atom_conf**: use the format of {"Z": orbital description}. This means do *NOT* include the number of sites, or their ids. 

It should also be noted that convolution is performed separately, inside of FANTASTX, so you must define the convolution params inside the main yaml file. 

!!! example "fdmnes.yml"
    ```YAML
    # General XANES information which will be converted into cards internally. This section
    # is the same for all XANES yamls.
    cluster_radius: 5.0
    structure_type: "molecule" # Molecule or Crystal
    core_hole_site_element: "Fe"
    core_hole_site_id: 1 # which target element site is the center of the calculation (1 .. n_element)
    edge: "K"

    # Direct FDMNES cards
    fdmnes_cards:
        Atom: null
        Atom_conf: {"26": "3 3 2 5.0 4 0 2. 4 1 1.", "6": "2 2 0 2 2 1 2.", "7": "2 2 0 2 2 1 3."}
        Green: null
        Range: "-5 0.2 7 0.8 50.0"
        Screening: "3 2 0.55"
        Multipolar: "Quadrupole"
        SCF: "include"
        Hubbard: "5.3 0.0 0.0"
        Self_abs: null
        Double_cor: null
        TDDFT: "include"
        Perdew: null
        Chfree: "include"
    ```

### FEFF yaml file

***

Below is a sample **FEFF** yaml file. As with the FDMNES yaml file, general information which must be included in all XANES yaml input files, regardless of the simulation method, is at the top. FEFF specific inputs are provided in the sub-section **feff_cards**. Again, as noted in the YAML itself, two conventions need to be followed:

- To include a card in the yaml file, but not in the final input file, assign it a value of `null`.

- To only include a card, with no associated value, such as including the line **Green**, assign it a value of `"include"`.

In this case, unliked with FDMNES, there are no special inputs which require further input.

!!! example "feff.yml"
    ```YAML
    # General XANES information which will be converted into cards internally. This section
    # is the same for all XANES yamls.
    cluster_radius: 5.0
    structure_type: "molecule" # Molecule or Crystal
    core_hole_site_element: "Fe"
    core_hole_site_id: 1 # which target element site is the center of the calculation (1 .. n_element)
    edge: "K"

    # Direct FDMNES cards
    feff_cards:
        CONTROL: "1 1 1 1 1 1"
        S02: "0.0"
        LDOS: null
        SCF: "3.5 1"
        XANES: "5.7 0.04 0.2"
        PRINT: "1 0 0 0 0 0"
        FMS: "4.5 1"
        RPATH: "-1"
        EXCHANGE: "5 1.5 0.0 2"
        CORRECTIONS: null
        FOLP: null
        COREHOLE: "RPA"
        MULTIPOLE: "2"
        MPSE: "2"
        EPS0: "78.4"
        OPCONS: "include"
        SFCONV: "include"
        MBCONV: null
        JUMPRM: null
        RSIGMA: null
        TDLDA: "0"
        NOHOLE: null
        EGRID: "-10 100 0.2"
        # CONFIG: "card 3\n0 Fe Ar 3d 2 2 4s 2\n1 N He 2s 2 2p 2 1\n2 C H2 2s 2 2p 2 1"
        ION: null # "0 0.15\nION 1 0.15\nION 2 0.15"
        PMBSE: null
        TDDFT: null
    ```