FANTASTX supports several XANES simulation codes. Currently those are: **FDMNES**, **FEFF**, and **VASP**. When you have decided which XANES code to use, a key ingredient to the FANTASTX run is the yaml input file which assigns all XANES simulation parameters. Below are sample yaml input files for each of the supported simulation codes.

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