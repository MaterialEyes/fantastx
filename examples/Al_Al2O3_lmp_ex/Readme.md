##### How to run this example?
Once you have Fantastx installed on your system, use the input file (gb_input.yaml) and submit_job script to run this example. Change the submit_job script with appropriate options, and according to the PBS/SLURM cluster.

```sh
qsub submit_job # on PBS cluster
```

##### What are the input files?
All the input parameters required for the search are provided using a single input yaml file. In this example, we call it gb_input.yaml. The input file parameters (and corresponding keywords) are different for different types of searches. This example contains the example input file for a search to find stable grain boundary interface structure that also provides a well-matched simulated TEM image to the experimental TEM image (courtesy: Dr. Ilke Arslan).

Path to several other input files as required by the algorithm should also provided as parameters in the input file. For example, the path of the directory which contains the input files for the LAMMPS or VASP calculations is provided with the keyword: *energy_files_path*.

##### What are the output files?
All the models investigated by the algorithm exist in an output directory called *calcs*. The calculations corresponding to a model are present in its directory specified using its label. The simulated data (TEM/STM/PDF) is also available in the same directory.

For each model, its inheritance, full composition, and the evaluated objective function values based on its energy (*Obj_0*) and experimental mismatch (*Obj_1*) are written to an output file called *data_file*.

##### How does the parallelization work?
The algorithm uses Dask-job_queue package to parallelize on PBS/SLURM clusters. On a local machine, it uses ProcessPoolExecutor. Dask is a python package that connects (from a master) to the PBS/SLURM scheduler to start jobs. Each job is called a Dask worker. Each worker can perform several I/O tasks as requested by the master.

The executable *run_fx.py* will be submitted as an initial job. Consider the node which runs this initial job as the master. The master will request the PBS/SLURM scheduler for workers. The job configurations for the workers are provided as input parameters in the input file. Each worker has separate resources and performs LAMMPS/VASP calculations independent of other workers.  The idea is to perform expensive calculations on these workers and run the algorithm using the output from workers and continue with more calculations.

##### What are the other input files?
The other input files can be divided according to their specific purpose in the algorithm.
<br>
For performing energy calculations -
<br>
    &nbsp;&nbsp;&nbsp;&nbsp; in.min & potential_file for LAMMPS
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp; INCAR, KPOINTS, POTCAR(s) for VASP
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp; keyword: *energy_files_path*
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp; Refer [this](xxxxxxxxx) example input file
    <br>

For PDF simulation and comparison - <br>
&nbsp;&nbsp;&nbsp;&nbsp; experimental_pdf.txt <br>
&nbsp;&nbsp;&nbsp;&nbsp; keyword: *exp_pdf_file* <br>
&nbsp;&nbsp;&nbsp;&nbsp; Refer [this](xxxxxxxxx) for PDF simulation example

For TEM simulation and comparison - <br>
&nbsp;&nbsp;&nbsp;&nbsp; progress_file.txt; experimental_TEM.dm3; POSCAR_gb_ingrained <br>
&nbsp;&nbsp;&nbsp;&nbsp; keywords: *progress_file*; *dm3_path*; *init_gb_astr*
&nbsp;&nbsp;&nbsp;&nbsp; Refer [this](xxxxxxxxx) for TEM simulation example


##### How to use post-processing scripts?
TBD
