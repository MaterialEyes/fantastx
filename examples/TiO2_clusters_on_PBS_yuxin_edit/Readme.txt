# command to run this Fantastx toy example on Carbon
>>> python ex_script.py

Make the following changes according to the user and the cluster.

To run Fantastx example on Carbon:
	Change the energy_files_path to the according to the user in input file
	Change the energy_exec_cmd according to the cluster and binary in input file
	Change the "workers" keyword parameters in the input file according to the cluster
	Change the path to the lammps potential file in the in.min file (which is present in the energy_files_path)

Edit ~/.modules_el7 file in user's home directory. This is to ensure that the dask-workers, which run the LAMMPS/VASP calculations, can load the LAMMPS/VASP modules of the (Carbon) cluster and therefore can run the calculations within the workers. If this is not done, the following error occurs in worker error file - "lmp_mpi command not found"

# Add these lines in ~/.modules-el7 file:

module load --auto --silent lammps
module load --auto --silent vasp5
---------------------------------------------------------------------------
---------------------------------------------------------------------------
Here is how that file looks like --
(fx_521) [vkolluru@login6 EL7 ~]$ more .modules-el7
# Carbon software modules initialization.  This file is read by the module
# "profile/user", on CentOS-6 nodes, in the 2016 naming scheme.
#
# vim:syntax=tcl:
#
# https://wiki.anl.gov/cnm/HPC/Module_naming_scheme_2016

module load --auto --silent lammps
module load --auto --silent vasp5
(fx_521) [vkolluru@login6 EL7 ~]$
---------------------------------------------------------------------------
---------------------------------------------------------------------------
Lammps input files (in.min, Ti-O.tersoff.zbl) ; Fantastx input file (cluster_input.yaml) are the only input files for this example.

After running ex_script.py, Fantastx will create "calcs/" directory whcih contains LAMMPS calculations of all models created.

"data_file" contains the energy objective function (E_lammps - ∑ Ni.µi), and the "inheritance" of each model. "random" suggessts that the model is created radnomly for the initial population. Only one parent label means the model is created by "basinhopping" method. Two parent labels means that it is created by "mating" operations.
