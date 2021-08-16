To run this Fantastx example on SLURM based cluster:

	Change the energy_files_path in input file to the current directory (where in.min file and potential file exists) according to the user
	Change the energy_exec_cmd in input file according to the cluster and binary
	Change the "workers" keyword parameters in the input file according to the cluster and user account details
	Change the path to the lammps potential file in the in.min file (which is present in the energy_files_path)


Lammps input files (in.min, Ti-O.tersoff.zbl) ; Fantastx input file (cluster_input.yaml) are the only input files for this example.

After running ex_script.py, Fantastx will create "calcs/" directory whcih contains LAMMPS calculations of all models created.

"data_file" contains the energy objective function (E_lammps - ∑ Ni.µi), and the "inheritance" of each model. "random" suggessts that the model is created radnomly for the initial population. Only one parent label means the model is created by "basinhopping" method. Two parent labels means that it is created by "mating" operations.
