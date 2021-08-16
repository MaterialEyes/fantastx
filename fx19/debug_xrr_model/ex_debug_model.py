import os, yaml, sys
import datetime
import random, copy

import time, gc
import numpy as np
from time import sleep

import numpy as np
from pymatgen.core.structure import Structure, Lattice
from pymatgen.core.composition import Composition 

from fx19 import inputs
from fx19.run_ops import *
from fx19 import structure_record
from fx19 import experimental_simulation

main_path = os.getcwd()
sys.path.insert(1, '/Users/klweaver/Code/XRay/FoxPy')

# read input file and make input dictionary
with open('/Users/klweaver/Code/FANTASTX/fantastx/fx19/debug_xrr_model/debug_xrr.yaml') as ifile:
#with open('debug_xrr.yaml') as ifile:
    i_dict = yaml.load(ifile, Loader=yaml.FullLoader)
    i_dict['main_path'] = main_path

#test comment commit

#structure_files = ['POSCAR_rand_1', 'POSCAR_rand_2', 'POSCAR_rand_3']
structure_files = ['/Users/klweaver/Code/XRay/FoxPy/input/TT-rt13/rt13.cif']
pymatgen_structures = [Structure.from_file(s) for s in structure_files]

reg_id = structure_record.register_id()

surface_models = [structure_record.model(s, reg_id) for s in pymatgen_structures]

# make experimental_simulation object(s)
exp_sim_methods = ['PDF', 'GB_STEM', 'PRISM', 'GSASII', 'FEFF', 'XRR']
if 'exp_sim_1' in i_dict:
    if i_dict['exp_sim_1'] in exp_sim_methods:
        method_1 = i_dict['exp_sim_1']
        if not 'exp_sim_1_params' in i_dict:
            print ('exp_sim_1_params not provided. They are mandatory.')
        if method_1 == 'XRR':
            Xsim1_params = inputs.get_foxpy_params(i_dict, 'exp_sim_1_params')
            Xsim_1 = experimental_simulation.xrr_foxpy(Xsim1_params)
        else: 
            Xsim_1 = None




for model in surface_models:
    print("Model ID: " + str(model.label))
    model.tot_en = np.random.uniform(1000, 1200)
    print("Total Energy: " + str(model.tot_en))
    model.converged = True

    if Xsim_1:
        model.Xsim1 = Xsim_1.name
        model, Xsim_val = Xsim_1.evaluate_obj(model)

    #energy objective
    model.obj0_val = np.random.uniform(50, 120)
    #experimental objective
    model.obj1_val = Xsim_val
    model.overall_val = 0.5*model.obj1_val+0.5*model.obj0_val
    #Wanted to equally weight first two objectives - but this fxn doesn't do this.  It does nothing b/c some weights are zero
    #model.set_overall_val([0.5,0.5,0,0,0])
   
    print("Objective: " + str(model.overall_val))
    model.selection_prob = np.random.uniform(0.2, 1)
    print("Selection Probability: " + str(model.selection_prob)+"\n")
