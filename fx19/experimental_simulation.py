
from __future__ import division, unicode_literals, print_function

import scipy
from scipy import optimize
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from diffpy.Structure import loadStructure
from diffpy.srfit.pdf import PDFContribution
from diffpy.srfit.fitbase import FitRecipe, FitResults

from ingrained.structure import Bicrystal
from ingrained.optimize import CongruityBuilder
from ingrained import image_ops
import cv2

import numpy as np
import os

class pdf_of_model(object):
    """
    This class contains functions to calculate the PDF fits it to the
    experimental pdf. Uses residual to calculate objective function.
    """
    def __init__(self, pdf_params):

        # main path as in energy.py
        self.name='PDF'
        self.main_path = pdf_params['main_path']
        self.pdf_sim_dir = None


        # set default values for the parameters
        self.stretch = 1.1
        self.Uiso_val = 0.01
        self.scale = 1.0
        self.delta1 = 0.5
        self.delta2 = 3.0
        self.qdamp = 0.005
        self.fix_atom_coords = True
        # default bounds_dict
        lb_ub_dict = {}
        lb_ub_dict['Uiso_val'] = [0.00001, 0.05]
        lb_ub_dict['scale'] = [0, 100.0]
        lb_ub_dict['delta1'] = [0, 5.0]
        lb_ub_dict['delta2'] = [2.0, 5.0]
        lb_ub_dict['qdamp'] = [0, 0.5]
        self.var_bounds = lb_ub_dict

        # minimization method from scipy.optimize.minimize i.e., one of strings
        # ['CG', 'BFGS', 'L-BFGS-B', 'SLSQP']
        self.minimize_method = 'CG'
        # Range parameters of the PDF function
        self.xmin = 1.5
        self.xmax = 7.5
        self.dx = 0.01
        # PDF Qmin and Qmax
        self.Qmin = 1.0
        self.Qmax = 50.0
        # elemental symbols of species as a list
        self.symbols = None

        # path to experimental pdf file
        self.exp_pdf_file = pdf_params['exp_pdf_file']
        # stretch factor to stretch the structure
        if 'stretch' in pdf_params:
            self.stretch = pdf_params['stretch']
        # values for initialization of minimization of scalar function
        # Uiso vals for all species
        if 'Uiso_val' in pdf_params:
            self.Uiso_val = pdf_params['Uiso_val']
        # scale factor for the PDF
        if 'scale' in pdf_params:
            self.scale = pdf_params['scale']
        # delta1 value
        if 'delta1' in pdf_params:
            self.delta1 = pdf_params['delta1']
        # delta2 value
        if 'delta2' in pdf_params:
            self.delta2 = pdf_params['delta2']
        # peak dampening factor
        if 'qdamp' in pdf_params:
            self.qdamp = pdf_params['qdamp']
        # Fix the atomic coordinates while minimization, (bool) default True
        if 'fix_atom_coords' in pdf_params:
            self.fix_atom_coords = pdf_params['fix_atom_coords']
        # minimize method from ['CG'(default), 'BFGS', 'L-BFGS-B', 'SLSQP']
        if 'minimize_method' in pdf_params:
            self.minimize_method = pdf_params['minimize_method']

        # set range parameters to provided pdf_params if available
        if 'xmin' in pdf_params:
            self.xmin = pdf_params['xmin']
        if 'xmax' in pdf_params:
            self.xmax = pdf_params['xmax']
        if 'dx' in pdf_params:
            self.dx = pdf_params['dx']

        # set Qmin and Qmax from Pdf_params
        if 'Qmin' in pdf_params:
            self.Qmin = pdf_params['Qmin']
            self.Qmax = pdf_params['Qmax']

        # set lower and upper bounds for fitting variables
        if 'var_bounds' in pdf_params:
            self.var_bounds = pdf_params['var_bounds']


    def write_temp_cif(self, stretch, model):
        """
        Writes temp.cif file in the pdf simulation directory from model.astr

        Args:
        stretch - (float) The ratio by which to stretch the structure when
                  creating temp.cif
                  Eg: 1.1 --> stretches 10% ; 0.9 --> shrinks by 10%
        model - model object
        """
        # use the relaxed structure from energy calculation
        astr = model.astr
        symbols = astr.symbol_set
        self.symbols = symbols
        frac_coords = astr.frac_coords
        # stretch the structure
        stretched_coords = frac_coords * stretch
        # make new structure object with stretched_coords
        new_structure = Structure(astr.lattice, astr.species, stretched_coords)
        # write new_structure to a temporary cif file
        # diffpy only able to read cif format
        cif_writer = CifWriter(new_structure)
        temp_cif = self.pdf_sim_dir + '/temp.cif'
        cif_writer.write_file(temp_cif)


    def get_PDF_obj(self):
        """
        Make diffpy.srfit.pdfcontribution.PDFContribution object from temp.cif
        """
        temp_cif = self.pdf_sim_dir + '/temp.cif'
        # create a diffpy structure object
        diffpy_str = loadStructure(temp_cif)
        # create a PDFContribution object; here 'atoms' is name of the object
        PDF = PDFContribution('atoms')
        # upload the experimental pdf data
        PDF.loadData(self.exp_pdf_file)
        PDF.setCalculationRange(xmin=self.xmin, xmax=self.xmax, dx=self.dx)
        # add stretched structure to PDF object
        PDF.addStructure("atoms", diffpy_str, periodic=False)
        PDF.setQmin(self.Qmin)
        PDF.setQmax(self.Qmax)

        return PDF


    def fit_and_residual(self, PDF, type=None):
        """
        get diffpy.srfit.fitbase.FitRecipe object

        Args:
        PDF (PDFContribution) - PDF object from get_PDF_obj
        type - (str) 'opt_stretch' when optimizing stretch factor
             - None for anything else

        (NOTE: made PDF and Fit two separate functions for convenience)
        """
        symbols = self.symbols
        Fit = FitRecipe()
        Fit.addContribution(PDF)
        Uisos = []
        for sym in symbols:
            Uisos.append('Uiso_{}'.format(sym))
            Fit.newVar('Uiso_{}'.format(sym), value=self.Uiso_val, fixed=False)

        for atom in PDF.atoms.phase.atoms:
            for sym in symbols:
                if atom.element == sym:
                    Fit.constrain(atom.Uiso, 'Uiso_{}'.format(sym))

        # Set all Uiso values to provided or default Uiso_val
        for p in Uisos:
            fit_param = Fit._parameters[p]
            fit_param.setValue(self.Uiso_val)
            fit_param.bounds = self.var_bounds['Uiso_val']

        # add existing PDF variables as Fit parameters and setValue
        Fit.addVar(PDF.scale, self.scale, fixed=False)
        Fit.addVar(PDF.atoms.delta1, self.delta1, fixed=False)
        Fit.addVar(PDF.atoms.delta2, self.delta2, fixed=False)
        Fit.addVar(PDF.qdamp, self.qdamp, fixed=False)

        # Set lower and upper bounds for variables
        Fit.scale.bounds = self.var_bounds['scale']
        Fit.delta1.bounds = self.var_bounds['delta1']
        Fit.delta2.bounds = self.var_bounds['delta2']
        Fit.qdamp.bounds = self.var_bounds['qdamp']

        # Initialize atomic coordinate with one fixed central atom
        for i in range(len(PDF.atoms.phase.atoms)):
            pdf_atom = PDF.atoms.phase.atoms[i]
            for cc in ['x', 'y', 'z']:
                vname = cc + '_' + pdf_atom.name
                cc_var = pdf_atom.get(cc)
                Fit.addVar(cc_var, name=vname, tag='xyz', fixed=True)

        Fit.boundsToRestraints(sig = 0.00001)
        # Turn off printout of iteration number.
        Fit.clearFitHooks()

        init_Uisos = []
        for i in range(len(Uisos)):
            init_Uisos.append(self.Uiso_val)
        init_rem = [self.scale, self.delta1, self.delta2, self.qdamp]
        init_params = init_Uisos + init_rem

        # optimize params using scipy
        init_params = np.array(init_params)
        # For finding stretch residual
        if type == 'opt_stretch':
            result = scipy.optimize.minimize(Fit.scalarResidual,
                                             init_params,
                                             method=self.minimize_method,
                                             tol=1e-3,
                                             options={'maxiter':20})
            residual = Fit.scalarResidual(fitted_params)
            return (residual / 600) ** .5
        else:
            result = scipy.optimize.minimize(Fit.scalarResidual, init_params,
                                             method=self.minimize_method)
            # get residue from the fitted params
            fitted_params = result.x
            residual = Fit.scalarResidual(fitted_params)

            # write the pdf comparison data to a file
            r = Fit.atoms.profile.x
            g_obs = Fit.atoms.profile.y
            g_calc = Fit.atoms.evaluate()
            g_diff = g_obs - g_calc

            with open(self.pdf_sim_dir + '/pdf_dataNow.txt', 'w') as f:
                f.write('radius \tg_exp \tg_calc \tg_diff \n' )
                for i, item in enumerate(r):
                    f.write(str(r[i])[:4] + '\t ' + str(g_obs[i])[:6] + '\t ' \
                            + str(g_calc[i])[:6] + '\t ' + str(g_diff[i])[:6] \
                            + ' \n')

            return fitted_params, (residual / 600) ** .5

    def get_stretch_residual(self, stretch, model, type=None):
        """
        A wrapper function around fit_and_residual to optimize stretch outside
        main parameters optimization within FitRecipe

        Args:
        stretch - (float) The ratio by which to stretch the structure when
                  creating temp.cif
                  Eg: 1.1 --> stretches 10% ; 0.9 --> shrinks by 10%
        model - model object
        type - (str) 'opt_stretch' when optimizing stretch factor
             - None for anything else
        """
        """
        print (stretch)
        try:
            iter_stretch = float(stretch[0])
        except:
            print ('Key Error: 0')
            print ('Optimization complete!')
            iter_stretch = stretch.x[0]
        """
        self.write_temp_cif(stretch, model)
        # load exp_pdf and temp_cif to PDF object
        PDF = self.get_PDF_obj()
        # fit params and get residual
        if type=='opt_stretch':
            residual = self.fit_and_residual(PDF, type=type)
            return residual
        else:
            fitted_params, residual = self.fit_and_residual(PDF, type=type)
            return fitted_params, residual


    def evaluate_obj(self, model):
        """
        Fits calculated and experimental pdf and returns model.
        Objective function value is added to model attributes.
        Residual is the objective function value for pdf.

        Args:
        model - model object
        """
        main_path = self.main_path
        pdf_sim = main_path + '/calcs/' + str(model.label) + '/pdf_sim'
        os.mkdir(pdf_sim)
        self.pdf_sim_dir = pdf_sim

        init_stretch = np.array([self.stretch])
        # minimize the stretch_residual and fit best stretch
        opt_stretch = scipy.optimize.minimize_scalar(
                                              self.get_stretch_residual,
                                              bounds=(0.97, 1.05),
                                              args=(model, 'opt_stretch'),
                                              method='bounded'
                                              )

        # Get final residual and fitted params for the record
        fitted_params, final_res = self.get_stretch_residual(opt_stretch.x,
                                                                  model)

        # the order of exp_sims is from Xsim1 -> Xsim2 -> ...
        # Hence, obj1val -> ob2_val -> ... for assigning evaluated sims
        if model.Xsim1 == 'PDF':
            model.obj1_val = float(final_res)
        elif model.Xsim2 == 'PDF':
            model.obj2_val = float(final_res)
        elif model.Xsim3 == 'PDF':
            model.obj3_val = float(final_res)
        elif model.Xsim4 == 'PDF':
            model.obj4_val = float(final_res)

        return model, fitted_params


class gb_ingrained(object):
    """
    Uses "ingrained" module to make the simulated TEM image of the model
    """
    def __init__(self, gb_ingrained_params):
        """
        the gb_ingrained_params is a dictionary of the user-provided
        parameters in the input_file.yaml

        1. The experimental image is from the POSCAR_init_gb provided by the
        user
        2. The optimized parameters for the initial image matching should be
        provided in gb_ingrained_params
            Ex: IW = 0.111786813217729      # image width
                DF = 1.1488391624705199     # defocus
                PX = 0.13293286291986323    # pixel size from the .dm3 file
                left_x, right_x, bot_y, top_y = 0.05, 0.95, 0.4, 0.75
                border_reduce = (left_x, right_x, bot_y, top_y)

            The inital optimization prints out these params. Use them as it is.
        """
        self.name = 'GB_STEM'
        ingrained_keys = ['iw', 'df', 'px', 'border_reduce']
        for k in ingrained_keys:
            if k not in gb_ingrained_params.keys():
                print ("Provide all the optimized ingrained parameters.")

        self.iw = gb_ingrained_params['iw']
        self.df = gb_ingrained_params['df']
        self.px = gb_ingrained_params['px']
        self.border_reduce = gb_ingrained_params['border_reduce']

        self.main_path = gb_ingrained_params['main_path']
        self.init_gb_path = gb_ingrained_params['init_gb_path']
        self.dm3_path = gb_ingrained_params['dm3_path']
        bicrys_ref = Bicrystal(self.init_gb_path)
        congruity_ref = CongruityBuilder(bicrys_ref, dm3=self.dm3_path)
        congruity_ref.fit(interface_width=self.iw, defocus=self.df,
                          border_reduce=self.border_reduce, pixel_size=self.px,
                          save_experiment=self.main_path+"/gb_im_ref.jpg")

        # Temporary hard coded cropping of image to interface region
        im_ref = cv2.imread(self.main_path+"/gb_im_ref.jpg", 0)
        self.im_ref = im_ref[55:170, :]
        cv2.imwrite(self.main_path+"/gb_im_ref.jpg", self.im_ref)


    def scale_gb_astr(self, tested_scales, factor=0.01):
        """
        Once we evaluate the gb_astr, if residual is greater than the threshold,
        we stretch or squeeze the model.

        tested_scales: a list of (float) values that have already been tested
        on the model
        factor: the factor by which to scale the model
        """
        pass

    def evaluate_obj(self, model):
        """
        This function is a must for overall Fantastx run. All classes in this
        module should have this function.

        NOTE: Please refer to the pdf_of_model class and run.py for how this
        works.

        Returns model after setting attribute obj()_val = residual
        """
        relax_path = self.main_path + '/calcs/' + str(model.label) + '/relax'
        bicrys_model = Bicrystal(relax_path+'/POSCAR_relaxed')
        filename = relax_path+"/gb_im_model.jpg"
        bicrys_model.convolution_HAADF(filename=filename,
                                       dm3=self.dm3_path,
                                       pixel_size=self.px,
                                       interface_width=self.iw,
                                       defocus=self.df,
                                       border_reduce=self.border_reduce)

        im_model = cv2.imread(filename, 0)
        im_model = im_model[55:170, :]
        cv2.imwrite(filename, im_model)
        #TODO: sigma value should be fixed?
        score = image_ops.score_vifp(self.im_ref, im_model, sigma=2)

        # the order of exp_sims is from Xsim1 -> Xsim2 -> ...
        # Hence, obj1val -> ob2_val -> ... for assigning evaluated sims
        if model.Xsim1 == 'GB_STEM':
            model.obj1_val = float((1 - score)*100) # Minimizing the obj vals
        elif model.Xsim2 == 'GB_STEM':
            model.obj2_val = float((1 - score)*100)
        elif model.Xsim3 == 'GB_STEM':
            model.obj3_val = float((1 - score)*100)
        elif model.Xsim4 == 'GB_STEM':
            model.obj4_val = float((1 - score)*100)

        return model, score
