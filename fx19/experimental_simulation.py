
from __future__ import division, unicode_literals, print_function

from fx19 import distance_check as dc
from scipy import optimize as scipy_optimize
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
try:
    from diffpy.Structure import loadStructure
    from diffpy.srfit.pdf import PDFContribution
    from diffpy.srfit.fitbase import FitRecipe, FitResults
    import matplotlib.pyplot as plt
except ImportError:
    print('Install Diffpy-CMI for PDF simulation. Otherwise ignore..')

# For preprocessing experimental image
# from skimage.transform import rescale
try:
    from skimage import restoration
    from skimage.exposure import equalize_adapthist

    from ingrained.structure import Bicrystal
    from ingrained.optimize import CongruityBuilder
    import ingrained.image_ops as iop
    import cv2
except ImportError:
    print ('Install scikit-image, Ingrained, opencv for TEM simulation.'
           ' Otherwise ignore..')

from math import floor
import numpy as np
import os


class pdf_of_model(object):
    """
    This class contains functions to calculate the PDF and fit it to the
    experimental pdf. Uses the residual to calculate objective function.

    Args:

    pdf_params (dict): A dictionary of parameters used for fitting PDF using
    Diffpy.
    """

    def __init__(self, pdf_params):
        """
        """
        # main path as in energy.py
        self.name = 'PDF'
        self.main_path = pdf_params['main_path']
        self.pdf_sim_dir = None

        # Default isotropic ADP value for all species
        self.Uiso_val = 0.009
        # default structure scale factor
        self.scale = 1.0
        # quadratic term related to sharpness of first peak (from pdfgui manual)
        self.delta2 = 3.87
        # exp. instrument (peak-damping) parameter (default from pdfgui manual)
        self.qdamp = 0.043
        self.fit_coords = False
        # default bounds_dict
        lb_ub_dict = {}
        lb_ub_dict['Uiso_val'] = [0.00001, 0.11]
        lb_ub_dict['scale'] = [0.8, 1.2]
        lb_ub_dict['delta2'] = [2.0, 10.0]
        lb_ub_dict['qdamp'] = [0.001, 0.1]
        self.var_bounds = lb_ub_dict

        # minimization method from scipy_optimize.minimize i.e., one of strings
        # ['L-BFGS-B', 'SLSQP']
        self.minimize_method = 'L-BFGS-B' # or 'SLSQP' only
        # Range parameters of the PDF function (x-axis)
        self.xmin = 1.5
        self.xmax = 7.5
        self.dx = 0.01
        # PDF Qmin and Qmax (y-axis)
        self.Qmin = 1.0
        self.Qmax = 50.0
        # elemental symbols of species as a list
        self.symbols = None

        # path to experimental pdf file
        self.exp_pdf_file = pdf_params['exp_pdf_file']
        # values for initialization of minimization of scalar function
        # Uiso vals for all species
        if 'Uiso_val' in pdf_params:
            self.Uiso_val = pdf_params['Uiso_val']
        # scale factor for the PDF
        if 'scale' in pdf_params:
            self.scale = pdf_params['scale']
        # delta2 value
        if 'delta2' in pdf_params:
            self.delta2 = pdf_params['delta2']
        # peak dampening factor
        if 'qdamp' in pdf_params:
            self.qdamp = pdf_params['qdamp']
        # whether to fit atomic coordinates as variables in FitRecipe
        if 'fit_coords' in pdf_params:
            self.fit_coords = pdf_params['fit_coords']
        # minimize method from ['L-BFGS-B', 'SLSQP']
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
            self.Qmax = pdf_params['Qmax']

        # set lower and upper bounds for fitting variables
        if 'var_bounds' in pdf_params:
            vbs = pdf_params['var_bounds']
            for a_key in vbs.keys():
                self.var_bounds[a_key] = vbs[a_key]

    def write_temp_cif(self, model):
        """
        Writes temp.cif file in the pdf simulation directory from model.astr

        Args:

        model (obj): structure_record.model() object for which PDF simulation
        will be done
        """
        # use the relaxed structure from energy calculation
        astr = model.astr
        self.symbols = astr.symbol_set

        # write new_structure to a temporary cif file
        # diffpy only able to read cif format
        temp_init = self.pdf_sim_dir + '/temp_init.cif'
        cif_writer = CifWriter(astr)
        cif_writer.write_file(temp_init)

    def get_PDF_obj(self, cif_file):
        """
        Make diffpy.srfit.pdfcontribution.PDFContribution object from temp.cif
        """
        # create a diffpy structure object
        diffpy_str = loadStructure(cif_file)
        # create a PDFContribution object; here 'atoms' is name of the object
        PDF = PDFContribution('nanocluster')
        # upload the experimental pdf data
        PDF.loadData(self.exp_pdf_file)
        PDF.setCalculationRange(xmin=self.xmin, xmax=self.xmax, dx=self.dx)
        # add stretched structure to PDF object
        PDF.addStructure("generator", diffpy_str, periodic=False)
        PDF.setQmin(self.Qmin)
        #PDF.setQmax(self.Qmax)

        return PDF

    def fit_variables_recipe(self, PDF, cif_file):
        """
        Performs optimization of PDF variables like qdamp, delta2, scale and
        ADP (Uisos) using diffpy.srfit.fitbase.FitRecipe object.

        This function call should be preceeded by get_PDF_obj function. The PDF
        object should contain a structure that is optimized previously using
        fit_coords_recipe()

        (NOTE: made PDF and Fit two separate functions for convenience)

        Returns fitted_params after PDF fit and the residual

        Args:

        PDF (PDFContribution) - PDF object from get_PDF_obj
        """
        symbols = self.symbols
        Fit = FitRecipe()
        Fit.addContribution(PDF)
        Uisos = []
        for sym in symbols:
            Uisos.append('Uiso{}'.format(sym))
            Fit.newVar('Uiso{}'.format(sym), value=self.Uiso_val, fixed=False)

        for atom in PDF.generator.phase.atoms:
            for sym in symbols:
                if atom.element == sym:
                    Fit.constrain(atom.Uiso, 'Uiso{}'.format(sym))

        # Set all Uiso values to provided or default Uiso_val
        #for p in Uisos:
        #    fit_param = Fit._parameters[p]
        #    fit_param.setValue(self.Uiso_val)

        # add existing PDF variables as Fit parameters and setValue
        Fit.addVar(PDF.generator.scale, self.scale, fixed=False)
        Fit.addVar(PDF.generator.delta2, self.delta2, fixed=False)
        Fit.addVar(PDF.qdamp, self.qdamp, fixed=False)

        bounds=[self.var_bounds['Uiso_val'],
                self.var_bounds['scale'],
                self.var_bounds['delta2'],
                self.var_bounds['qdamp']]

        # Add coordinates as fixed variables
        pymat_str = Structure.from_file(cif_file)
        cc = pymat_str.cart_coords
        center = [(cc[:, 0].max() + cc[:, 0].min())/2,
                  (cc[:, 1].max() + cc[:, 1].min())/2,
                  (cc[:, 2].max() + cc[:, 2].min())/2]
        dists = [dc.dist(c, center) for c in cc]
        center_ind = np.argmin(dists)

        # Fix central atom coords and optimize all other atom coords
        for i, pdf_atom in enumerate(PDF.generator.phase.atoms):
            for cc in ['x', 'y', 'z']:
                vname = cc + '_' + pdf_atom.name
                cc_var = pdf_atom.get(cc)
                if int(i) == int(center_ind):
                    Fit.addVar(cc_var, name=vname, tag='xyz', fixed=True)
                else:
                    Fit.addVar(cc_var, name=vname, tag='xyz', fixed=True)

        # Turn all bounded parameters into restraints with uncertainty sigma
        Fit.boundsToRestraints(sig=0.01)
        # Turn off printout of iteration number.
        Fit.clearFitHooks()

        # For finding stretch residual
        result = scipy_optimize.minimize(Fit.scalarResidual,
                                         Fit.getValues(),
                                         method=self.minimize_method,
                                         tol=1e-3,
                                         bounds=bounds,
                                         options={'maxiter': 100000})
        # get residue from the fitted params
        fitted_params = result.x
        residual = Fit.scalarResidual(fitted_params)

        # write the pdf comparison data to a file
        r = Fit.nanocluster.profile.x
        g_obs = Fit.nanocluster.profile.y
        g_calc = Fit.nanocluster.evaluate()
        g_diff = g_obs - g_calc

        diffzero = -0.8 * max(g_obs) * np.ones_like(g_obs)
        diff = g_obs - g_calc + diffzero

        with open(self.pdf_sim_dir + '/pdf_data.txt', 'w') as f:
            f.write('radius \tg_exp \tg_calc \tg_diff \n')
            for i, item in enumerate(r):
                f.write(str(r[i])[:4] + '\t ' + str(g_obs[i])[:6] + '\t '
                        + str(g_calc[i])[:6] + '\t ' + str(g_diff[i])[:6]
                        + ' \n')

        plt.plot(r, g_obs, 'bo', label="G(r) Target")
        plt.plot(r, g_calc, 'r-', label="G(r) Fit")
        plt.plot(r, diff, 'g-', label="G(r) diff")
        plt.plot(r, diffzero, 'k-')
        plt.xlabel(r"$r (\AA)$")
        plt.ylabel(r"$G (\AA^{-2})$")
        plt.legend(loc=1)

        plt.savefig(fname=self.pdf_sim_dir + '/pdf_fit.png')
        plt.close()

        return fitted_params, residual

    def fit_coords_recipe(self, PDF):
        """
        Performs optimization ofatomic coordiantes of a structure. This does
        not try to optimize the PDF related variables. Take the energy_code
        relaxed structure, and create a PDF object. Then perform
        fit_variables_recipe() using the resulting fitted coordinates from this.

        Returns nothing. (Writes temp_opt.cif to the pdf_sim_dir)

        Args:

        PDF (PDFContribution) - PDF object from get_PDF_obj
        """
        symbols = self.symbols
        Fit = FitRecipe()
        Fit.addContribution(PDF)
        Uisos = []
        for sym in symbols:
            Uisos.append('Uiso{}'.format(sym))
            Fit.newVar('Uiso{}'.format(sym), value=self.Uiso_val, fixed=True)

        for atom in PDF.generator.phase.atoms:
            for sym in symbols:
                if atom.element == sym:
                    Fit.constrain(atom.Uiso, 'Uiso{}'.format(sym))

        # Set all Uiso values to provided or default Uiso_val
        #for p in Uisos:
        #    fit_param = Fit._parameters[p]
        #    fit_param.setValue(self.Uiso_val)

        # add existing PDF variables as Fit parameters and setValue
        Fit.addVar(PDF.generator.scale, self.scale, fixed=True)
        Fit.addVar(PDF.generator.delta2, self.delta2, fixed=True)
        Fit.addVar(PDF.qdamp, self.qdamp, fixed=True)

        # We 'fixed' all variables defined so far. So no bounds needed for them
        bounds=[]

        # Add coordinates as variables and then optimize
        temp_init = self.pdf_sim_dir + '/temp_init.cif'
        pymat_str = Structure.from_file(temp_init)
        cc = pymat_str.cart_coords
        center = [(cc[:, 0].max() + cc[:, 0].min())/2,
                  (cc[:, 1].max() + cc[:, 1].min())/2,
                  (cc[:, 2].max() + cc[:, 2].min())/2]
        dists = [dc.dist(c, center) for c in cc]
        center_ind = np.argmin(dists)

        # Fix central atom coords and optimize all other atom coords
        for i, pdf_atom in enumerate(PDF.generator.phase.atoms):
            for cc in ['x', 'y', 'z']:
                vname = cc + '_' + pdf_atom.name
                cc_var = pdf_atom.get(cc)
                if int(i) == int(center_ind):
                    Fit.addVar(cc_var, name=vname, tag='xyz', fixed=True)
                else:
                    Fit.addVar(cc_var, name=vname, tag='xyz', fixed=False)
        # add bounds for new variables of atom coords
        fc = np.delete(pymat_str.frac_coords, center_ind, 0)
        for i in fc.flatten():
            bounds.append([i-0.05, i+0.05])

        # Turn all bounded parameters into restraints with uncertainty sigma
        Fit.boundsToRestraints(sig=0.001)
        # Turn off printout of iteration number.
        Fit.clearFitHooks()

        # For finding stretch residual
        result = scipy_optimize.minimize(Fit.scalarResidual,
                                         Fit.getValues(),
                                         method=self.minimize_method,
                                         tol=1e-3,
                                         bounds=bounds,
                                         options={'maxiter': 100000})
        # get residue from the fitted params
        fitted_params = result.x
        residual = Fit.scalarResidual(fitted_params)

        # Write output structure with new coordinates
        fcs = result.x
        fcs_new = np.insert(fcs, center_ind,
                            pymat_str.frac_coords[center_ind], axis=0)
        #fcs_new = np.concatenate((pymat_str.frac_coords[center_ind], fcs))
        fcs_new = fcs_new.reshape(14, 3)
        astr_varied = Structure(pymat_str.lattice, pymat_str.species,
                                fcs_new, coords_are_cartesian=False)
        opt_cif = self.pdf_sim_dir + '/temp_opt.cif'
        opt_pos = self.pdf_sim_dir + '/POSCAR_opt_pdf'
        astr_varied.to(filename=opt_cif)
        astr_varied.to(filename=opt_pos)

    def evaluate_obj(self, model):
        """
        Fits calculated and experimental pdf and returns model. Uses the
        Residual as the objective function value for pdf. Objective function
        value is added to model attributes obj1_val.

        Returns model, fitted_params

        Args:

        model (obj): structure_record.model() object for which energy
                     evaluation will be done
        """
        main_path = self.main_path
        pdf_sim = main_path + '/calcs/' + str(model.label) + '/pdf_sim'
        os.mkdir(pdf_sim)
        self.pdf_sim_dir = pdf_sim

        # write temp_init.cif to pdf_sim_dir
        self.write_temp_cif(model)
        # load exp_pdf and temp_init.cif to PDF object
        cif_file = pdf_sim + '/temp_init.cif'
        PDF = self.get_PDF_obj(cif_file)

        if self.fit_coords is True:
            # fit the atomic coordinates using Diffpy
            self.fit_coords_recipe(PDF)
            # use the new cif file created with optimized coords
            cif_file = pdf_sim + '/temp_opt.cif'
            PDF = self.get_PDF_obj(cif_file)

        # fit the PDF variables
        fitted_params, residual = self.fit_variables_recipe(PDF, cif_file)

        # the order of exp_sims is from Xsim1 -> Xsim2 -> ...
        # Hence, obj1val -> ob2_val -> ... for assigning evaluated sims
        if model.Xsim1 == 'PDF':
            model.obj1_val = float(residual)
        elif model.Xsim2 == 'PDF':
            model.obj2_val = float(residual)
        elif model.Xsim3 == 'PDF':
            model.obj3_val = float(residual)
        elif model.Xsim4 == 'PDF':
            model.obj4_val = float(residual)

        return model, fitted_params


class gb_ingrained(object):
    """
    This class is used to simulate STEM image of a grain boundary structure
    with "Ingrained" package
    """

    def __init__(self, gb_ingrained_params):
        """
        The gb_ingrained_params is a dictionary of the user-provided
        parameters from the input_file.yaml.

        A separate Ingrained optimization should be performed prior to running
        FANTASTX to get an initial grain boundary structure which will be used
        as starting structure to make new models. Along with it, the final
        optimium parameters are taken from the progress_file of the Ingrained
        optimization.

        Ex:



        """
        self.name = 'GB_STEM'

        self.main_path = gb_ingrained_params['main_path']
        self.init_gb_path = gb_ingrained_params['init_gb_path']
        if not self.init_gb_path:
            print('Provide path to ingrained optimized initial '
                  'grain boundary structure')

        # get either progress_file or ing_opt_params from input
        self.progress_file = gb_ingrained_params['progress_file']
        self.opt_params = gb_ingrained_params['ing_opt_params']
        if not self.progress_file and not self.opt_params:
            print('Provide ingrained optimization progress as progress_file'
                  ' or sim params of optimized solution')

        self.dm3_path = gb_ingrained_params['dm3_path']
        if not self.dm3_path:
            print('Provide path (dm3_path) to experimental image')

        # Prepare experimental image
        # (make sure this procedure matches the procedure in 'run.py')
        image_data = iop.image_open(self.dm3_path)
        exp_img = iop.apply_rotation(image_data['Pixels'], 1)
        exp_img = iop.scale_pixels(exp_img, mode='rescale')
        exp_img = restoration.wiener(exp_img, np.ones((7, 7))/3.5, 1300)
        exp_img = equalize_adapthist(exp_img, clip_limit=0.005)

        bicrys_ref = Bicrystal(poscar_file=self.init_gb_path)
        congruity = CongruityBuilder(sim_obj=bicrys_ref, exp_img=exp_img)

        # Get solutions from text file
        if self.progress_file:
            progress = np.genfromtxt(self.progress_file, delimiter=',')
            best_idx = int(np.argmin(progress[:, -1]))
            x = progress[best_idx]
            xfit = x[1:-1]
            xfit = [a for a in xfit[:-2]] + [int(a) for a in xfit[-2::]]

        # TODO: Find why we set self.opt_params[1] = 0
        if not self.opt_params:
            self.opt_params = xfit.copy()
            self.opt_params[1] = 0
        else:
            xfit = self.opt_params.copy()
        xfit[1] = 0
        sim_img, sim_struct, exp_patch, shift_score, stable_idxs = \
            congruity.fit_gb(sim_params=xfit, bias_y=1E-4)

        sim_struct.to(filename='POSCAR_init_fitted', fmt='poscar')

        #np.save(self.main_path + '/whole_exp.npy', exp_patch)
        #np.save(self.main_path + '/whole_sim_init.npy', sim_img)

        # Temporarily "hard-coded" exp interface region for VASP runs
        # Load prev_whole_exp.npy that is from the LAMMPS runs
        # exp_prev = np.load('prev_whole_exp.npy')
        # in y & x directions # TODO: remove hard-coded values
        exp_patch_for_vasp = exp_img[459:584, 249:374] #exp_prev[152:279, 12:]
        #exp_patch_for_vasp = exp_prev
        self.im_ref = exp_patch_for_vasp
        match_ssim = iop.score_ssim(sim_img, self.im_ref)
        print("Score SSIM (POSCAR_init vs exp image): {}".format(match_ssim))

    def evaluate_obj(self, model):
        """
        This function simulated the TEM image of a grain boundary model. Then,
        compares it with the experimental TEM image (target). The objective
        function is (1 - SSIM score) which is assigned as a model attribute
        (obj1_val).

        This function is a part of the API for all classes in
        experimental_simulation module.

        Returns model object

        Args:

        model (obj): structure_record.model() object for which TEM simulation
                     is obtained and a mismatch score is assigned
        """
        relax_path = self.main_path + '/calcs/' + str(model.label) + '/relax'
        # Initialize a Bicrystal object from relaxed structure
        bicrys_model = Bicrystal(poscar_file=relax_path+'/POSCAR_relaxed')
        # Simulate an image
        im_model, __ = bicrys_model.simulate_image(sim_params=self.opt_params)
        np.save(relax_path + '/model_sim.npy', im_model)

        # im_model = im_model[132:300] # TODO: remove hard-coded values
        try:
            score = iop.score_ssim(im_model, self.im_ref)
        except ValueError:
            im_model, im_ref = self.crop_dims(im_model, self.im_ref)
            score = iop.score_ssim(im_model, im_ref)
            print('Adjusted image dimensions for model {}'.format(model.label))

        filename = relax_path+"/gb_im_model.jpg"
        cv2.imwrite(filename, im_model)

        # the order of exp_sims is from Xsim1 -> Xsim2 -> ...
        # Hence, obj1val -> ob2_val -> ... for assigning evaluated sims
        if model.Xsim1 == 'GB_STEM':
            model.obj1_val = float((score)*100)  # Minimizing the obj vals
        elif model.Xsim2 == 'GB_STEM':
            model.obj2_val = float((score)*100)
        elif model.Xsim3 == 'GB_STEM':
            model.obj3_val = float((score)*100)
        elif model.Xsim4 == 'GB_STEM':
            model.obj4_val = float((score)*100)

        return model, score

    def evaluate_obj_two_models(self, model_one, model_two):
        '''
        This function calculates the SSIM for the STEM images of two models.
        Used to cluster models using SSIM scores as distance metrics.

        Returns the SSIM score.

        Args:

        model_one and model_two (model objs): models which are being compared.
        '''
        relax_path_one = self.main_path + '/calcs/' + \
            str(model_one.label) + '/relax'
        relax_path_two = self.main_path + '/calcs/' + \
            str(model_two.label) + '/relax'

        # Load simulated images
        im_one = np.load(relax_path_one + '/model_sim.npy')
        im_two = np.load(relax_path_two + '/model_sim.npy')

        cropped_im_one = im_one[177:259, :]  # expanded from 197:239
        cropped_im_two = im_two[177:259, :]

        # Calculate score
        try:
            score = iop.score_ssim(cropped_im_one, cropped_im_two)
        except ValueError:
            cropped_im_one, cropped_im_two = self.crop_dims(
                cropped_im_one, cropped_im_two)
            score = iop.score_ssim(cropped_im_one, cropped_im_two)
            print(
                f'Adjusted image dimensions for models {model_one.label}'
                + f' and {model_two.label}')

        return float((score) * 100)

    def crop_dims(self, img, ref):
        """
        This function is used to adjust the 'shape' of the simulated image or
        the target image to have them both equal. The dimensions in x, y are
        altered such that minimum number of pixels are lost overall.

        NOTE: Since, the lattice in POSCAR is maintained same across all models
        (ISIF=2), the simulated image should be same (or only different by
        couple of pixels in each dimension)

        Returns the simulated image and target image with equal dimensions

        Args:

        img (2D arr): the simulated TEM image

        ref (2D arr): the target experimental TEM image
        """
        diff_pix_x = img.shape[0] - ref.shape[0]
        diff_pix_y = img.shape[1] - ref.shape[1]

        if diff_pix_x < 0:  # img is smaller & ref should be cropped
            ref = ref[floor(-1*diff_pix_x/2): floor(diff_pix_x/2), :]
        elif diff_pix_x > 0:  # ref is smaller & img should be cropped
            img = img[floor(diff_pix_x/2): floor(-1*diff_pix_x/2), :]

        if diff_pix_y < 0:  # img is smaller & ref should be cropped
            ref = ref[:, floor(-1*diff_pix_y/2): floor(diff_pix_y/2)]
        if diff_pix_y > 0:  # ref is smaller & img should be cropped
            img = img[:, floor(diff_pix_y/2): floor(-1*diff_pix_y/2)]

        return img, ref
