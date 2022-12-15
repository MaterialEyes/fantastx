from __future__ import division, unicode_literals, print_function

from fx19 import distance_check as dc
from scipy import optimize as scipy_optimize
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
import matplotlib.pyplot as plt
try:
    from pyobjcryst import loadCrystal
    from diffpy.srfit.pdf import PDFContribution
    from diffpy.srfit.pdf import DebyePDFGenerator, PDFGenerator
    from diffpy.srfit.fitbase import Profile
    from diffpy.srfit.fitbase import FitRecipe
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
    print('Install scikit-image, Ingrained, opencv for TEM simulation.'
          ' Otherwise ignore..')

try:
    from xtk import simulate, optimization, convolution, processing
    from xtk import distance as xtk_distance
except ImportError:
    print("Install [xtk](https://github.com/MaterialEyes/xtk) to perform "
          " XANES simulations.")

try:
    import GSASIIscriptable as G2sc
except ImportError:
    print('Install GSASIIscriptable for powder diffraction simulation.'
          ' Otherwise ignore..')

from math import floor
import numpy as np
import os
import yaml
import subprocess as sp
from scipy.interpolate import CubicSpline, UnivariateSpline
from ase.data import atomic_numbers
from fx19.fingerprinting import DistanceCalculator
import re
from collections import Counter
from pymatgen.core.lattice import Lattice
import shutil

DEBUG = True


class xanes_of_model(object):
    """
    This class contains functions to calculate XANES spectra, either the
    raw spectra or the difference spectra (essential for XTA analysis, and
    useful for raw XANES analysis as well).

    These functions are all built around the package
    [xtk](https://github.com/MaterialEyes/xtk), which currently supports the
    XANES simulation codes:

    - FDMNES

    - FEFF

    This class also performs post-simulation smoothing and convolution.
    Smoothing is performed with either a cubic or univariate spline, and
    convolution can be performed with either a Gaussian or Lorentzian. For
    the Gaussian, the FWHM must be provided in units of eV. For the
    Lorentzian, the width is energy dependent, and more parameters must be
    provided. For a reference, refer to [this](../xanes) page.

    Arguments:

        xanes_params (dict): A dictionary of parameters used.
    """

    def __init__(self, xanes_params):
        """
        """
        print("Initializing XANES module.")
        # main path as in energy.py
        self.name = 'XANES'
        self.set_params(xanes_params)

    def set_params(self, xanes_params):
        """
        Set all parameters, throwing exceptions for missing parameters.

        Arguments:
            xanes_params (dict): dictionary of all XANES parameters
        """
        try:
            self.main_path = xanes_params['main_path']
            self.simulation_code = xanes_params['simulation_code']
            self.input_yaml_filepath = xanes_params['input_yaml_filepath']
            self.comparison_spectra_type =\
                xanes_params['comparison_spectra_type']
            self.exp_filepath = xanes_params['exp_filepath']

            self.code_folder = xanes_params['code_folder']
            # e.g. "./mpirun_fdmnes -np 4"
            self.exec_cmd = xanes_params['exec_cmd']
        except KeyError:
            raise KeyError('Error, did not provide all essential XANES keys.')

        if self.simulation_code == "FEFF":
            self.simulator = simulate.Feff(self.input_yaml_filepath)
        elif self.simulation_code == "FDMNES":
            self.simulator = simulate.Fdmnes(self.input_yaml_filepath)
        else:
            self.simulator = None

        if self.comparison_spectra_type == "difference":
            if "comp_base_ref_filepath" in xanes_params:
                self.comp_base_ref_filepath =\
                    xanes_params["comp_base_ref_filepath"]
            else:
                raise KeyError('Error, did not provide comp_base_ref_filepath'
                               ' key.')

        if 'exec_cmd' in xanes_params:
            self.exec_cmd = xanes_params['exec_cmd']
        else:
            self.exec_cmd = "./mpirun_fdmnes -np 4"

        # only needed for FEFF
        if 'mpi_cmd' in xanes_params:
            self.mpi_cmd = xanes_params['mpi_cmd']
        else:
            if self.simulation_code == "FEFF":
                raise KeyError('Error, did not provide mpi_cmd key.')
            else:
                self.mpi_cmd = None

        # quantify the difference between the experimental spectra and the
        # simulated spectra
        if 'spectra_distance_metric' in xanes_params:
            try:
                self.dc = xtk_distance.DistanceCalculator(
                    xanes_params['spectra_distance_metric'])
            except AssertionError:
                print("Error. Invalid spectra_distance_metric applied.")
                self.dc = xtk_distance.DistanceCalculator('euclidean')
        else:
            # options are any of those in fingerprinting.DistanceCalculator
            self.dc = xtk_distance.DistanceCalculator('euclidean')

        # set the spline to fit to the spectra
        self.sp = processing.SpectraProcessing(
            spline_mesh=np.arange(7110, 7165, 0.1))
        if 'spline_mesh_params' in xanes_params:
            try:
                spline_min = xanes_params['spline_mesh_params'][0]
                spline_max = xanes_params['spline_mesh_params'][1]
                spline_step = xanes_params['spline_mesh_params'][2]
                self.sp.spline_mesh = np.arange(
                    spline_min, spline_max, spline_step)
            except KeyError:
                print("Error! Missing spline_mesh_param key.")

        # set the convolution parameters
        if 'convolution' in xanes_params:
            try:
                self.convolver = convolution.Convolution(
                    kernel_type=xanes_params['convolution']['kernel'],
                    x_dependence=xanes_params['convolution']['x_dependence'],
                    kernel_fwhm_args=xanes_params['convolution']['arguments'])
            except KeyError:
                raise KeyError("Missing convolution key!")
            try:
                self.extract_fermi_energy =\
                    xanes_params['convolution']['extract_fermi_energy']
            except KeyError:
                print("Error! Missing extract_fermi_energy key.")
                self.extract_fermi_energy = False
        else:
            self.extract_fermi_energy = False
            if self.simulation_code == "FEFF":
                self.convolver = convolution.Convolution(
                    kernel_type="lorentzian",
                    g_ch=0.3,
                    kernel_step=0.2
                )
            elif self.simulation_code == "FDMNES":
                self.extract_fermi_energy = True
                params = {'g_ch': 1.33,
                          'g_max': 15,
                          'e_cent': 23.5,
                          'e_larg': 23.5,
                          'fermi_energy': 0
                          }
                self.convolver = convolution.Convolution(
                    kernel_type="lorentzian",
                    x_dependence="arctan",
                    kernel_fwhm_args=params
                )

        # set optimization parameters
        self.optimize_simulation = True
        if self.optimize_simulation or\
                self.comparison_spectra_type == "difference":
            if 'optimization_params' in xanes_params:
                op = xanes_params['optimization_params']
                self.optimizer = optimization.Optimizer(
                    metric=op['metric'],
                    opt_method=op['opt_method'],
                    opt_options=op['opt_options'],
                    opt_window=op['opt_window']
                )
                self.opt_bounds = op['opt_bounds']
            else:
                self.optimizer = optimization.Optimizer(
                    metric='euclidean'
                )
                self.opt_bounds = {
                    'shift': (-10, 10),
                    'scale': (0.01, 10.0),
                    'g_ch': (0.75, 5),
                    'g_max': (5, 20),
                    'e_cent': (5, 50),
                    'e_larg': (5, 50),
                    'fermi_energy': (-10, 10)}

            self.shift_independently = False
            self.scale_independently = False
            if self.comparison_spectra_type == "difference":
                if 'shift_targ' in self.opt_bounds:
                    self.shift_independently = True
                if 'scale_targ' in self.opt_bounds:
                    self.scale_independently = True

        self.constant_broadening = False

        self.cutting_energy_correction = 0.0  # was -6.0

        # Gather experimental data
        self.exp_data = self.gather_experimental_data(self.exp_filepath)
        self.exp_spline = self.sp.fit_spline(self.exp_data['Energy'],
                                             self.exp_data['Mu'],
                                             type="cubic")

        if self.comparison_spectra_type == "difference":
            self.sim_base_data = simulate.get_experiment_results(
                self.comp_base_ref_filepath, headers=["Energy", 'Mu'],
                data_line=0, sortcolumn=0)
            self.sim_base_spline = self.sp.fit_spline(
                self.sim_base_data['Energy'],
                self.sim_base_data['Mu'],
                type="cubic")

    def _set_optimization_parameters(self, opt_params=None):
        """
        Set the optimization parameters

        Arguments:
            opt_params (dict): optimization parameter settings
        """
        pass

    def gather_experimental_data(self, filepath, delimit=None,
                                 columns=['Energy', 'Mu']):
        """
        Read in the experimental data that will be compared against, and
        perform basic processing steps if necessary.
        """
        if self.comparison_spectra_type == "direct":
            experiment_data = simulate.get_experiment_results(
                filepath, headers=columns, data_line=0, sortcolumn=0,
                delimiter=delimit)
        else:
            if hasattr(filepath, "__iter__") and type(filepath) is not str:
                experiment_data_base = simulate.get_experiment_results(
                    filepath[0], headers=columns, data_line=0, sortcolumn=0,
                    delimiter=delimit)
                experiment_data = simulate.get_experiment_results(
                    filepath[1], headers=columns, data_line=0, sortcolumn=0,
                    delimiter=delimit)
                experiment_data['Mu'] = experiment_data['Mu']\
                    - experiment_data_base['Mu']
            else:
                # hitting direct comparison
                experiment_data = simulate.get_experiment_results(
                    filepath, headers=columns, data_line=0, sortcolumn=0,
                    delimiter=delimit)
        return experiment_data

    def run_simulation(self, model):
        """
        Prepare and run XANES simulation using xtk.
        """
        simulation_path = model.relax_path + "/" + self.simulation_code
        exec_cmd = self.exec_cmd.split()
        os.mkdir(simulation_path)
        # iron_three = False
        # oxygen = False
        # for i in model.astr.sites:
        #     if i.specie.symbol == "Fe":
        #         if i.specie.oxi_state == 3:
        #             iron_three = True
        #     elif i.specie.symbol == "O":
        #         oxygen = True
        # if iron_three and oxygen:
        #     self.simulator.param_dict['feff_cards']['ION'] =\
        #         '0 0.15\nION 1 0.05\nION 2 0.05\nION 3 -0.03\nION 4 -0.03'
        # elif iron_three and not oxygen:
        #     self.simulator.param_dict['feff_cards']['ION'] =\
        #         "0 0.15\nION 1 0.05\nION 2 0.05"
        # elif oxygen and not iron_three:
        #     self.simulator.param_dict['feff_cards']['ION'] =\
        #         "3 -0.03\nION 4 -0.03"
        # else:
        #     self.simulator.param_dict['ION'] = None
        self.simulator.prepare_simulation(
            model.astr, simulation_path, True, self.code_folder, self.mpi_cmd)

        if not DEBUG:
            results = self.simulator.run(exec_cmd, simulation_path)

            # clean the simulation directory after use if FEFF
            if self.simulation_code == "FEFF":
                shutil.copy(simulation_path + "/FEFF/xmu.dat", simulation_path + "/xmu.dat")
                shutil.copy(simulation_path + "/FEFF/feff.inp", simulation_path + "/feff.inp")
                shutil.rmtree(simulation_path + "/FEFF")
            return results

    def evaluate_obj(self, model):
        """
        This function performs the full XANES simulation of the model.
        It prepares the input file, executes the simulation code, and
        then calculates the user-specified objective function. This
        objective can be a straightforward comparison between the
        experimental and simulated spectra, using a user-specified
        distance calculation. Alternatively, it can be a comparison
        between the experimental and simulated difference spectra. This
        is essential to comparing to XTA experiments, where the spectra
        is the time-resolved difference between the excited spectra and
        the reference spectra.

        Which mode is used depends on the users choice of the
        `comparison_spectra_type` variable, which is defined in the
        FANTASTX yaml.

        This function is a part of the API for all classes in
        experimental_simulation module.

        Arguments:
            model (obj): structure_record.model() object for which the XANES
             simulation is obtained and a mismatch score is assigned

        Returns:
            (model obj, float):
            - the model whose objective was calculated
            - the objective value
        """

        if not DEBUG:
            results = self.run_simulation(model)

            lowest_distance = np.inf
            for n, i in enumerate(results):
                x_result = i['x_array']
                y_result = i['y_array'] if 'tddft_y_array' not in i else\
                    i['tddft_y_array']
                spline_result = self.sp.fit_spline(
                    x_result, y_result, type="cubic")

                if self.extract_fermi_energy and\
                        self.simulation_code == "FDMNES":
                    # set fermi energy of convolver. Add edge energy as
                    # "fermi_energy" is relative to the energy
                    self.convolver.fermi_energy =\
                        i['edge_energy'] + i['fermi_energy']

                    if self.opt_bounds['fermi_energy'][1] <\
                            self.convolver.fermi_energy:
                        self.opt_bounds['fermi_energy'][0] += i['edge_energy']
                        self.opt_bounds['fermi_energy'][1] += i['edge_energy']

                # options:
                # - align spectra (optimizing scale and shift)
                # - convolve spectra (optimize or not)
                distance = np.inf
                if self.comparison_spectra_type == "direct":
                    if not self.optimize_simulation:
                        # convolve spectra
                        convolved_y = self.convolver.convolve_function(
                            self.sp.spline_mesh, spline_result)
                        # align spectra
                        exp_peaks = self.sp.locate_peaks(
                            self.sp.spline_mesh, self.exp_spline)
                        aligned_x, aligned_y = self.sp.match_first_peak(
                            self.sp.spline_mesh, convolved_y, exp_peaks)
                        spline_result = self.sp.fit_spline(
                            aligned_x, aligned_y, type="cubic")
                        distance = self.dc.calculate(
                            spline_result, self.exp_spline)
                    else:
                        # optimize
                        spline_result, result =\
                            self.optimizer.optimize_post_simulation_parameters(
                                self.sp.spline_mesh,
                                spline_result,
                                self.exp_spline,
                                self.convolver,
                                self.sp,
                                self.opt_bounds
                            )
                        distance = result.fun
                elif self.comparison_spectra_type == "difference":
                    if not self.optimize_simulation:
                        # convolve spectra
                        convolved_y = self.convolver.convolve_function(
                            self.sp.spline_mesh, spline_result)
                        spline_diff = convolved_y - self.sim_base_spline
                        distance = self.dc.calculate(
                            spline_diff, self.exp_spline)

                    else:
                        spline_result, spline_diff, result =\
                            self.optimizer.optimize_difference(
                                spline_result,
                                self.sim_base_spline,
                                self.exp_spline,
                                self.convolver,
                                self.sp,
                                self.opt_bounds,
                                shift_independently=self.shift_independently,
                                scale_independently=self.scale_independently)
                        distance = result.fun

                if distance < lowest_distance:
                    lowest_distance = distance

                print(f"Score for run {n}: "
                      f"{float((distance)*100)}")
                np.save(model.relax_path + "/model_sim_spectra_" +
                        str(n) + ".npy", spline_result)

                # if self.comparison_spectra_type == "direct":
                fig, axes = plt.subplots(1, 1)
                fig.set_size_inches(10, 10)
                axes.plot(self.sp.spline_mesh,
                          self.sim_base_spline, marker=".",
                          linestyle="-", label="Simulation base")
                axes.plot(self.sp.spline_mesh, spline_result, marker=".",
                          linestyle="--", label="Simulation result")
                axes.set_ylabel("Absorbance (arbitrary units)", fontsize=24)
                axes.set_xlabel(
                    "Energy (eV)", fontsize=24)
                axes.set_xlim(
                    (self.sp.spline_mesh[0], self.sp.spline_mesh[-1]))
                axes.set_ylim((0, 2.5))
                axes.legend(bbox_to_anchor=(0.48, 0.85),
                            loc="lower left", fontsize=20)
                plt.setp(axes.get_xticklabels(), fontsize=20)
                plt.setp(axes.get_yticklabels(), fontsize=16)
                filename = model.relax_path + "/" +\
                    "experiment_vs_sim_spectra_" +\
                    str(n) + ".png"
                plt.savefig(filename, format="png", dpi=300)

                if self.comparison_spectra_type == "difference":
                    fig, axes = plt.subplots(1, 1)
                    fig.set_size_inches(10, 10)
                    axes.plot(self.sp.spline_mesh, self.exp_spline, marker=".",
                              linestyle="-", label="Experiment")
                    axes.plot(self.sp.spline_mesh, spline_diff, marker=".",
                              linestyle="--", label=self.simulation_code)
                    axes.set_ylabel(
                        r"$\Delta$ Absorbance (arbitrary units)", fontsize=24)
                    axes.set_xlabel(
                        "Energy (eV)", fontsize=24)
                    axes.set_xlim(
                        (self.sp.spline_mesh[0], self.sp.spline_mesh[-1]))
                    exp_y_max = np.amax(self.exp_spline)
                    exp_y_min = np.amin(self.exp_spline)
                    axes.set_ylim((1.1*exp_y_min, 1.1*exp_y_max))
                    axes.legend(bbox_to_anchor=(0.48, 0.85),
                                loc="lower left", fontsize=20)
                    plt.setp(axes.get_xticklabels(), fontsize=20)
                    plt.setp(axes.get_yticklabels(), fontsize=16)
                    filename = model.relax_path + "/" +\
                        "experiment_vs_sim_spectra_diff_" +\
                        str(n) + ".png"
                    plt.savefig(filename, format="png", dpi=300)

            if model.Xsim1 == 'XANES':
                # Minimizing the obj vals
                model.obj1_val = float((lowest_distance)*100)
            elif model.Xsim2 == 'XANES':
                model.obj2_val = float((lowest_distance)*100)
            elif model.Xsim3 == 'XANES':
                model.obj3_val = float((lowest_distance)*100)
            elif model.Xsim4 == 'XANES':
                model.obj4_val = float((lowest_distance)*100)

            return model, lowest_distance
        else:
            self.run_simulation(model)
            lowest_distance = np.random.uniform(0, 5)
            if model.Xsim1 == 'XANES':
                # Minimizing the obj vals
                model.obj1_val = float((lowest_distance)*100)
            elif model.Xsim2 == 'XANES':
                model.obj2_val = float((lowest_distance)*100)
            elif model.Xsim3 == 'XANES':
                model.obj3_val = float((lowest_distance)*100)
            elif model.Xsim4 == 'XANES':
                model.obj4_val = float((lowest_distance)*100)

            return model, lowest_distance


class pdf_of_model(object):
    """
    This class contains functions to calculate the PDF and fit it to the
    experimental pdf. Uses the residual to calculate objective function.

    Arguments:

        pdf_params (dict): A dictionary of parameters used for fitting PDF
         using **Diffpy**.
    """

    def __init__(self, pdf_params):
        """
        """
        # main path as in energy.py
        self.name = 'PDF'
        self.main_path = pdf_params['main_path']
        self.pdf_sim_dir = None

        # Default isotropic ADP value for all species
        self.Biso_val = 0.71
        # default structure scale factor
        self.scale = 1.0
        # quadratic term related to sharpness of first peak
        # (from pdfgui manual)
        self.delta2 = 3.87
        # exp. instrument (peak-damping) parameter (default from pdfgui manual)
        self.qdamp = 0.043  # G(r) intensity decereases with r
        self.fit_coords = False
        # default bounds_dict
        lb_ub_dict = {}
        # note: can alternately define bounds by [a_low_bound, a_high_bound]
        lb_ub_dict['a'] = 0.02
        lb_ub_dict['b'] = 0.02
        lb_ub_dict['c'] = 0.02
        # Biso_val = 8*pi**2 * Uiso_val
        lb_ub_dict['Biso_val'] = [78.96*0.00001, 200.096*0.01]
        lb_ub_dict['scale'] = [0.5, 1.5]
        lb_ub_dict['delta2'] = [2.0, 5.0]
        lb_ub_dict['qdamp'] = [0.001, 0.1]
        self.var_bounds = lb_ub_dict

        # minimization method from scipy_optimize.minimize i.e., one of strings
        # ['L-BFGS-B', 'SLSQP']
        self.minimize_method = 'L-BFGS-B'  # or 'SLSQP' only
        # Range parameters of the PDF function (x-axis)
        self.xmin = 0.5
        self.xmax = 10.0
        self.dx = 0.01
        # PDF Qmin and Qmax
        self.Qmin = 0.0  # G(r) goes below zero for Qmin > 0
        self.Qmax = 50.0  # G(r) gets wavy for smaller Qmax
        # elemental symbols of species as a list
        self.symbols = None

        # path to experimental pdf file
        self.exp_pdf_file = pdf_params['exp_pdf_file']
        # values for initialization of minimization of scalar function
        # Biso vals for all species
        if 'Biso_val' in pdf_params:
            self.Biso_val = pdf_params['Biso_val']
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
            self.Qmin = pdf_params['Qmin']
        if 'Qmax' in pdf_params:
            self.Qmax = pdf_params['Qmax']

        # set lower and upper bounds for fitting variables
        if 'var_bounds' in pdf_params:
            vbs = pdf_params['var_bounds']
            for a_key in vbs.keys():
                self.var_bounds[a_key] = vbs[a_key]

        self.min_box_abc = None
        if 'min_box_abc' in pdf_params:
            self.min_box_abc = pdf_params['min_box_abc']

        # tolerance to relax each x/y/z coordinate of an atom coordinates
        self.coord_tol = 0.1
        if 'coord_tol' in pdf_params:
            self.coord_tol = pdf_params['coord_tol']

        self.make_supercell = False
        self.periodic = True

    def write_temp_cif(self, model):
        """
        Writes temp.cif file in the pdf simulation directory from model.astr

        Arguments:

            model (obj): structure_record.model() object for which the PDF
            simulation will be done
        """
        # use the relaxed structure from energy calculation
        astr = model.astr.copy()
        if self.make_supercell:
            astr.make_supercell((2, 2, 2))
        if self.min_box_abc is not None:
            for axis in range(3):
                species = astr.species
                if astr.lattice.abc[axis]/self.min_box_abc[axis] < 1:

                    new_cart_coords = astr.cart_coords.copy().tolist()
                    translate_to_center = self.min_box_abc[axis]/2 -\
                        astr.lattice.abc[axis]/2
                    for i in new_cart_coords:
                        if i[axis] < 0:
                            i[axis] += astr.lattice.abc[axis]
                        i[axis] += translate_to_center

                    latt_matrix = astr.lattice.matrix
                    new_latt_matrix = latt_matrix.copy()
                    new_latt_matrix[axis][axis] = self.min_box_abc[axis]
                    new_latt = Lattice(new_latt_matrix)
                    astr.lattice = new_latt

                    for i, new_coords in enumerate(new_cart_coords):
                        specie = species[i]
                        astr.replace(i, specie, new_coords,
                                     coords_are_cartesian=True)

        self.symbols = astr.symbol_set

        # write new_structure to a temporary cif file
        # diffpy only able to read cif format
        temp_init = self.pdf_sim_dir + '/temp_init.cif'
        cif_writer = CifWriter(astr)
        cif_writer.write_file(temp_init)

    def get_PDFContribution_obj(self, cif_file):
        """
        Make diffpy.srfit.pdfcontribution.PDFContribution object from temp.cif
        """
        # Make profile object
        profile = Profile()
        # Add data through parser to the profile
        profile.loadtxt(self.exp_pdf_file)
        profile.setCalculationRange(xmin=self.xmin, xmax=self.xmax, dx=self.dx)

        # Make generator object
        diffpy_str = loadCrystal(cif_file)
        if self.periodic:
            generator = PDFGenerator('generator_name')
        else:
            generator = DebyePDFGenerator('generator_name')
        generator.setStructure(diffpy_str, periodic=self.periodic)
        generator.setQmax(self.Qmax)
        generator.setQmin(self.Qmin)

        # Make contribution object
        # The FitContribution
        contribution = PDFContribution("contribution_name")
        contribution.addProfileGenerator(generator)
        contribution.setProfile(profile, xname="r")

        return contribution

    def fit_variables_recipe(self, contribution, fitted_params=None):
        """
        Performs optimization of PDF variables like qdamp, delta2, scale and
        ADP (Bisos) using diffpy.srfit.fitbase.FitRecipe object.

        This function call should be preceeded by get_PDF_obj function.

        !!! note

            Made PDF and Fit two separate functions for convenience

        Args:

            PDF (PDFContribution) - PDF object from get_PDF_obj

        Returns:

            tuple:
            - fitted_params after PDF fit
            - residual
        """
        symbols = self.symbols
        generator = contribution.generator_name
        recipe = FitRecipe()
        recipe.addContribution(contribution)
        """
        # if fitted_params is given, use it
        if fitted_params is not None:
            Biso_val, init_scale, init_delta2, init_qdamp = fitted_params
        else:
            Biso_val, init_scale, init_delta2, init_qdamp = \
                        self.Biso_val, self.scale, self.delta2, self.qdamp
        """
        # Add the three lattice vectors as variables to the fit recipe
        spacegroup_pars = contribution.generator_name.phase.sgpars
        lattice = contribution.generator_name.phase.getLattice()

        a = lattice.a.getValue()
        b = lattice.b.getValue()
        c = lattice.c.getValue()

        # If we have a cubic lattice then constrain it to remain cubic
        cubic_lattice = False
        alpha = lattice.alpha.getValue()
        beta = lattice.beta.getValue()
        gamma = lattice.gamma.getValue()
        if np.isclose(a, b) and np.isclose(a, c):
            if np.isclose(alpha, np.pi/2) and\
                np.isclose(beta, np.pi/2) and\
                    np.isclose(gamma, np.pi/2):
                cubic_lattice = True
                lattice.constrain(lattice.b, lattice.a)
                lattice.constrain(lattice.c, lattice.a)
                lattice.alpha.setConst(True, np.pi/2.)
                lattice.beta.setConst(True, np.pi/2.)
                lattice.gamma.setConst(True, np.pi/2.)

        # allow diffpy to shift the lattice parameters
        for par in spacegroup_pars.latpars:
            if par.name in ['a', 'b', 'c']:
                recipe.addVar(par, fixed=False)

        Bisos = []
        for sym in symbols:
            Bisos.append('Biso{}'.format(sym))
            recipe.newVar('Biso{}'.format(sym),
                          value=self.Biso_val,
                          fixed=False).boundRange(
                              self.var_bounds['Biso_val'][0],
                              self.var_bounds['Biso_val'][1])

        for scatterer in contribution.generator_name.phase.scatterers:
            for sym in symbols:
                if sym in scatterer.name:
                    recipe.constrain(scatterer._parameters['Biso'],
                                     'Biso{}'.format(sym))

        # Set all Biso values to provided or default Biso_val
        # for p in Bisos:
        #    fit_param = recipe._parameters[p]
        #    fit_param.setValue(self.Biso_val)

        # add existing PDF variables as Fit parameters and setValue
        recipe.addVar(generator.scale, self.scale, fixed=False)
        recipe.addVar(generator.delta2, self.delta2, fixed=False)
        recipe.addVar(generator.qdamp, self.qdamp, fixed=False)

        # Set lower and upper bounds for variables
        # For lattice parameters, set bound for each parameter
        if type(self.var_bounds['a']) is list:
            recipe.a.bounds = self.var_bounds['a']
        else:
            recipe.a.bounds = [a - self.var_bounds['a']*a,
                               a + self.var_bounds['a']*a]

        if not cubic_lattice:
            if type(self.var_bounds['b']) is list:
                recipe.b.bounds = self.var_bounds['b']
            else:
                recipe.b.bounds = [b - self.var_bounds['b']*b,
                                   b + self.var_bounds['b']*b]
            if type(self.var_bounds['c']) is list:
                recipe.c.bounds = self.var_bounds['c']
            else:
                recipe.c.bounds = [c - self.var_bounds['c']*c,
                                   c + self.var_bounds['c']*c]
        recipe.scale.bounds = self.var_bounds['scale']
        recipe.delta2.bounds = self.var_bounds['delta2']
        recipe.qdamp.bounds = self.var_bounds['qdamp']

        # Turn all bounded parameters into restraints with uncertainty sigma
        recipe.boundsToRestraints(sig=0.001)
        # Turn off printout of iteration number.
        recipe.clearFitHooks()

        # For finding stretch residual
        result = scipy_optimize.minimize(recipe.scalarResidual,
                                         recipe.getValues(),
                                         method=self.minimize_method,
                                         tol=1e-3,
                                         options={'maxiter': 100000})
        # get residue from the fitted params
        fitted_params = result.x
        residual = recipe.scalarResidual(fitted_params)

        return fitted_params, residual, recipe

    def fit_coords_recipe(self, contribution, fitted_params):
        """
        Performs optimization ofatomic coordiantes of a structure.
        This optimizes previous variables along with atomic coordinates.
        Take the energy_code
        relaxed structure, and create a PDF object. Then perform
        fit_variables_recipe() using the resulting fitted coordinates from
        this.

        The PDF object should contain the optimum variables for the main four
        variables obtained from fit_variables_recipe()

        Returns nothing. (Writes temp_opt.cif to the pdf_sim_dir)

        Arguments:

            PDF (PDFContribution) - PDF object from get_PDF_obj
        """
        symbols = self.symbols
        generator = contribution.generator_name
        recipe = FitRecipe()
        recipe.addContribution(contribution)

        # Add the three lattice vectors as variables to the fit recipe
        spacegroup_pars = contribution.generator_name.phase.sgpars
        lattice = contribution.generator_name.phase.getLattice()

        a = lattice.a.getValue()
        b = lattice.b.getValue()
        c = lattice.c.getValue()
        # If we have a cubic lattice then constrain it to remain cubic
        cubic_lattice = False
        alpha = lattice.alpha.getValue()
        beta = lattice.beta.getValue()
        gamma = lattice.gamma.getValue()
        if np.isclose(a, b) and np.isclose(a, c):
            if np.isclose(alpha, np.pi/2) and\
                np.isclose(beta, np.pi/2) and\
                    np.isclose(gamma, np.pi/2):
                cubic_lattice = True
                lattice.constrain(lattice.b, lattice.a)
                lattice.constrain(lattice.c, lattice.a)
                lattice.alpha.setConst(True, np.pi/2.)
                lattice.beta.setConst(True, np.pi/2.)
                lattice.gamma.setConst(True, np.pi/2.)

        for par in spacegroup_pars.latpars:
            if par.name in ['a', 'b', 'c']:
                recipe.addVar(par, fixed=False)

        Bisos = []
        for sym in symbols:
            Bisos.append('Biso{}'.format(sym))
            recipe.newVar('Biso{}'.format(sym),
                          value=self.Biso_val,
                          fixed=False).boundRange(
                              self.var_bounds['Biso_val'][0],
                              self.var_bounds['Biso_val'][1])

        for scatterer in contribution.generator_name.phase.scatterers:
            for sym in symbols:
                if sym in scatterer.name:
                    recipe.constrain(scatterer._parameters['Biso'],
                                     'Biso{}'.format(sym))

        # add existing PDF variables as Fit parameters and setValue
        recipe.addVar(generator.scale, self.scale, fixed=False)
        recipe.addVar(generator.delta2, self.delta2, fixed=False)
        recipe.addVar(generator.qdamp, self.qdamp, fixed=False)

        # Set lower and upper bounds for variables
        # For lattice parameters, set bound for each parameter
        if type(self.var_bounds['a']) is list:
            recipe.a.bounds = self.var_bounds['a']
        else:
            recipe.a.bounds = [a - self.var_bounds['a']*a,
                               a + self.var_bounds['a']*a]

        if not cubic_lattice:
            if type(self.var_bounds['b']) is list:
                recipe.b.bounds = self.var_bounds['b']
            else:
                recipe.b.bounds = [b - self.var_bounds['b']*b,
                                   b + self.var_bounds['b']*b]
            if type(self.var_bounds['c']) is list:
                recipe.c.bounds = self.var_bounds['c']
            else:
                recipe.c.bounds = [c - self.var_bounds['c']*c,
                                   c + self.var_bounds['c']*c]
        recipe.scale.bounds = self.var_bounds['scale']
        recipe.delta2.bounds = self.var_bounds['delta2']
        recipe.qdamp.bounds = self.var_bounds['qdamp']

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
        for i, pdf_atom in enumerate(contribution.generator_name.phase.scatterers):
            for cc in ['x', 'y', 'z']:
                vname = cc + '_' + pdf_atom.name
                cc_var = pdf_atom.get(cc)
                if int(i) == int(center_ind):
                    recipe.addVar(cc_var, name=vname, tag='xyz', fixed=True)
                else:
                    recipe.addVar(cc_var, name=vname, tag='xyz', fixed=False)

        # add bounds for new variables of atom coords
        for par in recipe.iterPars(r"x_|y_|z_"):
            par.bounds = [par.value-self.coord_tol, par.value+self.coord_tol]

        # Turn all bounded parameters into restraints with uncertainty sigma
        recipe.boundsToRestraints(sig=0.001)
        # Turn off printout of iteration number.
        recipe.clearFitHooks()

        # For finding stretch residual
        result = scipy_optimize.minimize(recipe.scalarResidual,
                                         recipe.getValues(),
                                         method=self.minimize_method,
                                         tol=1e-3,
                                         options={'maxiter': 100000})
        # get residue from the fitted params
        fitted_params = result.x
        residual = recipe.scalarResidual(fitted_params)

        # Write output structure with new coordinates
        fcs = result.x[-1*(len(pymat_str)-1)*3:]
        fcs_new = np.insert(fcs, center_ind,
                            pymat_str.frac_coords[center_ind], axis=0)
        # fcs_new = np.concatenate((pymat_str.frac_coords[center_ind], fcs))
        fcs_new = fcs_new.reshape(len(pymat_str), 3)
        astr_varied = Structure(pymat_str.lattice, pymat_str.species,
                                fcs_new, coords_are_cartesian=False)
        opt_cif = self.pdf_sim_dir + '/temp_opt.cif'
        opt_pos = self.pdf_sim_dir + '/POSCAR_opt_pdf'
        astr_varied.to(filename=opt_cif)
        astr_varied.to(filename=opt_pos)

        return fitted_params, residual, recipe

    def plot_pdf(self, recipe, name):
        """
        """
        # Plotting pdf
        r = recipe.contribution_name.profile.x
        g_obs = recipe.contribution_name.profile.y
        g_calc = recipe.contribution_name.evaluate()

        g_diff = g_obs - g_calc

        diffzero = -0.8 * max(g_obs) * np.ones_like(g_obs)
        diff = g_obs - g_calc + diffzero

        pdf_data_name = name + '_pdf_data.txt'
        with open(self.pdf_sim_dir + '/' + pdf_data_name, 'w') as f:
            f.write('radius \tg_exp \tg_calc \tg_diff \n')
            for i, item in enumerate(r):
                f.write(str(r[i])[:4] + '\t ' + str(g_obs[i])[:6] + '\t '
                        + str(g_calc[i])[:6] + '\t ' + str(g_diff[i])[:6]
                        + ' \n')

        plt.figure(figsize=(10, 6))
        plt.plot(r, g_obs, 'bo', label="G(r) Target")
        plt.plot(r, g_calc, 'r-', label="G(r) Fit", linewidth=3)
        plt.plot(r, diff, 'c-', label="G(r) diff", linewidth=3)
        plt.plot(r, diffzero, 'k-', linewidth=1)
        plt.xlabel(r"$r (\AA)$", fontsize=20)
        plt.ylabel(r"$G (\AA^{-2})$", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(loc=1, fontsize=15)
        plt.tight_layout()

        pdf_plot_name = name + '_pdf_plot.png'
        plt.savefig(fname=self.pdf_sim_dir + '/' + pdf_plot_name)
        plt.close()

    def evaluate_obj(self, model):
        """
        Fits calculated and experimental pdf and returns model. Uses the
        Residual as the objective function value for pdf. Objective function
        value is added to model attributes obj1_val.

        Arguments:

            model (obj): structure_record.model() object for which energy
             evaluation will be done

        Returns:

            tuple:
            - model
            - fitted_params after PDF fit
        """
        main_path = self.main_path
        pdf_sim = main_path + '/calcs/' + str(model.label) + '/pdf_sim'
        os.mkdir(pdf_sim)
        self.pdf_sim_dir = pdf_sim

        # write temp_init.cif to pdf_sim_dir
        self.write_temp_cif(model)
        # load exp_pdf and temp_init.cif to PDF object
        cif_file = pdf_sim + '/temp_init.cif'
        contribution = self.get_PDFContribution_obj(cif_file)

        # NOTE: First fit the main four variables only. Second fit main four +
        # coords as variables. This is to get best solution wrt main variables.
        # Then some local solution with second fitting..
        fitted_params, residual, recipe = self.fit_variables_recipe(
            contribution)
        # write initial fitted parameters to a file
        with open(pdf_sim + '/initial_fitted_params.txt', 'w') as f:
            for item in fitted_params:
                f.write(str(float(item)) + '\n')
            f.write('Residual: {}'.format(residual))
        self.plot_pdf(recipe, 'initial')

        if self.fit_coords is True:
            # reset contribution to fit again with coords
            contribution = self.get_PDFContribution_obj(cif_file)
            # fit the atomic coordinates using Diffpy
            fitted_params, residual, recipe = self.fit_coords_recipe(
                contribution, fitted_params)
            # # write initial fitted parameters to a file
            with open(pdf_sim + '/final_fitted_params.txt', 'w') as f:
                for item in fitted_params[:7]:
                    f.write(str(item) + '\n')
                f.write('Residual: {}'.format(residual))
            self.plot_pdf(recipe, 'final')

        # fit the PDF variables
        # fitted_params, residual, Fit = self.fit_variables_recipe(PDF,
        #                            cif_file, fitted_params=fitted_params)
        # CK Debug
        # print ('step 4: \n', fitted_params)
        # self.plot_pdf(Fit, 'step_4')

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
        if self.dm3_path is None:
            print('Error! No path (dm3_path) provided to experimental image')
            print('Trying to refer to hard-coded reference ')
            print('"inputs/whole_exp.npy" instead.')
            if os.path.exists(self.main_path + '/inputs/whole_exp.npy'):
                exp_prev = np.load(self.main_path + '/inputs/whole_exp.npy')
                if exp_prev.ndim == 3:
                    exp_prev = np.mean(exp_prev, axis=2)
                    self.im_ref = exp_prev[:, :-1]
                else:
                    self.im_ref = exp_prev
            else:
                print('No hard coded reference file found.')
        else:
            # Prepare experimental image
            # (make sure this procedure matches the procedure in 'run.py')
            image_data = iop.image_open(self.dm3_path)
            exp_img = iop.apply_rotation(image_data['Pixels'], 1)
            exp_img = iop.scale_pixels(exp_img, mode='rescale')
            exp_img = restoration.wiener(exp_img, np.ones((7, 7))/3.5, 1300)
            exp_img = equalize_adapthist(exp_img, clip_limit=0.005)

            bicrys_ref = Bicrystal(poscar_file=self.init_gb_path)
            congruity = CongruityBuilder(sim_obj=bicrys_ref, exp_img=exp_img)

            # Get ingrained optimization parameters from the best fit in
            # the ingrained progress file
            if self.progress_file:
                progress = np.genfromtxt(self.progress_file, delimiter=',')
                best_idx = int(np.argmin(progress[:, -1]))
                x = progress[best_idx]
                xfit = x[1:-1]
                xfit = [a for a in xfit[:-2]] + [int(a) for a in xfit[-2::]]

            # TODO: Find why we set self.opt_params[1] = 0
            if self.opt_params is None:
                if self.progress_file is None:
                    print("Error! No ingrained optimization parameters"
                          " provided, and no ingrained progress file found!")
                    xfit = None
                else:
                    self.opt_params = xfit.copy()
                    self.opt_params[1] = 0
            else:
                xfit = self.opt_params.copy()
            xfit[1] = 0
            sim_img, sim_struct, exp_patch, shift_score, stable_idxs = \
                congruity.fit_gb(sim_params=xfit, bias_y=1E-4)

            sim_struct.to(filename='POSCAR_init_fitted', fmt='poscar')

            np.save(self.main_path + '/whole_exp.npy', exp_patch)
            np.save(self.main_path + '/whole_sim_init.npy', sim_img)

            # in y & x directions # TODO: remove hard-coded values
            # exp_patch_for_vasp = exp_img[459:584,
            #                              249:374]  # exp_prev[152:279, 12:]

            if exp_prev.ndim == 3:
                exp_prev = np.mean(exp_prev, axis=2)
                self.im_ref = exp_prev[:, :-1]
            else:
                self.im_ref = exp_prev

        self.do_scell = True   # Set to False if using a 1x3 supercell
        # Make sim TEM from init_gb
        if self.do_scell:
            # ss = Structure.from_file(self.init_gb_path)
            # ss.make_supercell((1, 3, 1))
            # temp_file = self.main_path + '/inputs/POSCAR_temp_scell'
            # self.temp_file = temp_file
            # ss.to(filename=temp_file)
            bicrys_model = Bicrystal(poscar_file=self.init_gb_path)
            bicrys_model.structure.make_supercell((1, 3, 1))
        else:
            bicrys_model = Bicrystal(poscar_file=self.init_gb_path)
        sim_img, __ = bicrys_model._get_image_cell(
            defocus=self.opt_params[2],
            interface_width=self.opt_params[1],
            pix_size=self.opt_params[0],
            view=False)
        sim_img = sim_img[32:132]

        try:
            match_ssim = iop.score_ssim(sim_img, self.im_ref)
        except ValueError:
            sim_img, im_ref = self.crop_dims(sim_img, self.im_ref)
            match_ssim = iop.score_ssim(sim_img, im_ref)
            print('Adjusted image dimensions for initial model')
        # match_ssim = iop.score_ssim(sim_img, self.im_ref)
        print("Score SSIM (POSCAR_init vs exp image): {}".format(match_ssim))

    def evaluate_obj(self, model):
        """
        This function simulated the TEM image of a grain boundary model. Then,
        compares it with the experimental TEM image (target). The objective
        function is (1 - SSIM score) which is assigned as a model attribute
        (obj1_val).

        This function is a part of the API for all classes in
        experimental_simulation module.

        Arguments:

            model (obj): structure_record.model() object for which TEM
             simulation is obtained and a mismatch score is assigned

        Returns:

            (structure_record.model(), float):
            - The model object being evaluated
            - the SSIM score which is the objective
        """
        relax_path = self.main_path + '/calcs/' + str(model.label) + '/relax'
        if self.do_scell is True:
            bicrys_model = Bicrystal(poscar_file=relax_path+'/POSCAR_relaxed')
            bicrys_model.structure.make_supercell((1, 3, 1))
        else:
            bicrys_model = Bicrystal(poscar_file=relax_path+'/POSCAR_relaxed')
        # Simulate an image
        # Initialize a Bicrystal object from relaxed structure
        # bicrys_model = Bicrystal(poscar_file=relax_path+'/POSCAR_relaxed')
        # Simulate an image
        im_model, __ = bicrys_model._get_image_cell(
            defocus=self.opt_params[2],
            interface_width=self.opt_params[1],
            pix_size=self.opt_params[0],
            view=False)
        im_model = im_model[32:132]
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

        Arguments:

            model_one (structure_record.model()): first model being compared

            model_two (structure_record.model()): second model being compared

        Returns:

            float: the SSIM score which is the objective
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
        This function is used to adjust the `shape` of the simulated image or
        the target image to have them both equal. The dimensions in x, y are
        altered such that minimum number of pixels are lost overall.

        !!! note
            Since, the lattice in POSCAR is maintained same across all models
            (ISIF=2), the simulated image should be same (or only different by
            couple of pixels in each dimension)

        Returns the simulated image and target image with equal dimensions

        Arguments:

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


class xrd_of_model(object):
    """
    This class contains functions to calculate the powder diffraction pattern (neutron or X-ray) of a crystal structure and calculate the similarity descriptor against experimental data.

    Arguments:

        xrd_params (dict): A dictionary of parameters used for simulating XRD
         using **GSASII scriptable**.
    """

    def __init__(self, xrd_params):
        """
        """

        # main path as in energy.py
        self.name = 'XRD'
        self.main_path = xrd_params['main_path']
        self.xrd_sim_dir = None

        print(xrd_params)
#        open('params', 'w').write(str(xrd_params))

        # path to provided files
        self.exp_xrd_file = xrd_params['exp_xrd_file']
        self.instr_param_file = xrd_params['instr_param_file']

        # GSAS related arguments
        self.xmin = 15
        self.xmax = 65
        self.npoints = 1250
        self.xmin_fit = 20
        self.xmax_fit = 60
        self.npoints_fit = 2001
        self.scale = 100
        self.score_method = 'res_fit'

        if 'xmin' in xrd_params:
            self.xmin = xrd_params['xmin']  # min x value for simulation
        if 'xmax' in xrd_params:
            self.xmax = xrd_params['xmax']  # max x value for simulation
        if 'npoints' in xrd_params:
            # make sure it is a integer
            self.npoints = int(xrd_params['npoints'])
        if 'xmin_fit' in xrd_params:
            # min x value for fitting/normalization
            self.xmin_fit = xrd_params['xmin_fit']
        if 'xmax_fit' in xrd_params:
            # min x value for fitting/normalization
            self.xmax_fit = xrd_params['xmax_fit']
        if 'npoints_fit' in xrd_params:
            # make sure it is a integer
            self.npoints_fit = int(xrd_params['npoints_fit'])
        if 'scale' in xrd_params:
            self.scale = xrd_params['scale']  # scaling factor for histogram

    def read_histogram(self, filename):
        """
        Read the simulated histogram data from GSASII-genrated file

        Args:
            filename (string): absolute path to the histogram data file

        Returns:
            Array: (N,2) array of floats
        """
        data_raw = open(filename).readlines()
        flag = [data_raw.index(l) for l in data_raw if 'weight' in l][0]
        data_raw = data_raw[flag+1:]
        data_raw = [[eval(n) for n in l[:-1].split(',')] for l in data_raw]
        return np.array(data_raw)[:, :2]

    def xrd_normalize(self, data, scale='minmax'):
        if scale == 'minmax':
            return (data - data.min())/(data.max()-data.min())
        if scale == 'freq':
            return (data - data.min())/(data - data.min()).max()

    def xrd_similarity_metrics_new(self, data_exp, data_sim, scale='minmax'):
        """
        Calculate the similarity metric between the simulated and experimental neutron/XRD data.
        Optional: normalizing and vertically translating the simulated data, using the curve_fit function, to align better with the experimental data.

        Args:
            data_exp (array): experimental diffraction pattern data as a (2, N) array
            data_sim (array): simulated diffraction pattern data as a (2, N) array

        Returns:
            (res_sim, res_fit, emd_sim, emd_fit) -> tuple of 4 floats
            res_sim: residual between experimental data and raw simulated data
            res_fit: residual between experimental data and fited/aligned simulated data
            emd_sim: Earth mover's distance between experimental data and raw simulated data
            emd_fit: Earth mover's distance between experimental data and fited/aligned simulated data
        """
        from scipy import interpolate, stats

        f_sim = interpolate.interp1d(data_sim[:, 0], data_sim[:, 1])
        f_exp = interpolate.interp1d(data_exp[:, 0], data_exp[:, 1])
        x = np.linspace(self.xmin_fit, self.xmax_fit, self.npoints_fit)

        # residual or earth mover's distance
        # between exp & sim or sim with transformation
        return abs(f_exp(x)-f_sim(x)).mean(),\
            abs(self.xrd_normalize(f_exp(x), scale) - self.xrd_normalize(f_sim(x), scale)).mean(),\
            stats.wasserstein_distance(f_sim(x), f_exp(x)),\
            stats.wasserstein_distance(self.xrd_normalize(
                f_sim(x), scale), self.xrd_normalize(f_exp(x), scale))

    def xrd_similarity_metrics(self, data_exp, data_sim, scaled=True):
        """
        Calculate the similarity metric between the simulated and experimental neutron/XRD data.
        Optional: normalizing and vertically translating the simulated data, using the curve_fit function, to align better with the experimental data.

        Args:
            data_exp (array): experimental diffraction pattern data as a (2, N) array
            data_sim (array): simulated diffraction pattern data as a (2, N) array

        Returns:
            (res_sim, res_fit, emd_sim, emd_fit) -> tuple of 4 floats
            res_sim: residual between experimental data and raw simulated data
            res_fit: residual between experimental data and fited/aligned simulated data
            emd_sim: Earth mover's distance between experimental data and raw simulated data
            emd_fit: Earth mover's distance between experimental data and fited/aligned simulated data
        """
        from scipy import interpolate, optimize, stats

        f_sim = interpolate.interp1d(data_sim[:, 0], data_sim[:, 1])
        f_exp = interpolate.interp1d(data_exp[:, 0], data_exp[:, 1])
        x = np.linspace(self.xmin_fit, self.xmax_fit, self.npoints_fit)
        # transforming the raw simulated data to align
        def f_fit(x, a, b): return f_sim(x)*a + b
        popt, pcov = optimize.curve_fit(f_fit, x, f_exp(x))

        # residual or earth mover's distance
        # between exp & sim or sim with transformation
        if scaled:

            return (abs(f_exp(x)-f_sim(x))).mean() / (data_exp[:, 1].max() - data_exp[:, 1].min()),\
                (abs(f_exp(x)-f_fit(x, *popt))).mean() / (data_exp[:, 1].max() - data_exp[:, 1].min()),\
                stats.wasserstein_distance(f_sim(x), f_exp(x)) / (data_exp[:, 1].max() - data_exp[:, 1].min()),\
                stats.wasserstein_distance(
                    f_fit(x, *popt), f_exp(x)) / (data_exp[:, 1].max() - data_exp[:, 1].min())
        else:
            return (abs(f_exp(x)-f_sim(x))).mean(),\
                (abs(f_exp(x)-f_fit(x, *popt))).mean(),\
                stats.wasserstein_distance(f_sim(x), f_exp(x)),\
                stats.wasserstein_distance(f_fit(x, *popt), f_exp(x))

    def evaluate_obj(self, model):
        """
        This function simulated the powder diffraction pattern of a crystal structure. Then, compares it with the experimental pattern (target). A objective functions, measuring similarity with target, is assigned as a model attribute (obj1_val).

        This function is a part of the API for all classes in
        experimental_simulation module.

        Arguments:

            model (obj): structure_record.model() object for which TEM
             simulation is obtained and a mismatch score is assigned

        Returns:

            (structure_record.model(), float):
            - The model object being evaluated
            - the similarity metric (score) which is the objective
        """
        main_path = self.main_path
        xrd_sim_dir = main_path + '/calcs/' + str(model.label) + '/xrd_sim'
        os.mkdir(xrd_sim_dir)

        # write the structure as cif file in the simulation dir
        temp_init = xrd_sim_dir + '/temp_init.cif'
        cif_writer = CifWriter(model.astr.copy())
        cif_writer.write_file(temp_init)

        # Create GSASII project
        gpx = G2sc.G2Project(filename=f'{xrd_sim_dir}/{model.label}.gpx')
        phase0 = gpx.add_phase(
            f'{xrd_sim_dir}/temp_init.cif',
            phasename=str(model.label),
            fmthint='CIF'
        )

        # Simulate power diffraction histogram and write the data
        hist1 = gpx.add_simulated_powder_histogram(
            f'{model.label} XRD simulation',
            self.instr_param_file,
            self.xmin, self.xmax, Npoints=self.npoints,
            phases=gpx.phases(), scale=self.scale
        )
        gpx.do_refinements()   # calculate pattern
        gpx.save()
        gpx.histogram(0).Export(
            f'{xrd_sim_dir}/data_{model.label}', '.csv', 'hist')  # data
        gpx.histogram(0).Export(
            f'{xrd_sim_dir}/refl_{model.label}', '.csv', 'refl')  # reflections

        # post-processing of histogram data
        # calculate and report the desired scoring function
        data_sim = self.read_histogram(f'{xrd_sim_dir}/data_{model.label}.csv')
        data_exp = np.loadtxt(self.exp_xrd_file)
        res_sim, res_fit, emd_sim, emd_fit = self.xrd_similarity_metrics(
            data_exp, data_sim)
        open(f'{xrd_sim_dir}/log', 'a').write(
            f'res_sim: {res_sim}\nres_fit: {res_fit}\nemd_sim: {emd_sim}\nemd_fit: {emd_fit}\n')

        if self.score_method == 'res_sim':  # residual vs. raw simulated data
            score = float(res_sim)
        if self.score_method == 'res_fit':  # residual vs. fitted/normalized simulated data
            score = float(res_fit)
        if self.score_method == 'emd_sim':  # Earth mover's distance vs. raw sim. data
            score = float(emd_sim)
        if self.score_method == 'emd_fit':  # Earth mover's distance vs. fitted sim. data
            score = float(emd_fit)

        # the order of exp_sims is from Xsim1 -> Xsim2 -> ...
        # Hence, obj1val -> ob2_val -> ... for assigning evaluated sims
        if model.Xsim1 == 'XRD':
            model.obj1_val = score
        elif model.Xsim2 == 'XRD':
            model.obj2_val = score
        elif model.Xsim3 == 'XRD':
            model.obj3_val = score
        elif model.Xsim4 == 'XRD':
            model.obj4_val = score

        return model, score
