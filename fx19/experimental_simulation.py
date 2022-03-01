
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
import subprocess as sp
from scipy.interpolate import CubicSpline, UnivariateSpline
from ase.data import atomic_numbers

class xanes_of_model(object):
    """
    This class contains functions to calculate the XANES, compare it against
    the previously computed [Fe(CN)6]-4 spectra, and compare the difference
    spectra against the experimental difference spectra.

    Args:

    xanes_params (dict): A dictionary of parameters used.
    """

    def __init__(self, xanes_params):
        """
        """
        print("Initializing XANES module.")
        # main path as in energy.py
        self.name = 'XANES'
        self.main_path = xanes_params['main_path']

        if 'fdmnes_folder' in xanes_params:
            self.fdmnes_folder = xanes_params['fdmnes_folder']
        else:
            self.fdmnes_folder = "/mnt/c/Users/dunru/Research/XANES/parallel_fdmnes"

        if 'experiment_fe_2_filepath' in xanes_params:
            self.fe_2_filepath = xanes_params["experiment_fe_2_filepath"] + "/experiment_fe2+.dat"
        else:
            self.fe_2_filepath = self.main_path + "/experiment_fe2+.dat"

        if 'computational_fe_2_filepath' in xanes_params:
            self.comp_fe_2_filepath = xanes_params["computational_fe_2_filepath"] + "/computational_fe2+.dat"
        else:
            self.comp_fe_2_filepath = self.main_path + "/computational_fe2+.dat"

        if 'experiment_fe_3_filepath' in xanes_params:
            self.fe_3_filepath = xanes_params["experiment_fe_3_filepath"] + "/experiment_fe3+.dat"
        else:
            self.fe_3_filepath = self.main_path + "/experiment_fe3+.dat"

        if 'fdmnes_exec_cmd' in xanes_params:
            self.fdmnes_exec_cmd = xanes_params['fdmnes_exec_cmd']
        else:
            self.fdmnes_exec_cmd = "./mpirun_fdmnes -np 4"

        self.spline_mesh = np.arange(7110, 7165, 0.25)

        # Gather experimental data
        self.arrays_fe_2, self.maxes_fe_2 = self.read_in_experimental_spectra(self.fe_2_filepath)
        self.arrays_fe_3, self.maxes_fe_3 = self.read_in_experimental_spectra(self.fe_3_filepath)
        fe_2_spline = self.fit_spline(self.arrays_fe_2[0], self.arrays_fe_2[1], self.spline_mesh, "cubic")
        fe_3_spline = self.fit_spline(self.arrays_fe_3[0], self.arrays_fe_3[1], self.spline_mesh, "cubic")
        self.experiment_dif_spectra = fe_3_spline - fe_2_spline

        print("Gathered experimental data.")

        # Gather fe_2 spectra
        self.comp_arrays_fe_2, _ = self.read_in_calculated_spectra(self.comp_fe_2_filepath, self.maxes_fe_2)
        self.comp_fe_2_spline = self.fit_spline(self.comp_arrays_fe_2[0], self.comp_arrays_fe_2[1], self.spline_mesh, "cubic")

        print("Gathered pre-computed computational data.")

    def fwhm2sigma(self, fwhm):
        return fwhm / np.sqrt(8 * np.log(2))


    def lorentzian_broadening(self, E, g_ch, g_m, E_cent, E_larg, E_f):
        eps = (E - E_f)/E_cent
        return g_ch + g_m*(0.5 + 1/np.pi*np.arctan(np.pi/3*g_m/E_larg*(eps-1/eps**2)))


    def create_lorentzian_kernel(self, g_ch, g_m, E_cent, E_larg, E_f):
        x_for_kernel = np.arange(-10, 10)
        gammas = self.lorentzian_broadening(
            x_for_kernel, g_ch, g_m, E_cent, E_larg, E_f)
        kernel = 1/np.pi*(0.5*gammas)/((x_for_kernel)**2 + (0.5*gammas)**2)
        kernel_above_thresh = kernel > 0.0001
        finite_kernel = kernel[kernel_above_thresh]
        finite_kernel = finite_kernel / finite_kernel.sum()
        kernel_n_below_0 = int((len(finite_kernel) - 1) / 2.)

        return finite_kernel, kernel_n_below_0


    def create_gaussian_kernel(self, fwhm):
        '''
        Create a gaussian kernel for convolution
        '''
        # create gaussian kernel
        sigma = self.fwhm2sigma(fwhm)
        x_for_kernel = np.arange(-10, 10)
        kernel = np.exp(-(x_for_kernel) ** 2 / (2 * sigma ** 2))
        kernel_above_thresh = kernel > 0.0001
        finite_kernel = kernel[kernel_above_thresh]
        finite_kernel = finite_kernel / finite_kernel.sum()
        kernel_n_below_0 = int((len(finite_kernel) - 1) / 2.)

        return finite_kernel, kernel_n_below_0


    def convolve_with_gaussian(self, fwhm, y_array):
        '''
        Convolve spectra with a gaussian

        Returns convolved spectra
        '''
        n_points = len(y_array)
        finite_kernel, kernel_n_below_0 = self.create_gaussian_kernel(fwhm)
        convolved_y = np.convolve(y_array, finite_kernel)
        smoothed_y = convolved_y[kernel_n_below_0:(
            n_points + kernel_n_below_0)]

        return smoothed_y


    def convolve_with_lorentzian(self, y_array, g_ch, g_m, E_cent, E_larg, E_f):
        '''
        Convolve spectra with a lorentzian

        Returns convolved spectra
        '''
        n_points = len(y_array)
        finite_kernel, kernel_n_below_0 = self.create_lorentzian_kernel(
            g_ch, g_m, E_cent, E_larg, E_f)
        convolved_y = np.convolve(y_array, finite_kernel)
        smoothed_y = convolved_y[kernel_n_below_0:(
            n_points + kernel_n_below_0)]

        return smoothed_y


    def fit_spline(self, x_array, y_array, x_mesh, type):
        '''
        Fit a cubic spline to the spectra, and use it
        to interpolate points onto a pre-defined mesh.

        Returns the spline points on x_mesh
        '''
        if type == "cubic":
            cs = CubicSpline(x_array, y_array)
            # us = UnivariateSpline(x_array, y_array, s=0.01)
            new_data = cs(x_mesh)
        else:
            us = UnivariateSpline(x_array, y_array, s=0.0001)
            new_data = us(x_mesh)
        return new_data

    def read_in_experimental_spectra(self, file_path):
        lines = open(file_path, "r").read().splitlines()
        x_list = []
        y_list = []
        for line in lines:
            newline = line.split()
            if len(newline) != 0 and newline[0] != "#":
                x = float(newline[0])*1000
                y = float(newline[1])
                x_list.append(x)
                y_list.append(y)
        x_array = np.array(x_list)
        y_array = np.array(y_list)

        smoothed_y = self.convolve_with_gaussian(0.5, y_array)

        y_max = np.amax(smoothed_y)
        x_max = x_array[np.argmax(smoothed_y)]

        return (x_array, smoothed_y), (x_max, y_max)

    def read_in_calculated_spectra(self, file_path, experimental_maxes):
        lines = open(file_path, "r").read().splitlines()
        x_list = []
        y_list = []
        energy_val = 0
        for line_index, line in enumerate(lines):
            newline = line.split()
            if line_index == 0:
                energy_val = float(newline[0])
            if line_index > 1:
                x = float(newline[0]) + energy_val
                y = float(newline[1])*100
                x_list.append(x)
                y_list.append(y)
        x_array = np.array(x_list)
        y_array = np.array(y_list)

        smoothed_y = self.convolve_with_lorentzian(
            y_array, 1.33, 10.0, 30, 30, -8)

        y_max = np.amax(smoothed_y)
        x_max = x_array[np.argmax(smoothed_y)]
        if np.argmax(smoothed_y) < 70:
            print("Had to trim arrays.")
            test_y_array = smoothed_y[70:]
            x_max = x_array[np.argmax(test_y_array) + 70]
            y_max = np.amax(test_y_array)

            smoothed_y = smoothed_y[30:]
            x_array = x_array[30:]

        scale_factor = experimental_maxes[1] / y_max
        shift_factor = experimental_maxes[0] - x_max

        scaled_y = smoothed_y * scale_factor
        shifted_x = x_array + shift_factor

        return (shifted_x, scaled_y), (scale_factor, shift_factor)

    def prepare_fdmnes(self, relax_path, fdmnes_path):
        # Define the directory containing the VASP poscar, and define the filename
        # which will match the FDMNES outputs
        vaspfile = relax_path + "/POSCAR_relaxed"

        fdmnes_input_folder = relax_path + "/FDMNES_in/"
        fdmnes_output_folder = relax_path + "/FDMNES_out/"
        try:
            os.mkdir(fdmnes_input_folder)
            print("Created FDMNES input directory.")
        except FileExistsError:
            print("Error. Input directory already exists.")
        try:
            os.mkdir(fdmnes_output_folder)
            print("Created FDMNES input directory.")
        except FileExistsError:
            print("Error. Output directory already exists.")

        fdmnes_input_filename = fdmnes_input_folder + "run_fdmnes.inp"
        fdmnes_abbr_input_filename = fdmnes_input_filename # fdmnes_folder + "run" + sys.argv[1] + ".inp"
        fdmnes_output_filename = fdmnes_output_folder + "run_fdmnes_result"
        fdmfile_filename = relax_path + "/fdmfile.txt"
        fdmnes_mpirun_filename = relax_path + "/mpirun_fdmnes"


        #############################################
        #  Write the fdmfile.txt file #
        fdmfile = open(fdmfile_filename, "w+")
        fdmfile.write("1\n")
        fdmfile.write(fdmnes_abbr_input_filename + "\n")
        fdmfile.close()

        #############################################
        # Write the fdmnes input file #

        # Define parameters for the calculation
        cluster_radius = 6.5
        structure_type = "molecule"
        structure_id = "0"
        if structure_type == "molecule":
            structure_id = "1"
        edge = "K"
        molecule_radius = "3.5"
        core_hole_site = "Fe"  # string or site index

        fdmnes_cards = {
            "Atom": 0, # if we want to define the valence orbitals ourselves (corresponds to electronic_densities below)
            "Atom_conf": 1, #alternate way of defining the valence orbitals
            "Green": 0, # if we want to use the multiple scattering mode
            "Range": 1, # if we want to define the energy range (corresponds to e_grid below)
            "Multipolar": 1,
            "SCF": 1,
            "Self_abs": 0,
            "Double_cor": 0,
            "Convolution" : 0,
            "TDDFT": 1,
            "Hubbard": 1,
            "Perdew": 0,
            "Chfree": 1
        }

        fdmnes_card_values = {
            #"electronic_densities": {"13": "2 3 0 2 3 1 1", "8": "2 2 0 2 2 1 4"},
            "electronic_densities": {"26": "3 3 2 5.5 4 0 1.5 4 1 1.", "6": "2 2 0 2 2 1 2.", "7": "2 2 0 2 2 1 3."},
            "e_grid": "-5 0.2 7 0.8 50.0",
            "multipole_expansion": "Quadrupole",
            "Lmax_tddft": "2",
            "hubbard_U": "5.3 0.0 0.0"
        }

        fdmnes_inputs = {
            "Filout": fdmnes_output_filename,
            "Radius": cluster_radius,
            "Edge": "K"
        }


        # Read in POSCAR
        filename = vaspfile
        structure = Structure.from_file(filename)

        # Absorber and core_hole_coords are determined based on structure
        core_hole_index = 0
        core_hole_coords = [0,0,0]
        for n, site in enumerate(structure.sites):
            specie = site.specie.symbol
            if specie == core_hole_site:
                core_hole_index = n
                core_hole_coords = np.copy(site.coords)
                absorber_index = n + 1
        fdmnes_inputs["Absorber"] = str(absorber_index)

        inputfile = open(fdmnes_input_filename, "w+")
        separator = " "
        for key, value in fdmnes_inputs.items():
            inputfile.write(key + "\n")
            inputfile.write(str(value) + "\n" + "\n")

        for key, value in fdmnes_cards.items():
            if value == 1 and key != "Convolution":
                if key != "Multipolar":
                    inputfile.write(key + "\n")
                if key == "Atom":
                    if "electronic_densities" in fdmnes_card_values.keys():
                        for sub_key, sub_value in fdmnes_card_values["electronic_densities"].items():
                            inputfile.write(sub_key + " " + sub_value + "\n")
                    else:
                        print("Need to add electronic_densities to fdmnes_card_values!")
                if key == "Atom_conf":
                    if "electronic_densities" in fdmnes_card_values.keys():
                        all_atom_counts = {}
                        all_atom_indices = {}
                        atom_index = 1
                        for site in structure.sites:
                            specie = site.specie.symbol
                            an = str(atomic_numbers[specie])
                            if an in all_atom_counts:
                                all_atom_counts[an] += 1
                            else:
                                all_atom_counts[an] = 1
                            if an in all_atom_indices:
                                all_atom_indices[an].append(str(atom_index))
                            else:
                                all_atom_indices[an] = [str(atom_index)]
                            atom_index += 1

                        for sub_key, sub_value in fdmnes_card_values["electronic_densities"].items():
                            # Need to get number of atoms and their indices
                            atom_count = str(all_atom_counts[sub_key])
                            atom_indices = " ".join(all_atom_indices[sub_key])
                            inputfile.write(atom_count + " " + atom_indices + " " + sub_value + "\n")
                    else:
                        print("Need to add electronic_densities to fdmnes_card_values!")
                if key == "Range":
                    if "e_grid" in fdmnes_card_values.keys():
                        inputfile.write(fdmnes_card_values["e_grid"] + "\n")
                    else:
                        print("Need to add e_grid to fdmnes_card_values!")
                if key == "TDDFT":
                    if "Lmax_tddft" in fdmnes_card_values.keys():
                        inputfile.write("Lmax_tddft\n")
                        inputfile.write(fdmnes_card_values["Lmax_tddft"] + "\n")
                if key == "Multipolar":
                    inputfile.write(fdmnes_card_values["multipole_expansion"] + "\n")
                if key == "Hubbard":
                    inputfile.write(fdmnes_card_values["hubbard_U"] + "\n")

                inputfile.write("\n")

        # create atoms card
        if structure_id == "1":
            inputfile.write("Molecule\n")
        else:
            inputfile.write("Crystal\n")

        # grab cartesian coordinates of lattice
        abc = structure.lattice.abc
        angles = structure.lattice.angles
        l_vals = str(abc[0]) + " " + str(abc[1]) + " " + str(abc[2])
        angle_vals = str(angles[0]) + " " + str(angles[1]) + " " + str(angles[2])
        inputfile.write("    " + l_vals + " " + angle_vals + "\n")

        # for site in structure.sites:
        #     print(site.coords)

        for site in structure.sites:
            specie = site.specie.symbol
            an = atomic_numbers[specie]
            coords = site.coords
            mc = []
            for i in range(3):
                coords[i] -= core_hole_coords[i]
                #coords[i] -= abc[i]/2
                if coords[i] > abc[i]/2:
                    mc.append((coords[i] - abc[i])/abc[i])
                else:
                    mc.append(coords[i]/abc[i])
            inputfile.write(str(an) + "  " + str(mc[0]) + " " + str(mc[1]) + " " + str(mc[2]) + "\n")

        inputfile.write("\n")

        if fdmnes_cards["Convolution"] == 1:
            inputfile.write("Convolution"+ "\n" + "\n")
            inputfile.write("Gamma_max\n")
            inputfile.write("7.5\n\n")

        inputfile.write("END\n")
        inputfile.close()

        ######################################
        # Write the mpirun_fdmnes file
        mpifile = open(fdmnes_mpirun_filename, "w+")
        mpifile.write("#!/bin/bash\n")
        mpifile.write(f"fdmnesDir={fdmnes_path}\n")
        mpifile.write("IFS=$'\\n'\n")
        mpifile.write(". \"${fdmnesDir}/mpirt/bin/intel64/mpivars.sh\"\n")
        mpifile.write("\"${fdmnesDir}/mpirt/bin/intel64/mpirun\" $* \"${fdmnesDir}/fdmnes_mpi_linux64\"\n")
        mpifile.write("IFS=$' \\t\\n'\n")
        mpifile.close()

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
        relax_path = model.relax_path

        # Prepare FDMNES input file and run simulation
        self.prepare_fdmnes(relax_path, self.fdmnes_folder)

        fdmnes_exec = self.fdmnes_exec_cmd.split()
        with open(relax_path + '/log_fdmnes.{}'.format(model.label), 'w') as log_file:
            fdmnes_job = sp.Popen(
                fdmnes_exec, stdout=sp.PIPE, stderr=sp.STDOUT, cwd=relax_path)
            for each_line in fdmnes_job.stdout:
                line = each_line.decode('utf-8')
                log_file.write(line)
        # wait for the calculation to finish
        fdmnes_job.wait()

        # Reference XANES simulation against experiment
        xanes_result_path = relax_path + "/FDMNES_out/run_fdmnes_result_tddft.txt"
        self.comp_arrays_fe_3, (scale_factor, shift_factor) = self.read_in_calculated_spectra(xanes_result_path, self.maxes_fe_3)
        self.comp_fe_3_spline = \
            self.fit_spline(self.comp_arrays_fe_3[0], self.comp_arrays_fe_3[1], self.spline_mesh, "cubic")

        print("Read in calculated spectra.")

        # # Shift curves in order to minimize root mean square of difference spectras
        compare_indices = (self.spline_mesh <= 7135) & (self.spline_mesh >= 7110)
        lowest_rms = np.inf
        lowest_shift = shift_factor
        best_scale = scale_factor
        lowest_spline = None
        scale_factor_og = scale_factor
        for i in range(-50, 50):
            for j in range(-5, 5):
                y_array = self.comp_arrays_fe_3[1]*(1 + 0.01*j/scale_factor_og)
                x_array = self.comp_arrays_fe_3[0] - i*0.01
                new_spline = self.fit_spline(
                    x_array, y_array, self.spline_mesh, "cubic")
                compare_spline = self.comp_fe_2_spline[compare_indices]
                fdmnes_dif_spectra = new_spline[compare_indices] - \
                    compare_spline
                rms = np.sqrt(np.sum(np.square(fdmnes_dif_spectra -
                                            self.experiment_dif_spectra[compare_indices]))/len(fdmnes_dif_spectra))

                if rms < lowest_rms:
                    lowest_rms = rms
                    lowest_spline = new_spline

        # lowest_spline_array = np.array(lowest_spline[self.spline_mesh])
        # print(lowest_spline_array)
        # print(lowest_spline)
        print(f"RMS score: {float((lowest_rms)*100)}")
        np.save(relax_path + "/model_sim_spectra.npy", lowest_spline)

        # the order of exp_sims is from Xsim1 -> Xsim2 -> ...
        # Hence, obj1val -> ob2_val -> ... for assigning evaluated diff spectra
        if model.Xsim1 == 'XANES':
            model.obj1_val = float((lowest_rms)*100)  # Minimizing the obj vals
        elif model.Xsim2 == 'XANES':
            model.obj2_val = float((lowest_rms)*100)
        elif model.Xsim3 == 'XANES':
            model.obj3_val = float((lowest_rms)*100)
        elif model.Xsim4 == 'XANES':
            model.obj4_val = float((lowest_rms)*100)

        return model, lowest_rms


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
