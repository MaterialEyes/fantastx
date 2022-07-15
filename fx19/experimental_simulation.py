from __future__ import division, unicode_literals, print_function

from fx19 import distance_check as dc
from scipy import optimize as scipy_optimize
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
try:
    from pyobjcryst import loadCrystal
    from diffpy.srfit.pdf import PDFContribution
    from diffpy.srfit.pdf import DebyePDFGenerator
    from diffpy.srfit.fitbase import Profile
    from diffpy.srfit.fitbase import FitRecipe
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
    print('Install scikit-image, Ingrained, opencv for TEM simulation.'
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


class xanes_of_model(object):
    """
    This class contains functions to calculate XANES spectra, either the
    raw spectra or the difference spectra (essential for XTA analysis, and
    useful for raw XANES analysis as well). 

    Three different XANES simulation codes are currently supported:

    - FDMNES

    - FEFF

    - VASP

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
        self.main_path = xanes_params['main_path']
        self.simulation_code = xanes_params['simulation_code']
        self.input_yaml_filepath = xanes_params['input_yaml_filepath']
        self.comparison_spectra_type = xanes_params['comparison_spectra_type']

        if 'code_folder' in xanes_params:
            self.code_folder = xanes_params['code_folder']
        else:
            self.code_folder =\
                "/mnt/c/Users/dunru/Research/XANES/parallel_fdmnes"

        if self.comparison_spectra_type == "difference":
            # 3 files need to be read in: base and excited experimental
            # reference spectra, and the computational base spectra.
            if "exp_base_ref_filepath" in xanes_params:
                self.exp_base_ref_filepath =\
                    xanes_params["exp_base_ref_filepath"]
            else:
                self.exp_base_ref_filepath = "/experiment_base_ref.dat"
            if "exp_exc_ref_filepath" in xanes_params:
                self.exp_exc_ref_filepath =\
                    xanes_params["exp_exc_ref_filepath"]
            else:
                self.exp_exc_ref_filepath = "/experiment_exc_ref.dat"
            if "comp_base_ref_filepath" in xanes_params:
                self.comp_base_ref_filepath =\
                    xanes_params["comp_base_ref_filepath"]
            else:
                self.comp_base_ref_filepath = "/computational_base_ref.dat"
        else:
            if "exp_base_ref_filepath" in xanes_params:
                self.exp_base_ref_filepath =\
                    xanes_params["exp_base_ref_filepath"]
            else:
                self.exp_base_ref_filepath = "/experiment_base_ref.dat"

        if 'exec_cmd' in xanes_params:
            self.exec_cmd = xanes_params['exec_cmd']
        else:
            self.exec_cmd = "./mpirun_fdmnes -np 4"

        if 'spectra_distance_metric' in xanes_params:
            self.distance_calculator = DistanceCalculator(
                xanes_params['spectra_distance_metric'])
        else:
            # options are any of those in fingerprinting.DistanceCalculator
            self.distance_calculator = DistanceCalculator('rmse')

        if 'spline_mesh_params' in xanes_params:
            spline_min = xanes_params['spline_mesh_params'][0]
            spline_max = xanes_params['spline_mesh_params'][1]
            spline_step = xanes_params['spline_mesh_params'][2]
            self.spline_mesh = np.arange(spline_min, spline_max, spline_step)
            self.mesh_step = spline_step
        else:
            self.spline_mesh = np.arange(7110, 7165, 0.1)
            self.mesh_step = 0.1

        if 'convolution_params' in xanes_params:
            self.convolution_type = xanes_params['convolution_params'][0]
            self.convolution_params = xanes_params['convolution_params'][1]
            self.extract_cutting_energy = xanes_params['convolution_params'][2]
        else:
            self.convolution_type = 'lorentzian'
            self.convolution_params = [1.33, 15., 23.5, 23.5, -8]
            self.extract_cutting_energy = True
        self.cutting_energy_correction = -6.
        self.refine_alignment_using_second_peak = True
        self.refine_alignment_using_difference_spectra = False
        self.comparison_window = [7110., 7147.]

        # Gather experimental data
        self.exp_base_arrays, self.exp_base_peaks =\
            self.read_in_experimental_spectra(self.exp_base_ref_filepath)
        exp_base_spline = self.fit_spline(
            self.exp_base_arrays[0], self.exp_base_arrays[1], "cubic")
        if self.comparison_spectra_type == "difference":
            self.exp_exc_arrays, self.exp_exc_peaks =\
                self.read_in_experimental_spectra(self.exp_exc_ref_filepath)
            exp_exc_spline = self.fit_spline(
                self.exp_exc_arrays[0], self.exp_exc_arrays[1], "cubic")
            self.exp_dif_spline = exp_exc_spline - exp_base_spline
            self.exp_dif_reshaped_spline = np.reshape(
                self.exp_dif_spline, (-1, 1))
        else:
            self.exp_base_spline = exp_base_spline
            self.exp_base_reshaped_spline = np.reshape(
                self.exp_base_spline, (-1, 1))

        print("Gathered experimental data.")

        if self.comparison_spectra_type == "difference":
            # Gather pre-computed computational base spectra
            self.comp_base_arrays, _ = self.read_in_calculated_spectra(
                self.comp_base_ref_filepath, self.exp_base_peaks)
            self.comp_base_spline = self.fit_spline(
                self.comp_base_arrays[0], self.comp_base_arrays[1], "cubic")
            print("Gathered pre-computed computational data.")

    def fwhm2sigma(self, fwhm):
        '''
        Converts the full width half maximum into a sigma for gaussian
        convolution.

        Arguments:

            fwhm (float): full width half maximum

        Returns:

            float: the gaussian sigma
        '''
        return fwhm / np.sqrt(8 * np.log(2))

    def lorentzian_broadening(self, E, g_ch, g_m, E_cent, E_larg, E_f):
        '''
        Calculates the energy dependent broadening parameter for
        Lorentzian broadening. For reference, refer to this
        [paper](https://hal.archives-ouvertes.fr/hal-00687301/document).
        In this method, the broadening depends on both the core-hole width
        as well as the spectral width of the final state. This spectral
        width is approximated by an arctangent.

        Arguments:

            E (float): energy

            g_ch (float): core-hole broadening width

            g_m (float): maximum height of the arctangent

            E_cent (float): the energy of the arctangent inflection point

            E_larg (float): the inclination of the arctangent

            E_f (float): the effective Fermi energy (also referred to as
              the cutting energy)

        Returns:

            float: the energy dependent broadening parameter
        '''
        eps = (E - E_f)/E_cent
        return g_ch + g_m*(0.5 +
                           1/np.pi *
                           np.arctan(np.pi/3*g_m/E_larg*(eps-1/eps**2)))

    def _locate_peaks(self, x_array, y_array):
        '''
        Finds the first and second peaks of the spectra post-edge. 

        The first peak of the spectra is adjusted to be the point closest to
        the maximum intensity point where the first derivative is zero. The
        spectra is first fitted with a spline, so as to correspond with
        the final mesh which will be used.

        The first derivative is then calculated numerically as:

        $f'(x) = f(x+h) - \dfrac{f(x-h)}{2*h}$

        The second derivative is then calculated numerically as:

        $f''(x) = \dfrac{f(x+h) - 2*f(x) + f(x-h)}{h^2}$

        The zero-crossing of the first derivative is then estimated
        by approximating the second-derivative as constant in this
        narrow mesh interval.

        The second peak of the spectra is found by simply looking for
        inflection points in the first derivative, and choosing the one
        with the maximal y value apart from the first peak.

        Arguments:

            x_array (iterable): the bin locations of the spectra

            y_array (iterable): the bin heights of the spectra

        Returns:

            float: the estimated x-coordinate of the peak
        '''
        spline_y = self.fit_spline(x_array, y_array, "cubic")
        y_max = np.amax(spline_y)
        max_indice = np.argmax(spline_y)
        x_max = self.spline_mesh[np.argmax(spline_y)]

        # now find the second_derivative maximum
        peak_derivative = (spline_y[max_indice + 1] -
                           spline_y[max_indice - 1])/(2*self.mesh_step)
        peak_second_derivative = (
            spline_y[max_indice + 1] -
            2*spline_y[max_indice] +
            spline_y[max_indice - 1]) / (self.mesh_step**2)
        zero_derivative_adjustment = (-peak_derivative)/peak_second_derivative
        x_max = x_max + zero_derivative_adjustment

        # Second peak: found by looking at first derivative inflection points
        derivatives = []
        for i in range(1, len(y_array) - 1):
            d = (y_array[i+1] - y_array[i-1]) / (2 * self.mesh_step)
            derivatives.append(d)

        inflection_points = []
        for i in range(1, len(derivatives)):
            d1 = derivatives[i-1]
            d2 = derivatives[i]
            if d1*d2 < 0 or np.isclose(d1*d2, 0.0):
                inflection_points.append(i)

        high_e_inflection_points = [
            i for i in inflection_points if i > max_indice]
        y_vals = y_array[high_e_inflection_points]

        y_max_two = np.amax(y_vals)
        ip = np.argmax(y_vals)
        max_indice_two = high_e_inflection_points[ip]
        x_max_two = x_array[max_indice_two]

        return [(x_max, y_max), (x_max_two, y_max_two)]

    def create_lorentzian_kernel(self, g_ch, g_m, E_cent, E_larg, E_f):
        '''
        Creates a Lorentzian kernel for convolution. Uses the energy-
        dependent broadening parameter described in this [paper](https:
        //hal.archives-ouvertes.fr/hal-00687301/document).

        Arguments:

            g_ch (float): core-hole broadening width

            g_m (float): maximum height of the arctangent

            E_cent (float): the energy of the arctangent inflection point

            E_larg (float): the inclination of the arctangent

            E_f (float): the effective Fermi energy (also referred to as
              the cutting energy)

        Returns:
            (array, int):
            - the convolution kernel
            - integer used for shifting the final
            convolved spectra to remove the zero points

        '''
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

        Arguments:

            fwhm (float): the broadening energy

        Returns:
            (array, int):
            - the convolution kernel
            - integer used for shifting the final convolved spectra
            to remove the zero points
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

    def convolve_with_gaussian(self, y_array, fwhm):
        '''
        Convolves a XANES spectra with a gaussian.

        Arguments:

            y_array (array): the absorption profile to be convolved.

            fwhm (float): the broadening energy

        Returns:

            array: the convolved absorption profile
        '''
        n_points = len(y_array)
        finite_kernel, kernel_n_below_0 = self.create_gaussian_kernel(fwhm)
        convolved_y = np.convolve(y_array, finite_kernel)
        smoothed_y = convolved_y[kernel_n_below_0:(
            n_points + kernel_n_below_0)]

        return smoothed_y

    def convolve_with_lorentzian(self, y_array,
                                 g_ch, g_m, E_cent, E_larg, E_f):
        '''
        Convolve a XANES spectra with a lorentzian

        Arguments:

            y_array (array): the absorption profile to be convolved.

            g_ch (float): core-hole broadening width

            g_m (float): maximum height of the arctangent

            E_cent (float): the energy of the arctangent inflection point

            E_larg (float): the inclination of the arctangent

            E_f (float): the effective Fermi energy (also referred to as
              the cutting energy)

        Returns:

            array: the convolved absorption profile
        '''
        n_points = len(y_array)
        finite_kernel, kernel_n_below_0 = self.create_lorentzian_kernel(
            g_ch, g_m, E_cent, E_larg, E_f)
        convolved_y = np.convolve(y_array, finite_kernel)
        smoothed_y = convolved_y[kernel_n_below_0:(
            n_points + kernel_n_below_0)]

        return smoothed_y

    def fit_spline(self, x_array, y_array, type):
        '''
        Fit a spline to the spectra, and uses it to interpolate points
        onto a pre-defined mesh (`self.spline_mesh`).

        Arguments:

            x_array (array): spectra energy values

            y_array (array): spectra absorption values

            type (string): which type of spline should be fit to the
            spectra. Options are `cubic` and `univariate`.

        Returns:

            array: the spline points on self.spline_mesh
        '''
        if type not in ["cubic", "univariate"]:
            print("Error. Tried to fit spline with a keyword that was"
                  "not 'cubic' or 'univariate'. Using the default of 'cubic'.")
            type = "cubic"

        if type == "cubic":
            cs = CubicSpline(x_array, y_array)
            # us = UnivariateSpline(x_array, y_array, s=0.01)
            new_data = cs(self.spline_mesh)
        else:
            us = UnivariateSpline(x_array, y_array, s=0.0001)
            new_data = us(self.spline_mesh)
        return new_data

    def read_in_experimental_spectra(self, file_path):
        '''
        Reads in the experimental spectra from the .dat file, convolves
        it with a Gaussian with broadening of 0.5 eV, and returns the
        x- and y-coords of the first peak maximum (taken to be the energy
        value where the absorption profile has zero derivative).

        Arguments:

            file_path (string): the path to the .dat file

        Returns:
            (tuple, tuple):
            - the convolved spectra
            - the x- and y-coords of the first peak maximum.
        '''
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

        peaks = self._locate_peaks(x_array, smoothed_y)

        return (x_array, smoothed_y), peaks

    def read_in_calculated_spectra(self, file_path, experimental_peaks):
        '''
        Reads in the calculated spectra from the simulation file. This
        spectra is then convolved using the user-specified parameters. 
        The x- and y-coords of the first peak maximum, taken to be the
        energy value where the first derivative is zero, are then
        extracted. These values are then used to scale and shift the
        spectra to align with the provided experimental max values.

        Functionality also exists to adjust the convolution in order to
        match the heights of the second peaks of the simulated and
        experimental spectra.

        Arguments:

            file_path (string): the path to the simulated spectra data file.

            experimental_peaks (tuple): the x- and y-coords of the first and
            second peaks of the convolved experimental spectra.

        Returns:
            (tuple, tuple):
            - the convolved spectra, shifted and scaled to match the
            experimental spectra.
            - the values by which the spectra was shifted and scaled.
        '''
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

        shifted_x, scaled_y =\
            self._convolve_and_align_spectra(x_array, y_array)

        if self.convolution_type == "lorentzian":
            if self.extract_cutting_energy:
                # Grab the fermi level to cut with
                match = None
                cycle_index = 19
                while match is None:
                    pattern = re.compile(f"Cycle  {cycle_index}")
                    bav_file = file_path[:-9] + "bav.txt"
                    lines = open(bav_file, "r").read().splitlines()
                    for line in lines:
                        match = re.search(pattern, line)
                        if match is not None:
                            fermi_energy = float(line.split()[5])
                            fermi_energy += self.cutting_energy_correction
                            self.convolution_params[4] = fermi_energy
                            break
                    cycle_index -= 1
                    if cycle_index == 10:
                        fermi_energy = self.cutting_energy_correction
                        self.convolution_params[4] = fermi_energy
                        break
            smoothed_y = self.convolve_with_lorentzian(
                y_array, *self.convolution_params)
        else:
            smoothed_y = self.convolve_with_gaussian(
                y_array, *self.convolution_params)

        max_indice = np.argmax(smoothed_y)
        if max_indice < 30:
            print("Had to trim arrays.")
            smoothed_y = smoothed_y[30:]
            x_array = x_array[30:]

        peaks = self._locate_peaks(x_array, smoothed_y)

        scale_factor = experimental_peaks[0][1] / peaks[0][1]
        shift_factor = experimental_peaks[0][0] - peaks[0][0]

        scaled_y = smoothed_y * scale_factor
        shifted_x = x_array + shift_factor

        if self.refine_alignment_using_second_peak:
            adjust_attempts = 0
            smallest_diff = np.inf
            adjustment = 0.0

            if self.convolution_type == "lorentzian":
                conv_params = self.convolution_params.copy()
                adjustment = -0.1
            else:
                conv_params = self.convolution_params
                adjustment = 0.01

            while smallest_diff > 0.01 and adjust_attempts < 250:
                if self.convolution_type == "lorentzian":
                    conv_params[4] += adjustment
                    smoothed_y = self.convolve_with_lorentzian(
                        y_array, *conv_params)
                else:
                    conv_params += adjustment
                    smoothed_y = self.convolve_with_gaussian(
                        y_array, *conv_params)

                peaks = self._locate_peaks(x_array, smoothed_y)

                sc_factor = experimental_peaks[0][1] / peaks[0][1]
                sh_factor = experimental_peaks[0][0] - peaks[0][0]

                sc_y = smoothed_y * sc_factor
                sh_x = x_array + sh_factor

                adjust_attempts += 1
                diff = abs(peaks[1][1] * sc_factor - experimental_peaks[1][1])
                if diff < smallest_diff:
                    smallest_diff = diff
                    shifted_x = sh_x
                    shift_factor = sh_factor
                    scaled_y = sc_y
                    scale_factor = sc_factor

        return (shifted_x, scaled_y), (scale_factor, shift_factor)

    def prepare_fdmnes(self, model, fdmnes_path):
        '''
        Function which prepares the FDMNES input file as well as the
        mpi file if running on a computer which does not have mpi
        installed.
        Filenames are standardized for all FANTASTX runs. However, the
        FDMNES inputs themselves are defined through a yaml file for
        user friendliness.

        Arguments:

            model (obj): `model` which is the target of FDMNES

            fdmnes_path (string): path to the folder containing the fdmnes
            mpi executable.
        '''
        ################################################
        # Define the filenames for all FDMNES operations
        fdmnes_input_folder = model.relax_path + "/FDMNES_in/"
        fdmnes_output_folder = model.relax_path + "/FDMNES_out/"
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

        #############################################
        # Open the FDMNES input yaml file #
        with open(self.input_yaml_filepath) as ifile:
            fdmnes_dict = yaml.load(ifile, Loader=yaml.FullLoader)

        # First, check to see if multiple screening values will be considered
        number_screenings = 1
        if "Screening" in fdmnes_dict["fdmnes_cards"].keys():
            if type(fdmnes_dict["fdmnes_cards"]["Screening"]) is list:
                number_screenings = len(
                    fdmnes_dict["fdmnes_cards"]["Screening"])

        #############################################
        # Prepare all filenames #
        fdmnes_input_filenames = []
        fdmnes_abbr_input_filenames = []
        fdmnes_output_filenames = []
        for s in range(number_screenings):
            fdmnes_input_filenames.append(
                fdmnes_input_folder + "run_fdmnes_" + str(s) + ".inp")
            fdmnes_abbr_input_filenames.append(
                fdmnes_input_folder + "run_fdmnes_" + str(s) + ".inp")
            fdmnes_output_filenames.append(
                fdmnes_output_folder + "run_fdmnes_result_" + str(s)
            )
        fdmfile_filename = model.relax_path + "/fdmfile.txt"
        fdmnes_mpirun_filename = model.relax_path + "/mpirun_fdmnes"

        #############################################
        #  Write the fdmfile.txt file #
        fdmfile = open(fdmfile_filename, "w+")
        fdmfile.write(str(number_screenings) + "\n")
        for s in range(number_screenings):
            fdmfile.write(fdmnes_abbr_input_filenames[s] + "\n")
        fdmfile.close()

        ###################################################
        # Remove counter ions from molecule if they exist #
        fdmnes_astr = model.astr.copy()
        if model.molecule_representation is not None:
            if "counter_ions" in model.molecule_representation:
                fdmnes_astr.remove_sites(
                    model.molecule_representation["counter_ions"])

        #############################################
        # Write the fdmnes input file(s) #

        for s in range(number_screenings):
            fdmnes_headers = {
                "Filout": fdmnes_output_filenames[s],
                "Radius": fdmnes_dict["cluster_radius"],
                "Edge": fdmnes_dict["edge"]
            }

            # Absorber and core_hole_coords are determined based on structure
            core_hole_index = 1
            absorption_site = ""
            core_hole_coords = [0, 0, 0]
            for n, site in enumerate(fdmnes_astr.sites):
                specie = site.specie.symbol
                if specie == fdmnes_dict["core_hole_site_element"]:
                    if core_hole_index == fdmnes_dict["core_hole_site_id"]:
                        core_hole_coords = np.copy(site.coords)
                        absorption_site = str(n+1)
                        fdmnes_headers['Absorber'] = absorption_site
                    core_hole_index += 1

            inputfile = open(fdmnes_input_filenames[s], "w+")
            for key, value in fdmnes_headers.items():
                inputfile.write(str(key) + "\n" + str(value) + "\n\n")

            for key, value in fdmnes_dict["fdmnes_cards"].items():
                print(f"key: {key}; value: {value}")
                if value is not None:
                    if value == "include":
                        inputfile.write(key + "\n")
                    else:
                        if key == "Atom":
                            inputfile.write(key + "\n")
                            # create oxidation state separated substates
                            oxi_confs = {}
                            for sub_key, sub_value in value.items():
                                # if key corresponds to central atom, check for
                                # ionic charge
                                underscore_index = sub_key.find("_")
                                atomic_id = sub_key
                                if underscore_index != -1:
                                    atomic_id = sub_key[:underscore_index]
                                    charge = int(
                                        sub_key[underscore_index + 1:])

                                    if atomic_id in oxi_confs:
                                        oxi_confs[atomic_id][charge] = sub_value
                                    else:
                                        oxi_confs[atomic_id] = {
                                            charge: sub_value}
                                else:
                                    oxi_confs[atomic_id] = sub_value

                            for atomic_id, val in oxi_confs.items():
                                if type(val) is dict:
                                    # determine dominant oxidation state in the structure
                                    oxi_states = []
                                    for site in fdmnes_astr.sites:
                                        number = site.specie.number
                                        if number == int(atomic_id):
                                            if hasattr(site.specie, 'oxi_state'):
                                                oxi_states.append(
                                                    int(site.specie.oxi_state))
                                    if len(oxi_states) == 0:
                                        key = list(val.keys())[0]
                                        inputfile.write(
                                            atomic_id + " " + val[key] + "\n")
                                    else:
                                        oxi_state = Counter(
                                            oxi_states).most_common(1)[0][0]
                                        inputfile.write(
                                            atomic_id + " " + val[oxi_state] + "\n")
                                else:
                                    inputfile.write(
                                        atomic_id + " " + val + "\n")
                        elif key == "Atom_conf":
                            inputfile.write(key + "\n")
                            all_atom_counts = {}
                            all_atom_indices = {}
                            atom_index = 1
                            found_keys = []
                            for site in fdmnes_astr.sites:
                                an = str(site.specie.number)
                                oxidized = hasattr(site.specie, 'oxi_state')
                                if oxidized:
                                    ox_an = an + "_" + \
                                        str(int(site.specie.oxi_state))
                                    if ox_an in value.keys():
                                        an = ox_an
                                    else:
                                        if an not in value.keys():
                                            print("Note! No atom_conf for "
                                                  f"oxidation state {ox_an} and "
                                                  "no default state found for"
                                                  f" atomic number {an} either.")
                                        else:
                                            print("Note! No atom_conf for "
                                                  f"oxidation state {ox_an}. Using"
                                                  "default state found.")
                                found_keys.append(an)
                                if an in all_atom_counts:
                                    all_atom_counts[an] += 1
                                else:
                                    all_atom_counts[an] = 1
                                if an in all_atom_indices:
                                    all_atom_indices[an].append(
                                        str(atom_index))
                                else:
                                    all_atom_indices[an] = [str(atom_index)]
                                atom_index += 1

                            for sub_key, sub_value in value.items():
                                # Need to get number of atoms and their indices
                                if sub_key in found_keys:
                                    atom_count = str(all_atom_counts[sub_key])
                                    atom_indices = " ".join(
                                        all_atom_indices[sub_key])
                                    inputfile.write(
                                        atom_count + " " +
                                        atom_indices + " " +
                                        sub_value + "\n")
                        elif key == "Multipolar":
                            if type(value) is str:
                                inputfile.write(value + "\n")
                            else:
                                for sub_value in value:
                                    inputfile.write(sub_value + "\n")
                        elif key == "Screening":
                            if type(value) is str:
                                inputfile.write(key + "\n")
                                inputfile.write(value + "\n")
                            else:
                                inputfile.write(key + "\n")
                                inputfile.write(value[s] + "\n")
                        else:
                            inputfile.write(key + "\n")
                            inputfile.write(value + "\n")
                    inputfile.write("\n")

            # create atoms card
            inputfile.write(fdmnes_dict["structure_type"] + "\n")

            # grab cartesian coordinates of lattice
            abc = fdmnes_astr.lattice.abc
            angles = fdmnes_astr.lattice.angles
            l_vals = str(abc[0]) + " " + str(abc[1]) + " " + str(abc[2])
            angle_vals = str(angles[0]) + " " + \
                str(angles[1]) + " " + str(angles[2])
            inputfile.write("    " + l_vals + " " + angle_vals + "\n")
            for site in fdmnes_astr.sites:
                specie = site.specie.symbol
                an = atomic_numbers[specie]
                coords = np.copy(site.coords)
                mc = []
                for i in range(3):
                    coords[i] -= core_hole_coords[i]
                    if coords[i] > abc[i]/2:
                        mc.append((coords[i] - abc[i])/abc[i])
                    else:
                        mc.append(coords[i]/abc[i])
                inputfile.write(
                    str(an) + "  " + str(mc[0]) +
                    " " + str(mc[1]) +
                    " " + str(mc[2]) + "\n")
            inputfile.write("\n")

            inputfile.write("END\n")
            inputfile.close()

        ######################################
        # Write the mpirun_fdmnes file. Only needed if on local computer.
        mpifile = open(fdmnes_mpirun_filename, "w+")
        mpifile.write("#!/bin/bash\n")
        mpifile.write(f"fdmnesDir={fdmnes_path}\n")
        mpifile.write("IFS=$'\\n'\n")
        mpifile.write(". \"${fdmnesDir}/mpirt/bin/intel64/mpivars.sh\"\n")
        mpifile.write(
            "\"${fdmnesDir}/mpirt/bin/intel64/mpirun\" "
            "$* \"${fdmnesDir}/fdmnes_mpi_linux64\"\n")
        mpifile.write("IFS=$' \\t\\n'\n")
        mpifile.close()

        # Make the mpifile executable
        make_exc_string = "chmod +rx " + fdmnes_mpirun_filename
        make_exc_command = make_exc_string.split()
        sp.call(make_exc_command)

        return number_screenings

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

        Args:

        model (obj): structure_record.model() object for which TEM simulation
                     is obtained and a mismatch score is assigned

        Returns:
            (model obj, float):
            - the model whose objective was calculated
            - the objective value
        """

        # Prepare FDMNES input file and run simulation
        num_files = self.prepare_fdmnes(model, self.code_folder)

        exec_cmd = self.exec_cmd.split()
        with open(model.relax_path + '/log_fdmnes.{}'.format(model.label),
                  'w') as log_file:
            fdmnes_job = sp.Popen(
                exec_cmd,
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                cwd=model.relax_path)
            for each_line in fdmnes_job.stdout:
                line = each_line.decode('utf-8')
                log_file.write(line)
        # wait for the calculation to finish
        fdmnes_job.wait()

        # Reference XANES simulation(s) against experiment
        results = []
        for xanes_run in range(num_files):
            xanes_result_path = model.relax_path +\
                "/FDMNES_out/run_fdmnes_result_" +\
                str(xanes_run) + "_tddft.txt"
            reference_peaks = self.exp_base_peaks
            if self.comparison_spectra_type == "difference":
                reference_peaks = self.exp_exc_peaks
            self.model_comp_arrays, (scale_factor, shift_factor) =\
                self.read_in_calculated_spectra(xanes_result_path,
                                                reference_peaks)
            self.model_comp_spline = \
                self.fit_spline(
                    self.model_comp_arrays[0],
                    self.model_comp_arrays[1],
                    "cubic")

            print("Read in calculated spectra.")

            compare_indices = \
                (self.spline_mesh <= self.comparison_window[1]) & (
                    self.spline_mesh >= self.comparison_window[0])
            lowest_spectra_distance = np.inf
            lowest_spline = None
            if self.comparison_spectra_type == "difference":
                if self.refine_alignment_using_difference_spectra:
                    scale_factor_og = scale_factor
                    for i in range(-50, 50):
                        for j in range(-5, 5):
                            y_array = self.model_comp_arrays[1] * \
                                (1 + 0.01*j/scale_factor_og)
                            x_array = self.model_comp_arrays[0] - i*0.01
                            new_spline = self.fit_spline(
                                x_array, y_array, "cubic")
                            compare_spline =\
                                self.comp_base_spline[compare_indices]
                            fdmnes_dif_spectra =\
                                new_spline[compare_indices] - compare_spline
                            fdmnes_dif_spectra_array = np.reshape(
                                fdmnes_dif_spectra, (-1, 1))
                            spectra_distance = self.distance_calculator.create(
                                fdmnes_dif_spectra_array,
                                self.exp_dif_reshaped_spline[compare_indices])

                            if spectra_distance < lowest_spectra_distance:
                                lowest_spectra_distance = spectra_distance
                                lowest_spline = new_spline
                else:
                    compare_spline = self.comp_base_spline[compare_indices]
                    fdmnes_dif_spline =\
                        self.model_comp_spline[compare_indices] - \
                        compare_spline
                    fdmnes_dif_reshaped_spline = np.reshape(
                        fdmnes_dif_spline, (-1, 1))
                    lowest_spectra_distance = self.distance_calculator.create(
                        fdmnes_dif_reshaped_spline,
                        self.exp_dif_reshaped_spline[compare_indices])
                    lowest_spline = self.model_comp_spline
            else:
                fdmnes_compare_spline = self.model_comp_spline[compare_indices]
                fdmnes_compare_array = np.reshape(
                    fdmnes_compare_spline, (-1, 1)
                )
                lowest_spectra_distance = self.distance_calculator.create(
                    fdmnes_compare_array,
                    self.exp_base_reshaped_spline[compare_indices]
                )
                lowest_spline = self.model_comp_spline

            print(f"RMS score for run {xanes_run}: "
                  f"{float((lowest_spectra_distance)*100)}")
            results.append(lowest_spectra_distance)
            np.save(model.relax_path + "/model_sim_spectra_" +
                    str(xanes_run) + ".npy", lowest_spline)

            fig, axes = plt.subplots(1, 1)
            fig.set_size_inches(10, 10)
            axes.plot(self.spline_mesh, self.exp_base_spline, marker=".",
                      linestyle="-", label="Experiment")
            axes.plot(self.spline_mesh, lowest_spline, marker=".",
                      linestyle="--", label="FDMNES")
            axes.set_ylabel("Absorbance (arbitrary units)", fontsize=24)
            axes.set_xlabel(
                "Energy (eV)", fontsize=24)
            axes.set_xlim((self.spline_mesh[0], self.spline_mesh[-1]))
            axes.set_ylim((0, 2.0))
            axes.legend(bbox_to_anchor=(0.48, 0.85),
                        loc="lower left", fontsize=20)
            plt.setp(axes.get_xticklabels(), fontsize=20)
            plt.setp(axes.get_yticklabels(), fontsize=16)
            filename = model.relax_path + "/" + "experiment_vs_sim_spectra_" +\
                str(xanes_run) + ".png"
            plt.savefig(filename, format="png", dpi=300)

        lowest_spectra_distance = min(results)
        # the order of exp_sims is from Xsim1 -> Xsim2 -> ...
        # Hence, obj1val -> ob2_val -> ... for assigning evaluated diff spectra
        if model.Xsim1 == 'XANES':
            # Minimizing the obj vals
            model.obj1_val = float((lowest_spectra_distance)*100)
        elif model.Xsim2 == 'XANES':
            model.obj2_val = float((lowest_spectra_distance)*100)
        elif model.Xsim3 == 'XANES':
            model.obj3_val = float((lowest_spectra_distance)*100)
        elif model.Xsim4 == 'XANES':
            model.obj4_val = float((lowest_spectra_distance)*100)

        return model, lowest_spectra_distance


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
        # quadratic term related to sharpness of first peak (from pdfgui manual)
        self.delta2 = 3.87
        # exp. instrument (peak-damping) parameter (default from pdfgui manual)
        self.qdamp = 0.043  # G(r) intensity decereases with r
        self.fit_coords = True
        # default bounds_dict
        lb_ub_dict = {}
        lb_ub_dict['a'] = [19.0, 21.0]
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

        # tolerance to relax each x/y/z coordinate of an atom coordinates
        self.coord_tol = 0.1
        if 'coord_tol' in pdf_params:
            self.coord_tol = pdf_params['coord_tol']

    def write_temp_cif(self, model):
        """
        Writes temp.cif file in the pdf simulation directory from model.astr

        Arguments:

            model (obj): structure_record.model() object for which the PDF
            simulation will be done
        """
        # use the relaxed structure from energy calculation
        astr = model.astr
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
        generator = DebyePDFGenerator('generator_name')
        generator.setStructure(diffpy_str)
        generator.setQmax(self.Qmax)
        generator.setQmin(self.Qmin)

        # Make contribution object
        # The FitContribution
        contribution = PDFContribution("contribution_name")
        contribution.addProfileGenerator(generator)
        contribution.setProfile(profile, xname="r")

        return contribution

    def fit_variables_recipe(self, contribution, cif_file, fitted_params=None):
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

        # Set all Biso values to provided or default Biso_val
        # for p in Bisos:
        #    fit_param = recipe._parameters[p]
        #    fit_param.setValue(self.Biso_val)

        # add existing PDF variables as Fit parameters and setValue
        recipe.addVar(generator.scale, self.scale, fixed=False)
        recipe.addVar(generator.delta2, self.delta2, fixed=False)
        recipe.addVar(generator.qdamp, self.qdamp, fixed=False)

        # Set lower and upper bounds for variables
        recipe.a.bounds = self.var_bounds['a']
        # recipe.b.bounds = self.var_bounds['a']
        # recipe.c.bounds = self.var_bounds['a']
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
        recipe.a.bounds = self.var_bounds['a']
        # recipe.b.bounds = self.var_bounds['a']
        # recipe.c.bounds = self.var_bounds['a']
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
        #fcs_new = np.concatenate((pymat_str.frac_coords[center_ind], fcs))
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
            contribution, cif_file)
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
            # use the new cif file created with optimized coords
            #cif_file = pdf_sim + '/temp_opt.cif'
            #contribution = self.get_PDF_obj(cif_file)

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

        # self.dm3_path = gb_ingrained_params['dm3_path']
        # if not self.dm3_path:
        #     print('Provide path (dm3_path) to experimental image')

        # # Prepare experimental image
        # # (make sure this procedure matches the procedure in 'run.py')
        # image_data = iop.image_open(self.dm3_path)
        # exp_img = iop.apply_rotation(image_data['Pixels'], 1)
        # exp_img = iop.scale_pixels(exp_img, mode='rescale')
        # exp_img = restoration.wiener(exp_img, np.ones((7, 7))/3.5, 1300)
        # exp_img = equalize_adapthist(exp_img, clip_limit=0.005)

        # bicrys_ref = Bicrystal(poscar_file=self.init_gb_path)
        # congruity = CongruityBuilder(sim_obj=bicrys_ref, exp_img=exp_img)

        # Get solutions from text file
        # if self.progress_file:
        #     progress = np.genfromtxt(self.progress_file, delimiter=',')
        #     best_idx = int(np.argmin(progress[:, -1]))
        #     x = progress[best_idx]
        #     xfit = x[1:-1]
        #     xfit = [a for a in xfit[:-2]] + [int(a) for a in xfit[-2::]]

        # TODO: Find why we set self.opt_params[1] = 0
        # if not self.opt_params:
        #     self.opt_params = xfit.copy()
        #     self.opt_params[1] = 0
        # else:
        #     xfit = self.opt_params.copy()
        # xfit[1] = 0
        # sim_img, sim_struct, exp_patch, shift_score, stable_idxs = \
        #     congruity.fit_gb(sim_params=xfit, bias_y=1E-4)

        # sim_struct.to(filename='POSCAR_init_fitted', fmt='poscar')

        # np.save(self.main_path + '/whole_exp.npy', exp_patch)
        # np.save(self.main_path + '/whole_sim_init.npy', sim_img)

        # Temporarily "hard-coded" exp interface region for VASP runs
        # Load prev_whole_exp.npy that is from the LAMMPS runs
        exp_prev = np.load('inputs/whole_exp.npy')
        if exp_prev.ndim == 3:
            exp_prev = np.mean(exp_prev, axis=2)
            self.im_ref = exp_prev[:, :-1]
        else:
            self.im_ref = exp_prev

        # in y & x directions # TODO: remove hard-coded values
        # exp_patch_for_vasp = exp_img[459:584,
        #                              249:374]  # exp_prev[152:279, 12:]

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
