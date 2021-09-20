from __future__ import division, unicode_literals, print_function
from typing import no_type_check_decorator

from numpy.random.mtrand import random_sample
from scipy.spatial.distance import MetricInfo

"""
This module contains functions to conduct multi-objective search. 
It contains a pool of structures which contains "Population" and 
"Archive" sub-groups. The "Population" group is the primary breeding
pool for genetic operations. The "Archive" group is an elite population
which ensures that the structure search is always conducting genetic
operations with at least one structure on the Pareto front. The
algorithm is steady-state, adding one child structure at a time. 

The primary multi-objective search algorithm is epsilon-MOEA. 
[Citation: Deb K, Mohan M, Mishra S. Evol Comput. 2005 Winter;13(4):501-25]
However, until the "Population" sub-group has reached the steady-state
capacity, the "Archive" remains uninitialized, and a linear
selection protocol is used instead. In this protocol, every child
structure is added to the "Population" and is assigned a selection
probability which corresponds to its distance to the minimums of each
objective function. Parents for genetic operations are then chosen 
randomly. 

Once the "Population" has reached steady-state capacity, full epsilon-MOEA
is employed. For a thorough explanation, see the citation above. 

Note: single-objective search is also supported to be consistent with
selection.py. 
"""
import numpy as np
import random
import math
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from scipy.optimize import minimize
from dscribe.kernels import REMatchKernel
from dscribe.descriptors import SOAP
from ase.ga.ofp_comparator import OFPComparator
from pymatgen.io.ase import AseAtomsAdaptor
from fx19 import distance_check as dc


class Comparator(object):
    '''
    A modified version of epsilon dominance. The epsilon boxes are still used,
    but now both structures within an epsilon box are kept unless they are
    classified as nearly identical by comparing their global fingerprints.
    '''

    def __init__(self, label='bag-of-bonds', tolerances=None):
        self.label = label

        # Assign default tolerance values if none are provided
        if tolerances is None:
            tolerances = {}
            tolerances["valle-oganov"] = 1e-3
            tolerances["bag-of-bonds"] = [.02, 0.7]
            tolerances["rematch-soap"] = 1e-3

        self.tolerances = tolerances
        self.comp = None
        self.kernel_gen = None

    def set_soap_descriptor(self, _species, soap_values=None):
        if soap_values is None:
            self.desc = SOAP(species=_species, rcut=5.0, nmax=9, lmax=6,
                             sigma=0.5, periodic=True, crossover=True, sparse=False)
        else:
            _rcut = 5.0
            _nmax = 9
            _lmax = 6
            _sigma = 0.5
            if "rcut" in soap_values:
                _rcut = soap_values["rcut"]
            if "nmax" in soap_values:
                _nmax = soap_values["nmax"]
            if "lmax" in soap_values:
                _lmax = soap_values["lmax"]
            if "sigma" in soap_values:
                _sigma = soap_values["sigma"]
            self.desc = SOAP(species=_species,
                             rcut=_rcut, nmax=_nmax,
                             sigma=_sigma, periodic=True,
                             crossover=True, sparse=False)

    def set_valle_oganov_comparator(self, comp_values=None):
        if comp_values is None:
            self.comp = OFPComparator(n_top=None, dE=None,
                                      cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                                      pbc=[True, True, True], sigma=0.05, nsigma=4,
                                      recalculate=False)
        else:
            _n_top = None
            _dE = None
            _cos_dist_max = 1e-3
            _rcut = 10.
            _binwidth = 0.05
            _pbc = [True, True, True]
            _sigma = 0.05
            _nsigma = 4
            _recalculate = False

            if 'n_top' in comp_values:
                _n_top = comp_values['n_top']
            if 'dE' in comp_values:
                _dE = comp_values['dE']
            if 'cos_dist_max' in comp_values:
                _cos_dist_max = comp_values['cos_dist_max']
            if 'rcut' in comp_values:
                _rcut = comp_values['rcut']
            if 'binwidth' in comp_values:
                _binwidth = comp_values['binwidth']
            if 'pbc' in comp_values:
                _pbc = comp_values['pbc']
            if 'sigma' in comp_values:
                _sigma = comp_values['sigma']
            if 'nsigma' in comp_values:
                _nsigma = comp_values['nsigma']
            if 'recalculate' in comp_values:
                _recalculate = comp_values['recalculate']
            self.comp = OFPComparator(n_top=_n_top, dE=_dE,
                                      cos_dist_max=_cos_dist_max, rcut=_rcut,
                                      binwidth=_binwidth, pbc=_pbc, sigma=_sigma,
                                      nsigma=_nsigma, recalculate=_recalculate)

    def set_rematch_kernel_generator(self, kg_values=None):
        if kg_values is None:
            self.kernel_gen = REMatchKernel(
                metric="linear", alpha=1, threshold=1e-6)
        else:
            _metric = "linear"
            _alpha = 1
            _threshold = 1e-6
            if 'metric' in kg_values:
                _metric = kg_values['metric']
            if 'alpha' in kg_values:
                _alpha = kg_values['alpha']
            if 'threshold' in kg_values:
                _threshold = kg_values['threshold']
            self.kernel_gen = REMatchKernel(
                metric=_metric, alpha=_alpha, threshold=_threshold)

    def compare_fingerprints(self, test_model, ref_model):
        '''
        Compare the fingerprints between two structures. Each model contains a fingerprint
        dictionary, which has all the relevant information for the appropriate fingerprint.
        In the case of the Valle-Oganov fingerprint, this information is contained within
        an ASE Atoms structure.
        In the case of the bag-of-bonds fingerprint, this information is contained within
        a pair_cor dictionary. 
        In the case of the REMatch SOAP kernel, this information is contained within a set
        of normalized soap descriptors called normed_features.

        Returns a tuple of length 2. Only the bag-of-bonds comparison completely fills the tuple.
        '''
        if self.label == "valle-oganov":
            return (self.comp._compare_structure(test_model.fingerprint["ase"], ref_model.fingerprint["ase"]),)

        elif self.label == "rematch-soap":
            return (self.kernel_gen.create([test_model.fingerprint["normed_features"], ref_model.fingerprint["normed_features"]]),)

        elif self.label == "bag-of-bonds":
            # print(f"{self.fingerprint_label} fingerprint is being calculated.")
            pair_cor1 = test_model.pair_cor
            pair_cor2 = ref_model.pair_cor
            total_cum_diff = 0.
            max_diff = 0
            for n in pair_cor1.keys():
                cum_diff = 0.
                norm_factor = pair_cor1[n][0]
                dists1 = pair_cor1[n][1]
                dists2 = pair_cor2[n][1]
                assert len(dists1) == len(dists2)
                if len(dists1) == 0:
                    continue
                diff = np.abs(dists1 - dists2)
                sum = np.abs(dists1 + dists2)
                cum_diff = np.sum(diff)
                cum_sum = np.sum(sum)
                max_diff_key = np.max(diff)
                if max_diff_key > max_diff:
                    max_diff = max_diff_key
                total_cum_diff += norm_factor * 2 * cum_diff / cum_sum
            return (total_cum_diff, max_diff)

    def compare_models(self, test_model, ref_model):
        '''
        Runs comparison of models, utilizing the appropriate tolerance parameters
        depending on the global fingerprint used.
        Returns 0 if the models are exactly same, 1 if the models are the same
        within tolerance, and -1 if they are not within tolerance of each other.
        '''
        try:
            comparison = self.compare_fingerprints(test_model, ref_model)
        except:
            # models did not contain the same number of atoms (bag-of-bonds)
            return -1
        if self.label == "valle-oganov":
            if np.isclose(comparison, 0.0, atol=1e-5):
                return 0
            elif comparison < self.tolerances["valle-oganov"]:
                return 1
            else:
                return -1

        elif self.label == "bag-of-bonds":
            if np.isclose(comparison[0], 0.0, atol=1e-5) and np.isclose(comparison[1], 0.0, atol=1e-5):
                return 0
            elif comparison[0] < self.tolerances["bag-of-bonds"][0] and \
                    comparison[1] < self.tolerances["bag-of-bonds"][1]:
                return 1
            else:
                return -1
        elif self.label == "rematch-soap":
            # rematch kernel is a matrix. Here we only use one of (identical)
            # off-diagonal matrix elements to calculate the structure distance
            compare_fm = comparison[0][1]  # The cross-similarity
            distance = math.sqrt(2 - 2*compare_fm)
            if np.isclose(distance, 0.0, atol=1e-5):
                return 0
            elif distance < self.tolerances["rematch-soap"]:
                return 1
            else:
                return -1


class ParetoDominance(object):
    def get_nondominated_solutions(self, population):
        """
        Source: https://github.com/QUVA-Lab/artemis/blob/peter/artemis/general/pareto_efficiency.py

        Return all non-dominated solutions (the Pareto front) from
        a set of models. 

        param costs: An (n_points, n_costs) array

        returns: A (n_points, ) boolean array, indicating whether each point is
                 Pareto efficient
        """
        is_efficient = np.ones(population.size, dtype=bool)
        # Iterate once through the population to assemble the array of objective values
        objectives = []
        for model in population.models:
            # TODO: make flexible with number of objectives
            objectives.append([model.obj0_val, model.obj1_val])

        obj_array = np.array(objectives)

        for index, objs in enumerate(obj_array):
            if is_efficient[index]:
                # Keep any point with a lower cost
                is_efficient[is_efficient] = np.any(
                    obj_array[is_efficient] < objs, axis=1)
                is_efficient[index] = True  # And keep self

        return list(itertools.compress(population.models, is_efficient))

    def alt_nondominance(self, population):
        is_efficient = np.ones(population.size, dtype=bool)
        for index, model in enumerate(population.models):
            if is_efficient[index]:
                flags = [
                    self.compare(m, model) for m in list(itertools.compress(population.models, is_efficient))]
                # keep any point which either dominated the model or was non-dominated
                equal_or_better = [x < 0 or x == 0 for x in flags]
                is_efficient[is_efficient] = equal_or_better
                is_efficient[index] = True  # and keep self

        return list(itertools.compress(population.models, is_efficient))

    def compare(self, test_model, ref_model):
        '''

        Outputs:
        Returns -1 if test_model dominates the ref_model
        Returns 0 if both non-dominated
        Returns +1 if test_model dominated by the ref_model
        '''

        dominate_test = False
        dominate_ref = False

        # TODO: make flexible with number of objectives
        for n in range(2):

            if n == 0:
                test_obj = test_model.obj0_val
                ref_obj = ref_model.obj0_val
            elif n == 1:
                test_obj = test_model.obj1_val
                ref_obj = ref_model.obj1_val

            if test_obj < ref_obj:
                dominate_test = True
                # Check for non-domination
                if dominate_ref:
                    return 0

            elif test_obj > ref_obj:
                dominate_ref = True
                # Check for non-domination
                if dominate_test:
                    return 0

        # Otherwise one dominates the other, return the appropriate value
        if dominate_test:
            return -1
        else:
            return 1

    def choose_non_dominated(self, model1, model2):
        '''
        Outputs:
        The non-dominated model, or a random selection if both non-dominated
        '''
        dominate1 = False
        dominate2 = False

        # TODO: make flexible with number of objectives
        for n in range(2):

            if n == 0:
                model1_obj = model1.obj0_val
                model2_obj = model2.obj0_val
            elif n == 1:
                model1_obj = model1.obj1_val
                model2_obj = model2.obj1_val

            if model1_obj < model2_obj:
                dominate1 = True
                # Check for non-domination
                if dominate2:
                    model_num = np.random.randint(1, 3)
                    if model_num == 1:
                        return model1
                    elif model_num == 2:
                        return model2

            elif model1_obj > model2_obj:
                dominate2 = True
                # Check for non-domination
                if dominate1:
                    model_num = np.random.randint(1, 3)
                    if model_num == 1:
                        return model1
                    elif model_num == 2:
                        return model2

        # Otherwise one dominates the other, return the appropriate value
        if dominate1:
            return model1
        else:
            return model2


class EpsilonDominance(object):
    def __init__(self, epsilons=None):
        # Assign default epsilons if none are provided
        if epsilons is None:
            self.epsilons = [.1, .1]
        else:
            self.epsilons = epsilons

    def get_nondominated_solutions(self, population):
        """
        Source: https://github.com/QUVA-Lab/artemis/blob/peter/artemis/general/pareto_efficiency.py

        Return all non-dominated solutions (the Pareto front) from
        a set of models. 

        param costs: An (n_points, n_costs) array

        returns: A (n_points, ) boolean array, indicating whether each point is
                 Pareto efficient
        """
        is_efficient = np.ones(population.size, dtype=bool)
        for index, model in enumerate(population.models):
            if is_efficient[index]:
                flags = [
                    self.compare(m, model) for m in list(itertools.compress(population.models, is_efficient))]
                # keep any point which either dominated the model or was non-dominated
                equal_or_better = [x < 0 or x == 0 for x in flags]
                is_efficient[is_efficient] = equal_or_better
                is_efficient[index] = True  # and keep self

        return list(itertools.compress(population.models, is_efficient))

    def compare(self, test_model, ref_model):
        '''
        Outputs:
        Returns -1 if test_model dominates the ref_model
        Returns 0 if both non-dominated
        Returns +1 if test_model dominated by the ref_model
        '''

        dominate_test = False
        dominate_ref = False

        # TODO: make flexible with number of objectives

        for n in range(2):
            epsilon = float(self.epsilons[n % len(self.epsilons)])

            if n == 0:
                test_val = math.floor(test_model.obj0_val / epsilon)
                ref_val = math.floor(ref_model.obj0_val / epsilon)
            elif n == 1:
                test_val = math.floor(test_model.obj1_val / epsilon)
                ref_val = math.floor(ref_model.obj1_val / epsilon)

            if test_val < ref_val:
                dominate_test = True
                # Check for non-domination (but not same epsilon box)
                if dominate_ref:
                    return 0

            elif test_val > ref_val:
                dominate_ref = True
                # Check for non-domination (but not same epsilon box)
                if dominate_test:
                    return 0

        # If neither one is better than the other at all, they are in the same box
        if not dominate_ref and not dominate_test:
            # Check for distance to box corner
            d_test = 0.0
            d_ref = 0.0

            # TODO: make flexible with number of objectives
            for n in range(2):
                epsilon = float(self.epsilons[n % len(self.epsilons)])
                if n == 0:
                    test_obj = test_model.obj0_val
                    ref_obj = ref_model.obj0_val
                elif n == 1:
                    test_obj = test_model.obj1_val
                    ref_obj = ref_model.obj1_val

                test_eps_val = math.floor(test_obj / epsilon)
                ref_eps_val = math.floor(ref_obj / epsilon)

                d_test += (test_obj - test_eps_val*epsilon)**2
                d_ref += (ref_obj - ref_eps_val*epsilon)**2

            if d_test < d_ref:
                return -1
            else:
                return 1

        # Otherwise one dominates the other, return the appropriate value
        elif dominate_test:
            return -1
        else:
            return 1


class StructuralEpsilonDominance(object):
    '''
    A modified version of epsilon dominance. The epsilon boxes are still used,
    but now both structures within an epsilon box are kept unless they are
    classified as nearly identical by comparing their global fingerprints.
    '''

    def __init__(self, comparator=Comparator(), epsilons=None):
        # Assign default epsilons if none are provided
        if epsilons is None:
            self.epsilons = [.1, .1]
        else:
            self.epsilons = epsilons

        # store comparator object
        self.comparator = comparator

    def get_nondominated_solutions(self, population):
        """
        Source: https://github.com/QUVA-Lab/artemis/blob/peter/artemis/general/pareto_efficiency.py

        Return all non-dominated solutions (the Pareto front) from
        a set of models. 

        param costs: An (n_points, n_costs) array

        returns: A (n_points, ) boolean array, indicating whether each point is
                 Pareto efficient
        """
        is_efficient = np.ones(population.size, dtype=bool)
        for index, model in enumerate(population.models):
            if is_efficient[index]:
                flags = [
                    self.compare(m, model) for m in list(itertools.compress(population.models, is_efficient))]
                print(f"Flags are: {flags}")
                # keep any point which either dominated the model or was non-dominated
                equal_or_better = [x < 0 or x == 0 for x in flags]
                print(f"Dominated is: {equal_or_better}")
                is_efficient[is_efficient] = equal_or_better
                print(f"is_efficient is now: {is_efficient}")
                is_efficient[index] = True  # and keep self

        # is_efficient = np.ones(population.size, dtype=bool)
        # # Iterate once through the population to assemble the array of objective values
        # objectives = []
        # for model in population.models:
        #     # TODO: make flexible with number of objectives
        #     objectives.append([model.obj0_val, model.obj1_val])

        # obj_array = np.array(objectives)

        # for index, objs in enumerate(obj_array):
        #     if is_efficient[index]:
        #         # Keep any point with a lower cost
        #         is_efficient[is_efficient] = np.any(
        #             obj_array[is_efficient] < objs, axis=1)
        #         is_efficient[index] = True  # And keep self

        return list(itertools.compress(population.models, is_efficient))

    def compare(self, test_model, ref_model):
        '''
        Outputs:
        Returns -1 if test_model dominates the ref_model
        Returns 0 if both non-dominated
        Returns +1 if test_model dominated by the ref_model
        '''

        dominate_test = False
        dominate_ref = False

        # TODO: make flexible with number of objectives

        for n in range(2):
            epsilon = float(self.epsilons[n % len(self.epsilons)])

            if n == 0:
                test_val = math.floor(test_model.obj0_val / epsilon)
                ref_val = math.floor(ref_model.obj0_val / epsilon)
            elif n == 1:
                test_val = math.floor(test_model.obj1_val / epsilon)
                ref_val = math.floor(ref_model.obj1_val / epsilon)

            if test_val < ref_val:
                dominate_test = True
                # Check for non-domination (but not same epsilon box)
                if dominate_ref:
                    return 0

            elif test_val > ref_val:
                dominate_ref = True
                # Check for non-domination (but not same epsilon box)
                if dominate_test:
                    return 0

        # If neither one is better than the other at all, they are in the same box
        if not dominate_ref and not dominate_test:
            # Check for structural similarity. Note: if the model was exactly the same
            # as a population member, it was already ruled out.
            # If models fall within similarity tolerance, then keep model which is closest
            # to the corner of the epsilon box
            # Otherwise, keep both models
            similarity = self.comparator.compare_models(test_model, ref_model)
            if similarity > 0:
                print(
                    f"Models {test_model.label} and {ref_model.label} are similar within tolerance. Checking proximity to epsilon box corner.")
                d_test = 0.0
                d_ref = 0.0

                # TODO: make flexible with number of objectives
                for n in range(2):
                    epsilon = float(self.epsilons[n % len(self.epsilons)])
                    if n == 0:
                        test_obj = test_model.obj0_val
                        ref_obj = ref_model.obj0_val
                    elif n == 1:
                        test_obj = test_model.obj1_val
                        ref_obj = ref_model.obj1_val

                    print(
                        f"Non-floored objective values are: {test_obj} and {ref_obj}")

                    test_eps_val = math.floor(test_obj / epsilon)
                    ref_eps_val = math.floor(ref_obj / epsilon)

                    print(
                        f"Floored objective values are : {test_eps_val} and {ref_eps_val}.")

                    d_test += (test_obj / epsilon - test_eps_val)**2
                    d_ref += (ref_obj / epsilon - ref_eps_val)**2

                    print(f"Distances are: {d_test} and {d_ref}")

                if d_test < d_ref or np.isclose(d_test, d_ref, atol=1e-5):
                    return -1
                else:
                    return 1
            elif similarity == 0:
                # models are identical, only keep the old model
                return 1
            else:
                return 0

        # Otherwise one dominates the other, return the appropriate value
        elif dominate_test:
            return -1
        else:
            return 1


class Pool(object):
    """
    A pool of structures which are used for genetic crossing.

    Maintain two lists:
    Population - Diverse selection of models
    Archive - Non-dominated models contained within the population.
            - Separated on the Pareto front by epsilon boxes

    The initial population is created by descending along the gradient
    in both objective functions if possible. Then epsilon-MOEA is
    employed as the multi-objective optimization algorithm once the
    population size reaches the steady-state level.
    """

    def __init__(self, pool_params):
        """
        Initialize a pool with the following parameters set from i_dict:
        Capacity (int) - the steady-state population size for epsilon-MOEA
        Weights (list) - the weights of the objective functions for linear
                         selection protocol.
        Epsilons (list) - the epsilon values defining the grid for epsilon
                          dominance. Used for the "Archive" only. 
        Tolerances (dict) - the tolerance values used for global structural
                            fingerprint comparison to determine similarity
                            of structures.

        Here the "Population" and "Archive" objects are also initialized.
        The "Population" uses ParetoDominance and a zero-tolerance comparator, 
        and the "Archive" uses EpsilonDominance and the i_dict tolerances.
        """
        energy_pkg = pool_params['energy_pkg']

        # Get the capacity of the population
        if 'capacity' not in pool_params:
            if energy_pkg == 'vasp':
                self.capacity = 50
            else:  # energy_code == 'lammps' or 'gulp':
                self.capacity = 500
        else:
            if 'capacity' in pool_params:
                self.capacity = pool_params['capacity']

        # Define weights for the objective functions
        if 'weights' not in pool_params:
            self.weights = [1, 1, 1, 1, 1]
        else:
            self.weights = pool_params['weights']
            while len(self.weights) != 5:
                self.weights.append(1)

        if 'epsilons' not in pool_params:
            self.epsilons = [1, 1]
        else:
            self.epsilons = pool_params['epsilons']

        if 'fingerprint_params' not in pool_params:
            # no fingerprint comparisons are going to be made
            self.comparator = None
            self.population = Population(
                self.capacity, ParetoDominance(), self.weights, None)
            self.archive = Archive(EpsilonDominance(
                self.epsilons))
        else:
            fp_params = pool_params['fingerprint_params']
            fp_label = fp_params['label']
            tolerance = {fp_label: fp_params['tolerance']}
            self.comparator = Comparator(label=fp_label, tolerances=tolerance)
            if fp_label == "valle-oganov":
                if 'comp_values' in fp_params:
                    self.comparator.set_valle_oganov_comparator(
                        fp_params['comp_values'])
                else:
                    self.comparator.set_valle_oganov_comparator()
            elif fp_label == "rematch-soap":
                if 'soap_values' in fp_params:
                    self.comparator.set_soap_descriptor(_species=fp_params['species'],
                                                        soap_values=fp_params['soap_values']
                                                        )
                else:
                    self.comparator.set_soap_descriptor(
                        _species=fp_params['species'])

                if 'kg_values' in fp_params:
                    self.comparator.set_rematch_kernel_generator(
                        fp_params['kg_values'])
                else:
                    self.comparator.set_rematch_kernel_generator()

            self.population = Population(
                self.capacity, ParetoDominance(), self.weights, self.comparator
            )
            self.archive = Archive(
                StructuralEpsilonDominance(
                    self.comparator, self.epsilons)
            )

    def add_to_pool(self, model, select, sim_ids=None):
        """
        Attempt to add the model to the pool. If single objective
        function search, or population has not reached capacity yet,
        add the model only to the population. If multi objective function
        search, and population has reached capacity, add model to both 
        the population and the archive.

        Args:
        model: model object to be added
        select (obj): Select object
        sim_ids (list of integers): simulation ids Eg: [1] for one Xsim
        """
        # If global fingerprint comparison is going to be made, calculate
        # fingerprint for model
        if self.comparator is not None:
            self.create_fingerprint(model)
        no_prior_epsilon = True
        # If population contains at least one model, check to make sure that
        # the model is unique.
        unique = True
        if self.population.size >= 1:
            unique = self.population.check_uniqueness(model)
        if unique:
            # If population size is less than 10, add any models created
            if self.population.size < 10:
                print(f'New Model {model.label} added to population')
                self.population.extend(model)
                if select.type == "single":
                    self.population.good_pool.append(model)

            # If population size less than capacity, or single-objective function search,
            # use linear method instead of epsilon-MOEA. Note: the model will
            # be appended no matter what here. However, the model selection criteria
            # will be different than usual.
            elif 10 <= self.population.size < self.capacity or select.type == "single":
                self.population.basic_addition_to_population(
                    model, select, sim_ids=sim_ids)
                print(f'New model {model.label} added to population based'
                      ' on their objective values only!')

            # Othewise, perform usual epsilon-MOEA
            else:
                no_prior_epsilon = False
                self.population.add_to_population(model)
                self.archive.add_to_archive(model)

            # If size has now reached capacity, then add models to archive
            # in preparation for epsilon-MOEA
            if self.population.size == self.population.capacity and no_prior_epsilon:
                #non_dominated_models = ParetoDominance().get_nondominated_solutions(self.population)
                # self.archive.initialize_models(non_dominated_models)
                print("Pool has reached steady-state capacity. Seeding the archive with the \
                structural-epsilon-non-dominated models.")
                self.archive.seed_archive(self.population)
                model_labels = [model.label for model in self.archive.models]
                print(f"Archive seeded with models: {model_labels}")
        else:
            print('New model {} rejected because it was the same as'
                  ' as another model in the population!'.format(model.label))

        return select

    def create_fingerprint(self, model):
        '''
        Create fingerprint for model
        '''
        if self.comparator.label == "valle-oganov":
            # print(f"{self.fingerprint_label} fingerprint is being calculated.")
            ase_atoms = AseAtomsAdaptor.get_atoms(model.astr)
            fp, typedic = self.comparator.comp._take_fingerprints(ase_atoms)
            ase_atoms.info['fingerprints'] = self.comparator.comp._json_encode(
                fp, typedic)
            model.ase = ase_atoms

        elif self.comparator.label == "rematch-soap":
            # print(f"{self.fingerprint_label} fingerprint is being calculated.")
            ase_atoms = AseAtomsAdaptor.get_atoms(model.astr)
            features = self.comparator.desc.create(ase_atoms)
            model.normed_features = normalize(features)

        elif self.comparator.label == "bag-of-bonds":
            # print(f"{self.fingerprint_label} fingerprint is being calculated.")
            lattice = model.astr.lattice
            species_set = model.astr.types_of_specie
            coord_sets = {}
            for specie in species_set:
                coords = [
                    site.coords for site in model.astr.sites if site.specie == specie]
                coord_sets[specie] = coords
            pair_cor = {}
            for n, specie1 in enumerate(species_set):
                for specie2 in species_set[n:]:
                    # Compare each specie1 to each specie2
                    dists = []
                    for n, i in enumerate(coord_sets[specie1]):
                        if specie1 == specie2:
                            for j in coord_sets[specie2][n+1:]:
                                dists.append(dc.dist_pbc(i, j, lattice))
                        else:
                            for j in coord_sets[specie2]:
                                dists.append(dc.dist_pbc(i, j, lattice))
                    dists.sort()
                    # if specie1 == specie2:
                    #     norm_factor = len(
                    #         coord_sets[specie1])/(2*len(astr_from_file))
                    # else:
                    norm_factor = (
                        len(coord_sets[specie1]) + len(coord_sets[specie2]))/(2*len(model.astr))

                    pair_cor[str(specie1) + "-" + str(specie2)
                             ] = (norm_factor, np.array(dists))
            model.pair_cor = pair_cor

    def provide_parent_models(self, select, num_parents):
        '''
        Provide parent models for mating operations. If archive has not been
        created, then provide both parents from the population. Otherwise,
        provide one parent from the population, and one parent from the archive.
        '''
        return select.get_parents(self, num_parents)


class Select(object):
    def __init__(self, select_obj_params):
        # 'single' or 'multi'
        self.type = select_obj_params['objective_fn_type']

        # Define weights for the objective functions
        if 'weights' not in select_obj_params:
            self.weights = [1, 1, 1, 1, 1]
        else:
            self.weights = select_obj_params['weights']
            while len(self.weights) != 5:
                self.weights.append(1)

        self.all_parent_labels = []

        if 'archive_pop_bh_ratio' not in select_obj_params:
            self.archive_pop_bh_ratio = 0.7
        else:
            self.archive_pop_bh_ratio = select_obj_params['archive_pop_bh_ratio']

        # Information required for single objective optimization:
        # store optimized k; gets updated every 100th model
        self.optimum_k = -1  # default
        self.adjust_k_every = 100  # default
        if 'adjust_k_every' in select_obj_params:
            self.adjust_k_every = select_obj_params['adjust_k_every']
        self.num_required_above_50 = 100  # default
        if 'num_required_above_50' in select_obj_params:
            self.num_required_above_50 = \
                select_obj_params['num_required_above_50']

    def linear_update_selection_probs(self, models, good_pool_capacity, sim_ids=None):
        '''
        Calculates the overall objective function of each model based on linear distance to
        lowest objective function values. Returns the "good pool" of models, which are the
        top models up to a number determined by the good_pool_capacity.
        Args:

        all_models: (list) of all models evaluated so far

        good_pool_capacity: (int) maximum number of models in good pool

        sim_ids: (list of ints) indices which specifies how many exp sim
        objective functions are used.
        '''
        # Call a function to get probs based only on obj0_val (single obj)
        if self.type == 'single':
            return self.update_probs_single_obj(models, good_pool_capacity)

        # Store all models objective function values separately
        # TODO: Make this compatible with more than 2 objectives
        model_labels = []
        all_v0, all_v1 = [], []
        for model in models:
            model_labels.append(model.label)
            all_v0.append(model.obj0_val)
            if sim_ids and 1 in sim_ids:
                all_v1.append(model.obj1_val)
            # TODO: Other objective function values should be added here

        # Make an array of objective function values and transpose
        vals = np.array([all_v0])
        if len(vals[0]) == len(all_v1):
            vals = np.array([all_v0, all_v1])  # n_obj_fns x n_models
        vals = vals.T  # n_models x n_obj_fns

        # normalize using MinMaxScaler
        scaler = MinMaxScaler()
        norm_vals = scaler.fit_transform(vals)
        # Maintain weights based on number of objective functions
        weights = np.array(self.weights[:len(vals[0])])
        # weighted normalized values
        weighted_norm_vals = norm_vals/weights

        all_models_values = np.array([sum(weighted_norm_vals[i])
                                      for i in range(len(weighted_norm_vals))])
        # get cutoff for good_pool
        if len(models) <= good_pool_capacity:
            cutoff_value = max(all_models_values)
        else:
            copy_vals = all_models_values.copy()
            copy_vals.sort()
            cutoff_value = copy_vals[good_pool_capacity]

        # get good pool
        good_pool, good_pool_values = [], []
        for model, value in zip(models, all_models_values):
            model.overall_val = value
            if value <= cutoff_value:
                good_pool.append(model)
                good_pool_values.append(value)

        # Update selection probabilities using a linear model
        linear_vals = good_pool_values
        linear_probs = (linear_vals - max(linear_vals)) / \
            (min(linear_vals) - max(linear_vals))
        selection_probs = linear_probs

        # Assign probabilities to the models
        for model, prob in zip(models, selection_probs):
            model.selection_prob = prob
        return good_pool

    def update_probs_single_obj(self, models, good_pool_capacity,
                                update_cutoff_only=False):
        """
        For a search with only one objective function, updates selection
        probabliites based on an exponential function.

        Args:
        all_models: (list) of all models evaluated so far

        good_pool_capacity: (int) maximum number of models in good pool
        """
        # Get all models obj0_val
        model_labels, all_v0 = [], []
        for model in models:
            model_labels.append(model.label)
            all_v0.append(model.obj0_val)
        self.minmax_obj0 = min(all_v0), max(all_v0)

        # Get cutoff value for good pool
        if len(all_v0) < good_pool_capacity:
            self.cutoff_value = max(all_v0)
        else:
            copy_v0 = all_v0.copy()
            copy_v0.sort()
            self.cutoff_value = copy_v0[good_pool_capacity - 1]

        # NOTE: When using the function for cutoff_value only; return
        if update_cutoff_only:
            return self.cutoff_value

        # Make an array of objective function values and transpose
        vals = np.array([all_v0])
        vals = vals.T  # n_models x n_obj_fns

        # normalize using MinMaxScaler
        scaler = MinMaxScaler()
        norm_vals = scaler.fit_transform(vals)
        norm_vals = norm_vals.T[0]

        # get good pool --> pick top (good_pool_capacity) models
        good_pool_labels = [i for _, i in sorted(zip(norm_vals, model_labels))]
        good_pool_labels = good_pool_labels[:good_pool_capacity]
        good_pool = [m for m in models if m.label in good_pool_labels]
        norm_vals.sort()
        good_pool_vals = norm_vals[:good_pool_capacity]

        # optimize contant (k) in the exponential function e^(-kx) such that
        # required number of models have probability greater than 0.5
        initial_k = [-1]
        if len(good_pool_labels) > self.adjust_k_every and \
                len(good_pool_labels) > self.num_required_above_50:
            res = minimize(Select._optimize_exponential_constant, initial_k,
                           args=(self.num_required_above_50, norm_vals),
                           method='Nelder-Mead', options={'maxiter': 100})
            opt_k = res.x[0]
            self.optimum_k = opt_k
            # Get probabilities by min max exponential function
            # using the opt_k
            exponential_probs = [(math.exp(opt_k * i) - math.exp(opt_k)) /
                                 (1 - math.exp(opt_k)) for i in good_pool_vals]
        else:
            opt_k = initial_k[0]
            # Get probabilities by simple exponential function
            # using opt_k = -1
            exponential_probs = [math.exp(opt_k * i) for i in good_pool_vals]
        selection_probs = exponential_probs

        # Assign probabilities to the models
        for model, prob in zip(good_pool, selection_probs):
            model.selection_prob = prob

        return good_pool

    @staticmethod
    def _optimize_exponential_constant(k, num_required_above_50,
                                       scaled_good_pool_values):
        """
        A scalar function to minimize the exponential constant to give required
        number of models with probability above 0.5 (or 50%).

        Args:

        k: (a list or an array) of the variable for minimize function (Eg: [-1])

        scaled_good_pool_values: (1D array or list) The objective function
                                 values of models in good pool scaled between 0
                                 and 1. For example, distances_from_hull for
                                 multiobjective optimization
        """
        # Get probs based on constant k
        exponential_probs = [math.exp(k[0]*i) for i in scaled_good_pool_values]
        num_above_50 = len([i for i in exponential_probs if i > 0.5])

        return (num_required_above_50 - num_above_50)**2

    def get_parents(self, pool, num_parents, same_ab=False, abs_tol=0.2):
        '''
        Provide requested number of parent models for mating operations. 
        If archive has not been created, then provide both parents from the population. 
        Otherwise, alternate providing one parent from the population, 
        and one parent from the archive.

        Args:

        pool            - pool object
        num_parents     - integer
        same_ab (bool)  - If num_parents > 1, species whether all parents should
                         have same a, b lattice vectors
        abs_tol (float) - The maximum value for the sum of absolute difference
                         between the "ab" of two lattice vectors
        '''
        if pool.archive.size == 0 or self.type == "single":
            parents = []
            while len(parents) < num_parents:
                new_parent = self.get_a_linear_parent(pool)
                if len(parents) == 0:
                    parents.append(new_parent)
                for existing_parent in parents:
                    if existing_parent.label == new_parent.label:
                        continue
                    if same_ab:
                        ab_1 = existing_parent.astr.lattice.matrix[:2]
                        ab_2 = new_parent.astr.lattice.matrix[:2]
                        diff = np.array(ab_1) - np.array(ab_2)
                        # return first match since keys are already shuffled
                        if np.absolute(diff).sum() < abs_tol:
                            parents.append(new_parent)
                    else:
                        parents.append(new_parent)
            return parents
        else:
            parents = []
            while len(parents) < num_parents:
                # alternate adding population and archive members
                if len(parents) % 2 == 0:
                    new_parent = pool.population.produce_model()
                else:
                    new_parent = pool.archive.produce_model()
                if len(parents) == 0:
                    parents.append(new_parent)
                    self.all_parent_labels.append(new_parent.label)
                else:
                    for existing_parent in parents:
                        if existing_parent.label == new_parent.label:
                            continue
                        if same_ab:
                            ab_1 = existing_parent.astr.lattice.matrix[:2]
                            ab_2 = new_parent.astr.lattice.matrix[:2]
                            diff = np.array(ab_1) - np.array(ab_2)
                            # return first match since keys are already shuffled
                            if np.absolute(diff).sum() < abs_tol:
                                parents.append(new_parent)
                                self.all_parent_labels.append(new_parent.label)
                        else:
                            parents.append(new_parent)
                            self.all_parent_labels.append(new_parent.label)
            return parents

    def get_a_parent(self, pool):
        # produce a parent model from the archive if it exists, otherwise from the population
        if pool.archive.size == 0:
            new_parent = pool.population.produce_model()
        else:
            r = np.random.uniform()
            if r < self.archive_pop_bh_ratio:
                # should weight selection from archive and from pool
                print("Producing archive model")
                new_parent = pool.archive.produce_model()
                print(f"Archive model is {new_parent.label}")
            else:
                print("Producing population model")
                new_parent = pool.population.produce_model()
                print(f"Population model is {new_parent.label}")
        self.all_parent_labels.append(new_parent.label)
        return new_parent

    def get_a_linear_parent(self, pool):
        """
        Returns exactly one parent from the population, selected at random. 
        If single objective function search, use the population good_pool. 
        Otherwise, draw from all models.
        """
        done = False
        models = pool.population.models
        if self.type == "single":
            models = pool.population.good_pool
        while not done:
            # randomly choose a parent
            parent = random.choice(models)
            if parent.selection_prob:
                if random.random() < parent.selection_prob:
                    if self.all_parent_labels.count(parent.label) < 200:
                        done = True
                        return parent

    def return_nd_pop_models(self, pool):
        population = pool.population
        return ParetoDominance().alt_nondominance(population)


class Population(object):
    """
    The entire population of models. The archive is a subset of the population.

    Maintain a list of all models contained
    """

    def __init__(self, capacity, dominance=ParetoDominance(), weights=[1, 1, 1, 1, 1], comparator=None):
        """
        Population capacity defines the steady-state level
        """
        self.capacity = capacity
        self.models = []
        self._dominance = dominance
        self.size = 0

        # Weights for linear addition to population
        self.weights = weights
        while len(self.weights) < 5:
            self.weights.append(1)

        # "Good pool" for linear portion of multi-objective search
        self.good_pool = []
        self.comparator = comparator

    def extend(self, model):
        '''
        Add a model to the population
        '''
        self.models.append(model)
        self.size += 1

    def check_uniqueness(self, model):
        '''
        Check whether a model is unique.
        Returns True if the model is unique, 
        return False if the model is the (exact) same as another model.
        Note: if the models are considered the same "within tolerance",
        this method still returns True. 
        '''
        if self.comparator is None:
            # No comparator, so automatic return True
            return True
        else:
            flags = [self.comparator.compare_models(
                model, m) for m in self.models]
            same = [f == 0 for f in flags]
            if any(same):
                return False
            else:
                return True

    def basic_addition_to_population(self, model, select, sim_ids=None):
        '''
        Linear addition based on proximity to lowest possible
        values in all objective functions
        '''
        self.extend(model)

        if select.type == 'multi':
            # Note: no need to worry about good_pool, since this method is only
            # used if the population has not reached capacity yet.
            self.models = select.linear_update_selection_probs(
                self.models,
                self.capacity,
                sim_ids=sim_ids)
            print('New model {} added: probs updated based on sum of'
                  ' normalized obj. values!'.format(model.label))

            return select

        elif select.type == 'single':
            # Get updated cutoff value to skip probs for a bad model
            cutoff_value = select.update_probs_single_obj(self.models,
                                                          self.capacity, update_cutoff_only=True)
            if cutoff_value >= model.obj0_val:
                to_good_pool = True
            else:
                to_good_pool = False

            if to_good_pool == True:
                # Add model to good_pool
                self.good_pool.append(model)
                good_pool_values = np.array(
                    [i.overall_val for i in self.good_pool])
                if select.type == 'single':
                    good_pool_values = np.array([i.obj0_val for i in
                                                 self.good_pool])

                if len(self.good_pool) > self.capacity:
                    # remove worst model from good_pool
                    remove_ind = np.argmax(good_pool_values)
                    demoted_label = self.good_pool[remove_ind].label
                    del self.good_pool[remove_ind]
                    print('New Model {} added to good_pool and Model {} demoted'
                          ' from good_pool'.format(model.label, demoted_label))
                else:
                    print('New Model {} added to good_pool'.format(model.label))

                # scale the good_pool_values using MinMaxScaler
                good_pool_values = good_pool_values.reshape(-1, 1)
                scaler = MinMaxScaler()
                scaled_values = scaler.fit_transform(good_pool_values)[:, 0]

                # optimize k for every 100th model (labels are continuous)
                if model.label % select.adjust_k_every == 0:
                    initial_k = select.optimum_k
                    # optimize k
                    res = minimize(Select._optimize_exponential_constant, initial_k,
                                   args=(select.num_required_above_50,
                                         scaled_values),
                                   method='Nelder-Mead', options={'maxiter': 100})
                    # Store new optimum k
                    select.optimum_k = res.x[0]

                opt_k = select.optimum_k
                # Update probabilities by min max exponential function using opt_k
                exponential_probs = [(math.exp(opt_k * i) - math.exp(opt_k)) /
                                     (1 - math.exp(opt_k)) for i in scaled_values]
                # Assign probabilities to the models
                for m, prob in zip(self.good_pool, exponential_probs):
                    m.selection_prob = prob

                return select

            if to_good_pool == False:
                print('New Model {} not added to good_pool'.format(model.label))

                return select

    def add_to_population(self, model):
        '''
        Test the addition of the model to the population

        Criteria:
        If model dominates a population member, replace
        If model is dominated by a population member, reject
        If neither, then replace a population member at random
        Note: it is possible for the model to simultaneously dominate pop
        member (a) and be dominated by pop member (b). This is possible
        due to the fact that the initial population members are
        not checked for domination. Over time, these dominated population
        members will be bred out of the population.

        Args:
        model: model object to be added
        sim_ids (list of integers): simulation ids Eg: [1] for one Xsim
        '''
        dominates = []
        dominated = False
        for index, m in enumerate(self.models):
            flag = self._dominance.compare(model, m)
            if flag < 0:
                dominates.append(index)
            elif flag > 0:
                dominated = True

        # If dominates any models, then replace one at random
        if len(dominates) > 0:
            del self.models[np.random.choice(dominates)]
            self.models.append(model)
            print(
                f"Model {model.label} appended to population by domination replacement.")
        # If does not dominate, but is not dominated, then replace any one
        # population member at random
        elif not dominated:
            del self.models[np.random.randint(0, self.size - 1)]
            self.models.append(model)
            print(
                f"Model {model.label} appended to population by non-domination replacement.")
        else:
            print(
                f"Model {model.label} not added to population because dominated by pop member (and does not dominate a pop member).")

    def produce_model(self):
        '''
        Produce a model for breeding. Choose two models, and return the 
        non-dominated model. If both are non-dominated, then return one 
        randomly. 
        '''

        [model_one, model_two] = np.random.choice(self.models, 2)
        return self._dominance.choose_non_dominated(model_one, model_two)


class Archive(object):
    '''
    An archive containing only the elite non-dominated models
    '''

    def __init__(self, dominance=StructuralEpsilonDominance(comparator=Comparator(),
                                                            epsilons=[1, 1])):
        """
        Initialize the archive with the following attributes:
        models (list) - a list of the non-dominated models
        _dominance - the dominance algorithm which is employed. This can
                     be either standard Pareto dominance, or epsilon
                     dominance. 
        size (int) - the number of models in the archive
        """

        self.models = []
        self._dominance = dominance
        self.size = 0

    def seed_archive(self, population):
        '''
        Seed the archive with the initial set of non-dominated models, according
        to the archive dominance criteria.
        '''
        self.models = self._dominance.get_nondominated_solutions(population)
        self.size = len(self.models)

    def initialize_models(self, non_dominated_models):
        '''
        Initialize the archive with a set of non-dominated models
        '''
        self.models = non_dominated_models
        self.size = len(non_dominated_models)

    def add_to_archive(self, model):
        '''
        Test the addition of the model to the population

        Criteria:
        If model dominates an archive member, replace
        If model is dominated by an archive member, reject
        If neither, then add the model to the archive

        Args:
        model: model object to be added
        sim_ids (list of integers): simulation ids Eg: [1] for one Xsim
        '''

        flags = [self._dominance.compare(model, m) for m in self.models]
        print(f"Archive flags: {flags}")
        labels = [m.label for m in self.models]
        print(f"Archive labels: {labels}")
        nondominated = [x == 0 for x in flags]
        dominated = [x > 0 for x in flags]

        if any(dominated):
            return False
        else:
            self.models = list(itertools.compress(
                self.models, nondominated)) + [model]
            self.size = len(self.models)
            print(f"Size updated. New size: {self.size}")
            return True

    def produce_model(self):
        '''
        Returns randomly selected model
        '''
        return np.random.choice(self.models)
