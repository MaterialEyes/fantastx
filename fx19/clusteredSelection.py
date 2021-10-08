from __future__ import division, unicode_literals, print_function
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
import copy

"""
This module contains functions to conduct multi-objective search.
It contains a pool of structures which contains the "Population"
sub-group of active models. The "Population" further divides the
models into clusters. These clusters are actively used in genetic
operations: mating operations can be conducted which pull models
from the same cluster, or different clusters.
Model selection is governed by non-domination ranking. Each model
is ranked twice:
First, the models are ranked by non-domination within the entire
population. Models within the last ranking tier are considered
to be candidates for replacement.
Second, the models are ranked by non-domination within their
respective clusters. This ranking determines their selection
probability, P, where P = exp(-cluster_rank). Ranking starts at zero.
Selection occurs using a roulette selection method. Models are
chosen at random from the population, and a second random number
is drawn. If the second random number is smaller than their selection
probability, then the model is selected. Otherwise, more random
models are drawn until one is successfully chosen.

Non-domination is calculated using "structural epsilon dominance",
though other dominance ranking methods are also included in the
below code. In this dominance mechanism, models are not allowed to
live in the same epsilon box of objective space if they are
structurally similar.

The algorithm is steady-state, adding one child structure at a time.

Note: single-objective search is also supported, using the original method
of V.S.C. Kolluru.

Note: currently only compositional clusters are supported,
support for hierarchical clusters will be added in a subsequent update.
"""


class Comparator(object):
    '''
    Class which handles all structural fingerprinting. Contains functions
    to create fingerprints, compare fingerprint, and compare models
    based on their fingerprints (whether local or global).
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
        self.zbounds = None

    def set_soap_descriptor(self, _species, soap_values=None):
        '''
        Creates the class SOAP descriptor object.

        Arguments:

        _species: (string array) containing the element names of
        all atomic species handled by the descriptor.

        soap_values: (dictionary) containing user-defined
        values for some or all SOAP descriptor parameters.
        '''
        if soap_values is None:
            self.desc = SOAP(species=_species, rcut=5.0, nmax=9, lmax=6,
                             sigma=0.5, periodic=True, crossover=True,
                             sparse=False)
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
                             lmax=_lmax, sigma=_sigma,
                             periodic=True, crossover=True, sparse=False)

    def set_valle_oganov_comparator(self, comp_values=None):
        '''
        Creates the class valle-oganov comparator object.

        Arguments:

        comp_values: (dictionary) containing user-defined
        values for some or all valle-oganov comparator parameters.
        '''
        if comp_values is None:
            self.comp = OFPComparator(n_top=None, dE=None,
                                      cos_dist_max=1e-3, rcut=10.,
                                      binwidth=0.05, pbc=[True, True, True],
                                      sigma=0.05, nsigma=4, recalculate=False)
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
                                      binwidth=_binwidth, pbc=_pbc,
                                      sigma=_sigma, nsigma=_nsigma,
                                      recalculate=_recalculate)

    def set_rematch_kernel_generator(self, kg_values=None):
        '''
        Creates the class SOAP REMatch kernel generator object, to map
        local SOAP descriptors to a global descriptor.

        Arguments:

        kg_values: (dictionary) containing user-defined values for
        some or all REMatch kernel generator parameters.
        '''
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
        Compare the fingerprints between two structures. Each model
        contains a fingerprint dictionary, assigned using the
        create_fingerprint function, which has all the relevant
        information for the appropriate fingerprint. In the case of the
        Valle-Oganov fingerprint, this information is contained within
        an ASE Atoms structure. In the case of the bag-of-bonds
        fingerprint, this information is contained within a pair_cor
        dictionary. In the case of the REMatch SOAP kernel, this
        information is contained within a set of normalized soap
        descriptors called normed_features.

        Returns a tuple of length 2, quantifying the fingerprint
        similarity. Only the bag-of-bonds comparison completely
        fills the tuple, all other comparisons only result in one
        similarity or distance metric.

        Note: the order of the test_model and ref_models are irrelevant,
        the function will return the same value if models A and B are
        swapped.

        Args:

        test_model (obj): structure_record.model() A for the comparison.

        ref_model (obj): structure_record.model() B for the comparison.
        '''
        if self.label == "valle-oganov":
            return (self.comp._compare_structure(test_model.fingerprint["ase"],
                                                 ref_model.fingerprint["ase"]),
                    )

        elif self.label == "rematch-soap":
            return (self.kernel_gen.create([test_model.fingerprint[
                "normed_features"],
                ref_model.fingerprint["normed_features"]]),)

        elif self.label == "bag-of-bonds":
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
        Runs comparison of models, utilizing the appropriate tolerance
        parameters depending on the global fingerprint used.

        Returns 0 if the models are exactly same, 1 if the models are
        the same within tolerance, and -1 if they are not within
        tolerance of each other.

        Note: test_model and ref_model are interchangeable, the same
        comparison result will be yielded if models A and B are swapped.

        Args:

        test_model (obj): structure_record.model() A for the comparison.

        ref_model (obj): structure_record.model() B for the comparison.
        '''
        try:
            comparison = self.compare_fingerprints(test_model, ref_model)
        except AssertionError:
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
            if np.isclose(comparison[0], 0.0, atol=1e-5) and \
                    np.isclose(comparison[1], 0.0, atol=1e-5):
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

    def create_fingerprint(self, model):
        '''
        Create the fingerprint for the model object.

        Args:

        model (obj): structure_record.model() which will be assigned a
        fingerprint.
        '''
        if self.label == "valle-oganov":
            ase_atoms = AseAtomsAdaptor.get_atoms(model.astr)
            fp, typedic = self.comp._take_fingerprints(ase_atoms)
            ase_atoms.info['fingerprints'] = self.comp._json_encode(
                fp, typedic)
            model.ase = ase_atoms

        elif self.label == "rematch-soap":
            ase_atoms = AseAtomsAdaptor.get_atoms(model.astr)
            features = self.desc.create(ase_atoms)
            model.normed_features = normalize(features)

        elif self.label == "bag-of-bonds":
            model_astr = copy.deepcopy(model.astr)
            if self.zbounds is not None:
                site_removal_indices = []
                for site_index, site in enumerate(model_astr.sites):
                    if site.coords[2] < self.zbounds[0] or \
                            site.coords[2] > self.zbounds[1]:
                        site_removal_indices.append(site_index)
                model_astr.remove_sites(site_removal_indices)

            lattice = model_astr.lattice
            species_set = model_astr.types_of_specie
            coord_sets = {}
            for specie in species_set:
                coords = [
                    site.coords for site in model_astr.sites
                    if site.specie == specie]
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
                    norm_factor = (
                        len(coord_sets[specie1]) + len(coord_sets[specie2])) \
                        / (2*len(model.astr))

                    pair_cor[str(specie1) + "-" + str(specie2)
                             ] = (norm_factor, np.array(dists))
            model.pair_cor = pair_cor


class ParetoDominance(object):
    '''
    Class which performs non-dominance calculations. Non-dominance is
    determined simply based on the relation between objective function
    values of two models. If all objective function values for model A
    are better (lower) than those of model B, then model A dominates
    model B. If at least one (but not all) model B objective function
    value is better than that of model A, then the models are
    non-dominated.
    '''

    def get_nondominated_solutions(self, models):
        """
        Source: https://github.com/QUVA-Lab/artemis/blob/peter/artemis
        /general/pareto_efficiency.py

        Return all non-dominated solutions (the Pareto front) from
        a set of models.

        Returns the list of non-dominated models, and the list of the models
        which are dominated by at least one other model. 

        Args:

        models (list of objs): the set of structure_record.model()s
        for which non-domination will be determined.
        """
        is_efficient = np.ones(len(models), dtype=bool)
        # Iterate once through the population to
        # assemble the array of objective values
        objectives = []
        for model in models:
            # TODO: make flexible with number of objectives
            objectives.append([model.obj0_val, model.obj1_val])

        obj_array = np.array(objectives)

        for index, objs in enumerate(obj_array):
            if is_efficient[index]:
                # Keep any point with a lower cost
                is_efficient[is_efficient] = np.any(
                    obj_array[is_efficient] < objs, axis=1)
                is_efficient[index] = True  # And keep self

        non_dominated_solutions = list(
            itertools.compress(models, is_efficient))
        dominated_solutions = list(
            itertools.compress(models, np.invert(is_efficient)))
        return non_dominated_solutions, dominated_solutions

    def rank_models(self, models, starting_rank, flag):
        '''
        Function which recursively ranks models according to
        non-domination. 

        Returns non-dominated (rank 0) members of the set of models.
        Flag will determine which selection rank (and selection
        probability if relevant) is updated.

        Args:

        models (list of objs): the structure_record.model()s which will
        be ranked by non-domination.

        starting_rank (int): the rank which will be assigned to the
        non-dominated models.

        flag (string): indicates the set of models which is being ranked.
        Choices are "cluster", or "population".
        '''
        assert flag in ["cluster", "population"]
        nd_solutions, d_solutions = self.get_nondominated_solutions(models)
        for model in nd_solutions:
            if flag == "cluster":
                model.cluster_rank = starting_rank
                model.selection_prob = np.exp(-model.cluster_rank)
            elif flag == "population":
                model.rank = starting_rank
        if d_solutions:
            self.rank_models(d_solutions, starting_rank + 1, flag)

        if starting_rank == 0:
            return nd_solutions

    def alt_nondominance(self, population):
        """
        Inspired by: https://github.com/QUVA-Lab/artemis/blob/peter/artemis
        /general/pareto_efficiency.py

        An alternative non-dominance calculator which uses the self
        calculated comparison flags between models to determine
        non-dominance.

        Returns the list of non-dominated models.

        Args:

        population (obj): the population from which the
        non-dominated solutions are being obtained.
        """
        is_efficient = np.ones(population.size, dtype=bool)
        for index, model in enumerate(population.models):
            if is_efficient[index]:
                flags = [
                    self.compare(m, model) for m in
                    list(itertools.compress(population.models, is_efficient))]
                # keep any point which either dominated the model
                # or was non-dominated
                equal_or_better = [x < 0 or x == 0 for x in flags]
                is_efficient[is_efficient] = equal_or_better
                is_efficient[index] = True  # and keep self

        return list(itertools.compress(population.models, is_efficient))

    def compare(self, test_model, ref_model):
        '''
        Function which performs comparison between models

        Returns -1 if test_model dominates the ref_model
        Returns 0 if both non-dominated
        Returns +1 if test_model dominated by the ref_model

        Args:

        test_model (obj): structure_record.model() A for the comparison

        ref_model (obj): structure_record.model() B for the comparison
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
        Function which compares two models and returns the model
        which dominates the other. If the two models are
        non-dominated, returns one at random.

        Args:

        model1 (obj): structure_record.model() A for the comparison

        model2 (obj): structure_record.model() B for the comparison
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


class StructuralEpsilonDominance(object):
    '''
    Class which performs non-dominance calculations. Compared to 
    ParetoDominance, here non-dominance is determined based on two
    additional factors. First, non-dominance is calculated based on
    a grid which discretizes the objective function space. If two
    models occupy the same grid square (with side lengths of
    epsilon_a and epsilon_b), then the model which is closest to the
    corner of the square is considered to dominate the other model.
    HOWEVER, if the two models are structurally similar, then a flag
    is thrown. This has the advantage of only performing structural
    similarity checks for models which lie within the same grid box,
    which may save large amounts of computational expense depending
    on which fingerprinting method is used.
    '''

    def __init__(self, comparator=Comparator(), epsilons=None):
        '''
        Args:

        comparator (obj): instance of the comparator class
        which will perform all structural similarity checks

        epsilons (list of floats): the grid size of the multiobjective
        space. Should be the same length as the number of objectives,
        and should be in the same order as the objectives are assigned
        to the models.
        '''
        # Assign default epsilons if none are provided
        if epsilons is None:
            self.epsilons = [.1, .1]
        else:
            self.epsilons = epsilons

        # store comparator object
        self.comparator = comparator

    def get_nondominated_solutions(self, models):
        """
        Inspired by: https://github.com/QUVA-Lab/artemis/blob/peter/artemis
        /general/pareto_efficiency.py

        Returns the list of non-dominated models, and the list of the models
        which are dominated by at least one other model, based on comparison
        function flags.

        Args:

        models (list of objs): the set of structure_record.model()s for which
        non-domination will be determined.
        """
        is_efficient = np.ones(len(models), dtype=bool)
        for index, model in enumerate(models):
            if is_efficient[index]:
                flags = [
                    self.compare(m, model) for m in
                    list(itertools.compress(models, is_efficient))]
                # keep any point which either dominated the model
                # or was non-dominated
                equal_or_better = [x < 0 or x == 0 for x in flags]
                is_efficient[is_efficient] = equal_or_better
                is_efficient[index] = True  # and keep self

        non_dominated_solutions = list(
            itertools.compress(models, is_efficient))
        dominated_solutions = list(
            itertools.compress(models, np.invert(is_efficient)))
        return non_dominated_solutions, dominated_solutions

    def rank_models(self, models, model_level_structure, flag):
        '''
        Function which recursively ranks models according to
        non-domination. 

        Returns non-dominated (rank 0) members of the set of models.
        Flag will determine which selection rank (and selection
        probability if relevant) is updated.

        Args:

        models (list of objs): the structure_record.model()s which
        will be ranked by non-domination.

        starting_rank (int): the rank which will be assigned to the
        non-dominated models.

        flag (string): indicates the set of models which is being ranked.
        Choices are "cluster", or "population".
        '''
        assert flag in ["cluster", "population"]
        current_rank = len(model_level_structure)
        nd_solutions, d_solutions = self.get_nondominated_solutions(models)
        for model in nd_solutions:
            if flag == "cluster":
                model.cluster_rank = current_rank
                model.selection_prob = np.exp(-model.cluster_rank)
            elif flag == "population":
                model.rank = current_rank

        model_level_structure.append(nd_solutions)
        if d_solutions:
            self.rank_models(d_solutions, model_level_structure, flag)

    def update_model_levels(self, level_structure, new_model, flag):
        '''
        Function which updates the non-dominated level structure based
        on the new model. This is much more efficient than recalculating
        the entire non-dominance ranking of the set of models. Works as
        follows:

        Finds lowest tier in which the model is not dominated by any models.
        If the model dominates all models in the tier, then insert the model
        into the level structure in its own tier, bumping the dominated
        tier and all other tiers to a higher rank. 
        If the model dominates some of the models in the tier, then replace
        those models with the model, and continue the same process with the
        dominated models (beginning with the next tier).
        If the model is non-dominated with all models in the tier, then
        append the model to the tier with no other adjustments. 

        Arguments:

        level_structure (list of lists): List containing the list of models
        at each non-domination rank.

        new_model (obj): the new structure_record.model() to be added to
        the level structure.

        flag (string): determines which rank (and selection probability
        if relevant) is updated.
        '''
        T = [new_model]
        moved_levels_up = False
        for level_index in range(len(level_structure)):
            level = level_structure[level_index]
            dominated_models = []
            T_model_dominated = False
            for m in level:
                flags = [self.compare(m, T_model) for T_model in T]
                if -1 in flags:
                    T_model_dominated = True
                    break
                elif 1 in flags:
                    dominated_models.append(m)
                elif flag == 2:
                    print("SIMILARITY FLAG THROWN")
                    # model too similar, return False (non-unique)
                    return False

            if T_model_dominated:
                if level_index == len(level_structure) - 1:
                    level_structure.append(T)
                    for model in T:
                        if flag == "population":
                            model.rank = level_index + 1
                        elif flag == "cluster":
                            model.cluster_rank = level_index + 1
                            model.selection_prob = np.exp(
                                -model.cluster_rank)
                    break
                continue

            if len(dominated_models) == len(level):
                # all models were dominated, so shift this level
                # and all subsequent levels upward one level
                level_structure.insert(level_index, T)
                for model in T:
                    if flag == "population":
                        model.rank = level_index
                    elif flag == "cluster":
                        model.cluster_rank = level_index
                        model.selection_prob = np.exp(
                            -model.cluster_rank)
                moved_levels_up = True
                break

            elif len(dominated_models) == 0:
                for model in T:
                    level.append(model)
                    if flag == "population":
                        model.rank = level_index
                    elif flag == "cluster":
                        model.cluster_rank = level_index
                        model.selection_prob = np.exp(
                            -model.cluster_rank)
                break

            else:
                for model in dominated_models:
                    level.remove(model)
                for model in T:
                    level.append(model)
                    if flag == "population":
                        model.rank = level_index
                    elif flag == "cluster":
                        model.cluster_rank = level_index
                        model.selection_prob = np.exp(
                            -model.cluster_rank)
                T = dominated_models
                if level_index == len(level_structure) - 1:
                    level_structure.append(T)
                    break

        if moved_levels_up:
            for level in level_structure[level_index + 1:]:
                for model in level:
                    if flag == "population":
                        model.rank += 1
                    elif flag == "cluster":
                        model.cluster_rank += 1
                        model.selection_prob = np.exp(-model.cluster_rank)
        return True  # unique

    def compare(self, test_model, ref_model):
        '''
        Function which performs comparison between models

        Returns -1 if test_model dominates the ref_model
        Returns 0 if both non-dominated
        Returns +1 if test_model dominated by the ref_model
        Returns +2 if the models are identified as being the same as
        each other within the set tolerance limits of the comparator.

        Args:

        test_model (obj): structure_record.model() A for the comparison

        ref_model (obj): structure_record.model() B for the comparison
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

        # If neither one is better than the other at all,
        # they are in the same box
        if not dominate_ref and not dominate_test:
            # Check for structural similarity.
            # Note: if the model was exactly the same as a population member,
            # it was already ruled out. If models fall within similarity
            # tolerance, then keep model which is closest to the corner
            # of the epsilon box. Otherwise, keep both models
            similarity = self.comparator.compare_models(test_model, ref_model)
            if similarity >= 0:
                return 2
            else:
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
                    d_test += (test_obj / epsilon - test_eps_val)**2
                    d_ref += (ref_obj / epsilon - ref_eps_val)**2
                if d_test < d_ref or np.isclose(d_test, d_ref, atol=1e-5):
                    return -1
                else:
                    return 1

        # Otherwise one dominates the other, return the appropriate value
        elif dominate_test:
            return -1
        else:
            return 1


class Pool(object):
    """
    A pool of structures which are used for genetic crossing. While the
    pool does contain all models which are created during the evolutionary
    algorithm process, all active models are contained with a "Population"
    subgroup. 

    Intertwined with the Select class. The Pool handles all model addition
    to the population, while Select handles all parent selection from the
    population.

    The initial population is created by descending along the gradient
    in both objective functions if possible. Then a cluster-based MOEA is
    employed as the multi-objective optimization algorithm once the
    population size reaches the steady-state level.
    """

    def __init__(self, pool_params):
        """
        Initialize a pool with the following parameters set from i_dict:

        Capacity (int): the steady-state population size for epsilon-MOEA

        Weights (list): the weights of the objective functions for linear
        selection protocol.

        Epsilons (list): the epsilon values defining the grid for epsilon
        dominance. Used for the "Archive" only.

        cluster_obj (obj): the cluster object which will be used for
        clustering during the evolutionary algorithm process.

        fingerprint_params (dict): contains all necessary parameters to
        initialize the Comparator object for the population.

        Here the "Population" objects is also initialized.
        The "Population" uses StructuralEpsilonDominance, though simple
        ParetoDominance is also usable.
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

        self.all_models = []

        if 'cluster_obj' not in pool_params:
            self.cluster_obj = None
        else:
            self.cluster_obj = pool_params["cluster_obj"]

        if 'fingerprint_params' not in pool_params:
            # no fingerprint comparisons are going to be made
            self.comparator = None
            self.population = Population(
                self.capacity, ParetoDominance(), self.weights, None)
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
                    self.comparator.set_soap_descriptor(
                        _species=fp_params['species'],
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
            if 'zbounds' in fp_params:
                self.comparator.zbounds = fp_params['zbounds']
            else:
                self.comparator.zbounds = None

            self.population = Population(
                self.capacity, StructuralEpsilonDominance(
                    self.comparator, self.epsilons), self.weights,
                self.comparator, self.cluster_obj
            )

    def add_to_pool(self, model, select, sim_ids=None):
        """
        Attempt to add the model to the pool. If single objective
        function search, or population has not reached capacity yet,
        simply add the model to the population. If the population size
        has not reached a set minimum capacity then the model is added
        without any further steps. If the population is larger than the
        minimum capacity but has not reached its steady state capacity,
        then the model is also ranked within the population using a
        simple measure of the distance to the lowest possible objective
        function values. This is also used if single objective optimization
        is employed. 

        If multi objective function search, and population has reached
        steady-state capacity, add model to the population using the full
        procedure.

        Args:

        model (obj): structure_record.model() to add to the pool

        select (obj): instance of Select which will be updated by
        model addition.

        sim_ids (list of integers): simulation ids Eg: [1] for one Xsim
        """
        # If global fingerprint comparison is going to be made, calculate
        # fingerprint for model
        if self.comparator is not None:
            self.comparator.create_fingerprint(model)
        no_prior_epsilon = True
        self.all_models.append(model)
        # If population contains at least one model, but has not reached
        # capacity, check for uniqueness here. Otherwise, uniqueness
        # will be checked for internally when adding to the population.
        unique = True
        if 1 <= self.population.size < self.capacity:
            unique = self.population.check_uniqueness(model)
        if unique:
            # If population size is less than 10, add any models created
            if self.population.size < 10:
                print(f'New Model {model.label} added to population')
                self.population.extend(model)
                if select.type == "single":
                    self.population.good_pool.append(model)

            # If population size less than capacity,
            # or single-objective function search,
            # use linear method instead of clustered-selection.
            # Note: the model will be appended no matter what here.
            # However, the model selection criteria will be different
            # than usual.
            elif 10 <= self.population.size < self.capacity \
                    or select.type == "single":
                self.population.basic_addition_to_population(
                    model, select, sim_ids)
                print(f'New model {model.label} added to population based'
                      ' on their objective values only!')

            # Othewise, perform usual cluster-MOEA
            else:
                no_prior_epsilon = False
                # return non-domination flag with addition to population
                model_added = self.population.add_to_population(
                    model)

                # Perform auto-adaptive adjustment of
                # genetic operator probabilities
                if model_added and \
                        select.operator_assignment == "auto-adaptive":
                    # update operator probabilities in select
                    # Formula:
                    # P_i=(C_i + epsilon)/Sum_j=1->N_operators(C_j + epsilon)
                    # Here epsilon = 1
                    operator_counts = np.zeros(len(select.operator_hashmap))
                    for operator in self.population.operator_inheritance:
                        if operator != "random":
                            operator_counts[
                                select.operator_hashmap[operator]
                            ] += 1
                        else:
                            operator_counts += 1/len(select.operator_hashmap)
                    divisor = self.population.non_dominated_size + \
                        len(select.operator_hashmap)
                    select.operator_frequencies = [
                        (count + 1)/divisor for count in operator_counts]

            # If size has now reached capacity, then add models to archive
            # in preparation for epsilon-MOEA
            if self.population.size == self.population.capacity \
                    and no_prior_epsilon:
                self.population.init_clustering_and_ranking()
        else:
            print('New model {} rejected because it was the same as'
                  ' as another model in the population!'.format(model.label))
        return select

    def provide_parent_models(self, select, num_parents):
        '''
        Provide parent models for mating operations.

        Args:

        select (obj): the instance of select which is used
        to select the parents.

        num_parents (int): the number of parents to choose.
        '''
        return select.get_parents(self, num_parents)


class Select(object):
    '''
    Class which handles choosing parents for mating operations.

    Selection is done using a roulette scheme, where a random model
    is selected, and then the model is chosen if a second number
    is lower than the models pre-determined selection probability.

    In the case of multi-objective optimization, this selection
    probability is determined based on the non-domination rank of
    the model within its cluster. 

    In the case of single objective optimization, uses the evaluated
    attributes of a model to assign the selection probability. 
    '''

    def __init__(self, select_obj_params):
        '''
        Args:

        select_obj_params (dictionary): all params which are needed to
        create the object. All parameters have default values which are
        assigned if they are not found in the dictionary.

        Some functionality is not included if not listed in the params
        dictionary. Namely, auto-adaptive operator selection (where the 
        relative frequency of each operator in the evolutionary process
        will be updated based on the operators which were used to create
        the non-dominated [or otherwise elite] solutions) can be used
        if listed in the dictionary, but will not be used otherwise.
        '''
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

        # Store operator information for auto-adaptively adjusting the operator
        # frequency for genetic operations
        if 'operators' not in select_obj_params:
            self.operators = ['perturb_sites', 'fraction_slice']
        else:
            self.operators = select_obj_params['operators']
        if 'operator_assignment' in select_obj_params:
            self.operator_assignment = select_obj_params['operator_assignment']
        else:
            self.operator_assignment = "fixed"

        self.operator_hashmap = {
            key: index for index, key in enumerate(self.operators)}
        if 'operator_frequencies' in select_obj_params:
            self.operator_frequencies = select_obj_params[
                'operator_frequencies'
            ]
        else:
            self.operator_frequencies = [
                1/len(self.operators)]*len(self.operators)

    def linear_update_selection_probs(self, models, good_pool_capacity,
                                      sim_ids=None):
        '''
        Calculates the overall objective function of each model based
        on linear distance to lowest objective function values. Returns
        the "good pool" of models, which are the top models up to a
        number determined by the good_pool_capacity.

        Args:

        all_models (list): of all models evaluated so far

        good_pool_capacity (int): maximum number of models in good pool

        sim_ids (list of ints): indices which specifies how many exp sim
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

        good_pool_capacity (int): maximum number of models in good pool

        update_cutoff_only (bool): Returns only cutoff value when True
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
        A scalar function to minimize the exponential constant to give
        required number of models with probability above 0.5 (or 50%).

        Args:

        k: (a list or an array) of the variable for
        minimize function (Eg: [-1])

        scaled_good_pool_values: (1D array or list) The objective function
        values of models in good pool scaled between 0 and 1.
        """
        # Get probs based on constant k
        exponential_probs = [math.exp(k[0]*i) for i in scaled_good_pool_values]
        num_above_50 = len([i for i in exponential_probs if i > 0.5])

        return (num_required_above_50 - num_above_50)**2

    def get_parents(self, pool, num_parents,
                    same_cluster=None, same_ab=False, abs_tol=0.2):
        '''
        Provide requested number of parent models for mating operations.
        Always chooses the first parent from the non-dominated solutions,
        then chooses the next parent from all models probabilistically. 
        The selection method for those models is a "roulette" method,
        where the model is selected at random, then a second random number
        is drawn and checked against the models selection probability. If
        the number is smaller than the selection probability, then the
        model is successfully chosen. Otherwise, another model is randomly
        drawn, until success is achieved.

        Returns the parents in a list.

        Args:

        pool (obj): the pool whose population is being drawn from.

        num_parents (int): how many parents to draw.

        same_cluster (bool): whether to choose the 2nd parent from the
        same cluster as the initial parent, or a different cluster than the
        initial parent.

        same_ab (bool): If num_parents > 1, specifies whether all parents 
        should have same a, b lattice vectors

        abs_tol (float): If num_parents > 1, specifies the the maximum value
        for the sum of the absolute difference between the "ab" of two
        lattice vectors
        '''
        if pool.population.size != pool.capacity or self.type == "single":
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
            first_parent = pool.population.produce_model(
                non_dominated=True
            )
            if same_cluster:
                # if cluster requirements are in place, it is required
                # that the first model comes from a multi-model cluster
                while first_parent.cluster not in \
                        pool.population.multi_model_clusters:
                    first_parent = pool.population.produce_model(
                        non_dominated=True)
            parents.append(first_parent)
            self.all_parent_labels.append(first_parent.label)

            while len(parents) < num_parents:
                # produce model differently if cluster requirements
                # are in place
                if same_cluster is None:
                    new_parent = pool.population.produce_model(
                        non_dominated=False)
                    # print(f"same_cluster none: {new_parent}")
                else:
                    new_parent = \
                        pool.population.produce_model(False,
                                                      parents[0].cluster,
                                                      same_cluster)
                    # print(f"same_cluster {same_cluster}: {new_parent}")

                # if model was not selected, new_parent will be None,
                # so continue loop
                if new_parent is not None:
                    for existing_parent in parents:
                        if existing_parent.label == new_parent.label:
                            continue
                        if same_ab:
                            ab_1 = existing_parent.astr.lattice.matrix[:2]
                            ab_2 = new_parent.astr.lattice.matrix[:2]
                            diff = np.array(ab_1) - np.array(ab_2)
                            # return first match since keys are
                            # already shuffled
                            if np.absolute(diff).sum() < abs_tol:
                                parents.append(new_parent)
                                self.all_parent_labels.append(new_parent.label)
                        else:
                            parents.append(new_parent)
                            self.all_parent_labels.append(new_parent.label)
            return parents

    def get_a_parent(self, pool):
        '''
        Function to draw a single parent from the pool's population, using
        the population's "produce_model" function. Only one parent is
        being chosen, so it is selected from the non-dominated models.

        Args:

        pool (obj): the pool whose population is being drawn from.
        '''
        # produce a parent model from the non-dominated pool models
        new_parent = pool.population.produce_model(non_dominated=True)
        self.all_parent_labels.append(new_parent.label)
        return new_parent

    def get_a_linear_parent(self, pool):
        """
        When the pool's capacity is below the steady-state capacity, model
        selection probabilities are assigned based on the distance from
        the lowest possible objective function values. Roulette selection
        is used, where a model is chosen at random, and then a 2nd random
        is drawn to determine if the model is selected for mating. If
        rejected, a different random model is chosen. Unlike the function
        "get_a_parent", here the model is drawn from the populations
        "good_pool" or entire set of models instead of from the
        non-dominated set.

        Args:

        pool (obj): the pool from which the model is being drawn.
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


class Population(object):
    """
    Class which manages the active set of models which are being used
    for mating operations. Here cluster selection is used. In cluster
    selection, every model is assigned to a cluster. Within each cluster,
    the models are ranked according to non-domination. The non-domination
    ranks are used to assign selection probabilities. In this manner, the
    non-dominated members of each cluster are equally weighted for
    selection in order to fully explore the compositional space.

    Contains functions to initialize and assign clusters to models, rank
    models within their own clusters as well as assign an overall rank to
    each model, choose a model to replace, add a model to the population,
    and produce a model for mating.

    The non-domination rankings are contained within a "level structure",
    which is a list of lists. The list at index 0 contains all rank 0 models,
    the list at index 1 contains all rank 1 models, etc. 
    """

    def __init__(self, capacity, dominance=ParetoDominance(),
                 weights=[1, 1, 1, 1, 1], comparator=None, cluster_obj=None):
        """
        Args:

        capacity (int): the number of models which are held in the
        population when it has reached the steady-state level.

        dominance (Dominance object): the dominance class instance
        which computes all pareto quantities, including ranking the models.

        weights (list of floats): the weights of each objective function.
        Only used for linear selection.

        comparator (Comparator object): the comparator class instance
        which handles all model similarity checks.

        cluster_obj (Clustering object): the cluster class instance
        which handles the clustering of all models. NOTE: currently only
        compositional clustering is supported for use with this selection
        algorithm.
        """
        self.capacity = capacity
        self.models = []
        self.model_level_structure = []
        self.non_dominated_models = []
        self.cluster_models = {}
        self.multi_model_clusters = []
        self._dominance = dominance
        self.size = 0
        self.non_dominated_size = 0
        self.cluster_models_hierarchies = {}

        # Weights for linear addition to population
        self.weights = weights
        while len(self.weights) < 5:
            self.weights.append(1)

        # "Good pool" for linear portion of multi-objective search
        self.good_pool = []
        self.comparator = comparator

        # cluster_obj for clustering models
        self.cluster_obj = cluster_obj

        self.operator_inheritance = []

    def extend(self, model):
        '''
        Add a model to the population.

        Args:

        model (obj): the structure_record.model() which is being added.
        '''
        self.models.append(model)
        self.size += 1

    def check_uniqueness(self, model, exact=True):
        '''
        Check whether a model is unique.

        Returns True if the model is unique, returns False if the model
        is the same (or "similar" if exact is False) as another model.

        Args:

        model (obj): the structure_record.model() for which uniqueness
        is being tested.

        exact (boolean): if True, models are considered unique if they
        are not exactly the same as another model. If False, models are
        considered unique if they are not the same as another model
        within tolerance limits.
        '''
        if self.comparator is None:
            # No comparator, so automatic return True
            return True
        else:
            flags = [self.comparator.compare_models(
                model, m) for m in self.models]
            if exact:
                same = [f == 0 for f in flags]
            else:
                same = [f >= 0 for f in flags]
            if any(same):
                return False
            else:
                return True

    def init_clustering_and_ranking(self):
        '''
        Initialize the cluster object by seeding it with the population
        models currently present. Also ranks the models, both on
        the population level, as well as on the individual cluster level.
        '''
        self.cluster_models, self.multi_model_clusters = \
            self.cluster_obj.initialize_clusters(self.models)

        self._dominance.rank_models(
            self.models, self.model_level_structure, "population")
        for cluster in self.cluster_models.keys():
            cluster_model_levels = []
            self._dominance.rank_models(
                self.cluster_models[cluster], cluster_model_levels, "cluster")
            self.cluster_models_hierarchies[cluster] = cluster_model_levels
            self.non_dominated_models.extend(cluster_model_levels[0])

        self.non_dominated_size = len(self.non_dominated_models)

    def basic_addition_to_population(self, model, select, sim_ids=None):
        '''
        Linear addition based on proximity to lowest possible values in
        all objective functions. Only used when the population has
        not yet reached its steady state capacity.

        Args:

        model (obj): the structure_record.model() being added to the
        population.

        select (obj): the select instance which is being used to
        assign selection probabilities to the models.

        sim_ids (list of ints): indices which specifies how many exp sim
        objective functions are used.
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
            cutoff_value = \
                select.update_probs_single_obj(self.models,
                                               self.capacity,
                                               update_cutoff_only=True)
            if cutoff_value >= model.obj0_val:
                to_good_pool = True
            else:
                to_good_pool = False

            if to_good_pool is True:
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
                    print(f'New Model {model.label} added to good_pool'
                          + f'and Model {demoted_label} demoted from'
                          + 'good_pool')
                else:
                    print(f'New Model {model.label} added to good_pool')

                # scale the good_pool_values using MinMaxScaler
                good_pool_values = good_pool_values.reshape(-1, 1)
                scaler = MinMaxScaler()
                scaled_values = scaler.fit_transform(good_pool_values)[:, 0]

                # optimize k for every 100th model (labels are continuous)
                if model.label % select.adjust_k_every == 0:
                    initial_k = select.optimum_k
                    # optimize k
                    res = minimize(Select._optimize_exponential_constant,
                                   initial_k,
                                   args=(select.num_required_above_50,
                                         scaled_values),
                                   method='Nelder-Mead',
                                   options={'maxiter': 100})
                    # Store new optimum k
                    select.optimum_k = res.x[0]

                opt_k = select.optimum_k
                # Update probabilities by min max
                # exponential function using opt_k
                exponential_probs = [(math.exp(opt_k * i) - math.exp(opt_k)) /
                                     (1 - math.exp(opt_k))
                                     for i in scaled_values]
                # Assign probabilities to the models
                for m, prob in zip(self.good_pool, exponential_probs):
                    m.selection_prob = prob

                return select

            if to_good_pool is False:
                print(f'New Model {model.label} not added to good_pool')

                return select

    def choose_worst_model(self, exclude_model=None,
                           use_cumulative_rank=False):
        '''
        Chooses which model will be replaced by the new model. Uses the
        overall non-domination level structure, not the individual cluster
        level structures, to make its decision. Candidate models for
        removal are the models in the highest tier of the overall level
        structure.

        Arguments:

        exclude_model (obj): if included, it is checked to make sure
        that this structure_record.model() is not chosen. Necessary when the
        new population model is added to the highest tier of the level
        structure.

        use_cumulative_rank (bool): whether to use basic non-dominated rank,
        or to use the sum of the basic rank and the cluster rank. This can
        be helpful when diversity in the clusters present is highly desired,
        as removed models will always come from the clusters with the most
        members, so long as that cluster is present in the highest
        non-domination tier.
        '''
        candidates = self.model_level_structure[-1]
        exclude_index = None
        if exclude_model is not None:
            if exclude_model in candidates:
                exclude_index = candidates.index(exclude_model)
                print(
                    "Model was added to back of level structure! "
                    + f"Index: {exclude_index}")
        if len(candidates) == 1:
            return self.model_level_structure.pop(-1)[0]
        if not use_cumulative_rank:
            index_options = np.arange(len(candidates))
            chosen_index = np.random.choice(index_options)
            while (exclude_index is not None) \
                    and (chosen_index == exclude_index):
                chosen_index = np.random.choice(index_options)
            print(f"Chosen index: {chosen_index}")
            return self.model_level_structure[-1].pop(chosen_index)
        else:
            worst_candidates = [candidates[0]]
            index_options = [0]
            worst_rank = candidates[0].rank + candidates[0].cluster_rank
            for index, candidate in enumerate(candidates[1:]):
                rank = candidate[0].rank + candidates[0].cluster_rank
                if rank > worst_rank:
                    worst_candidates = [candidate]
                    index_options = [index+1]
                    worst_rank = rank
                elif rank == worst_rank:
                    worst_candidates.append(candidate)
                    index_options.append(index + 1)
            chosen_index = np.random.choice(index_options)
            while (exclude_index is not None) \
                    and (chosen_index == exclude_index):
                chosen_index = np.random.choice(index_options)
            print(f"Chosen index: {chosen_index}")
            return self.model_level_structure[-1].pop(chosen_index)

    def add_to_population(self, model):
        '''
        Attempt to add a new model to the population.
        If the model is dominated by a population member in the
        highest non-domination tier, or is too similar to another
        population member, then reject its addition.

        Otherwise, it is accepted, and replaces a current population
        member. The population member which is replaced is chosen
        using the choose_worst_model function.

        Upon addition (/removal of the current population member),
        all non-domination ranks are re-evaluated. When possible,
        the update_model_levels function of the dominance calculator
        is used in order to preserve computational efficiency.

        Returns bool which indicates whether the model was added (True)
        or not (False)

        Args:

        model (obj): structure_record.model() which is being tested for
        addition.
        '''
        dominated = False
        too_similar = False
        # compare to models in last tier to make sure model would not be
        # completely dominated if added
        for m in self.model_level_structure[-1]:
            flag = self._dominance.compare(model, m)
            if flag == 1:
                dominated = True
                break
            elif flag == 2:
                too_similar = True
                break

        if not dominated and not too_similar:
            # First attempt to add new model. This will trigger any similarity
            # comparisons in at most O(N) time if the model is too similar to
            # any models currently in the model_level_structure.
            model_unique = self._dominance.update_model_levels(
                self.model_level_structure, model, "population")

            if model_unique:
                # Remove worst model
                worst_model = self.choose_worst_model(exclude_model=model)
                self.models.remove(worst_model)
                self.cluster_models, self.multi_model_clusters, update_levels \
                    = self.cluster_obj.remove_model(
                        worst_model, self.cluster_models_hierarchies)

                if update_levels:
                    cluster_model_levels = []
                    self._dominance.rank_models(
                        self.cluster_models[worst_model.cluster],
                        cluster_model_levels,
                        "cluster")
                    self.cluster_models_hierarchies[worst_model.cluster] \
                        = cluster_model_levels

                # Add new model
                self.models.append(model)
                self.cluster_models, self.multi_model_clusters \
                    = self.cluster_obj.append_model(
                        model)

                if len(self.cluster_models[model.cluster]) != 1:
                    self._dominance.update_model_levels(
                        self.cluster_models_hierarchies[model.cluster],
                        model,
                        "cluster")
                else:
                    self.cluster_models_hierarchies[model.cluster] = [
                        [model]]
                    model.cluster_rank = 0
                    model.selection_prob = np.exp(-model.cluster_rank)

                # update cluster non-dominated models
                self.non_dominated_models = []
                for hierarchy in self.cluster_models_hierarchies.values():
                    self.non_dominated_models.extend(hierarchy[0])
                self.non_dominated_size = len(self.non_dominated_models)

                # update operator inheritance frequency
                self.operator_inheritance = [
                    model.made_by for model in self.non_dominated_models]

                return True
            else:
                print(
                    f"Model {model.label} not added to population "
                    + "because not unique!")
                return False

        # If it is completely dominated,
        # then do not add to the population at all
        else:
            print(
                f"Model {model.label} not added to population because "
                + "dominated by pop member "
                + "(and does not dominate a pop member).")
            return False

    def produce_model(self, non_dominated, cluster=None, same_cluster=False):
        '''
        Produce a model for breeding. Choose two models, and return the
        non-dominated model. If both are non-dominated, then return one
        randomly.

        Arguments:
        non_dominated (bool): Whether to choose a model from the
        non-dominated models or the entire population.

        cluster (int): cluster to which the first parent model belonged.
        Necessary if requiring that the model comes from either the same
        cluster or a different cluster.

        same (bool): whether the model needs to belong to the same
        cluster (True) as the first parent model, or a different
        cluster (False).
        '''
        if non_dominated:
            return np.random.choice(self.non_dominated_models)
        else:
            # Refer to cluster dictionary to get models
            if same_cluster:
                models = self.cluster_models[cluster]
                chosen_model = np.random.choice(models)
                # determine if model will be selected or not
                r = np.random.uniform()
                if r < chosen_model.selection_prob:
                    return chosen_model
            else:
                # choose random key from clusters
                chose_cluster = False
                while not chose_cluster:
                    # print(list(self.cluster_models.keys()))
                    random_cluster = np.random.choice(
                        list(self.cluster_models.keys()))
                    if random_cluster != cluster:
                        chose_cluster = True
                # Next choose random model from cluster
                models = self.cluster_models[random_cluster]
                chosen_model = np.random.choice(models)
                # Determine if model will be selected or not
                r = np.random.uniform()
                if r < chosen_model.selection_prob:
                    return chosen_model
