"""
This module contains funcitons to update the pool, evaluate pareto front in
case of multi-objective optimization and assigns/updates selection probability
of models
"""

from __future__ import division, unicode_literals, print_function
import numpy as np
import random, copy
from math import sqrt, exp
# import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from scipy.optimize import minimize
from scipy.spatial import ConvexHull  # , convex_hull_plot_2d

from dscribe.kernels import REMatchKernel
from dscribe.descriptors import SOAP
from ase.ga.ofp_comparator import OFPComparator
from pymatgen.io.ase import AseAtomsAdaptor

from fx19 import distance_check as dc


class Pool(object):
    """
    A pool of structures that are evaluated. Parents will be selected from this
    pool.

    Maintain two lists:
    good_pool - has limited capacity
              - used for selection

    all_models - incldues a list of all the models evaluated thus far

    NOTE: The best and worst models are chosen based on the model attribute
    "overall_value"
    """

    def __init__(self, pool_params):
        """
        The capacity of the pool is taken from pool_params (i_dict)

        Args:

        pool_params (dict): Dictionary of parameters required to create Pool
                            object. Ex: {'capacity': 200}
        """
        energy_pkg = pool_params['energy_pkg']
        if 'capacity' not in pool_params:
            if energy_pkg == 'vasp':
                self.capacity = 50
            else:  # energy_code == 'lammps' or 'gulp':
                self.capacity = 500
        else:
            if 'capacity' in pool_params:
                self.capacity = pool_params['capacity']

        self.all_models = []
        self.good_pool = []

        if 'fingerprint_params' not in pool_params:
            # no fingerprint comparisons are going to be made
            self.comparator = None
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

    def add_to_pool(self, model, select, sim_ids=None):
        """
        This function adds the given model to the good_pool if
        pool capacity is not full. When full, replaces the worst model in
        good_pool if the new model is better.

        Part I
        ------------
        Get to_good_pool for the specific scenario.

            If multi-obj && no. of models < num_models_before_pareto -->
                update selection probs according to sum of normalized
                obj values

            If single-obj --> update_probs_single_obj

            If multi-obj && >num_models_before_pareto -->
                do select.add_new_model()
                # checks if model changes selection probs (if pareto optimal)

        PART II
        ------------
        if to_good_pool is True && no. of models > capacity -->
                demote worst model
                (NOTE: For single-obj, overall_val and obj0_val are same)
                scale overall_vals for models in good_pool
                optimize exponent "k" every 100th model
                get exponential probs (from scaled overall vals)
                update selection probs

        if to_good_pool is False:
                Do not add to good_pool && do nothing

        if to_good_pool is None:
                (NOTE: For single-obj search, to_good_pool would not be None)
                Model is pareto optimal
                Update all selection probs

        Args:

        model (obj): structure_record.model() object to be added

        select (obj): Select object

        sim_ids (list of integers): simulation ids. Eg: [1] for one Xsim
        """
        # If global fingerprint comparison is going to be made, calculate
        # fingerprint for model
        if self.comparator is not None:
            self.comparator.create_fingerprint(model)
        # Add model to all_models
        self.all_models.append(model)

        if len(self.all_models) < 10:
            print('New Model {} made by {} added to good pool'.format(
                                                model.label, model.made_by))
            self.good_pool = self.all_models

            return select

        # If multi-objective search, use linear method before dist_from_pareto
        # call update selection probs which uses linear method before pareto
        if select.type == 'multi':
            if len(self.all_models) <= select.num_models_before_pareto:
                self.good_pool = select.update_all_selection_probs(
                    self.all_models,
                    self.capacity,
                    sim_ids=sim_ids)
                print('New model {} made by {} added: probs updated'
                      ' based on sum of normalized obj. values!'.format(
                                                model.label, model.made_by))

                return select

        if select.type == 'single':
            # Get updated cutoff value to skip probs for a bad model
            cutoff_value = select.update_probs_single_obj(
                self.all_models, self.capacity,
                update_cutoff_only=True)
            if cutoff_value >= model.obj0_val:
                to_good_pool = True
            else:
                to_good_pool = False
        else:
            # check if model changes existing selection probs
            to_good_pool, model = select.add_new_model(model, sim_ids=sim_ids)

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
                print('New Model {} made by {} added to good pool '
                      'and Model {} demoted from good_pool'.format(
                                model.label, model.made_by, demoted_label))
            else:
                print('New Model {} made by {} added to good pool'.format(
                                            model.label, model.made_by))

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
            # Update probabilities by min max exponential function using opt_k
            exponential_probs = [(exp(opt_k * i) - exp(opt_k)) /
                                 (1 - exp(opt_k)) for i in scaled_values]
            # Assign probabilities to the models
            for m, prob in zip(self.good_pool, exponential_probs):
                m.selection_prob = prob

            return select

        if to_good_pool is False:
            print('New Model {} made by {} not added to good pool'.format(
                                                model.label, model.made_by))
            return select

        if to_good_pool is None:
            # model is pareto optimal
            self.good_pool = select.update_all_selection_probs(self.all_models,
                                                               self.capacity,
                                                               sim_ids=sim_ids)
            if len(self.good_pool) == 0 or select.type == 'single':
                # if update fails due to too few points for convex hull
                print('New Model {} made by {} added to good pool'.format(
                                                model.label, model.made_by))
                self.good_pool = self.all_models
            else:
                print('New Model {} made by {} is pareto efficient!'.format(
                                                model.label, model.made_by))
                if select.operator_assignment == "auto-adaptive":
                    # update operator probabilities in select
                    # Formula:
                    # P_i = (C_i + epsilon)/Sum_j=1->N_operators(C_j+epsilon)
                    # Here epsilon = 1
                    operator_counts = np.zeros(
                        len(select.operator_hashmap))
                    for operator in select.operator_inheritance:
                        if operator != "random":
                            operator_counts[
                                select.operator_hashmap[operator]
                            ] += 1
                        else:
                            operator_counts += 1 / \
                                len(select.operator_hashmap)
                    divisor = len(select.pareto_labels[-1]) + \
                        len(select.operator_hashmap)
                    select.operator_frequencies = [
                        (count + 1)/divisor for count in operator_counts]
                    print(
                        f"Operator frequencies: {select.operator_frequencies}")

            return select


class Select(object):
    """
    Uses the evaluated attributes of a model to assign selection probabilities.
    Distance from the pareto front is used to evaluate selection probability of
    a model. When a model is pareto optimal, the selection probabilities gets
    updated for all models in good_pool.

    If single objective - all the weights would be zero and the obj0_val will
    be overall_val.
    """

    def __init__(self, select_obj_params):
        """
        The dictionary of parameters to make a Select object should be
        provided.

        Eg: select_obj_params = {'objective_fn_type': 'multi'
                                 'num_required_above_50': 30
                                 'num_models_before_pareto': 80
                                 'adjust_k_every': 100}

        Some functionality is not included if not listed in the params
        dictionary. Namely, auto-adaptive operator selection (where the
        relative frequency of each operator in the evolutionary process
        will be updated based on the operators which were used to create
        the pareto-optimal solutions) can be used if listed in the
        dictionary, but will not be used otherwise.
        """
        # 'single' or 'multi'
        self.type = select_obj_params['objective_fn_type']
        # set defaults
        self.num_required_above_50 = 100  # default
        self.num_models_before_pareto = 200  # default
        def_weights = [1, 1, 1, 1, 1]  # [w0, w1, w2, w3, w4]

        if 'weights' not in select_obj_params:
            self.weights = def_weights
        else:
            self.weights = select_obj_params['weights']

        if 'num_required_above_50' in select_obj_params:
            self.num_required_above_50 = \
                select_obj_params['num_required_above_50']

        # number of models to do linear probs before switching to pareto
        if 'num_models_before_pareto' in select_obj_params:
            self.num_models_before_pareto = \
                select_obj_params['num_models_before_pareto']

        # Making sure that dummy weights are in place, since needed by obj fn
        if not len(self.weights) == 5:
            for i in range(5 - len(self.weights)):
                self.weights.append(1)

        # Store pareto points & convex hull points here
        self.pareto_points = None
        self.hull_points = None
        self.cutoff_value = None

        # Store min, max of obj0 & obj1
        self.minmax_obj0 = None
        self.minmax_obj1 = None

        # store optimized k; gets updated every 100th model
        self.optimum_k = -1  # default
        self.adjust_k_every = 100  # default
        if 'adjust_k_every' in select_obj_params:
            self.adjust_k_every = select_obj_params['adjust_k_every']

        # store the parent ids to control number of times a model can be parent
        self.all_parent_labels = []

        # store all sets of pareto points labels
        self.pareto_labels = []

        # Store operator information for mating operations
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

    def add_new_model(self, model, sim_ids=None):
        """
        For a multi-obj search, calculated the cutoff value and returns the
        to_good_pool based on which the new model is added to Pool.

        1. Get weighted normalized x, y for the new_model

        2. if point is on pareto front -> to_good_pool is None

        3. if point not on pareto front
                - get distance_from_hull
                - if distance_from_hull > cutoff_value
                        > to_good_pool is False
                - if distance from hull <= cutoff value
                        > to_good_pool is True (i.e., add model to good_pool)

        Args:

        model (obj): structure_record.model() object

        sim_ids (list of integers): simulation ids. Eg: [1] for one Xsim
        """
        # check if pareto_points or other class attributes exist
        if self.pareto_points is None or self.hull_points is None:
            return None, model

        model_obj0 = model.obj0_val
        if sim_ids and 1 in sim_ids:
            model_obj1 = model.obj1_val
        # For single-obj search, this function is not called at all
        # else:
        #    print('Single objective function optimization.'
        #          'TODO: Follow different routine..')
        #    return 0, model

        # normalize
        model_obj0 = (model_obj0 - self.minmax_obj0[0]) / \
            (self.minmax_obj0[1] - self.minmax_obj0[0])
        # divide by weights (because obj vals are minimized)
        model_obj0 = model_obj0 / self.weights[0]

        model_obj1 = (model_obj1 - self.minmax_obj1[0]) / \
            (self.minmax_obj1[1] - self.minmax_obj1[0])
        model_obj1 = model_obj1 / self.weights[1]

        if self.is_point_on_pareto((model_obj0, model_obj1)):
            return None, model

        # Assuming 2D pareto front from here
        # Get maximum x & maximum y hull points
        [Px, Py] = self.hull_points[np.argmax(self.hull_points, axis=0)]

        # slope of line between Px, Py
        m_pxpy = (Px[1] - Py[1]) / (Px[0] - Py[0])

        dist_from_hull = self.get_dist_from_hull(
            (model_obj0, model_obj1), m_pxpy)

        # set model's overall value
        model.overall_val = dist_from_hull

        if dist_from_hull > self.cutoff_value:
            to_good_pool = False
            return to_good_pool, model

        to_good_pool = True
        return to_good_pool, model

    def update_probs_single_obj(self, all_models, good_pool_capacity,
                                update_cutoff_only=False):
        """
        For a search with only one objective function, updates selection
        probabliites based on an exponential function.

        Args:

        all_models: (list) of all models evaluated so far

        good_pool_capacity: (int) maximum number of models in good pool

        update_cutoff_only: (bool) Returns only cutoff value when True
        """
        # Get all models obj0_val
        model_labels, all_v0 = [], []
        for model in all_models:
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
        good_pool = [m for m in all_models if m.label in good_pool_labels]
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
            exponential_probs = [(exp(opt_k * i) - exp(opt_k)) /
                                 (1 - exp(opt_k)) for i in good_pool_vals]
        else:
            opt_k = initial_k[0]
            # Get probabilities by simple exponential function
            # using opt_k = -1
            exponential_probs = [exp(opt_k * i) for i in good_pool_vals]
        selection_probs = exponential_probs

        # Assign probabilities to the models
        for model, prob in zip(good_pool, selection_probs):
            model.selection_prob = prob

        return good_pool

    def update_all_selection_probs(self, all_models,
                                   good_pool_capacity, sim_ids=None):
        """
        For a single-obj search, calls update_probs_single_obj() method.

        For a multi-obj search, calculates the overall value of each model,
        which is the distance from the pareto front. Updates selection
        probabilities based on an exponential distribution by optimizing a
        constant such that to maintain required number of models with
        probability above 50%.

        Args:

        all_models: (list) of all models evaluated so far

        good_pool_capacity: (int) maximum number of models in good pool

        sim_ids: (list of ints) indices which specifies how many exp sim
                                objective functions are used.

        Steps:
        --------------------

        1. Get obj0_vals, obj1_vals of all models

        2. normalize obj0_vals and obj1_vals separately

        3. assigns probabilities directly based on sum of its obj vals if total
        models less than 1000 models. This samples the PE landscape evenly than
        dist from pareto for smaller population at initial stages.

        4. If number of models is less than 1000, skip steps 5 - 13

        5. make a 2D pareto plot

        6. get pareto optimal points

        7. make convex hull; get pareto points that are on convex hull

        8. Get Px, Py - max & min points on convex hull

        9. Find slope of line perpendicular to PxPy line

        10. Find distance of each point from the convex hull along this
        perpendicular line

        11. Find cutoff distance from hull value to separate good_pool models

        12. Normalize the values (distances from hull) of all good_pool models

        13. Assign probabilities based on exp(kX). Default k = -1. However,
        optimize "k" once in "adjust_k_every" steps to get required number of
        models with probability greater than 0.5.

        14. Assign probabilities to models in good pool

        """
        # NOTE: Currently only supports pareto distance in 2D (with 2 objective
        # functions). So, dist_from_pareto is distance from a line in 2D. It
        # becomes distance from a plane in 3D and so on.. as objective function
        # dimensions increase.

        # Skip whole process if less points,
        if len(all_models) < 10:
            return []

        # Call a function to get probs based only on obj0_val (single obj)
        if self.type == 'single':
            return self.update_probs_single_obj(all_models, good_pool_capacity)

        # Store all models objective function values separately
        model_labels = []
        all_v0, all_v1 = [], []
        for model in all_models:
            model_labels.append(model.label)
            all_v0.append(model.obj0_val)
            if sim_ids and 1 in sim_ids:
                all_v1.append(model.obj1_val)
            # TODO: Other objective function values should be added here

        self.minmax_obj0 = min(all_v0), max(all_v0)
        # Make an array of objective function values and transpose
        vals = np.array([all_v0])
        if len(vals[0]) == len(all_v1):
            self.minmax_obj1 = min(all_v1), max(all_v1)
            vals = np.array([all_v0, all_v1])  # n_obj_fns x n_models
        vals = vals.T  # n_models x n_obj_fns

        # normalize using MinMaxScaler
        scaler = MinMaxScaler()
        norm_vals = scaler.fit_transform(vals)
        # Maintain weights based on number of objective functions
        weights = self.weights[:len(vals[0])]
        weights = np.array([weights])
        # weighted normalized values
        weighted_norm_vals = norm_vals/weights

        if len(all_models) > self.num_models_before_pareto:
            # Get indices of points (models) which are pareto efficient
            pareto_true_inds = Select._is_pareto_efficient(weighted_norm_vals)
            pareto_points_inds = [i for i, b in enumerate(pareto_true_inds)
                                  if b == True]
            pareto_points = [list(weighted_norm_vals[i]) for i in
                             pareto_points_inds]
            # store pareto optimal points as class attribute
            self.pareto_points = pareto_points
            # store pareto point labels to track how the pareto front changes
            pareto_labels = [model_labels[i] for i in pareto_points_inds]
            self.pareto_labels.append(pareto_labels)

            # Add origin in the beginning to get convex hull
            # visible from origin
            pareto_points = [[0 for i in range(len(pareto_points[0]))]] + \
                pareto_points
            # Sort by the first objective function
            pareto_points.sort()
            pareto_points = np.array(pareto_points)

            if self.operator_assignment == "auto-adaptive":
                pareto_models = [all_models[i] for i in pareto_points_inds]
                # update operator inheritance based on pareto points
                self.operator_inheritance = [
                    model.made_by for model in pareto_models]

            try:
                # Make convex hull with pareto points
                hull = ConvexHull(pareto_points, qhull_options='QG0')
            except:
                return []
            # Get visible facets from (0, 0). Each facet is bounded by 2 points
            visible_facets = []
            hull_points = []
            for facet in hull.simplices[hull.good]:
                vis_facets = hull.points[facet]
                visible_facets.append(vis_facets)
                if tuple(vis_facets[0]) not in hull_points:
                    hull_points.append(tuple(vis_facets[0]))
                if tuple(vis_facets[1]) not in hull_points:
                    hull_points.append(tuple(vis_facets[1]))

            hull_points = np.array(hull_points)
            self.hull_points = hull_points
            self.visible_facets = visible_facets

            # Assuming 2D pareto front from here
            # Get maximum x & maximum y hull points
            [Px, Py] = hull_points[np.argmax(hull_points, axis=0)]

            # slope of line between Px, Py
            m_pxpy = (Px[1] - Py[1]) / (Px[0] - Py[0])

            distances_from_hull = []
            for data_of_model in weighted_norm_vals:
                min_dist = Select.get_dist_from_hull(
                    self, data_of_model, m_pxpy)
                distances_from_hull.append(min_dist)

            # Get the cutoff value (distance_from_hull) for good_pool
            # TODO: Use a better way to get the cutoff_value
            # Using partition rather than sort is preferred here
            if len(all_models) <= good_pool_capacity:
                cutoff_value = max(distances_from_hull)
            else:
                # copy_vals = distances_from_hull.copy()
                # copy_vals.sort()
                # cutoff_value = copy_vals[good_pool_capacity]
                hull_distance_array = np.array(distances_from_hull)
                hull_distance_array.partition(good_pool_capacity)
                cutoff_value = hull_distance_array[good_pool_capacity]
            self.cutoff_value = cutoff_value

            # Make good_pool with models with less than cutoff value
            # Clear both pools
            good_pool = []
            good_pool_values = []
            for model, value in zip(all_models, distances_from_hull):
                model.overall_val = value
                if value <= cutoff_value:
                    good_pool.append(model)
                    good_pool_values.append(value)

            # Calculate probabilities to models in good_pool; 0 for others
            good_pool_values = np.array(good_pool_values).reshape(-1, 1)
            # scale the good_pool_values using MinMaxScaler
            scaled_good_pool_values = \
                scaler.fit_transform(good_pool_values)[:, 0]

            # optimize contant (k) in the exponential function e^(-kx)
            # such that required number of models have probability
            # greater than 0.5
            initial_k = [-1]
            if len(good_pool) > self.adjust_k_every and \
                    len(good_pool) > self.num_required_above_50:
                res = minimize(Select._optimize_exponential_constant,
                               initial_k,
                               args=(self.num_required_above_50,
                                     scaled_good_pool_values),
                               method='Nelder-Mead',
                               options={'maxiter': 100})
                opt_k = res.x[0]
                self.optimum_k = opt_k
                # Get probabilities by min max exponential function
                # using the opt_k
                exponential_probs = [(exp(opt_k * i) - exp(opt_k)) /
                                     (1 - exp(opt_k)) for i in
                                     scaled_good_pool_values]
            else:
                opt_k = initial_k[0]
                # Get probabilities by simple exponential function
                # using opt_k = -1
                exponential_probs = [exp(opt_k * i) for i in
                                     scaled_good_pool_values]
            selection_probs = exponential_probs

        else:  # if len(all_models) <= 1000
            all_models_values = np.array([sum(weighted_norm_vals[i])
                                          for i in
                                          range(len(weighted_norm_vals))])

            # get cutoff for good_pool
            if len(all_models) <= good_pool_capacity:
                cutoff_value = max(all_models_values)
            else:
                copy_vals = all_models_values.copy()
                copy_vals.sort()
                cutoff_value = copy_vals[good_pool_capacity]
            self.cutoff_value = cutoff_value

            # get good pool
            good_pool, good_pool_values = [], []
            for model, value in zip(all_models, all_models_values):
                model.overall_val = value
                if value <= cutoff_value:
                    good_pool.append(model)
                    good_pool_values.append(value)

            # use linear method to provide selection probs
            linear_vals = good_pool_values
            linear_probs = (linear_vals - max(linear_vals)) / \
                (min(linear_vals) - max(linear_vals))
            selection_probs = linear_probs

        # Assign probabilities to the models
        for model, prob in zip(good_pool, selection_probs):
            model.selection_prob = prob

        return good_pool

    def is_point_on_pareto(self, new_point):
        """
        Returns True if a new point is below or on the pareto front
        Uses the known pareto front saved in self.pareto_points

        Args:

        new_point: (list/tuple) of a 2D point weighted normalized
                                            [obj0_val, obj1_val]
        """
        new_point = np.array(new_point)
        for pp in self.pareto_points:
            if not np.any(new_point < pp):
                return False
        return True

    def get_dist_from_hull(self, new_point, m_pxpy):
        """
        Returns distance of a point from the convex hull.

        Args:

        new_point: (list/tuple) of a 2D point weighted normalized
                                            [obj0_val, obj1_val]
        m_pxpy: (float) the slope of the line connecting the extrema of the
                                                              pareto front
        """

        # get equation of line perpendicular to PxPy & passes through model
        c_xy = new_point[1] - (-1/m_pxpy) * new_point[0]
        xy_line = (-1/m_pxpy, c_xy)

        # find distance of the model from each visible facet along this line
        distances = []
        for facet in self.visible_facets:
            # get equation (m, c) for a facet
            facet_line = Select._line_from_points(facet[0], facet[1])
            # Get point of intersection with facet_line
            x0y0 = Select._point_on_two_lines(facet_line, xy_line)
            distances.append(Select._dist_from_point(x0y0, new_point))
        return min(distances)

    @staticmethod
    def _is_pareto_efficient(costs):
        """
        Source:
        https://github.com/QUVA-Lab/artemis/blob/peter/artemis
        /general/pareto_efficiency.py

        Find the pareto-efficient points

        param costs: An (n_points, n_costs) array

        returns: A (n_points, ) boolean array, indicating whether each point is
                 Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Keep any point with a lower cost
                is_efficient[is_efficient] = np.any(
                    costs[is_efficient] < c, axis=1)
                is_efficient[i] = True  # And keep self
        return is_efficient

    @staticmethod
    def _optimize_exponential_constant(k, num_required_above_50,
                                       scaled_good_pool_values):
        """
        A scalar function to minimize the exponential constant to give required
        number of models with probability above 0.5 (or 50%).

        Args:

        k: (a list or an array) of the variable for minimize function
           (Eg: [-1])

        scaled_good_pool_values: (1D array or list) The objective function
                                 values of models in good pool scaled between 0
                                 and 1. For example, distances_from_hull for
                                 multiobjective optimization
        """
        # Get probs based on constant k
        exponential_probs = [exp(k[0]*i) for i in scaled_good_pool_values]
        num_above_50 = len([i for i in exponential_probs if i > 0.5])

        return (num_required_above_50 - num_above_50)**2

    @staticmethod
    def _dist_from_point(P1, P2):
        """
        Returns distance between two points

        Args:

        P: point one
        xy: point two
        """
        return sqrt((P1[0] - P2[0]) ** 2 + (P1[1] - P2[1]) ** 2)

    @staticmethod
    def _point_on_two_lines(line1, line2):
        """
        Returns the point of intersection of two lines

        Args:

        line1: list/tuple (m1, c1) for line y = m1x + c1
        line2: list/tuple (m2, c2) for line y = m2x + c2
        """
        m1, c1 = line1
        m2, c2 = line2

        # y coordinate in point of intersection of two lines
        y_c = (m1 * c2 - m2 * c1) / (m1 - m2)
        # substitute y_c in y = m1x + c to get x_c
        x_c = (y_c - c1) / m1

        return x_c, y_c

    @staticmethod
    def _line_from_points(p1, p2):
        """
        Get equation of line given two points

        Args:

        p1: tuple/list of point 1
        p2: tuple/list of point 2
        """
        # slope
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        c = p1[1] - m * p1[0]

        return m, c

    def get_parents(self, pool, num_parents, same_cluster=None,
                                        same_ab=False, abs_tol=0.2):
        """
        Selects requested number of parents based on their probabilities
        Returns a list of parents

        TODO: Add clustering to similar to epsilonSelection
        TODO: Add same_ab for surface geometry runs

        Args:

        pool - Pool() object
        num_parents - integer number of parents
        same_ab (bool) - If num_parents > 1, species whether all parents should
                         have same a, b lattice vectors
        abs_tol (float) - The maximum value for the sum of absolute difference
                         between the "ab" of two lattice vectors
        """
        parents = []
        while len(parents) < num_parents:
            new_parent = self.get_a_parent(pool)
            if len(parents) == 0:
                parents.append(new_parent)
            for existing_parent in parents:
                if existing_parent.label == new_parent.label:
                    continue
                ab_1 = existing_parent.astr.lattice.matrix[:2]
                ab_2 = new_parent.astr.lattice.matrix[:2]
                diff = np.array(ab_1) - np.array(ab_2)
                # return first match since keys are already shuffled
                if np.absolute(diff).sum() < abs_tol:
                    parents.append(new_parent)
        return parents

    def get_a_parent(self, pool):
        """
        Returns exactly one parent
        """
        done = False
        while not done:
            # randomly choose a parent
            parent = random.choice(pool.good_pool)
            if parent.selection_prob:
                if random.random() < parent.selection_prob:
                    if self.all_parent_labels.count(parent.label) < 200:
                        done = True
                        return parent


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

        test_model (obj): structure_record.model() A for the comparison

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
        except (AssertionError, KeyError):
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
            distance = sqrt(2 - 2*compare_fm)
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

    def check_uniqueness(self, model, all_models, exact=True):
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
        # create a new fingerprint got the model
        # either before (new_model) or after relaxation (relaxed_astr)
        self.create_fingerprint(model)
        # compare
        flags = [self.compare_models(model, m) for m in all_models]
        if exact:
            same = [f == 0 for f in flags]
        else:
            same = [f >= 0 for f in flags]
        if any(same):
            return False
        else:
            return True
