from __future__ import division, unicode_literals, print_function

from numpy.random.mtrand import random_sample

"""
This module contains functions to conduct multi-objective search. 
It contains a pool of structures which contains "Population" and 
"Archive" sub-groups. The "Population" group is the primary breeding
pool for genetic operations. The "Archive" group is an elite population
which ensures that the structure search is always conducting genetic
operations with at least one structure on the Pareto front. The
algorithm is steady-state, adding one child structure at a time. 

The primary multi-objective search algorithm is epsilon-MOEA. 
[Citation: ]
However, until the "Population" sub-group has reached the steady-state
capacity, the "Archive" remains uninitialized, and a linear
selection protocol is used instead. In this protocol, every child
structure is added to the "Population" and is assigned a selection
probability which corresponds to its distance to the minimums of each
objective function. Parents for genetic operations are then chosen 
randomly. 

Once the "Population" has reached steady-state capacity, full epsilon-MOEA
is employed. For a thorough explanation, see the citation above. 
"""
import numpy as np
import random
import math
import itertools
from sklearn.preprocessing import MinMaxScaler


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

        Here the "Population" and "Archive" objects are also initialized.
        The "Population" uses ParetoDominance, and the "Archive" uses
        EpsilonDominance.
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

        # Define epsilons for the archive
        if 'epsilons' not in pool_params:
            self.epsilons = [.1, .1]
        else:
            self.epsilons = pool_params['epsilons']
            while len(self.epsilons) != 6:
                self.epsilons.append(.1)

        self.population = Population(
            self.capacity, ParetoDominance())
        self.archive = Archive(EpsilonDominance(self.epsilons))

    def add_to_pool(self, model):
        """
        Attempt to add the model to both the population and the archive.

        model: model object to be added

        sim_ids (list of integers): simulation ids Eg: [1] for one Xsim
        """
        no_prior_epsilon = True
        # If population size is less than 10, add any models created
        if self.population.size < 10:
            print('New Model {} added to population'.format(model.label))
            self.population.extend(model)

        # If population size less than capacity,  use linear method instead
        # of epsilon-MOEA
        elif 10 <= self.population.size < self.capacity:
            self.population.basic_addition_to_population(model)
            print('New model {} added to population based'
                  ' on their values only!'.format(model.label))

        # Othewise, perform usual epsilon-MOEA
        else:
            no_prior_epsilon = False
            self.population.add_to_population(model)
            self.archive.add_to_archive(model)

        # If size has now reached capacity, then add models to archive
        # in preparation for epsilon-MOEA
        if self.population.size == self.population.capacity and no_prior_epsilon:
            non_dominated_models = ParetoDominance().get_nondominated_solutions(self.population)
            self.archive.initialize_models(non_dominated_models)

    def provide_parent_models(self, num_parents):
        '''
        Provide parent models for mating operations. If archive has not been
        created, then provide both parents from the population. Otherwise,
        provide one parent from the population, and one parent from the archive.
        '''
        if self.archive.size == 0:
            return self.population.produce_linear_models(num_parents)
        else:
            population_member = self.population.produce_model()
            archive_member = self.archive.produce_model()
            return [population_member, archive_member]

    def return_random_pop_member(self):
        return self.population.return_single_linear_model()


class Population(object):
    """
    The entire population of models. The archive is a subset of the population.

    Maintain two lists:
    good_pool - has limited capacity
              - used for selection
    bad_pool - unlimited capacity (all remaining models)

    NOTE: The best and worst models are chosen based on the model attribute
    "overall_value"
    """

    def __init__(self, capacity, dominance=ParetoDominance(), weights=[1, 1, 1, 1, 1]):
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
        self.good_pool_size = 0

    def extend(self, model):
        '''
        Add a model to the population
        '''
        self.models.append(model)
        self.size += 1

    def basic_addition_to_population(self, model):
        '''
        Linear addition based on proximity to lowest possible
        values in all objective functions
        '''
        self.extend(model)

        # Store all models objective function values separately
        # TODO: Make this compatible with more than 2 objectives
        model_labels = []
        all_v0, all_v1 = [], []
        for model in self.models:
            model_labels.append(model.label)
            all_v0.append(model.obj0_val)
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

        # Update selection probabilities using a linear model
        linear_vals = np.array([sum(weighted_norm_vals[i])
                                for i in range(len(weighted_norm_vals))])
        linear_probs = (linear_vals - max(linear_vals)) / \
            (min(linear_vals) - max(linear_vals))
        selection_probs = linear_probs

        # Assign probabilities to the models
        for model, prob in zip(self.models, selection_probs):
            model.selection_prob = prob

    def add_to_population(self, model):
        '''
        Test the addition of the model to the population

        Criteria:
        If model dominates a population member, replace
        If model is dominated by a population member, reject
        If neither, then replace a population member at random

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
            del self.models[random.choice(dominates)]
            self.models.append(model)
        # If does not dominate, but is not dominated, then replace any one
        # population member at random
        elif not dominated:
            del self.models[random.randint(0, self.size - 1)]
            self.models.append(model)

    def produce_model(self):
        '''
        Produce a model for breeding. Choose two models, and return the 
        non-dominated model. If both are non-dominated, then return one 
        randomly. 
        '''

        [model_one, model_two] = random.sample(self.models, 2)
        return self._dominance.choose_non_dominated(model_one, model_two)

    def produce_linear_models(self, num_parents):
        '''
        If archive has not been created, then produce two models for breeding
        based on proximity to lowest possible objective values
        '''
        parents = []
        while len(parents) < num_parents:
            done = False
            new_parent = None
            while not done:
                # randomly choose a parent
                parent = random.choice(self.models)
                if parent.selection_prob:
                    if random.random() < parent.selection_prob:
                        done = True
                        new_parent = parent
            if len(parents) == 0:
                parents.append(new_parent)
            for existing_parent in parents:
                if existing_parent.label == new_parent.label:
                    continue
                else:
                    parents.append(new_parent)
        return parents

    def produce_single_linear_model(self):
        done = False
        while not done:
            # randomly choose a parent
            parent = random.choice(self.models)
            if parent.selection_prob:
                if random.random() < parent.selection_prob:
                    done = True
                    return parent



class Archive(object):
    '''
    An archive containing only the elite non-dominated models
    '''

    def __init__(self, dominance=EpsilonDominance([1, 1])):
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
        If model dominates a population member, replace
        If model is dominated by a population member, reject
        If neither, then replace a population member at random

        Args:
        model: model object to be added
        sim_ids (list of integers): simulation ids Eg: [1] for one Xsim
        '''

        flags = [self._dominance.compare(model, m) for m in self.models]
        nondominated = [x == 0 for x in flags]
        dominated = [x > 0 for x in flags]

        if any(dominated):
            return False
        else:
            self.models = list(itertools.compress(
                self.models, nondominated)) + [model]
            self.size += 1
            return True

    def produce_model(self):
        '''
        Returns randomly selected model
        '''
        return random.choice(self.models)


class ParetoDominance(object):
    def get_nondominated_solutions(self, population=Population(100)):
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
                ref_val = math.floor(test_model.obj1_val / epsilon)

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
