'''
This module contains all methods to calculate dominance relationships of
FANTASTX models. Currently, two dominance methods are implemented:

1. **Pareto dominance** (`ParetoDominance`): model A dominates model B if all
 of it's objective values are better than model B's (or vice-versa for model
 B dominating model A). It is non-dominated with B if at least one of its
 objective values is better than B's, but at least one of B's objective
 values is better than A's. 
2. **Epsilon dominance** (`EpsilonDominance`): same as Pareto dominance, but
 now the objective value space is discretized into $\epsilon$ grid boxes.
 Non-dominance is checked based on the corner (given by the floor or ceiling
 of each objective if minimizing or maximizing said objective) of the
 $\epsilon$ box that the model occupies. In the [original]()
 implementation, models were not allowed to occupy the same $\epsilon$ box,
 and the model that was kept was the model whose objective values were closest
 to the (possibly hyperdimensional) box corner. Here, the calculation of the
 distance to the box corner is kept, but models but only to determine
 dominance of models occupying the same box.

Both dominance methods have been further modified to facilitate
model structural comparisons. These structural comparisons are automatically
made if a `Comparator` object is provided to the dominance class. To enhance
efficiency, functionality has been included to only make structural
comparisons with other models whose objectives all lie within an $\epsilon$
box centered on the target model's objectives. This will always be the case
for `EpsilonDominance`, where these "structural" $\epsilon$ values will be
the same as those used for dominance calculations. For `ParetoDominance`,
these $\epsilon$ values need to be provided separately. If none are provided,
then structural comparisons are made with all other models.
'''

import numpy as np
import math
import itertools


class ParetoDominance(object):
    '''
    Base class for performing non-dominance calculations. Non-dominance is
    determined simply based on the relation between objective function
    values of two models. If all objective function values for model A
    are better (lower) than those of model B, then model A dominates
    model B. If at least one (but not all) model B objective function
    value is better than that of model A, then the models are
    non-dominated.

    It supports structural comparisons if a `Comparator` object is provided.
    These comparisons can either be made within a discretized region of
    objective space (an $\epsilon$ box just as in the $\epsilon$ dominance
    method), or with all other models. If no $\epsilon$ values are provided,
    then comparison is made with all other models by default.

    Beyond containing methods to perform non-dominance comparison between
    two models, the class also contains methods to:

    - Grab the sub-list of non-dominated models from a list of models
    - A method to rank a list of models by non-domination (in the form of
     a list of lists)
    - A method to efficiently update these non-domination rankings upon
     either model addition or deletion.
    '''

    def __init__(self, comparator=None, epsilons=None):
        '''
        Arguments:

            comparator (obj): instance of the comparator class which,
             if included, will perform all structural similarity checks

            epsilons (list): list with dimensionality that must match
             the number of objectives. E.g. for two objectives, would be:
              $[\epsilon_1, \epsilon_2]$. If provided, discretizes the
             objective space in order to limit structural comparisons. Only
             used if a comparator is also provided.
        '''
        # store comparator object and epsilons if they exist
        self.epsilons = None
        if comparator is not None:
            self.comparator = comparator
            self.make_similarity_checks = True
            self.epsilons = epsilons
        else:
            self.make_similarity_checks = False

    def get_nondominated_solutions(self, models):
        """
        Inspired by this [page](https://github.com/QUVA-Lab/artemis/
        blob/peter/artemis/general/pareto_efficiency.py)

        Returns the list of non-dominated models, and the list of the models
        which are dominated by at least one other model, based on comparison
        function flags.

        Arguments:

            models (list of objs): the list of `structure_record.model()`
             for which non-domination will be determined.
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

    def alt_nondominance(self, models):
        """
        Inspired by this [page](https://github.com/QUVA-Lab/artemis/
        blob/peter/artemis/general/pareto_efficiency.py)

        Return all non-dominated solutions (the Pareto front) from
        a set of models.

        Returns the list of non-dominated models, and the list of the models
        which are dominated by at least one other model.

        Arguments:

            models (list of objs): the list of `structure_record.model()`
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

    def compare(self, test_model, ref_model):
        '''
        Function which performs comparison between models. It returns:

        - -1 if test_model dominates the ref_model
        - 0 if both non-dominated
        - +1 if test_model dominated by the ref_model
        - +2 if the test_model is the same as the ref_model

        Similarity comparisons are only made for models which are inside
        the same $\epsilon$ box, which unlike $\epsilon$-domination, is
        centered on the reference model (to avoid missing comparisons between
        structures whose objective values lie at the edge of boxes). If no
        epsilons are provided, then the comparison is made for every model,
        assuming that a comparator object was provided in the first place.

        Arguments:

            test_model (obj): `structure_record.model()` A for the comparison

            ref_model (obj): `structure_record.model()` B for the comparison
        '''

        dominate_test = False
        dominate_ref = False

        outside_epsilon_box = False  # Determines if structural similarity
        # checks will be made
        for n in range(ref_model.num_of_obj):
            if n == 0:
                test_obj = test_model.obj0_val
                ref_obj = ref_model.obj0_val
            elif n == 1:
                test_obj = test_model.obj1_val
                ref_obj = ref_model.obj1_val
            elif n == 2:
                test_obj = test_model.obj2_val
                ref_obj = ref_model.obj2_val
            elif n == 3:
                test_obj = test_model.obj3_val
                ref_obj = ref_model.obj3_val
            elif n == 4:
                test_obj = test_model.obj4_val
                ref_obj = ref_model.obj4_val

            if self.make_similarity_checks:
                if self.epsilons is not None:
                    if abs(ref_obj - test_obj) > self.epsilons[n]/2:
                        outside_epsilon_box = True

                if n == ref_model.num_of_obj - 1:
                    if not outside_epsilon_box:
                        similarity = self.comparator.assess_models_similarity(
                            test_model, ref_model)
                        if similarity >= 0:
                            return 2

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

    def rank_models(self, models, model_level_structure, flag):
        '''
        Function which recursively ranks models according to
        non-domination.

        Updates model_level_structure, the variable corresponding
        to the model hierarchy, in place. If fed a blank list and
        the entire set of models, will create the model hierarchy
        from scratch.

        Arguments:

            models (list of objs): the structure_record.model()s which
             will be ranked by non-domination.

            model_level_structure (list): list which will contain model
             ranking hierarchy. Item 0 in the list is a list of all 0-rank
             models, item 1 in the list is a list of all 1-rank models, etc.

            flag (string): indicates the set of models which is being
             ranked. Choices are `"cluster"`, or `"population"`.
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

    def update_model_levels_with_insertion(self,
                                           level_structure, new_model, flag):
        '''
        Function which updates the non-dominated level structure based
        on the new model. This is much more efficient than recalculating
        the entire non-dominance ranking of the set of models. Works as
        follows:

        Finds lowest tier in which the model is not dominated by any models.
        - If the model dominates all models in the tier, then insert the model
        into the level structure in its own tier, bumping the dominated
        tier and all other tiers to a higher rank.

        - If the model dominates some of the models in the tier, then replace
        those models with the model, and continue the same process with the
        dominated models (beginning with the next tier).

        - If the model is non-dominated with all models in the tier, then
        append the model to the tier with no other adjustments.

        Arguments:

            level_structure (list of lists): List containing the list of
             models at each non-domination rank.

            new_model (obj): the new structure_record.model() to be added to
             the level structure.

            flag (string): determines which rank (and selection probability
             if relevant) is updated.

        Returns:

            boolean: True if model is unique, False if model is the same as
             another model.
        '''
        T = [new_model]
        moved_levels_up = False
        for level_index in range(len(level_structure)):
            level = level_structure[level_index]
            dominated_models = []
            T_model_dominated = False
            for m in level:
                domination_flags = [self.compare(m, T_model) for T_model in T]
                if -1 in domination_flags:
                    T_model_dominated = True
                    break
                elif 1 in domination_flags:
                    dominated_models.append(m)
                elif 2 in domination_flags:
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

    def update_model_levels_with_deletion(self,
                                          level_structure, old_model, flag):
        '''
        Function which updates the non-dominated level structure based
        on the removal of a model. This is much more efficient than
        recalculating the entire non-dominance ranking of the set of models.
        Works as follows:

        1. Starts from the model's tier. Removes the model, then looks at
         domination of models in the above tier.
        2. If any models are no longer dominated by any model in the current
         tier, then they are moved down into this tier.
        3. Repeat with each higher up tier.

        Arguments:

            level_structure (list of lists): List containing the list of
             models at each non-domination rank.

            old_model (obj): the old structure_record.model() to be removed
             from the level structure.

            flag (string): determines which rank (and selection probability
             if relevant) is updated.
        '''
        T = [old_model]
        if flag == "population":
            starting_level = old_model.rank
        elif flag == "cluster":
            starting_level = old_model.cluster_rank
        for level_index in range(starting_level, len(level_structure)):
            level = level_structure[level_index]
            for m in T:
                level.remove(m)
                T.remove(m)

            if level_index != len(level_structure) - 1:
                if len(level) == 0:
                    for higher_level in level_structure[level_index + 1:]:
                        for model in higher_level:
                            if flag == "population":
                                model.rank -= 1
                            elif flag == "cluster":
                                model.cluster_rank -= 1
                                model.selection_prob = np.exp(
                                    -model.cluster_rank)
                    level_structure.delete(level_index)
                else:
                    for m in level_structure[level_index + 1]:
                        domination_flags = [
                            self.compare(m, lm) for lm in level]
                        if 1 not in domination_flags:
                            T.append(m)
                            if flag == "population":
                                m.rank -= 1
                            elif flag == "cluster":
                                m.cluster_rank -= 1
                                m.selection_prob = np.exp(-m.cluster_rank)
            else:
                if len(level) == 0:
                    level_structure.delete(level_index)

            if len(T) == 0:
                break

    def choose_non_dominated(self, model1, model2):
        '''
        Function which compares two models and returns the model
        which dominates the other. If the two models are
        non-dominated, returns one at random.

        Arguments:

            model1 (obj): structure_record.model() A for the comparison

            model2 (obj): structure_record.model() B for the comparison
        '''
        dominate1 = False
        dominate2 = False

        # TODO: make flexible with number of objectives
        for n in range(model2.num_of_obj):

            if n == 0:
                model1_obj = model1.obj0_val
                model2_obj = model2.obj0_val
            elif n == 1:
                model1_obj = model1.obj1_val
                model2_obj = model2.obj1_val
            elif n == 2:
                model1_obj = model1.obj2_val
                model2_obj = model2.obj2_val
            elif n == 3:
                model1_obj = model1.obj3_val
                model2_obj = model2.obj3_val
            elif n == 4:
                model1_obj = model1.obj4_val
                model2_obj = model2.obj4_val

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


class EpsilonDominance(ParetoDominance):
    '''
    Class which performs non-dominance calculations. Compared to the
    base ParetoDominance class, here non-dominance is determined slightly
    differently. Now, non-dominance is calculated based on
    a grid which discretizes the objective function space. If two
    models occupy the same grid square (with side lengths of
    $\epsilon_1$ and $\epsilon_2$), then the model which is closest to the
    corner of the square is considered to dominate the other model.

    This discretization is also used for structural similarity checks. If
    two models which occupy the same grid square are structurally similar,
    then a flag is thrown. This has the advantage of only performing
    what could be computationally expensive structural similarity calculations
    for models which lie within the same grid box.
    '''

    def __init__(self, epsilons=[.1, .1], comparator=None):
        '''
        Arguments:

            epsilons (list of floats): the grid size of the multiobjective
             space. Should be the same length as the number of objectives,
             and should be in the same order as the objectives are assigned
             to the models. E.g. for two objectives, would be:
              $[\epsilon_1, \epsilon_2]$

            comparator (obj): instance of the comparator class
             which will perform all structural similarity checks

        '''
        # Assign default epsilons if none are provided
        if epsilons is None:
            self.epsilons = [.1, .1]
        else:
            self.epsilons = epsilons

        # store comparator object
        self.comparator = comparator
        if self.comparator is None:
            self.make_similarity_checks = False
        else:
            self.make_similarity_checks = True

    def compare(self, test_model, ref_model):
        '''
        Function which performs comparison between models. It returns:

        - -1 if test_model dominates the ref_model
        - 0 if both non-dominated
        - +1 if test_model dominated by the ref_model
        - +2 if the models are identified as being the same as
         each other within the set tolerance limits of the comparator.

        Arguments:

            test_model (obj): structure_record.model() A for the comparison

            ref_model (obj): structure_record.model() B for the comparison
        '''
        dominate_test = False
        dominate_ref = False

        outside_epsilon_box = False  # determines if structural comparison will
        # be made
        for n in range(ref_model.num_of_obj):
            epsilon = float(self.epsilons[n % len(self.epsilons)])

            if n == 0:
                test_val = math.floor(test_model.obj0_val / epsilon)
                ref_val = math.floor(ref_model.obj0_val / epsilon)
            elif n == 1:
                test_val = math.floor(test_model.obj1_val / epsilon)
                ref_val = math.floor(ref_model.obj1_val / epsilon)
            elif n == 2:
                test_val = math.floor(test_model.obj2_val / epsilon)
                ref_val = math.floor(ref_model.obj2_val / epsilon)
            elif n == 3:
                test_val = math.floor(test_model.obj3_val / epsilon)
                ref_val = math.floor(ref_model.obj3_val / epsilon)
            elif n == 4:
                test_val = math.floor(test_model.obj4_val / epsilon)
                ref_val = math.floor(ref_model.obj4_val / epsilon)

            if self.make_similarity_checks:
                if abs(ref_val - test_val) > epsilon/2:
                    outside_epsilon_box = True

                if n == ref_model.num_of_obj - 1:
                    if not outside_epsilon_box:
                        similarity = self.comparator.assess_models_similarity(
                            test_model, ref_model)
                        if similarity >= 0:
                            return 2

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
            d_test = 0.0
            d_ref = 0.0

            # TODO: make flexible with number of objectives
            for n in range(ref_model.num_of_obj):
                epsilon = float(self.epsilons[n % len(self.epsilons)])
                if n == 0:
                    test_obj = test_model.obj0_val
                    ref_obj = ref_model.obj0_val
                elif n == 1:
                    test_obj = test_model.obj1_val
                    ref_obj = ref_model.obj1_val
                elif n == 2:
                    test_obj = test_model.obj2_val
                    ref_obj = ref_model.obj2_val
                elif n == 3:
                    test_obj = test_model.obj3_val
                    ref_obj = ref_model.obj3_val
                elif n == 4:
                    test_obj = test_model.obj4_val
                    ref_obj = ref_model.obj4_val
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
