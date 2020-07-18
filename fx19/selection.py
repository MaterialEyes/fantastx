
from __future__ import division, unicode_literals, print_function

"""
This module contains funcitons to update the pareto front in case of multi
objective problem or the single objective funcition, selects required number of
parent structures
(This module comes after evaluation and before genetic operations)
"""
import numpy as np
import random

class Pool(object):
    """
    A pool of structures who are evaluated. Parents will be selected from here.

    Maintain two lists:
    good_pool - has limited capacity
              - used for selection
    bad_pool - unlimited capacity (all remaining models)

    NOTE: The best and worst models are chosen based on the model attribute
    "overall_value"
    """

    def __init__(self, pool_params):
        """
        pool capacity is set from i_dict
        """
        energy_pkg = pool_params['energy_pkg']
        if 'capacity' not in pool_params:
            if energy_pkg == 'vasp':
                self.capacity = 500
            else: # energy_code == 'lammps' or 'gulp':
                self.capacity = 100000
        else:
            if 'capacity' in pool_params:
                self.capacity = pool_params['capacity']

        self.good_pool = []
        # NOTE: bad_pool is empty until good_pool capacity is full
        self.bad_pool = []

    def get_sorted_pool(self, pool='good', attribute='overall_val',
                                                            reverse=False):
        """
        Sort good_pool or bad_pool based on an attribute
        """
        if pool == 'good':
            search_pool = self.good_pool
        elif pool == 'bad':
            search_pool = self.bad_pool
        # TODO: Check if need to reverse the sort or not
        # Smaller overall_val is at the beginning
        if attribute == 'overall_val':
            search_pool = sorted(search_pool, key=lambda x: x.overall_val,
                                                            reverse=reverse)
        if attribute == 'obj0_val':
            search_pool = sorted(search_pool, key=lambda x: x.obj0_val,
                                                            reverse=reverse)
        if attribute == 'obj1_val':
            search_pool = sorted(search_pool, key=lambda x: x.obj1_val,
                                                            reverse=reverse)
        if attribute == 'obj2_val':
            search_pool = sorted(search_pool, key=lambda x: x.obj2_val,
                                                            reverse=reverse)
        if attribute == 'obj3_val':
            search_pool = sorted(search_pool, key=lambda x: x.obj3_val,
                                                            reverse=reverse)
        if attribute == 'obj4_val':
            search_pool = sorted(search_pool, key=lambda x: x.obj4_val,
                                                            reverse=reverse)

        return search_pool # sorted

    def get_best_and_worst(self, pool='good', attribute='overall_val',
                           reverse=False):
        """
        From the current two pools, gives the best and worst model in each
        respectively

        pool (str): 'good' or 'bad'
        attribute (str): 'overall_val' or 'obj0_val' or 'obj1_val' or 'obj2_val'
                         'obj3_val' or 'obj4_val'
        reverse (bool): True if greater value is better for attribute
                        Otherwise False (Eg: Lower epa is better)
        """
        best_and_worst = {}
        sorted_pool = self.get_sorted_pool(pool=pool, attribute=attribute,
                                                        reverse=reverse)

        best_in_search_pool = sorted_pool[0]
        worst_in_search_pool = sorted_pool[-1]
        best_and_worst['best_model'] = best_in_search_pool
        best_and_worst['best_val'] = best_in_search_pool.__dict__[attribute]
        best_and_worst['worst_model'] = worst_in_search_pool
        best_and_worst['worst_val'] = worst_in_search_pool.__dict__[attribute]

        return best_and_worst

    def add_to_pool(self, model, select, pool='good', attribute='overall_val',
                    reverse=False, sims=None):
        """
        add the model to the good_pool or bad_pool
        if capacity is not full -> add to good_pool
        else -> compare with worst model in good_pool and add accordingly

        model: model object to be added
        pool (str): 'good' or 'bad'
        attribute (str): 'overall_val' or 'tot_en' or 'obj1_val' or 'obj2_val'
                         'obj3_val' or 'obj4_val'
        reverse (bool): True if greater value is better for attribute
                        Otherwise False (Eg: Lower epa is better)
        """
        demote_worst = False
        if len(self.good_pool) < self.capacity:
            self.good_pool.append(model)
            # update overall vals of all models in good_pool
            select.update_selection_probs(self.good_pool, sims=sims)
            print ('Model {} added to good_pool!\n'.format(model.label))
        else:
            #print ("Good_pool reached above capacity!!!\n")
            best_and_worst = self.get_best_and_worst(pool=pool,
                                                     attribute=attribute,
                                                     reverse=reverse)

            # get min, max and range of obj0_val and obj1_val
            obj0_vals = [m.obj0_val for m in self.good_pool]
            min_obj0, max_obj0 = min(obj0_vals), max(obj0_vals)
            obj1_vals = [m.obj1_val for m in self.good_pool]
            min_obj1, max_obj1 = min(obj1_vals), max(obj1_vals)

            # find overall_val of the model
            w0, w1, w2, w3, w4 = select.weights
            ov = ((model.obj0_val - min_obj0)/(max_obj0 - min_obj0))/w0 + \
                    ((model.obj1_val - min_obj1)/(max_obj1 - min_obj1))/w1
            model.overall_val = ov

            # compare with worst model and continue
            worst_model = best_and_worst['worst_model']
            worst_val = best_and_worst['worst_val']
            if model.__dict__[attribute] >= worst_val and reverse is True:
                self.good_pool.append(model)
                # update overall vals of all models in good_pool
                select.update_selection_probs(self.good_pool, sims=sims)
                demote_worst = True
            elif model.__dict__[attribute] <= worst_val and reverse is False:
                self.good_pool.append(model)
                # update overall vals of all models in good_pool
                select.update_selection_probs(self.good_pool, sims=sims)
                demote_worst = True
            else:
                self.bad_pool.append(model)
                print ('Model {} added to bad_pool directly.\n'.format(
                                                    model.label))
            if demote_worst:
                # demote the worst model to bad_pool
                self.bad_pool.append(worst_model)
                for i, rm_model in enumerate(self.good_pool):
                    if rm_model.label == worst_model.label:
                        self.good_pool.pop(i)
                print ('Model {} demoted to bad_pool and model {} added to '
                            'good_pool.\n'.format(worst_model.label, model.label))

    def clean_pool(self, sims):
        """
        Check if any of the required obj_vals are None
        remove those models from the good_pool
        Note: Not promoting any from bad_pool, the next model will be added to
        good_pool. If this method used, keeps the pool diverse
        """
        # this function is called if sims is not None
        # So, obj0 and obj1 would be present
        all_v0 = [m.obj0_val for m in self.good_pool]
        rem_inds_0 = [i for i, val in enumerate(all_v0) if val is None]
        all_v1 = [m.obj1_val for m in self.good_pool]
        rem_inds_1 = [i for i, val in enumerate(all_v1) if val is None]
        # check for 2, 3, 4 in sims and do this
        rem_inds_2 = []
        if 2 in sims:
            all_v2 = [m.obj2_val for m in self.good_pool]
            rem_inds_2 = [i for i, val in enumerate(all_v2) if val is None]
        # add for 3 and 4 as well
        rem_inds = list(set(rem_inds_0 + rem_inds_1 + rem_inds_2))
        rem_inds.sort(reverse=True)
        for i in rem_inds:
            self.good_pool.pop(i)

class Select(object):
    """
    Values of model from energy calculation and experimental_simulation are
    saved in the models attributes after evaluation. Use the evaluated
    attributes to assign selection probabilities to each model.
    The selection probabilities gets updated for all models after each model is
    evaulated.

    If single objective - all the weights would be zero and the obj0_val will
    be overall_val.
    """

    def __init__(self, select_obj_params):
        self.type = select_obj_params['type']    # 'single' or 'multi'
        # set defaults
        def_weights = [1, 1, 1, 1, 1] # [w0, w1, w2, w3, w4]
        def_temp = 10000
        if 'weights' not in select_obj_params:
            self.weights = def_weights
        else:
            self.weights = select_obj_params['weights']
        if 'temp' not in select_obj_params:
            self.temp = def_temp
        else:
            self.temp = select_obj_params['temp']

        # Making sure that dummy weights are in place, since needed by obj fn
        if not len(self.weights) == 5:
            for i in range(5 - len(self.weights)):
                self.weights.append(1)


    def update_selection_probs(self, good_pool, sims=None):
        """
        use f(E, delta) for all E, delta and get their respective selection
        probabilities. weights and temperatures are also included.
        """
        if len(good_pool) < 3:
            return False

        w0, w1, w2, w3, w4 = self.weights
        weights = np.array([[w0], [w1], [w2], [w3], [w4]])

        all_v0 = [m.obj0_val for m in good_pool]
        maxv, minv = max(all_v0), min(all_v0)
        # normalize obj0 between 0->1
        norm0 = np.array([(i-minv)/(maxv-minv) for i in all_v0])

        norm1 = np.zeros(len(norm0))
        norm2, norm3, norm4 = norm1, norm1, norm1
        if sims:
            all_v1 = [m.obj1_val for m in good_pool]
            if len(all_v1) == len(all_v0) and all(all_v1):
                maxv, minv = max(all_v1), min(all_v1)
                # normalize obj1 between 0->1
                norm1 = np.array([(i-minv)/(maxv-minv) for i in all_v1])
            else:
                return False

        # probs based on overall value
        # divide by weights --> more probability to more weightage obj_val
        # i.e., more weightage --> less overall_val --> more selection_prob
        norm_vals = np.array([norm0, norm1, norm2, norm3, norm4])
        vals = sum(norm_vals/weights)
        maxv, minv = vals.max(), vals.min()
        # normalize vals between 1->0
        probs = [(maxv-i)/(maxv-minv) for i in vals]

        for model, ov, prob in zip(good_pool, vals, probs):
            model.overall_val = ov
            model.selection_prob = prob
        return True


    def get_parents(self, pool, num_parents):
        """
        selects requested number of parents based on their probabilities

        pool - pool object
        num_parents - integer

        Returns a list of parents
        """
        parents = []
        while len(parents) < num_parents:
            new_parent = self.get_a_parent(pool)
            if len(parents) == 0:
                parents.append(new_parent)
            for existing_parent in parents:
                if existing_parent.label == new_parent.label:
                    continue
                else:
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
                    done = True
                    return parent
