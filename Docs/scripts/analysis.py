import itertools
import numpy as np


def read_in_data_file(file_path, type):
    '''
    Method to read in objective functions for the models from the data_file.
    Args:
    type (string) - type of optimization performed (multi or single). Possible values
                    taken are "single" and "multi"
    Outputs:
    obj_fncs (dictionary) - maps model objective functions values to model labels
    '''
    assert type in ["single", "multi"], "`type` must be 'single' or 'multi'."

    lines = open(file_path, "r").read().splitlines()
    index = 0
    obj_fncs = {}
    for line in lines:
        newline = line.split()
        if index > 0 and newline != []:
            if type == "single":
                obj_fncs[int(newline[0])] = [float(
                    newline[2]), float(newline[3])]
            else:
                # account for both single parent and two parent inheritance
                if len(newline) == 6:
                    obj_fncs[int(newline[0])] = [float(
                        newline[2]), float(newline[3]), float(newline[4])]
                elif len(newline) == 7:
                    obj_fncs[int(newline[0])] = [float(
                        newline[3]), float(newline[4]), float(newline[5])]
                else:
                    print("Bad line length!")
        index += 1
    return obj_fncs


def get_nondominated_solutions(obj_fncs, objs_to_use):
    """
    Inspired by this [page](https://github.com/QUVA-Lab/artemis/
    blob/peter/artemis/general/pareto_efficiency.py)

    Returns the list of non-dominated models, and the list of the models
    which are dominated by at least one other model, based on comparison
    function flags.

    Arguments:

        obj_fncs (dictionary): a dictionary mapping model indices to 
            lists of objective function values

        objs_to_use (list): indices of which objective function values to
            use to calculate non-dominance (i.e. [1,3] would be objective
            1 and objective 3)
    """
    model_indices = list(obj_fncs.keys())
    model_obj_fncs = list(obj_fncs.values())
    is_efficient = np.ones(len(model_indices), dtype=bool)
    for index, obj_fnc in enumerate(model_obj_fncs):
        if is_efficient[index]:
            flags = [
                compare_objs(m, obj_fnc, objs_to_use) for m in
                list(itertools.compress(model_obj_fncs, is_efficient))]
            # keep any point which either dominated the model
            # or was non-dominated
            equal_or_better = [x < 0 or x == 0 for x in flags]
            is_efficient[is_efficient] = equal_or_better
            is_efficient[index] = True  # and keep self

    non_dominated_indices = list(
        itertools.compress(model_indices, is_efficient))
    non_dominated_obj_fncs = list(
        itertools.compress(model_obj_fncs, is_efficient))
    dominated_indices = list(
        itertools.compress(model_indices, np.invert(is_efficient)))
    dominated_obj_fncs = list(
        itertools.compress(model_obj_fncs, np.invert(is_efficient)))
    return non_dominated_indices, dominated_indices, non_dominated_obj_fncs, dominated_obj_fncs


def compare_objs(obj_fncs1, obj_fncs2, objs_to_use):
    '''
    Function which performs comparison between models. It returns:

    - -1 if obj_fncs1 dominates obj_fncs2
    - 0 if both non-dominated
    - +1 if obj_fncs1 dominated by obj_fncs2

    Arguments:

        obj_fncs1 (list): first list of objective functions for comparison

        obj_fncs2 (list): second list of objective functions for comparison

        objs_to_use (list): indices of which objective function values to
            use to calculate non-dominance (i.e. [1,3] would be objective
            1 and objective 3)
    '''

    dominate_1 = False
    dominate_2 = False

    for n in range(len(obj_fncs1)):
        if n in objs_to_use:
            test_obj = obj_fncs1[n]
            ref_obj = obj_fncs2[n]

            if test_obj < ref_obj:
                dominate_1 = True
                # Check for non-domination
                if dominate_2:
                    return 0

            elif test_obj > ref_obj:
                dominate_2 = True
                # Check for non-domination
                if dominate_1:
                    return 0

    # Otherwise one dominates the other, return the appropriate value
    if dominate_1:
        return -1
    else:
        return 1
