
from __future__ import division, unicode_literals, print_function

import numpy as np
import math, copy

"""
This class checks the distance between all the atoms, the angles between
lattice vectors. This is a Divide and Conquer algorithm taken from (source):
https://medium.com/@andriylazorenko/closest-pair-of-points-in-python-79e2409fc0b2
and been modified
"""

def solution(x, y, z, min_dist, close_coords):
    x, y, z = list(x), list(y), list(z)
    a = list(zip(x, y, z))  # This produces list of tuples
    ax = sorted(a, key=lambda x: x[0])  # Presorting x-wise
    ay = sorted(a, key=lambda x: x[1])  # Presorting y-wise
    p1, p2, mi = closest_pair(ax, ay, min_dist, close_coords)  # Recursive D&C function
    return p1, p2, mi, close_coords

def dist(p1, p2):
    """
    calculates and returns the distance between two 3D points
    """
    if len(p1)==len(p2)==3: #if points are in 3D
        d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 +
                      (p1[2] - p2[2]) ** 2)
    #elif len(p1)==len(p2)==2: # if points are in 2D
    #    d = math.sqrt((p1[0] - p2[0])** 2 + (p1[1] - p2[1])** 2)
    return d


def brute(ax, min_dist, close_coords):
    """
    Calculates the min distance between points if the number of points are less
    than "3"

    ax: list of points sorted based on the x-coods
    """
    mi = dist(ax[0], ax[1])
    p1 = ax[0]
    p2 = ax[1]
    if mi < min_dist:
        close_coords.append([p1,p2])
    ln_ax = len(ax)
    if ln_ax == 2:
        return p1, p2, mi
    for i in range(ln_ax-1):
        for j in range(i + 1, ln_ax):
            if i != 0 and j != 1:
                d = dist(ax[i], ax[j])
                if d < mi:  # Update min_dist and points
                    mi = d
                    p1, p2 = ax[i], ax[j]
    return p1, p2, mi


def closest_pair(ax, ay, min_dist, close_coords):
    """
    Finds the closest pair of points and the minimum distance between them
    Divide and Conquer algorithm

    ax, ay and axz are sorted list of points w.r.t x-coords, y-coords and
    z-coords respectively
    """
    ln_ax = len(ax)  # It's quicker to assign variable
    if ln_ax <= 3:
        return brute(ax, min_dist, close_coords)  # A call to bruteforce comparison
    mid = ln_ax // 2  # Division without remainder, need int
    Qx = ax[:mid]  # Two-part split into Q and R
    Rx = ax[mid:]

    # Qx and Qy are sorted lists of Q, w.r.t x and y coords respectively
    Qy = list()
    Ry = list()
    for x in ay:  # split ay into 2 arrays using midpoint
        qx = set(Qx)
        if x in qx:
           Qy.append(x)
        else:
           Ry.append(x)

    # Call recursively both arrays after split
    (p1, q1, mi1) = closest_pair(Qx, Qy, min_dist, close_coords)
    (p2, q2, mi2) = closest_pair(Rx, Ry, min_dist, close_coords)
    # Determine smaller distance between points of 2 arrays
    if mi1 <= mi2:
        d = mi1
        mn = (p1, q1)
    else:
        d = mi2
        mn = (p2, q2)
    # Call function to account for points on the boundary
    (p3, q3, mi3) = closest_split_pair(ax, ay, d, mn, min_dist, close_coords)
    # Determine smallest distance for the array
    if d <= mi3:
        return mn[0], mn[1], d
    else:
        return p3, q3, mi3


def closest_split_pair(p_x, p_y, delta, best_pair, min_dist, close_coords):
    ln_x = len(p_x)  # store length - quicker
    mx_x = p_x[ln_x // 2][0]  # select midpoint on x-sorted array
    # Create a subarray of points not further than delta from
    # midpoint on x-sorted array
    s_y = [x for x in p_y if mx_x - delta <= x[0] <= mx_x + delta]
    best = delta  # assign best value to delta
    ln_y = len(s_y)  # store length of subarray for quickness
    for i in range(ln_y - 1):
        for j in range(i+1, min(i + 7, ln_y)):
            p, q = s_y[i], s_y[j]
            dst = dist(p, q)
            if dst < best:
                best_pair = p, q
                best = dst
            if dst < min_dist:
                close_coords.append([p,q])
    return best_pair[0], best_pair[1], best


def check_angles(astr, min_angle, max_angle):
    """
    Get the lattice vectors from the structure, get the angles between
    lattice vectors

    Returns True if the angles lie within min and max angles
    """
    pass

def astr_min_dist(astr, min_dist):
    """
    Returns True if atoms are too close
    """
    close_coords = []
    coords = astr.cart_coords
    p1, p2, dist, recheck_coords = solution(
                coords[:,0], coords[:,1], coords[:,2], min_dist, close_coords)
    if dist < min_dist:
        return True, recheck_coords
    else:
        return False, None

def coords_min_dist(coords, min_dist):
    """
    Returns True if atoms are too close

    coords: array of cartesian coords
    min_dist: min_dist for the given array
    """
    p1, p2, dist = solution(coords[:,0], coords[:,1], coords[:,2])
    if dist < min_dist:
        return True
    else:
        return False

def check_all_bonds(astr, min_dist_dict, cum_sum):
    """
    (Deprecated)
    Checks the species and corrensponding min bond distance
    Returns True if atoms are too close (less than minimum)

    astr: structure
    min_dist_dict: dictionary of min bond distances for all species
    """
    # get the maximum of all values in min dist dictionary
    max_of_min_dists = max(list(min_dist_dict.values()))
    # check all bonds and record the atoms with less than max(min dists)
    atoms_too_close, recheck_coords = astr_min_dist(astr, max_of_min_dists)
    # atoms_too_close is False if recheck_coords is None
    if recheck_coords is None:
        return atoms_too_close


    sp1_coords = np.round(astr.cart_coords[:cum_sum[0]], 3)
    if len(cum_sum) > 1:
        sp2_coords = np.round(astr.cart_coords[cum_sum[0]:cum_sum[1]], 3)
    if len(cum_sum) > 2:
        sp3_coords = np.round(astr.cart_coords[cum_sum[1]:cum_sum[2]], 3)
    if len(cum_sum) > 3:
        sp4_coords = np.round(astr.cart_coords[cum_sum[2]:cum_sum[3]], 3)
    if len(cum_sum) > 4:
        sp5_coords = np.round(astr.cart_coords[cum_sum[3]:cum_sum[4]], 3)

    # check if individual bonds are less than their corresponding min dist
    recheck_coords = np.round(np.array(recheck_coords), 3)
    # get species pairs for each pair of coords in recheck_coords
    species_pairs = []
    for pair in recheck_coords:
        sp_each_pair = []
        for coords in pair:
            if coords in sp1_coords:
                sp_each_pair.append('sp1')
            elif coords in sp2_coords:
                sp_each_pair.append('sp2')
            elif coords in sp3_coords:
                sp_each_pair.append('sp3')
            elif coords in sp4_coords:
                sp_each_pair.append('sp4')
            elif coords in sp5_coords:
                sp_each_pair.append('sp5')
            else:
                print('The species of the coords is not identified')
                print(coords)
        species_pairs.append(sp_each_pair)

    keys = ['sp1_sp1', 'sp1_sp2', 'sp1_sp3', 'sp1_sp4', 'sp1_sp5',
            'sp2_sp2', 'sp2_sp3', 'sp2_sp4', 'sp2_sp5',
            'sp3_sp3', 'sp3_sp4', 'sp3_sp5',
            'sp4_sp4', 'sp4_sp5',
            'sp5_sp5']

    for sp_pairs, coords_pq in zip(species_pairs, recheck_coords):
        d = dist(coords_pq[0], coords_pq[1])
        key_1 = sp_pairs[0] + '_' + sp_pairs[1]
        key_2 = sp_pairs[1] + '_' + sp_pairs[0]
        if key_1 in keys:
            min_d = min_dist_dict[key_1]
        elif key_2 in keys:
            min_d = min_dist_dict[key_2]
        if d < min_d:
            # There is a bond that is too short than its minimum allowed
            atoms_too_close = True
            return atoms_too_close

    # Reaches here if none of the bonds are smaller than their allowed minimums
    atoms_too_close = False
    return atoms_too_close

def one_to_many_distances(one_point, many_points, min_dist):
    """
    Checks the distances of one point to a list of many points

    one_point : coordinates of single point as list or an array
    many_points: list of other points
    min_dist: the minimum distance that is to be satisfied for all distances

    Returns False if the point is at less distance than min_dist. If satisfies
    min_dist requirement for all points in list, returns True.
    """
    for each_point in many_points:
        d = dist(one_point, each_point)
        if d < min_dist:
            return False
    return True

def satisfies_all_dists(new_point, new_sp, astr, min_dist_dict, species_dict,
                        remove_index=None):
    """
    Checks if new point satisfies all minimum distances specifically with
    each other atom already present in the astr.

    Args:

    new_point: (list/array) the coordinates of a the point which is checked
    new_sp: (str) species of the new coordniate
    astr: structure object wihtin which new coordinate will be checked
    min_dist_dict: dictionary of minimum distances with respect to different
                   species; from inputs
    species_dict: dictionary of species; from inputs
    remove_index: (int) The index of atom to be removed before doing distance
                  check
    """
    # Find if new_sp is specie1 or specie2 or ..
    for key in species_dict.keys():
        if new_sp == species_dict[key]['name']:
            x = 'sp' + key[-1] + '_'

    # Get individual min bond lengths of x i.e., 'sp1_' or 'sp2_' or ..
    # Add distances to min_dists and
    # add the coords of 'sp1_' or 'sp2_' or .. to coords_sets
    # Ex: min_dist 'sp1_sp2' added to min_dists and ..
    # all cart_coords of 'sp2' in astr are added to coords_sets simultaneously
    min_dists = []
    coords_sets = []
    for dist_key in min_dist_dict.keys():
        if x in dist_key:
            set_sp = 'species' + dist_key[-1]
            sp_name = species_dict[set_sp]['name']
            dist_set = min_dist_dict[dist_key]
            all_sites = copy.deepcopy(astr.sites)
            if remove_index:
                del all_sites[remove_index]
            coords_set = [i.coords for i in all_sites if i.specie.name==sp_name]
            min_dists.append(dist_set)
            coords_sets.append(coords_set)
            del all_sites

    bools = []
    for d, set in zip(min_dists, coords_sets):
        bools.append(one_to_many_distances(new_point, set, d))

    if not all(bools):
        return False
    else:
        return True
