from __future__ import division, unicode_literals, print_function

import numpy as np
import math

"""
This file contains functions to check the distance between all the atoms,
and the angles between lattice vectors. It uses a Divide and Conquer
algorithm taken from (source): https://medium.com/@andriylazorenko
/closest-pair-of-points-in-python-79e2409fc0b2
and been modified
"""


def solution(x, y, z, min_dist, close_coords):
    """
    Finds the closest pair of points out of all possible pairs, as well
    as all other points which are close within a threshold.

    Returns the two points, the minimum distance, and the close_coords
    object.

    Args:

    x,y,z (iterables): all x-, y-, and z-coords

    min_dist (float): distance_cutoff with which to consider points close

    close_coords (list): the container for all pairs of close coordinates.
    """
    x, y, z = list(x), list(y), list(z)
    a = list(zip(x, y, z))  # This produces list of tuples
    ax = sorted(a, key=lambda x: x[0])  # Presorting x-wise
    ay = sorted(a, key=lambda x: x[1])  # Presorting y-wise
    p1, p2, mi = closest_pair(ax, ay, min_dist, close_coords)
    return p1, p2, mi, close_coords


def dist(p1, p2):
    """
    Calculates and returns the distance between two 3D points.

    Args:

    p1 and p2: coordinates of the two 3D points. Can be provided as
    list, tuple or array.
    """
    if len(p1) == len(p2) == 3:  # if points are in 3D
        d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 +
                      (p1[2] - p2[2]) ** 2)
    # elif len(p1)==len(p2)==2: # if points are in 2D
    #    d = math.sqrt((p1[0] - p2[0])** 2 + (p1[1] - p2[1])** 2)
    return d


def brute(ax, min_dist, close_coords):
    """
    Calculates the min distance between points if the number of points are less
    than "3"

    Args:

    ax (list): list of points sorted based on the x-coods

    min_dist (float): distance cutoff within which to consider the points close

    close_coords (list): paired coordinates of points which are close.
    """
    mi = dist(ax[0], ax[1])
    p1 = ax[0]
    p2 = ax[1]
    if mi < min_dist:
        close_coords.append([p1, p2])
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
    using a Divide and Conquer algorithm.

    Returns the pair of points and the minimum distance.

    Args:

    ax (list): sorted list of points w.r.t x-coords

    ay (list): sorted list of points w.r.t y-coords

    min_dist (float): distance cutoff within which to consider points close

    close_coords (list): paired coordinates of points which are close.
    """
    ln_ax = len(ax)  # It's quicker to assign variable
    if ln_ax <= 3:
        # A call to bruteforce comparison
        return brute(ax, min_dist, close_coords)
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
                close_coords.append([p, q])
    return best_pair[0], best_pair[1], best


def check_angles(astr, min_angle, max_angle):
    """
    Get the lattice vectors from the structure, get the angles between
    lattice vectors

    Returns True if the angles lie within min and max angles
    """
    pass


def one_to_many_distances(one_point, many_points, min_dist):
    """
    Checks the distances of one point to a list of many points

    Returns False if the point is at less distance than min_dist. If satisfies
    min_dist requirement for all points in list, returns True.

    Args:

    one_point: coordinates of single point as list or an array

    many_points (list): list of coordinates of other points

    min_dist (float): the minimum distance that is to be satisfied for
    all distances
    """
    for each_point in many_points:
        d = dist(one_point, each_point)
        if d < min_dist:
            return False
    return True


def astr_min_dist(astr, min_dist):
    """
    Returns True if atoms are too close
    """
    close_coords = []
    coords = astr.cart_coords
    p1, p2, dist, recheck_coords = solution(
        coords[:, 0], coords[:, 1], coords[:, 2], min_dist, close_coords)
    if dist < min_dist:
        return True, recheck_coords
    else:
        return False, None


def check_all_bonds(astr, min_dist_dict, cum_sum):
    """
    Checks the species and corrensponding min bond distance
    Only used by initial population in cluster geometry

    Returns True if atoms are too close (less than minimum)

    Args:

    astr: pymatgen structure object

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
                print(f'The species of the coords {coords} is not identified')
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


def dist_pbc(p1, p2, lattice):
    """
    Calculates and returns the distance between two points
    in periodic boundary conditions, using minimum image representation.

    Args:

    p1 and p2: cartesian coordinates of the PeriodicSites. Preferred
    formats are numpy arrays, but can be provided as lists or tuples.

    lattice: pymatgen lattice object of the structure
    """
    if not type(p1) is np.ndarray or not type(p2) is np.ndarray:
        p1 = np.array(p1)
        p2 = np.array(p2)

    # convert each point into fractional lattice coordinates using pymatgen
    p1_frac = lattice.get_fractional_coords(p1)
    p2_frac = lattice.get_fractional_coords(p2)

    # alternately, solve it by hand
    # p1_frac = np.linalg.solve(lattice.matrix, p1)
    # p2_frac = np.linalg.solve(lattice.matrix, p2)

    # get vector connecting points, in the minimum image representation
    # will break if not all boundaries are periodic,
    # or if the norm of the vectors is larger than 0.5 * the min cell length
    v = p1_frac - p2_frac
    v -= np.floor(v + 0.5)
    v_minimage = np.dot(v, lattice.matrix)
    d = np.linalg.norm(v_minimage)

    return d


def dist_pbc_pymatgen(p1, p2, lattice):
    """
    Calculates and returns the distance between two points
    in periodic boundary conditions, using pymatgen lattice functions.
    Slower than dist_pbc(), but more robust

    Args:

    p1 and p2: cartesian coordinates of the PeriodicSites

    lattice: pymatgen lattice object of the structure
    """
    # convert each point into fractional lattice coordinates
    f1 = lattice.get_fractional_coords(p1)
    f2 = lattice.get_fractional_coords(p2)

    # get distance between the two fractional coordinates, returning the
    # distance and number of lattice translations required to shift the image
    d, _ = lattice.get_distance_and_image(f1, f2, None)

    return d


def get_all_image_distances(p1, p2, lattice, coords_are_cartesian=True):
    """
    Calculates the distance between a point and all periodic replicas of
    another point, using pymatgen lattice functions.

    Arguments:
        p1 and p2: coordinates of each point
        lattice (obj): pymatgen lattice object of the structure
        coords_are_cartesian (bool): True if p1 and p2 are cartesian

    Returns:
        (list, list): all periodic replica distances, and all corresponding
         images
    """
    if coords_are_cartesian:
        f1 = lattice.get_fractional_coords(p1)
        f2 = lattice.get_fractional_coords(p2)
    else:
        f1, f2 = p1, p2

    d, base_image = lattice.get_distance_and_image(f1, f2, None)
    reference = np.floor(f2 + base_image)
    signs = [-1 if i == 0 else 1 for i in reference]

    distances, images = [d], [base_image]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if i != 0 or j != 0 or k != 0:
                    new_image = np.multiply([i, j, k], signs) + base_image
                    new_d, _ = lattice.get_distance_and_image(f1, f2,
                                                              new_image)

                    distances.append(new_d)
                    images.append(new_image)

    return distances, images


def vector_connecting_points(p1, p2, lattice=None, coords_are_cartesian=False,
                             get_minimal_vector=False):
    """
    Calculates the vector connecting cartesian point p1 to cartesian point p2.
    If lattice is not `None`, then employs periodic boundary conditions.

    Arguments:
        p1 (iterable): cartesian coordinates of site 1
        p2 (iterable): cartesian coordinates of site 2
        lattice (obj): pymatgen lattice object that the points reside in
        get_minimal_vector (bool): True if want to reduce vector to the vector
         corresponding to the atomic coordinates with minimal image distance

    Returns:
        (array): the vector connecting p1 to p2, in cartesian coordinates
    """

    if lattice is None:
        cvec = p2 - p1
    else:

        if coords_are_cartesian:
            fvec = np.subtract(lattice.get_fractional_coords(p2),
                               lattice.get_fractional_coords(p1))
        else:
            fvec = np.subtract(p2, p1)

        if get_minimal_vector:
            fvec = np.subtract(fvec, np.round(fvec))
        cvec = lattice.get_cartesian_coords(fvec)

    return cvec


def one_to_many_distances_periodic(one_point, many_points, min_dist, lattice):
    """
    Checks the distances of one point to a list of many points

    one_point : cartesian coordinates of single point as list or an array
    many_points: list of cartesian coordinates of all other points
    min_dist: the minimum distance that is to be satisfied for all distances
    lattice: the lattice object corresponding to the pymatgen structure

    Returns False if the point is at less distance than min_dist. If satisfies
    min_dist requirement for all points in list, returns True.
    """
    for each_point in many_points:
        d = dist_pbc(one_point, each_point, lattice)
        if d < min_dist:
            return False
    return True


def get_bonded_neighbors(one_point, many_points, one_species,
                         many_species, inv_syms, max_dist_dict,
                         lattice,
                         coords_are_cartesian=True,
                         available_bonds=None):
    """
    Gets the bonded neighbors for an atom given full information for a set of
    atoms it is embedded in. Checks if bond limits are satisifed if provided

    Arguments:
        one_point (iterable): Cartesian coordinates of the new atom

        many_points (iterable): Cartesian coordinates of the atoms which
         currently reside in the structure

        one_species (str): species of the new atom

        many_species (iterable): strings corresponding to the species of the
         atoms which currently reside in the structure

        inv_syms (dict): the mapping of each atomic species to their
         designation in the input yaml file (sp1, sp2, etc)

        max_dist_dict (dict): dictionary of the maximum bond distances with
         respect to different species

        lattice (obj): Pymatgen `Lattice` object which contains the species.
         If provided, all distances are calculated using periodic boundary
         conditions.

        coords_are_cartesian (bool): True if the provided coordinates are
         cartesian, False, if the provided coordinates are fractional

    Returns:

        (dict): dictionary mapping bonded neighbor indices to bond vectors
         which go from the bonded neighbor to the target atom
    """
    sym1 = inv_syms[one_species]
    bonds = {}
    images = []
    for index, each_point in enumerate(many_points):
        sym2 = inv_syms[many_species[index]]
        key = sym1 + '_' + sym2
        if sym1 > sym2:
            key = sym2 + '_' + sym1
        ref_dist = 0.0
        if max_dist_dict[key] is not None:
            ref_dist = max_dist_dict[key]
        else:
            continue

        if lattice is None:
            if coords_are_cartesian:
                d = dist(one_point, each_point)
            else:
                print("Error! Provided coordinates are not cartesian, but"
                      " no lattice was provided.")
                return None
            if d <= ref_dist:
                if available_bonds is not None:
                    if available_bonds[index] == 0:
                        return None
                bonds[index] = [vector_connecting_points(
                    one_point, each_point, None)]
        else:
            dists, images = get_all_image_distances(
                one_point, each_point, lattice, coords_are_cartesian)

            # print(f"dists: {dists}, images: {images}")

            p1 = one_point
            p2 = each_point
            if coords_are_cartesian:
                p1 = lattice.get_fractional_coords(one_point)
                p2 = lattice.get_fractional_coords(each_point)

            new_bonds = 0
            for d, i in zip(dists, images):
                # print(f"distance: {d}, image: {i}")
                if d <= ref_dist:
                    vec = vector_connecting_points(
                        p1, p2 + i, lattice, False, False)
                    # print(
                    #     f"Passed sanity check: {np.isclose(np.linalg.norm(vec), d)}")
                    if index in bonds:
                        bonds[index].append(vec)
                    else:
                        bonds[index] = [vec]
                    if available_bonds is not None:
                        if available_bonds[index] == new_bonds:
                            return None
                    new_bonds += 1

    return bonds


def satisfies_all_dists_quick(one_point, many_points, one_species,
                              many_species, inv_syms, min_dist_dict,
                              max_dist_dict=None, lattice=None,
                              coords_are_cartesian=True):
    """
    Function to check that a new coordinate being added to an existing
    structure satisfies all minimum distance constraints, as well as maximum
    distance constraints if provided. To be used with initial_population and
    basinhopping methods. This is an alternate implementation of the function
    satisfies_all_dists, which looks for atoms within a sphere around a point
    before comparing distances. Both methods are O(N^2) but this method does
    reduce computation time.

    Arguments:

        one_point (iterable): Cartesian coordinates of the new atom

        many_points (iterable): Cartesian coordinates of the atoms which
         currently reside in the structure

        one_species (str): species of the new atom

        many_species (iterable): strings corresponding to the species of the
         atoms which currently reside in the structure

        inv_syms (dict): the mapping of each atomic species to their
         designation in the input yaml file (sp1, sp2, etc)

        min_dist_dict (dict): dictionary of minimum distances with respect to
         different species

        max_dist_dict (dict): dictionary of the maximum bond distances with
         respect to different species

        lattice (obj): Pymatgen `Lattice` object which contains the species.
         If provided, all distances are calculated using periodic boundary
         conditions.

        coords_are_cartesian (bool): True if the provided coordinates are
         cartesian, False, if the provided coordinates are fractional

        return_neighbors (bool): True if the indices of bonded neighbors
         should be checked for and returned.
    """
    sym1 = inv_syms[one_species]
    dists_ok = False
    # print(f"Symbol being tested: {sym1}")
    # print(f"Many points: {many_points}")
    # print(f"This point: {one_point}")
    for index, each_point in enumerate(many_points):
        if lattice is None:
            if coords_are_cartesian:
                d = dist(one_point, each_point)
            else:
                print("Error! Provided coordinates are not cartesian, but"
                      " no lattice was provided.")
                return None
        else:
            if coords_are_cartesian:
                d = dist_pbc_pymatgen(one_point, each_point, lattice)
                # d = dist_pbc(one_point, each_point, lattice)
            else:
                d, _ = lattice.get_distance_and_image(one_point,
                                                      each_point,
                                                      None)

        # d = dist(one_point, each_point)
        sym2 = inv_syms[many_species[index]]
        # print(f"Symbol of the other point: {sym2}")
        # print(f"Distance: {d}")
        key = sym1 + '_' + sym2
        if sym1 > sym2:
            key = sym2 + '_' + sym1

        if key in min_dist_dict:
            if d < min_dist_dict[key]:
                return False

        if max_dist_dict is not None:
            if max_dist_dict[key] is not None:
                if d <= max_dist_dict[key]:
                    dists_ok = True
        else:
            dists_ok = True

    # print(f"Distances ok: {dists_ok}")

    return dists_ok



def satisfies_all_dists(new_carts, existing_astr, element_syms,
                        min_dist_dict, max_dist_dict=None,
                        atom_index_in_astr=None,
                        new_carts_species=None):
    """
    Function to check that a new coordinate being added to an existing
    structure satisfies all the minimum and maximum distance constraints
    provided in the min_dist_dict and max_dist_dict. To be used with
    initial_population or basinhopping methods.
    Returns True if satisfies all constriants.

    Args:

    new_carts(list/array): Cartesian coordinates of the new atom to be added

    existing_astr (obj): Pymatgen structure object of the parent to which new
    coord is added

    element_syms (dict): dictionary of species which specifies the species
    index

    min_dist_dict (dict): dictionary of minimum distances with respect to
    different species

    max_dist_dict (dict): dictionary of maximum bond distances with respect to
    different species

    atom_index_in_astr (int): (For basinhopping only) The index of the atom in
    the parent structure that is perturbed

    new_carts_species (str): The species of the new atom as a string. If
    atom_index_in_astr is given, it is used to get the new atom species and
    overwrites this argument. At least one of these two parameters should
    be provided.

    """
    sphere_radius = max(min_dist_dict.values())
    if max_dist_dict is not None:
        sphere_radius = max(max_dist_dict.values())

    # Get all fractional coordinates of the existing structure
    all_frac_points = existing_astr.frac_coords
    all_species = existing_astr.species

    atoms_nearby = \
        existing_astr.lattice.get_points_in_sphere(all_frac_points,
                                                   new_carts,
                                                   sphere_radius)

    if atom_index_in_astr is not None:
        # Remove duplicate atom from the atoms nearby
        for i, atom_data in enumerate(atoms_nearby):
            if atom_data[2] == atom_index_in_astr:
                duplicate_atom_ind = i
                # del atoms_nearby[duplicate_atom_ind]
                atoms_nearby.pop(duplicate_atom_ind)
                break

    dists_nearby = [i[1] for i in atoms_nearby]
    inds_nearby = [i[2] for i in atoms_nearby]
    # Get the species of atoms nearby
    species_nearby = [all_species[i].name for i in inds_nearby]

    # Inverse of element_syms
    inv_syms = {v: 'sp' + str(k) for k, v in element_syms.items()}

    # Remove any extra species that are not in element_syms (Ex: substrate)
    species_nearby = [i for i in species_nearby if i in inv_syms]

    # Get species_keys_nearby
    species_keys_nearby = [inv_syms[each_sps] for each_sps in species_nearby]

    # Get new_atom_sym
    if atom_index_in_astr is not None:
        new_carts_species = existing_astr.species[atom_index_in_astr].name
    new_atom_sym = inv_syms[new_carts_species]

    dists_ok = len(species_keys_nearby) < 1
    for i, spx in enumerate(species_keys_nearby):
        dist = dists_nearby[i]
        # cover both 'sp1_sp2' & 'sp2_sp1'in key1 & key2
        key1 = new_atom_sym + '_' + spx
        key2 = spx + '_' + new_atom_sym
        if key1 in min_dist_dict:
            if dist < min_dist_dict[key1]:
                # print("FAILED")
                return False
        if key2 in min_dist_dict:
            if dist < min_dist_dict[key2]:
                # print("FAILED")
                return False
        # check max_dists as well if provided
        # make sure at least one atom is within relevant bond radius
        if max_dist_dict is not None:
            if key1 in max_dist_dict:
                if dist <= max_dist_dict[key1]:
                    dists_ok = True
            if key2 in max_dist_dict:
                if dist <= max_dist_dict[key2]:
                    dists_ok = True
        else:
            dists_ok = True

    return dists_ok

def check_interatom_dists(astr, species_dict, 
                          min_dist_dict,max_dist_dict):

    """
    Function for checking if all atoms meet the minimum and 
    maximum distance constraints set in the input.yaml file
    
    Args:

    astr (obj): Pymatgen structure object of the parent to which new
    coord is added

    element_syms (dict): dictionary of species which specifies the species
    index

    min_dist_dict (dict): dictionary of minimum distances with respect to
    different species

    max_dist_dict (dict): dictionary of maximum bond distances with respect to
    different species

    """

    dist_mat = astr.distance_matrix
    spec_trans={}
    specs = list(set([str(x) for x in astr.species]))
    for k in list(species_dict.keys()):
        spec_trans[species_dict[k]['name']]='sp'+k[7:]
    
    # Min dist check
    
    minDists=[]
    for i in range(astr.num_sites):
        minDists.append([])
        for j in range(astr.num_sites):
            try:
                minDists[i].append(min_dist_dict[
                               spec_trans[str(astr.species[i])]+'_'+
                               spec_trans[str(astr.species[j])]])
            except:
                minDists[i].append(min_dist_dict[
                               spec_trans[str(astr.species[j])]+'_'+
                               spec_trans[str(astr.species[i])]])
    diffs = np.array(dist_mat)-np.array(minDists)
    # Set diagonal to 1, else 0-X is always <0
    for i in range(len(diffs)):
        diffs[i][i]=1
    # If any distance is less than 0, atoms too close
    if np.any([np.any([x<0 for x in y]) for y in diffs]):
        return False 
    
    # Max dist check

    maxDists=[]
    for i in range(astr.num_sites):
        maxDists.append([])
        for j in range(astr.num_sites):
            try:
                maxDists[i].append(max_dist_dict[
                               spec_trans[str(astr.species[i])]+'_'+
                               spec_trans[str(astr.species[j])]])
            except:
                maxDists[i].append(max_dist_dict[
                               spec_trans[str(astr.species[j])]+'_'+
                               spec_trans[str(astr.species[i])]])
    diffs = np.array(dist_mat)-np.array(maxDists)
    # Set diagonal to 1, else 0-X is always <0
    for i in range(len(diffs)):
        diffs[i][i]=1
    # If any distance is less than 0, there is a nearest neighbor atom
    for atom in diffs:
        if np.any([x<0 for x in atom]):
            pass
        else:
            return False
    return True


