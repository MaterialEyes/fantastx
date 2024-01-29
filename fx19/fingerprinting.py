"""
This module handles all fingerprinting functions for FANTASTX.
Functionality is implemented in two classes:
    
- `DistanceCalculator`
- `Comparator`

The `DistanceCalculator` class contains methods to assess the
distance between two fingerprint vectors. It is essentially
a wrapper for the sklearn distance metrics. Currently, it is used
for the following fingerprints:

- Many-Body Tensor Representation (`mbtr`)
- Ewald Sum Matrix (`ewald-sum-matrix`)
- Sine Matrix (`sine-matrix`)

!!! Important "Take Notice"
    
    The `DistanceCalculator` class is not useful for fingerprints which
    need more sophisticated or otherwise specialized methods of distance
    comparison. These are:

    - Bag of Bonds (`bag-of-bonds`)
    - SOAP (`rematch-soap` and `average-soap`)
    - Valle-Oganaov (`valle-oganov`)

The `Comparator` class contains methods to create fingerprints for
structures, calculate the distance between the fingerprints of
different structures, and return appropriate flags to selection
algorithms. At a later date, functionality will also be added to
interface with a separate package which handles the calculation
of on-the-fly machine learning force fields.

---

"""

try:
    from dscribe.kernels import REMatchKernel, AverageKernel
    from dscribe.descriptors import SOAP, SineMatrix
    from dscribe.descriptors import EwaldSumMatrix, MBTR
except ImportError:
    print('Install Dscribe for structure comparison'
          'using dscribe fingerprints.')
from ase.ga.ofp_comparator import OFPComparator
from pymatgen.io.ase import AseAtomsAdaptor
import copy
from sklearn.preprocessing import normalize
from fx19 import distance_check as dc
import numpy as np
from sklearn.metrics import pairwise_distances, pairwise
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pymatgen.core.structure import Structure, Lattice


def cutoff(dist, dist_cut):
    if dist < dist_cut:
        return 0.5*np.cos(np.pi * dist / dist_cut) + 0.5
    else:
        return 0.0


class DistanceCalculator(object):
    """
    This class acts as a convenient wrapper for sklearn distance metrics.
    It is used to calculate the distance between two fingerprint vectors
    which are in the form of 1-D iterables.

    !!! note

        Any fingerprint fed into an instance of `DistanceCalculator` MUST
        be flattened! No dimensionality higher than 1-D is allowed.
    """

    def __init__(self, metric):
        """
        Each instance of `DistanceCalculator` is assigned a distance metric
        that it will be used to make comparisons with. The valid metrics are:

        - `manhattan`: the manhattan distance (the 1-norm)
        - `euclidean`: the euclidean distance (the 2-norm)
        - `cosine`: 1 - the cosine kernel similarity (basically the dot
         product)
        - `laplacian`: 1 - the laplacian kernel similarity
        - `gaussian`: 1 - the gaussian kernel similarity
        - `mae`: the mean absolute error
        - `mse`: the mean squared error
        - `rmse`: the root mean squared error
        - `r2_score`: the r^2 score
        """
        valid_metrics = ["manhattan", "euclidean", "cosine", "laplacian",
                         "gaussian", "mae", "mse", "rmse", "r2_score"]
        if metric not in valid_metrics:
            print("User assigned metric {metric} not contained in the the set"
                  "of valid DistanceCalculator metrics: {valid_metrics}."
                  "Default option 'euclidean' assigned.")
            self.metric = "euclidean"
        else:
            self.metric = metric

    def create(self, fingerprint1, fingerprint2):
        """
        Method to calculate the distance between two fingerprints.

        Arguments:

            fingerprint1 (array): first fingerprint
            fingerprint2 (array): second fingerprint

        Returns:

            float: the distance between the two fingerprints
        """
        if self.metric in ['euclidean', 'cosine', 'manhattan',
                           'laplacian', 'gaussian']:
            if fingerprint1.shape[0] != 1:
                fingerprint1 = fingerprint1.reshape(1, -1)
            if fingerprint2.shape[0] != 1:
                fingerprint2 = fingerprint2.reshape(1, -1)

        if self.metric == "euclidean":
            return pairwise_distances(fingerprint1,
                                      fingerprint2,
                                      metric="euclidean")[0][0]
        if self.metric == "cosine":
            return pairwise.cosine_distances(fingerprint1,
                                             fingerprint2)[0][0]
        if self.metric == "manhattan":
            return pairwise_distances(fingerprint1,
                                      fingerprint2,
                                      metric="manhattan")[0][0]
        if self.metric == "laplacian":
            return 1 - pairwise.laplacian_kernel(fingerprint1,
                                                 fingerprint2,
                                                 gamma=1e-2)[0][0]
        if self.metric == "gaussian":
            return 1 - pairwise.rbf_kernel(fingerprint1,
                                           fingerprint2,
                                           gamma=1e-2)[0][0]
        if self.metric == "rmse":
            return mean_squared_error(fingerprint1,
                                      fingerprint2,
                                      squared=False)
        if self.metric == "mse":
            return mean_squared_error(fingerprint1,
                                      fingerprint2,
                                      squared=True)
        if self.metric == "mae":
            return mean_absolute_error(fingerprint1,
                                       fingerprint2)
        if self.metric == "r2_score":
            return r2_score(fingerprint1,
                            fingerprint2)
        if self.metric == "chi2":
            f1 = np.abs(fingerprint1)
            f2 = np.abs(fingerprint2)
            return 0.5 * np.sum(np.square(f1 - f2)/(f1 + f2))


class Comparator(object):
    '''
    Class which handles all structural fingerprinting. Contains functions
    to create fingerprints, compare fingerprint, and compare models
    based on their fingerprints (whether local or global).

    For fingerprints which are not able to use the `DistanceCalculator` class
    to perform structural comparisons (i.e. local atomic fingerprints such
    as SOAP fingerprints), this class also contains functions to create their
    corresponding unique kernels for distance comparison. For `soap`
    fingerprints this can be either an *average* kernel or a *REMatch* kernel.
    For the `valle-oganov` fingerprints this is the kernel implemented in the
    ASE package. For the `bag-of-bonds` kernel, it is a specialized non-vector
    comparison. For details, refer to this `bag-of-bonds` [paper](https://pubs
    .acs.org/doi/10.1021/acs.jctc.6b01119).
    '''

    def __init__(self, label='bag-of-bonds', tolerances=None):
        self.label = label

        # Assign default tolerance values if none are provided
        if tolerances is None:
            tolerances = {}
            tolerances["valle-oganov"] = 1e-3
            tolerances["bag-of-bonds"] = [.02, 0.7]
            tolerances["rematch-soap"] = 1e-3
            tolerances['average-soap'] = 1e-3
            tolerances["mbtr"] = 1e-3
            tolerances['sine-matrix'] = 1e-3
            tolerances['ewald-sum-matrix'] = 1e-3

        self.tolerances = tolerances
        self.comp = None
        self.kernel_gen = None
        self.zbounds = None
        self.distance_calculator = None
        # whether to remove vacuum in all directions before fingerprint
        # to be used in cluster & surface geometries
        self.rem_vac = False  # defaults to False

    def set_soap_descriptor(self, _species, soap_values=None):
        '''
        Creates the class `soap` descriptor object.

        Arguments:

            _species (array of strs): containing the element names of
             all atomic species handled by the descriptor.

            soap_values (dict): containing user-defined
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

    def set_mbtr_descriptor(self, _species, mbtr_values=None):
        '''
        Creates the class `mbtr` descriptor object.

        Arguments:

            _species (array of strs) containing the element names of
             all atomic species handled by the descriptor.

            mbtr_values (dict): containing user-defined
             values for some or all MBTR descriptor parameters.
        '''
        if mbtr_values is None:
            self.desc = MBTR(
                species=_species,
                k1={
                    "geometry": {"function": "atomic_number"},
                    "grid": {"min": 0,
                             "max": 18,
                             "n": 100,
                             "sigma": 0.6},
                },
                k2={
                    "geometry": {"function": "inverse_distance"},
                    "grid": {"min": 0,
                             "max": 2,
                             "n": 100,
                             "sigma": 0.08},
                    "weighting": {"function": "exp",
                                  "scale": 0.5,
                                  "threshold": 1e-3},
                },
                k3={
                    "geometry": {"function": "cosine"},
                    "grid": {"min": -1,
                             "max": 1,
                             "n": 100,
                             "sigma": 0.03},
                    "weighting": {"function": "exp",
                                  "scale": 0.5,
                                  "threshold": 1e-3},
                },
                flatten=True,
                periodic=True,
                normalization="l2_each"
            )
        else:
            if "k1" not in mbtr_values and\
               "k2" not in mbtr_values and\
                    "k3" not in mbtr_values:
                print("Problem! Need to provide individual k-body term "
                      "dictionaries to set any mbtr parameters. "
                      "Using default settings. ")
                self.desc = MBTR(
                    species=_species,
                    k1={
                        "geometry": {"function": "atomic_number"},
                        "grid": {"min": 0,
                                 "max": 18,
                                 "n": 100,
                                 "sigma": 0.6},
                    },
                    k2={
                        "geometry": {"function": "inverse_distance"},
                        "grid": {"min": 0,
                                 "max": 2,
                                 "n": 100,
                                 "sigma": 0.08},
                        "weighting": {"function": "exp",
                                      "scale": 0.5,
                                      "threshold": 1e-3},
                    },
                    k3={
                        "geometry": {"function": "cosine"},
                        "grid": {"min": -1,
                                 "max": 1,
                                 "n": 100,
                                 "sigma": 0.03},
                        "weighting": {"function": "exp",
                                      "scale": 0.5,
                                      "threshold": 1e-3},
                    },
                    flatten=True,
                    periodic=True,
                    normalization="l2_each"
                )
            else:
                if "k1" not in mbtr_values:
                    mbtr_values["k1"] = None
                if "k2" not in mbtr_values:
                    mbtr_values["k2"] = None
                if "k3" not in mbtr_values:
                    mbtr_values["k3"] = None
                self.desc = MBTR(
                    species=_species,
                    k1=mbtr_values["k1"],
                    k2=mbtr_values["k2"],
                    k3=mbtr_values["k3"]
                )

    def set_sine_matrix_descriptor(self, sm_values=None):
        '''
        Creates the class `sine-matrix` descriptor object.

        Arguments:

            sm_values (dict): containing user-defined
            values for some or all Sine Matrix descriptor parameters.
        '''
        if sm_values is None:
            self.desc = SineMatrix(
                n_atoms_max=24,
                permutation="sorted_l2",
                sparse="False",
                flatten="True"
            )
        else:
            _n_atoms_max = 24
            _permutation = "sorted_l2"
            if "permutation" in sm_values:
                _permutation = sm_values["permutation"]
            if "n_atoms_max" in sm_values:
                _n_atoms_max = sm_values["n_atoms_max"]
            self.desc = SineMatrix(
                n_atoms_max=_n_atoms_max,
                permutation=_permutation,
                sparse="False",
                flatten="True")

    def set_ewald_sum_matrix_descriptor(self, esm_values=None):
        '''
        Creates the class `ewald-sum-matrix` descriptor object.

        Arguments:

            sm_values (dict): containing user-defined
            values for some or all Ewald Sum Matrix descriptor parameters.
        '''
        if esm_values is None:
            self.desc = EwaldSumMatrix(
                n_atoms_max=24,
                permutation="sorted_l2",
                sparse="False",
                flatten="True"
            )
        else:
            _n_atoms_max = 24
            _permutation = "sorted_l2"
            if "permutation" in esm_values:
                _permutation = esm_values["permutation"]
            if "n_atoms_max" in esm_values:
                _n_atoms_max = esm_values["n_atoms_max"]
            self.desc = EwaldSumMatrix(
                n_atoms_max=_n_atoms_max,
                permutation=_permutation,
                sparse="False",
                flatten="True")

    def set_valle_oganov_comparator(self, comp_values=None):
        '''
        Creates the class `valle-oganov` comparator object.

        Arguments:

            comp_values (dict): containing user-defined
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

    def set_kernel_generator(self, kg_values=None):
        '''
        Creates the class `soap` kernel generator object, to map
        local `soap` descriptors to a global descriptor. This kernel
        can be either an `average` or a `REMatch` kernel, depending
        on whether the user provides `average-soap` or `rematch-soap`
        as the fingerprint of choice. The difference between these kernels
        is as follows:

        - The `average` kernel assigns the global distance between
            two structures as the average of all possible local structural
            comparisons between the two structures. For instance, if there
            are 10 environments in structure B, then any local environment
            from structure A will be compared to all 10 environments in
            structure B, repeated for every local enviroment in structure
            A.

        - The `REMatch` kernel is an interpolation between a `best-match`
            comparison between structures, and an `average` comparison
            between structures. The `best-match` approach attempts to
            determine the best possible pairing of each local enviroment, and
            then use those distances as the global distance. It fails for
            structures which do not have the same number of enviroments, and
            the `REMatch` kernel corrects this problem using regularized
            entropy matching. For details, refer to this [paper](https://pubs
            .rsc.org/en/content/articlelanding/2016/CP/C6CP00415F).

        Arguments:

            kg_values (dict): containing user-defined values for
            some or all REMatch or Average kernel generator parameters.
        '''
        if kg_values is None:
            if self.label == "average-soap":
                self.kernel_gen = AverageKernel(
                    metric="linear", alpha=1, threshold=1e-6)
            elif self.label == "rematch-soap":
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

            if self.label == "average-soap":
                self.kernel_gen = AverageKernel(
                    metric=_metric, alpha=_alpha, threshold=_threshold)
            elif self.label == "rematch-soap":
                self.kernel_gen = REMatchKernel(
                    metric=_metric, alpha=_alpha, threshold=_threshold)

    def set_distance_calculator(self, distance_metric=None):
        '''
        Creates the `DistanceCalculator` object which will calculate the
        distance between models for the following fingerprints:

        - "Ewald Sum Matrix" (`ewald-sum-matrix`)
        - "Sine Matrix" (`sine-matrix`)
        - "Many Body Tensor Representation" (`mbtr`)
        '''
        if distance_metric is None:
            self.distance_calculator = DistanceCalculator("euclidean")
        else:
            self.distance_calculator = DistanceCalculator(distance_metric)

    def compare_fingerprints(self, test_model, ref_model):
        '''
        Compare the fingerprints between two structures. Each model
        contains a fingerprint dictionary, assigned using the
        `create_fingerprint` function, which has all the relevant
        information for the appropriate fingerprint. In the case of the
        Valle-Oganov fingerprint, this information is contained within
        an ASE Atoms structure. In the case of the bag-of-bonds
        fingerprint, this information is contained within a pair_cor
        dictionary. In the case of the REMatch or Average SOAP kernels,
        this information is contained within a set of normalized soap
        descriptors called normed_features.

        Returns a tuple of length 2, quantifying the fingerprint
        similarity. Only the `bag-of-bonds` comparison completely
        fills the tuple, all other comparisons only result in one
        similarity or distance metric.

        !!! note
            The order of the test_model and ref_models are irrelevant,
            the function will return the same value if models A and B are
            swapped.

        Arguments:

            test_model (obj): `structure_record.model()` A for the comparison

            ref_model (obj): `structure_record.model()` B for the comparison.

        Returns:

            (tuple): length 2 tuple which quantifies fingerprint similarity.
        '''
        if self.label == "valle-oganov":
            comp = self.comp
            return (comp._compare_structure(
                test_model.fingerprint["valle-oganov"],
                ref_model.fingerprint["valle-oganov"]),)

        elif self.label == "rematch-soap":
            kernel_gen = self.kernel_gen
            comparison = kernel_gen.create(
                [test_model.fingerprint["rematch-soap"],
                 ref_model.fingerprint["rematch-soap"]])
            distance = np.sqrt(
                comparison[0][0] + comparison[1][1] - 2*comparison[0][1])
            return (distance,)

        elif self.label == "average-soap":
            kernel_gen = self.kernel_gen
            comparison = kernel_gen.create(
                [test_model.fingerprint["average-soap"],
                 ref_model.fingerprint["average-soap"]])
            distance = np.sqrt(
                comparison[0][0] + comparison[1][1] - 2*comparison[0][1])
            return (distance,)

        elif self.label == "ewald-sum-matrix":
            distance_calculator = self.distance_calculator
            distance = distance_calculator.create(
                test_model.fingerprint["ewald-sum-matrix"],
                ref_model.fingerprint["ewald-sum-matrix"])
            return (distance,)

        elif self.label == "sine-matrix":
            distance_calculator = self.distance_calculator
            distance = distance_calculator.create(
                test_model.fingerprint["sine-matrix"],
                ref_model.fingerprint["sine-matrix"])
            return (distance,)

        elif self.label == "mbtr":
            distance_calculator = self.distance_calculator
            distance = distance_calculator.create(
                test_model.fingerprint["mbtr"],
                ref_model.fingerprint["mbtr"])
            print(f"Fingerprint distance: {distance}")
            return (distance,)

        elif self.label == "bag-of-bonds":
            pair_cor1 = test_model.fingerprint["bag-of-bonds"]
            pair_cor2 = ref_model.fingerprint["bag-of-bonds"]
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

    def assess_models_similarity(self, test_model, ref_model):
        '''
        Runs comparison of models, utilizing the appropriate tolerance
        parameters depending on the global fingerprint used.

        Return values:

        - 0 if the models are exactly same
        - 1 if the models are the same within tolerance
        - -1 if they are not within tolerance of each other.

        !!! note

            Test_model and ref_model are interchangeable, the same
            comparison result will be yielded if models A and B are swapped.

        Arguments:

            test_model (obj): structure_record.model() A for the comparison.

            ref_model (obj): structure_record.model() B for the comparison.

        Returns:

            (int): see above.
        '''
        try:
            comparison = self.compare_fingerprints(test_model, ref_model)
        except (AssertionError, KeyError):
            # models did not contain the same number of atoms (bag-of-bonds)
            return -1
        if self.label == "bag-of-bonds":
            if np.isclose(comparison[0], 0.0, atol=1e-5) and \
                    np.isclose(comparison[1], 0.0, atol=1e-5):
                return 0
            elif comparison[0] < self.tolerances["bag-of-bonds"][0] and \
                    comparison[1] < self.tolerances["bag-of-bonds"][1]:
                return 1
            else:
                return -1
        else:
            if np.isclose(comparison[0], 0.0, atol=1e-5):
                return 0
            elif comparison[0] < self.tolerances[self.label]:
                return 1
            else:
                return -1

    def create_fingerprint(self, model):
        '''
        Create the fingerprint for the model object. Automatically
        trims the model down to the active region if desired (according
        to `self.zbounds`), to save computational expense and make
        similarity checks more effective.

        Does not return anything. Instead, adds the fingerprint to the
        model's fingerprint dictionary. Currently, each model only ever
        has one fingerprint. However, it is possible to create every
        possible fingerprint for a given model, and run comparisons
        with every fingerprint.

        Arguments:

            model (obj): `structure_record.model()` which will be assigned a
            fingerprint.
        '''
        #
        model_astr = copy.deepcopy(model.astr)
        if self.zbounds is not None:
            site_removal_indices = []
            for site_index, site in enumerate(model_astr.sites):
                if site.coords[2] < self.zbounds[0] or \
                        site.coords[2] > self.zbounds[1]:
                    site_removal_indices.append(site_index)
            model_astr.remove_sites(site_removal_indices)

        if self.rem_vac:
            model_astr = self.remove_vacuum_in_cluster(model_astr)

        if self.label == "valle-oganov":
            ase_atoms = AseAtomsAdaptor.get_atoms(model_astr)
            fp, typedic = self.comp._take_fingerprints(ase_atoms)
            ase_atoms.info['fingerprints'] = self.comp._json_encode(
                fp, typedic)
            model.ase = ase_atoms
            model.fingerprint['valle-oganov'] = model.ase

        elif self.label == "average-soap":
            ase_atoms = AseAtomsAdaptor.get_atoms(model_astr)
            model.features = self.desc.create(ase_atoms)
            model.fingerprint['average-soap'] = model.features

        elif self.label == "rematch-soap":
            ase_atoms = AseAtomsAdaptor.get_atoms(model_astr)
            features = self.desc.create(ase_atoms)
            model.normed_features = normalize(features)
            model.fingerprint['rematch-soap'] = model.normed_features

        elif self.label == "sine-matrix":
            ase_atoms = AseAtomsAdaptor.get_atoms(model_astr)
            model.features = self.desc.create(ase_atoms)
            reshaped_features = model.features.reshape(-1, 1)
            normalized_features = normalize(reshaped_features)
            model.fingerprint["sine-matrix"] = model.features.reshape(1, -1)

        elif self.label == "ewald-sum-matrix":
            ase_atoms = AseAtomsAdaptor.get_atoms(model_astr)
            model.features = self.desc.create(ase_atoms)
            reshaped_features = self.features.reshape(-1, 1)
            normalized_features = normalize(reshaped_features)
            self.fingerprint["ewald-sum-matrix"] = normalized_features.reshape(
                1, -1)

        elif self.label == "mbtr":
            ase_atoms = AseAtomsAdaptor.get_atoms(model_astr)
            model.features = self.desc.create(ase_atoms)
            model.fingerprint["mbtr"] = model.features.reshape(1, -1)

        elif self.label == "hammer":
            nums = model_astr.atomic_numbers
            species = model_astr.types_of_specie
            num_species = len(species)

            # Create empty feature array
            model.features = np.zeros((model_astr.num_sites, 1 + num_species))
            species_indexing = {species: index +
                                1 for index, species in enumerate(species)}
            model.features[:, 0] = nums

            # Fingerprint is of the form: [Z, rho_A, rho_B]
            # Need to assemble rho terms
            # Each rho value is defined as:
            # exp(-r/λ)*cutoff(r)
            # and cutoff(r) = 1/2*cos(π*r/r_c) + 1/2 if r<r_c (0 otherwise)
            lattice = model_astr.lattice
            λ = 1.0
            r_cut = 11.9
            for col_index_one, site_one in enumerate(model_astr.sites[:-1]):
                coords_one = site_one.coords
                for col_index_prime, site_two in enumerate(
                        model_astr.sites[col_index_one + 1:]):
                    coords_two = site_two.coords
                    col_index_two = col_index_one + col_index_prime + 1
                    r = dc.dist_pbc(coords_one, coords_two, lattice)
                    factor = np.exp(-r/λ)*cutoff(r, r_cut)

                    # Need to assign rho value based on specie of other site
                    row_index_one = species_indexing[site_two.specie]
                    row_index_two = species_indexing[site_one.specie]

                    self.features[col_index_one, row_index_one] += factor
                    self.features[col_index_two, row_index_two] += factor

            model.fingerprint["hammer"] = self.features

        elif self.label == "bag-of-bonds":
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
                        / (2*len(model_astr))

                    pair_cor[str(specie1) + "-" + str(specie2)
                             ] = (norm_factor, np.array(dists))
            model.fingerprint['bag-of-bonds'] = pair_cor

    def check_model_uniqueness(self, model, all_models, exact=False):
        '''
        Check whether a model is unique.

        Arguments:

            model (obj): the `structure_record.model()` for which uniqueness
             is being tested.

            exact (boolean): if `True`, models are considered unique if they
             are not exactly the same as another model. If `False`, models
             are considered unique if they are not the same as another model
             within tolerance limits.

        Returns:

            boolean: True if the model is unique, False if the model is
             the same (or "similar" if `exact` is False) as another model
        '''
        # create a new fingerprint got the model
        # either before (new_model) or after relaxation (relaxed_astr)
        self.create_fingerprint(model)
        # compare
        flags = [self.assess_models_similarity(model, m) for m in all_models]
        if exact:
            same = [f == 0 for f in flags]
        else:
            same = [f >= 0 for f in flags]
        if any(same):
            return False
        else:
            return True

    def remove_vacuum_in_cluster(self, astr):
        """
        Checks if there is vacuum padding in any of the three directions and
        then removes it leaving a 2Å thickness in each direction.

        Args:

            astr (obj): Pymatgen structure object
        """
        xcarts, ycarts, zcarts = astr.cart_coords.T
        xthick = xcarts.max() - xcarts.min()
        ythick = ycarts.max() - ycarts.min()
        zthick = zcarts.max() - zcarts.min()

        newa, newb, newc = xthick+2, ythick+2, zthick+2
        new_latt = Lattice([[newa, 0, 0], [0, newb, 0], [0, 0, newc]])

        new_xcarts = xcarts - xcarts.min() + 1
        new_ycarts = ycarts - ycarts.min() + 1
        new_zcarts = zcarts - zcarts.min() + 1
        new_carts = np.array([new_xcarts, new_ycarts, new_zcarts]).T

        fp_astr = Structure(new_latt, astr.species, new_carts,
                            coords_are_cartesian=True)
        return fp_astr
