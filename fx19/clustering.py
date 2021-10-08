import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as ch
import matplotlib.colors as col
from collections import Counter


class hierarchical_clusterer(object):
    '''
    Cluster object which performs hierarchical clustering. Contains
    functions to read in models, cluster them into flat clusters based
    on similarity metrics (the distances between models in fingerprint
    space) and the user specified clustering method, and return those
    clusters for use in ML or GA applications.
    '''

    def __init__(self, params, xsim):
        '''
        Args:

        params (dictionary): contains all user-specified parameters.
        Must include the linkage method, cutoff type, and max number of
        clusters.

        xsim (experimental_simulation object): currently hierarchical
        clustering is implemented using the STEM SSIM comparison score
        between models as the distance metric, using the xsim object.
        '''
        self.linkage_method = params["linkage_method"]
        self.cutoff_type = params["cutoff_type"]
        self.max_clusters = params["max_clusters"]
        self.min_clusters = params["min_clusters"]
        self.max_incons_cutoff = params["max_incons_cutoff"]
        self.max_dist_cutoff = params["max_dist_cutoff"]
        self.min_cluster_occupancy = params["min_cluster_occupancy"]
        self.xsim = xsim
        self.num_items = 0
        self.num_clusters = 0

        # starting values for cutoffs
        self.inconsistency_cutoff = 1.5
        self.distance_cutoff = 10.5

        self.type = "hierarchical"

    def initialize_clusters(self, models):
        '''
        Function to initialize the cluster object. Name matches the
        same function in compositional_clusterer. 

        Args:

        models (list of model objects): the models which will be
        clustered initially.
        '''
        self.num_items = len(models)
        self.create_distance_matrix(models)
        return self.assign_clusters()

    def update_max_clusters(self, max_clusters):
        '''
        Function to update the max number of flat clusters which be
        allowed to be created.

        Args:

        max_clusters (int): the new number of max clusters
        '''
        if max_clusters < self.min_clusters:
            print("Cannot make max_clusters smaller than assigned "
                  + "min_clusters value.")
        else:
            self.max_clusters = max_clusters

    def read_in_objective_functions(self, file):
        '''
        Method to read in objective functions for the models from the
        data_file.

        Args:

        file (string): filename of the data_file which contains the
        objective function values for each model.
        '''
        lines = open(file, "r").read().splitlines()
        index = 0
        obj_fncs = {}
        for line in lines:
            newline = line.split()
            if index > 0 and newline != []:
                if len(newline) == 5:
                    # model derived from one parent
                    obj_fncs[int(newline[0])] = [float(
                        newline[2]), float(newline[3]), float(newline[4])]
                elif len(newline) == 6:
                    # model derived from two parents
                    obj_fncs[int(newline[0])] = [float(
                        newline[3]), float(newline[4]), float(newline[5])]
                else:
                    print("Bad line length!")
            index += 1
        return obj_fncs

    def create_distance_matrix(self, models):
        '''
        Method to create the matrix which defines the distances between
        structures in fingerprint space. For instance, when looking at
        the bag-of-bonds descriptor, the matrix contains the structure
        similarity values for every pair of structures.

        Args:

        models (list of model objs): The models which will be used to
        create the distance matrix.
        '''
        labels = [model.label for model in models]
        sort_args = np.argsort(labels)
        self.sorted_models = np.array(models)[sort_args].tolist()
        self.sorted_labels = np.array(labels)[sort_args].tolist()
        # compare all models to all other models
        n_models = len(models)
        max_ssim = 0
        min_ssim = 1
        self.distance_matrix = np.ones((n_models, n_models))
        for test_label, test_model in enumerate(self.sorted_models[:-1]):
            comparison_models = self.sorted_models[test_label+1:]
            self.distance_matrix[test_label][test_label] = 0.0
            for j, other_model in enumerate(comparison_models):
                other_label = test_label + j + 1
                try:
                    comparison = self.xsim.evaluate_obj_two_models(
                        test_model, other_model)
                    self.distance_matrix[test_label][other_label] = comparison
                    self.distance_matrix[other_label][test_label] = comparison
                    if comparison > max_ssim:
                        max_ssim = comparison
                    if comparison < min_ssim:
                        min_ssim = comparison
                except:
                    self.distance_matrix[test_label][other_label] = 100.0
                    self.distance_matrix[other_label][test_label] = 100.0
                    continue
        print("Max initial cluster ssim: ", max_ssim)
        print("Min initial cluster ssim: ", min_ssim)

    def edit_distance_matrix(self, added_model, removed_model):
        '''
        Edit distance matrix by removing the row and column corresponding
        to the removed model, and adding a new row and column corresponding
        to the new model.

        Args:

        added_model (obj): structure_record.model() which is being added.

        removed_model (obj): structure_record.model() which is being removed.
        '''
        removed_index = self.sorted_labels.index(removed_model.label)
        self.sorted_labels.pop(removed_index)
        self.sorted_models.pop(removed_index)
        # removed_index = np.where(self.sorted_labels == removed_model.label)
        # self.sorted_labels = np.delete(self.sorted_labels, removed_index)
        # self.sorted_models = np.delete(self.sorted_models, removed_index)
        rm_row_dm = np.delete(self.distance_matrix, removed_index, axis=0)
        rm_col_dm = np.delete(rm_row_dm, removed_index, axis=1)

        new_col = np.zeros(self.num_items - 1)
        new_row = np.zeros(self.num_items)
        for comp_label, comp_model in enumerate(self.sorted_models):
            try:
                comparison = self.xsim.evaluate_obj_two_models(
                    comp_model, added_model)
                new_col[comp_label] = comparison
                new_row[comp_label] = comparison
            except:
                new_col[comp_label] = 100.0
                new_row[comp_label] = 100.0
                continue

        stacked_col = np.column_stack((rm_col_dm, new_col))
        self.distance_matrix = np.vstack((stacked_col, new_row))
        self.sorted_labels

        self.sorted_labels.append(added_model.label)
        self.sorted_models.append(added_model)

    def assign_clusters(self):
        '''
        Assign clusters using hierarchical clustering based on the
        distance matrix which was previously calculated. 

        Returns cluster_models, a (dictionary) which contains the
        clusters as keys, and a list of the models belonging to each
        cluster as the values; also returns multi_model_clusters, a
        list which contains the keys for each cluster which contains
        more than one model.
        '''
        # Create linkage array
        linkage = ch.linkage(self.distance_matrix, method=self.linkage_method)

        # Create clusters. Use previously calculated cutoffs as rough starting
        # points to save some time
        self.inconsistency_cutoff -= 0.5
        self.distance_cutoff -= 10.0
        max_cluster_index = self.max_clusters + 1
        occupation = self.min_cluster_occupancy
        if self.cutoff_type == "inconsistent":
            while max_cluster_index > self.max_clusters \
                    or occupation < self.min_cluster_occupancy:
                clusters = ch.fcluster(linkage, self.inconsistency_cutoff,
                                       criterion='inconsistent', depth=30)
                self.inconsistency_cutoff += 0.1
                max_cluster_index = np.amax(clusters)
                if self.inconsistency_cutoff > self.max_incons_cutoff \
                        or max_cluster_index == self.min_clusters:
                    print("Unable to reach desired number of clusters "
                          + "or occupancy!")
                    break
                frequency_dict = Counter(clusters)
                frequencies = list(frequency_dict.values())
                occupation = np.min(frequencies)
        elif self.cutoff_type == "distance":
            while max_cluster_index > self.max_clusters \
                    or occupation < self.min_cluster_occupancy:
                # clusters = ch.fcluster(
                #     linkage, t=self.max_clusters, criterion="maxclust")
                clusters = ch.fcluster(linkage, self.distance_cutoff,
                                       criterion='distance')
                self.distance_cutoff += 0.5
                max_cluster_index = np.amax(clusters)
                if self.distance_cutoff > self.max_dist_cutoff \
                        or max_cluster_index == self.min_clusters:
                    print("Unable to reach desired number of clusters "
                          + "or occupancy!")
                    break
                frequency_dict = Counter(clusters)
                frequencies = list(frequency_dict.values())
                occupation = np.min(frequencies)
        elif self.cutoff_type == "maxclust_monocrit":
            R = ch.inconsistent(linkage, d=30)
            MI = ch.maxinconsts(linkage, R)
            clusters = ch.fcluster(
                linkage, t=self.max_clusters,
                criterion=self.cutoff_type, monocrit=MI)
            max_cluster_index = np.amax(clusters)
            frequency_dict = Counter(clusters)
            frequencies = list(frequency_dict.values())
            occupation = np.min(frequencies)

        self.num_clusters = max_cluster_index

        cluster_models = {i: [] for i in range(1, max_cluster_index + 1)}
        multi_model_clusters = []
        for index, cluster in enumerate(clusters):
            model = self.sorted_models[index]
            model.cluster = cluster
            if cluster not in cluster_models:
                cluster_models[cluster] = [model]
            else:
                cluster_models[cluster].append(model)
                if len(cluster_models[cluster]) == 2:
                    multi_model_clusters.append(cluster)
        return cluster_models, multi_model_clusters

    def update_clustering(self, new_model, old_model):
        '''
        A method to update the clustering by taking out the old model
        and adding the new model. 

        Returns the cluster_models and multi_model_clusters calculated
        by the assign_clusters() function. 

        Args:

        new_model (obj): the new structure_record.model() to add

        old_model (obj): the old structure_record.model() to remove
        '''
        self.edit_distance_matrix(new_model, old_model)
        return self.assign_clusters()

    def calculate_clustering(self, labels, obj_fncs):
        '''
        Method to calculate hierarchical clustering of structures.
        Will create a cluster dendrogram, as well as a visualization
        of the clusters in objective function space.

        Args:

        labels (list): the labels of each model contained in the
        distance matrix

        obj_fncs (dictionary): maps each label to the set of objective
        functions for that model.
        '''
        file_label = self.cutoff_type
        if labels is None:
            actual_labels = np.arange(1, len(self.distance_matrix)+1, 1)
        else:
            actual_labels = labels

        linkage = ch.linkage(self.distance_matrix, method=self.linkage_method)

        # print("Created linkage array.")

        inconsistency_cutoff = 1.0
        distance_cutoff = 1.0
        max_cluster_index = self.max_clusters + 1
        occupation = self.min_cluster_occupancy
        if self.cutoff_type == "inconsistent":
            while max_cluster_index > self.max_clusters \
                or max_cluster_index < self.min_clusters \
                    or occupation < self.min_cluster_occupancy:
                clusters = ch.fcluster(linkage, inconsistency_cutoff,
                                       criterion='inconsistent', depth=30)
                inconsistency_cutoff += 0.1
                if inconsistency_cutoff > self.max_incons_cutoff:
                    print("Unable to reach desired number of clusters")
                    break
                max_cluster_index = np.amax(clusters)

                frequency_dict = Counter(clusters)
                frequencies = list(frequency_dict.values())
                occupation = np.min(frequencies)
        elif self.cutoff_type == "distance":
            while max_cluster_index > self.max_clusters \
                or max_cluster_index < self.min_clusters \
                    or occupation < self.min_cluster_occupancy:
                # clusters = ch.fcluster(
                #     linkage, t=self.max_clusters, criterion="maxclust")
                clusters = ch.fcluster(linkage, distance_cutoff,
                                       criterion='distance')
                distance_cutoff += 0.5
                if distance_cutoff > self.max_dist_cutoff:
                    print("Unable to reach desired number of clusters")
                    break
                max_cluster_index = np.amax(clusters)
                frequency_dict = Counter(clusters)
                frequencies = list(frequency_dict.values())
                occupation = np.min(frequencies)
        elif self.cutoff_type == "maxclust_monocrit":
            R = ch.inconsistent(linkage, d=30)
            MI = ch.maxinconsts(linkage, R)
            clusters = ch.fcluster(
                linkage, t=self.max_clusters,
                criterion=self.cutoff_type, monocrit=MI)
            max_cluster_index = np.amax(clusters)
            frequency_dict = Counter(clusters)
            frequencies = list(frequency_dict.values())
            occupation = np.min(frequencies)

        print("Assigned clusters.")

        # Create the dendrogram
        dflt_col = "black"   # Unclustered gray
        # Use the tab20 colormap to assign colors
        colors = [i for i in plt.get_cmap('tab20').colors]
        hex_colors = [col.rgb2hex(i) for i in colors]
        leaf_colors = {}
        for n, i in enumerate(clusters):
            leaf_colors[n] = hex_colors[(i-1) % 20]
        link_cols = {}
        for i, i12 in enumerate(linkage[:, :2].astype(int)):
            c1, c2 = (link_cols[x] if x > len(linkage) else leaf_colors[x]
                      for x in i12)
            link_cols[i+1+len(linkage)] = c1 if c1 == c2 else dflt_col

        fig, axes = plt.subplots(1, 1, gridspec_kw={"hspace": 0.5})
        fig.set_size_inches(14, 10)
        # If labeling, use labels = actual_labels
        ch.dendrogram(linkage, labels=None, ax=axes, leaf_font_size=8,
                      color_threshold=None, above_threshold_color='y',
                      orientation='top',
                      link_color_func=lambda x: link_cols[x])
        axes.set_ylabel(r"SSIM Distance", fontsize=16)
        axes.set_ylim([0, 200.0])
        plt.title(
            r"Al/Al$_2$O$_3$ Grain Boundary Dendrogram", fontsize=24)
        folder = "/mnt/c/Users/dunru/GitHub/fantastx/epsilon_selection_Al2O3/"
        plt.savefig(folder + "Al2O3_SSIM_cluster_dendrogram_" + file_label
                    + ".png",
                    format="png", dpi=300)
        plt.show()

        # plot clustering on multi-objective plot
        cluster_properties = {}
        min_x = 0
        max_x = 0
        min_y = 0
        max_y = 0
        min_index = 100

        # Collect data to plot
        for index, label in enumerate(actual_labels):
            try:
                cluster = clusters[index]
                total_energy = obj_fncs[label][0]
                formation_energy = obj_fncs[label][1]
                exp = obj_fncs[label][2]
                color = leaf_colors[index]
                if cluster in cluster_properties:
                    cluster_properties[cluster]["Formation Energy"].append(
                        formation_energy)
                    cluster_properties[cluster]["Total Energy"].append(
                        total_energy)
                    cluster_properties[cluster]["Exp"].append(exp)
                    cluster_properties[cluster]["Color"].append(color)
                else:
                    cluster_properties[cluster] = {}
                    cluster_properties[cluster]["Formation Energy"] = [
                        formation_energy]
                    cluster_properties[cluster]["Total Energy"] = [
                        total_energy]
                    cluster_properties[cluster]["Exp"] = [exp]
                    cluster_properties[cluster]["Color"] = [color]

                if index < min_index:
                    min_x = formation_energy
                    max_x = formation_energy
                    min_y = exp
                    max_y = exp
                    min_index = index
                else:
                    if formation_energy < min_x:
                        min_x = formation_energy
                    if formation_energy > max_x:
                        max_x = formation_energy
                    if exp < min_y:
                        min_y = exp
                    if exp > max_y:
                        max_y = exp
            except KeyError:
                print("Error! Can't find label: ", label)
                continue

        # Plot data for each cluster
        fig, axes = plt.subplots(1, 1, gridspec_kw={"hspace": 0.5})
        fig.set_size_inches(14, 10)
        for key in sorted(cluster_properties.keys()):
            cluster_label = "Cluster " + str(key)
            cluster = cluster_properties[key]
            axes.scatter(cluster["Formation Energy"], cluster["Exp"],
                         s=10, marker="o", c=cluster["Color"],
                         label=cluster_label)
            index += 1

        axes.set_ylabel("STEM", fontsize=16)
        axes.set_xlabel(
            "Energy", fontsize=16)
        axes.set_xlim((min_x - 1, max_x + 1))
        axes.set_ylim((min_y - 0.1, max_y + 0.1))
        plt.setp(axes.get_xticklabels(), fontsize=13)
        plt.setp(axes.get_yticklabels(), fontsize=13)
        plt.legend(fontsize=16)
        plt.title(
            r"Clustering of Al/Al$_2$O$_3$ Grain Boundaries", fontsize=24)
        folder = "/mnt/c/Users/dunru/GitHub/fantastx/epsilon_selection_Al2O3/"
        plt.savefig(folder + "Al2O3_SSIM_clustering_" + file_label + ".png",
                    format="png", dpi=300)
        plt.close()

        # # Make a gif
        # gif_filenames = []
        # index = 0

        # # frames between transitions
        # n_frames = 6

        # # Plot data for each cluster
        # for key in sorted(cluster_properties.keys()):
        #     cluster_label = "Cluster " + str(key)
        #     # if key == last_cluster:
        #     #     cluster_label = "Unclustered (cluster " + str(key) + ")"
        #     cluster = cluster_properties[key]
        #     fig, axes = plt.subplots(1, 1, gridspec_kw={"hspace": 0.5})
        #     fig.set_size_inches(14, 10)
        #     axes.scatter(cluster["Formation Energy"], cluster["Exp"],
        #                  s=100, marker="o", c=cluster["Color"],
        #                  label=cluster_label)
        #     axes.set_ylabel("STEM", fontsize=16)
        #     axes.set_xlabel(
        #         "Energy", fontsize=16)
        #     axes.set_xlim((min_x - 1, max_x + 1))
        #     axes.set_ylim((min_y - 0.1, max_y + 0.1))
        #     plt.setp(axes.get_xticklabels(), fontsize=13)
        #     plt.setp(axes.get_yticklabels(), fontsize=13)
        #     plt.legend(fontsize=16)
        #     plt.title(
        #         r"Clustering of Al/Al$_2$O$_3$ Grain Boundaries",
        #         fontsize=24)
        #     folder = \
        #     "/mnt/c/Users/dunru/GitHub/fantastx/epsilon_selection_Al2O3/"
        #     filename = folder + "Al2O3_SSIM_clustering_" + \
        #         file_label + "_" + str(index) + ".png"
        #     plt.savefig(filename, format="png", dpi=300)
        #     plt.close()

        #     index += 1

        #     for i in np.arange(0, n_frames+1):
        #         gif_filenames.append(filename)

        # # assemble gif
        # print("Charts saved. Building gif.")
        # gif_filename = folder + "Al2O3_SSIM_clustering_" + file_label
        #                + ".gif"
        # with imageio.get_writer(gif_filename, mode="I") as writer:
        #     for filename in gif_filenames:
        #         image = imageio.imread(filename)
        #         writer.append_data(image)
        # print("Gif saved.")

        return (linkage, clusters)


class compositional_clusterer(object):
    '''
    Cluster object which performs compositional clustering. Simpler
    than hierarchical clustering, here models are grouped purely based
    on their atomic composition.

    Contains functions to read in models, assign clusters, and update
    clusters by removing and adding models.
    '''

    def __init__(self):
        self.clusters = {}
        self.multi_model_clusters = []
        self.type = "compositional"

    def initialize_clusters(self, models):
        '''
        Function to initialize the clusters.

        Args:

        models (list of model objs): the models which will be used to
        initially assign clusters.
        '''
        for model in models:
            composition = model.astr.composition.to_pretty_string()
            if composition in self.clusters:
                if len(self.clusters[composition]) == 1:
                    self.multi_model_clusters.append(composition)
                self.clusters[composition].append(model)
            else:
                self.clusters[composition] = [model]
            model.cluster = composition
        return self.clusters, self.multi_model_clusters

    def append_model(self, model):
        '''
        Function to append a new model to the clusters. Unlike
        hierarchical clustering, here assigning a new model to a cluster
        does not require updating the cluster assignments of all other
        models.

        Args:

        model (obj): the structure_record.model() which will be added.
        '''
        composition = model.astr.composition.to_pretty_string()
        if composition in self.clusters:
            if len(self.clusters[composition]) == 1:
                self.multi_model_clusters.append(composition)
            self.clusters[composition].append(model)
        else:
            self.clusters[composition] = [model]
        model.cluster = composition

        return self.clusters, self.multi_model_clusters

    def remove_model(self, model, cluster_model_levels):
        '''
        A function to remove a model. Here not only is the cluster
        itself updated, but the non-domination ranking of the
        cluster is also edited to be efficient.

        Returns the clusters dictionary which contains the cluster
        composition as keys and a list of the models in the cluster as
        values; returns multi_model_clusters, a list which contains the
        keys of each cluster which contains more than one model; and
        returns update_levels, a boolean which indicates whether the
        cluster level object will need further updating inside the
        selection algorithm. This occurs if the model was not removed
        from the tail of the cluster levels object.

        Args:

        model (obj): the structure_record.model() to be removed from the
        clusters. Must be a model currently contained in the cluster!

        cluster_model_levels (dictionary): the non-domination ranking
        of every cluster.
        '''
        update_levels = False
        composition = model.astr.composition.to_pretty_string()
        if len(self.clusters[composition]) == 1:
            self.clusters.pop(composition)
            if cluster_model_levels is not None:
                cluster_model_levels.pop(composition)
        else:
            if len(self.clusters[composition]) == 2:
                self.multi_model_clusters.remove(composition)
            self.clusters[composition].remove(model)
            # model may not be at end of cluster structure
            if cluster_model_levels is not None:
                for i in range(1, len(cluster_model_levels[composition]) + 1):
                    if model in cluster_model_levels[composition][-i]:
                        if len(cluster_model_levels[composition][-i]) == 1:
                            cluster_model_levels[composition].pop(-i)
                            if i != 1:
                                update_levels = True
                        else:
                            cluster_model_levels[composition][-i].remove(model)
                            if i != 1:
                                update_levels = True
                        break

        return self.clusters, self.multi_model_clusters, update_levels

    def update_clustering(self, model_add, model_remove,
                          cluster_model_levels=None):
        '''
        Function to update the clustering by both adding and removing
        models.

        For args, see append_model and remove_model.
        '''
        self.append_model(model_add)
        self.clusters, self.multi_model_clusters, update_levels = \
            self.remove_model(model_remove, cluster_model_levels)
        return self.clusters, self.multi_model_clusters, update_levels
