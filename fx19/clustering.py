import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as ch
import matplotlib.colors as col
import os
import imageio
from collections import Counter


class hierarchical_clusterer(object):
    def __init__(self, params, xsim):
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
        self.num_items = len(models)
        self.create_distance_matrix(models)
        return self.assign_clusters()

    def update_max_clusters(self, max_clusters):
        if max_clusters < self.min_clusters:
            print("Cannot make max_clusters smaller than assigned min_clusters value.")
        else:
            self.max_clusters = max_clusters

    def read_in_objective_functions(self, file):
        '''
        Method to read in objective functions for the models from the data_file.
        Args:
        file (string) - filename of the data_file which contains the objective
                        function values for each model.
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
        structures in fingerprint space. For instance, when looking at the
        bag-of-bonds descriptor, the matrix contains the structure similarity values
        for every pair of structures.

        Args:
        home_directory (string) - directory to read in models from.
        descriptor (string) - type of fingerprint descriptor to use to construct
                            distance matrix
        structure_type (string) - whether structures will be relaxed or unrelaxed
        paramDict (dictionary) - dictionary of various fingerprint parameters.
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
        Edit distance matrix by removing the row and column corresponding to the removed model,
        and adding a new row and column corresponding to the new model.
        '''
        print(f"Previous distance matrix: {self.distance_matrix}")
        print(
            f"Size of previous distance matrix: {self.distance_matrix.shape}")
        print(f"Sorted labels: {self.sorted_labels}")
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
        print(f"New distance matrix: {self.distance_matrix}")
        print(
            f"Size of new distance matrix: {self.distance_matrix.shape}")
        self.sorted_labels

        self.sorted_labels.append(added_model.label)
        self.sorted_models.append(added_model)

    def assign_clusters(self):
        '''
        Assign clusters using hierarchical clustering based on distance matrix
        '''
        # Create linkage array
        linkage = ch.linkage(self.distance_matrix, method=self.linkage_method)

        # Create clusters. Use previously calculated cutoffs as rough starting points to save some time
        self.inconsistency_cutoff -= 0.5
        self.distance_cutoff -= 10.0
        max_cluster_index = self.max_clusters + 1
        occupation = self.min_cluster_occupancy
        if self.cutoff_type == "inconsistent":
            while max_cluster_index > self.max_clusters or occupation < self.min_cluster_occupancy:
                clusters = ch.fcluster(linkage, self.inconsistency_cutoff,
                                       criterion='inconsistent', depth=30)
                self.inconsistency_cutoff += 0.1
                max_cluster_index = np.amax(clusters)
                if self.inconsistency_cutoff > self.max_incons_cutoff or max_cluster_index == self.min_clusters:
                    print("Unable to reach desired number of clusters or occupancy!")
                    break
                frequency_dict = Counter(clusters)
                frequencies = list(frequency_dict.values())
                occupation = np.min(frequencies)
        elif self.cutoff_type == "distance":
            while max_cluster_index > self.max_clusters or occupation < self.min_cluster_occupancy:
                # clusters = ch.fcluster(
                #     linkage, t=self.max_clusters, criterion="maxclust")
                clusters = ch.fcluster(linkage, self.distance_cutoff,
                                       criterion='distance')
                self.distance_cutoff += 0.5
                max_cluster_index = np.amax(clusters)
                if self.distance_cutoff > self.max_dist_cutoff or max_cluster_index == self.min_clusters:
                    print("Unable to reach desired number of clusters or occupancy!")
                    break
                frequency_dict = Counter(clusters)
                frequencies = list(frequency_dict.values())
                occupation = np.min(frequencies)
        elif self.cutoff_type == "maxclust_monocrit":
            R = ch.inconsistent(linkage, d=30)
            MI = ch.maxinconsts(linkage, R)
            clusters = ch.fcluster(
                linkage, t=self.max_clusters, criterion=self.cutoff_type, monocrit=MI)
            max_cluster_index = np.amax(clusters)
            frequency_dict = Counter(clusters)
            frequencies = list(frequency_dict.values())
            occupation = np.min(frequencies)

        self.num_clusters = max_cluster_index
        print(f"Num of clusters: {self.num_clusters}")
        # Now assign lists of models to clusters
        # cluster_models = {i: [] for i in range(1, max_cluster_index + 1)}
        # outside_cluster_models = {i: []
        #                           for i in range(1, max_cluster_index + 1)}
        # for index, cluster in enumerate(clusters):
        #     model = self.sorted_models[index]
        #     model.cluster = cluster
        #     if index not in cluster_models:
        #         cluster_models[index] = [model]
        #     else:
        #         cluster_models[index].append(model)
        #     for key in outside_cluster_models.keys():
        #         if key != cluster:
        #             if key not in outside_cluster_models:
        #                 outside_cluster_models[key] = [model]
        #             else:
        #                 outside_cluster_models[key].append(model)
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
        print(f"All clusters: {clusters}")
        print(f"Multi model clusters: {multi_model_clusters}")
        return cluster_models, multi_model_clusters

    def update_clustering(self, new_model, old_model):
        '''
        Update clustering by taking out the old model and adding
        the new model
        '''
        self.edit_distance_matrix(new_model, old_model)
        return self.assign_clusters()

    def calculate_clustering(self, labels, obj_fncs):
        '''
        Method to calculate clustering of structures, and create plots (of both
        the objective functions and the dendrogram of the hierarchy)
        '''
        file_label = self.cutoff_type
        if labels is None:
            actual_labels = np.arange(1, len(self.distance_matrix)+1, 1)
        else:
            actual_labels = labels

        linkage = ch.linkage(self.distance_matrix, method=self.linkage_method)

        print("Created linkage array.")

        inconsistency_cutoff = 1.0
        distance_cutoff = 1.0
        max_cluster_index = self.max_clusters + 1
        occupation = self.min_cluster_occupancy
        if self.cutoff_type == "inconsistent":
            while max_cluster_index > self.max_clusters or max_cluster_index < self.min_clusters or occupation < self.min_cluster_occupancy:
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
            while max_cluster_index > self.max_clusters or max_cluster_index < self.min_clusters or occupation < self.min_cluster_occupancy:
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
                linkage, t=self.max_clusters, criterion=self.cutoff_type, monocrit=MI)
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
        last_cluster = np.amax(clusters)
        leaf_colors = {}
        for n, i in enumerate(clusters):
            leaf_colors[n] = hex_colors[(i-1) % 20]
            # if i == last_cluster:
            #     leaf_colors[n] = dflt_col
        link_cols = {}
        for i, i12 in enumerate(linkage[:, :2].astype(int)):
            c1, c2 = (link_cols[x] if x > len(linkage) else leaf_colors[x]
                      for x in i12)
            link_cols[i+1+len(linkage)] = c1 if c1 == c2 else dflt_col

        fig, axes = plt.subplots(1, 1, gridspec_kw={"hspace": 0.5})
        fig.set_size_inches(14, 10)
        # If labeling, use labels = actual_labels
        ch.dendrogram(linkage, labels=None, ax=axes, leaf_font_size=8, color_threshold=None, above_threshold_color='y',
                      orientation='top', link_color_func=lambda x: link_cols[x])
        axes.set_ylabel(r"SSIM Distance", fontsize=16)
        axes.set_ylim([0, 200.0])
        plt.title(
            r"Al/Al$_2$O$_3$ Grain Boundary Dendrogram", fontsize=24)
        folder = "/mnt/c/Users/dunru/GitHub/fantastx/epsilon_selection_Al2O3/"
        plt.savefig(folder + "Al2O3_SSIM_cluster_dendrogram_" + file_label + ".png",
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
            except:
                print("Error! Can't find label: ", label)
                continue

        # Plot data for each cluster
        fig, axes = plt.subplots(1, 1, gridspec_kw={"hspace": 0.5})
        fig.set_size_inches(14, 10)
        for key in sorted(cluster_properties.keys()):
            cluster_label = "Cluster " + str(key)
            # if key == last_cluster:
            #     cluster_label = "Unclustered (cluster " + str(key) + ")"
            cluster = cluster_properties[key]
            axes.scatter(cluster["Formation Energy"], cluster["Exp"],
                         s=10, marker="o", c=cluster["Color"], label=cluster_label)
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
        #                  s=100, marker="o", c=cluster["Color"], label=cluster_label)
        #     axes.set_ylabel("STEM", fontsize=16)
        #     axes.set_xlabel(
        #         "Energy", fontsize=16)
        #     axes.set_xlim((min_x - 1, max_x + 1))
        #     axes.set_ylim((min_y - 0.1, max_y + 0.1))
        #     plt.setp(axes.get_xticklabels(), fontsize=13)
        #     plt.setp(axes.get_yticklabels(), fontsize=13)
        #     plt.legend(fontsize=16)
        #     plt.title(
        #         r"Clustering of Al/Al$_2$O$_3$ Grain Boundaries", fontsize=24)
        #     folder = "/mnt/c/Users/dunru/GitHub/fantastx/epsilon_selection_Al2O3/"
        #     filename = folder + "Al2O3_SSIM_clustering_" + \
        #         file_label + "_" + str(index) + ".png"
        #     plt.savefig(filename, format="png", dpi=300)
        #     plt.close()

        #     index += 1

        #     for i in np.arange(0, n_frames+1):
        #         gif_filenames.append(filename)

        # # assemble gif
        # print("Charts saved. Building gif.")
        # gif_filename = folder + "Al2O3_SSIM_clustering_" + file_label + ".gif"
        # with imageio.get_writer(gif_filename, mode="I") as writer:
        #     for filename in gif_filenames:
        #         image = imageio.imread(filename)
        #         writer.append_data(image)
        # print("Gif saved.")

        return (linkage, clusters)


class compositional_clusterer(object):
    def __init__(self):
        self.clusters = {}
        self.multi_model_clusters = []
        self.type = "compositional"

    def initialize_clusters(self, models):
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
        Remove the model, editing multi_model_clusters if the
        cluster the model was part of no longer contains at least 2
        models.
        Also update the cluster ranking of each model. 
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

    def update_clustering(self, model_add, model_remove, cluster_model_levels=None):
        self.append_model(model_add)
        self.remove_model(model_remove, cluster_model_levels)
        return self.clusters, self.multi_model_clusters
