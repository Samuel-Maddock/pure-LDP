from collections import Counter, OrderedDict

from pure_ldp.simulations.helpers.PureFOSimulation import PureSimulation

import numpy as np
import pandas as pd
import copy
import os, uuid
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import os
from collections import Counter
import math
import copy

class FrequencyOracleSimulation:
    def __init__(self, data, experiment_title, large_domain=False, calc_top_k=0, display_full_stats=False,
                 autosave=False):

        self.n = len(data)
        self.data = data

        self.experiment_title = experiment_title
        self.domain = np.arange(start=min(self.data), stop=max(self.data) + 1)
        self.bins = self.domain

        self.stats = []
        self.display_full_stats = display_full_stats
        self.autosave = autosave
        self.experiment_plot_data = []

        if len(self.bins) >= 8000:
            large_domain = True

        # Used for displaying data when the domain is large
        self.large_domain = large_domain
        self.data_counter = Counter(self.data)
        self.y_map = np.vectorize(lambda x: self.data_counter.get(x, 0))

        # Used to calculate statistics from the top k most frequent values
        # When 0 the stats are calculated from the whole dataset
        # Should be used when the domain is large but data is relatively small
        self.calc_top_k = calc_top_k

        self.name = "Frequency Oracle Simulation"
        self.reset()

    def reset(self):
        self.experiment_plot_data = []
        self.id = str(uuid.uuid4())
        self.plot_directory = "./plots/"
        self.stats_filename = self.plot_directory + "metrics/" + self.id + ".csv"
        self.stat_cache = {}

    def run_and_plot(self, experiment_list, display_stats_only=False):
        self._run(experiment_list)
        if display_stats_only:
            if self.autosave is False:
                self._save_experiment_stats()
        else:
            self._plot()

    def _run(self, experiment_list):
        for i in range(0, len(experiment_list)):
            experiment_name = experiment_list[i][0]
            custom_name = experiment_name
            params = experiment_list[i][1]

            if isinstance(experiment_name, tuple):
                experiment_name, custom_name = (
                experiment_name[0], experiment_name[0] + " " + experiment_name[1]) if len(experiment_name) == 2 \
                    else (experiment_name[0], experiment_name[1] + ": " + experiment_name[2])

            print("Running experiment", i + 1, ":", custom_name, "\n\tClient params:",
                  params["client_params"].__str__(),
                  "\n\tServer params:", params["server_params"].__str__(), "\n")

            experiment, experiment_output = self._run_experiment(experiment_name, copy.deepcopy(params))

            if self.autosave:  # If autosave, then experiment stats are cached and saved in each iteration
                try:
                    data = params["data"]
                    domain = np.arange(start=min(data), stop=max(data) + 1)
                except KeyError:
                    data = self.data
                    domain = self.bins

                experiment_data = experiment_output["plot_data"]

                row = self._generate_experiment_stats(experiment_list[i][0], data, experiment_data, domain)
                row = self._add_time_metrics(row, experiment_output)
                self._autosave(row)
            else:
                self.experiment_plot_data.append(((experiment_list[i][0], params), experiment_output))

    def _run_experiment(self, experiment_name, params):
        try:
            data = params["data"]
            domain = np.arange(start=min(data), stop=max(data) + 1)
        except KeyError:
            data = self.data
            domain = self.bins

        name_split = experiment_name.split("_")
        add_name = ["optimised", "threshold", "median"]

        # TODO: Rework this...
        for keyword in add_name:
            if keyword in name_split:
                name_split.remove(keyword)
                experiment_name = ""
                for index, name in enumerate(name_split):
                    if index > 0:
                        experiment_name += "_" + name
                    else:
                        experiment_name += name

        experiment = PureSimulation(experiment_name, params["client_params"], params["server_params"])

        return experiment, experiment.run(data, domain)


    def _plot(self):
        print("Plotting data...")
        figsize = (12, 20)

        fig, axs = plt.subplots(len(self.experiment_plot_data) + 1, figsize=figsize)
        colours = sns.color_palette("hls", len(self.experiment_plot_data) + 1)  # Generate colours for each plot

        if self.large_domain:
            axs[0].stackplot(self.bins, self.y_map(self.bins), color=colours[0], alpha=0.3)
        else:
            # Plotting a distplot of our integer data sampled from a normal dist
            sns.distplot(self.data, bins=self.bins, ax=axs[0], hist_kws={'ec': "black"}, color=colours[0],
                         label="Original", kde=False)

        axs[0].set_title(self.experiment_title)
        row_list = []
        for i, ldp_plot_data in enumerate(self.experiment_plot_data):
            experiment_name = ldp_plot_data[0][0]
            experiment_params = ldp_plot_data[0][1]

            try:
                data = experiment_params["data"]
                domain = np.arange(start=min(data), stop=max(data) + 1)
            except KeyError:
                data = self.data
                domain = self.bins

            experiment_data = ldp_plot_data[1]["plot_data"]

            if self.large_domain:
                axs[i + 1].stackplot(domain, experiment_data, color=colours[i + 1], alpha=0.4)
                sns.lineplot(domain, self.y_map(domain), ax=axs[i + 1], color=colours[0], alpha=1)
            else:
                # Plotting a distplot of the data produced from the experiment
                sns.distplot(data, bins=domain, ax=axs[i + 1], color=colours[0], hist_kws={'ec': "black"}, kde=False)
                sns.distplot(domain, bins=domain, ax=axs[i + 1], color=colours[i + 1],
                             hist_kws={'ec': "black", "weights": experiment_data}, kde=False, label=experiment_name)

            axs[i + 1].set_title(experiment_name)

        # # Plot the original kde of the data in the last axis
        # sns.distplot(self.data, bins=bins, hist=False, ax=axs[len(axs) - 1], color=colours[0])
        fig.legend()
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.2))

        fig.tight_layout()

        if not os.path.exists('plots'):
            os.mkdir('plots')

        filename = self.plot_directory + "normal_exp" + self.id + ".png"
        self._save_experiment_stats()

        plt.savefig(filename)

        plt.show()
        print("Plot Displayed...")

    def __calculate_error(self, data, ldp_data, domain, top_k=0):
        min_domain_val = min(domain)
        # Maps a data value x to {0, 1, ... , len(domain)-1} for indexing purposes
        if min_domain_val < 0:
            index_mapper = lambda x: x + abs(min_domain_val)
        else:
            index_mapper = lambda x: x - abs(min_domain_val)

        try:
            original_counter = Counter(data.tolist())
            original_freq_data = dict(original_counter)
        except AttributeError:
            original_counter = Counter(data)
            original_freq_data = dict(original_counter)

        ldp_freq_data = ldp_data
        mse = 0
        total_error = 0

        if self.calc_top_k != 0 and self.display_full_stats is False:
            values, _ = zip(*original_counter.most_common(self.calc_top_k))
        elif top_k != 0:
            values, _ = zip(*original_counter.most_common(self.calc_top_k))
        else:
            values = domain

        # Printing debug
        # sketch_help = list(values[0:20])
        # for item in sketch_help:
        #     print(str(item) + ": " + str(ldp_freq_data[index_mapper(item)]))

        errors = []

        for index, item in enumerate(values):
            error = abs(ldp_freq_data[index_mapper(item)] - original_freq_data.get(item, 0))
            errors.append(error)
            total_error += error
            mse += error ** 2

        avg_error = total_error / len(values)
        mse = mse / len(values)

        max_error = max(errors)
        max_item = domain[errors.index(max_error)]
        return max(errors), min(errors), max_item, avg_error, mse, np.median(errors)

    def _generate_experiment_stats(self, experiment_name, data, ldp_data, domain):
        row = OrderedDict()
        max_error, min_error, max_item, avg_error, mse, med_abs_dev = self.__calculate_error(data, ldp_data, domain)

        if isinstance(experiment_name, tuple):
            frequency_oracle, custom_name = experiment_name[len(experiment_name) - 2:]
        else:
            frequency_oracle = experiment_name
            custom_name = ""

        row["freq_oracle"] = frequency_oracle
        row["info"] = custom_name
        row["mse"] = mse
        row["average_error"] = avg_error

        if self.display_full_stats:
            k_max_error, k_min_error, k_max_item, k_avg_error, k_mse, k_med_abs_dev = self.__calculate_error(data,
                                                                                                             ldp_data,
                                                                                                             domain,
                                                                                                             self.calc_top_k)
            row["k_mse"] = k_mse
            row["k_average_error"] = k_avg_error

        row["median_abs_deviation"] = med_abs_dev
        row["max_error"] = max_error
        row["max_item"] = max_item
        row["min_error"] = min_error
        return row

    def _add_time_metrics(self, row, experiment_output):
        row["client_time"] = experiment_output["client_time"]

        row["server_init_time"] = experiment_output["server_init_time"]
        row["server_agg_time"] = experiment_output["server_agg_time"]
        row["server_est_time"] = experiment_output["server_est_time"]
        row["server_time"] = experiment_output["server_time"]

        row["total_time"] =  row["client_time"] + row["server_time"]

        return row

    def _generate_stats(self):
        row_list = []
        for i, ldp_plot_data in enumerate(self.experiment_plot_data):
            row = self.stat_cache.get(i, None)

            if row is None:
                experiment_name = ldp_plot_data[0][0]
                experiment_params = ldp_plot_data[0][1]
                experiment_output = ldp_plot_data[1]
                experiment_data = experiment_output["plot_data"]

                try:
                    data = experiment_params["data"]
                    domain = np.arange(start=min(data), stop=max(data) + 1)
                except KeyError:
                    data = self.data
                    domain = self.bins

                row = self._add_time_metrics(self._generate_experiment_stats(experiment_name, data, experiment_data, domain), experiment_output)
                self.stat_cache[i] = row

            row_list.append(row)
        return row_list

    def _save_experiment_stats(self):
        row_list = self._generate_stats()
        stats = pd.DataFrame(row_list)
        pd.set_option('display.max_rows', 0)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.4f}'.format)

        if self.calc_top_k != 0:
            print("Statistics are calculated based on the top", str(self.calc_top_k), "values in the dataset")

        filepath = self.stats_filename
        self.stats = stats

        print("Experiment Metrics are saved under:", filepath)
        print("\n", stats, "\n")

        if not os.path.exists('plots/metrics'):
            os.makedirs('plots/metrics')

        stats.to_csv(filepath)

    def _autosave(self, row):
        if not os.path.isfile(self.stats_filename):
            stats = pd.DataFrame([row])
        else:
            stats = pd.read_csv(self.stats_filename, index_col=0)
            stats = stats.append(row, ignore_index=True)

        pd.set_option('display.max_rows', 0)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.4f}'.format)

        if self.calc_top_k != 0 and self.display_full_stats is False:
            print("Statistics are calculated based on the top", str(self.calc_top_k), "values in the dataset")

        self.stats = stats

        print("Experiment Metrics are saved under:", self.stats_filename)
        print("\n", stats, "\n")

        self.stats.to_csv(self.stats_filename)
