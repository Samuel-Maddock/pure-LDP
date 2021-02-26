from .PureHHSimulation import PureHHSimulation

import os, uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import string

from collections import OrderedDict, Counter

class HeavyHitterExperiment:
    def __init__(self, word_length, word_sample_size, data=None, n=None, p=None, alphabet=None, prune_negative=True, metric_k =False, autosave=False):
        self.name = "Heavy Hitter Simulation"
        self.calc_top_k = 0
        self.metric_k = metric_k
        self.display_full_stats = False
        self.autosave = autosave
        self.prune_negative=True
        self.alphabet = alphabet
        self.word_length = word_length
        self.word_sample_size = word_sample_size

        self.padding_char = "*"

        self.reset()
        self.n = n
        self.p = p

        if data is None and self.n is not None and self.p is not None:
            self.data = self.__generate_dataset()
        else:
            self.data = data
            self.n = len(data)
            if alphabet is None:
                alphabet = list(string.ascii_lowercase)
            self.alphabet = alphabet

    def reset(self):
        self.experiment_plot_data = []
        self.id = str(uuid.uuid4())

        if not os.path.exists('plots'):
            os.mkdir('plots')

        if not os.path.exists('plots/metrics'):
            os.mkdir('plots/metrics')

        self.plot_directory = "plots/"
        self.stats_filename = self.plot_directory + "metrics/" + self.id + ".csv"
        self.stat_cache = {}

        return self

    def __generate_dataset(self):
        dataset = []
        alphabet_list = []

        for i in range(0, self.word_length):
            alphabet_list.append(self.alphabet)

        # Form all possible strings of length word_size from our given alphabet
        strings = list(set(map(lambda x: str().join(x), itertools.product(*alphabet_list))))

        def generate_words(n):
            words = set()
            for i in range(0, n):
                words.add(np.random.choice(strings))
            return list(words)

        words = generate_words(self.word_sample_size)

        geometric = Counter(np.random.geometric(p=self.p, size=self.n))

        for i, word in enumerate(words):
            frequency = geometric.get(i + 1)

            for j in range(0, frequency):
                dataset.append(word)  # Add the word to our dataset

        return dataset

    def _plot(self):
        freq_data = Counter(self.data)
        print("Plotting results...")

        figsize = (12, 20)
        fig, axs = plt.subplots(len(self.experiment_plot_data) + 1, figsize=figsize)
        ax1 = axs[0]

        # Plots the words and their frequencies in descending order
        x1, y1 = zip(*freq_data.most_common())
        color_palette = sns.cubehelix_palette(len(x1), start=.5, rot=-.75, reverse=True)
        sns.barplot(list(x1), list(y1), ax=ax1, palette=color_palette, order=list(x1))
        ax1.tick_params(rotation=45)
        ax1.set_xlabel("Words")
        ax1.set_ylabel("Word Count")
        ax1.set_title("ID:" + self.id + "\nWords and their frequencies in the dataset")

        for i, plot_data in enumerate(self.experiment_plot_data):
            experiment_name = plot_data[0][0]
            experiment_params = plot_data[0][1]
            heavy_hitter_data = plot_data[1]
            experiment = plot_data[2]

            ax = axs[i + 1]

            if len(heavy_hitter_data) == 0:
                heavy_hitter_data.add(("empty", 0))

            x, y = zip(*reversed(heavy_hitter_data))

            palette = self._generate_palette(color_palette, x1, x)

            # Plot the words discovered by the heavy hitter against estimated frequencies in descending order
            sns.barplot(list(x), list(y), ax=ax, palette=palette, order=list(x))
            ax.tick_params(rotation=45)
            ax.set_xlabel("Words Discovered")
            ax.set_ylabel("Estimated Word Count")

            if experiment_params.get("freq_oracle") is not None:
                experiment_name = experiment_name + " with " + experiment_params["freq_oracle"]

            ax.set_title(
                "Experiment: " + experiment_name)
            # + "\n Parameters: " + str(experiment_params) )

        self._save_experiment_stats()
        fig.tight_layout()
        filename = self.plot_directory + "exponential_exp" + self.id + ".png"
        plt.savefig(filename)
        plt.show()
        print("Plot saved, simulation ended...")

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
            params = experiment_list[i][1]

            k = params.get("k", None)
            alt_fo_name = params.get("alt_fo_name", None)

            params["method"] = "threshold"

            if k is not None:
                params["method"] = "top_k"

            if isinstance(experiment_name, tuple):
                info = experiment_name[1]
                params["info"] = info
                experiment_name = experiment_name[0]

            if alt_fo_name:
                experiment_freq_oracle = " with " + alt_fo_name
            else:
                experiment_freq_oracle = "" if params.get("freq_oracle", None) is None else " with " + params.get("freq_oracle")

            print("Running experiment", i + 1, ":", experiment_name + experiment_freq_oracle, "\nParams: \n %.1000s...\n" % params.__str__())

            experiment, experiment_output = self._run_experiment(experiment_name, params)

            if self.autosave:  # If autosave, then experiment stats are cached and saved in each iteration
                experiment_data = experiment_output[0]
                row = self._compute_metrics(experiment_list[i][0], Counter(self.data), experiment_data, params)
                row["client_time"] = experiment_output[1]
                row["server_time"] = experiment_output[2]
                row["total_time"] = experiment_output[1] + experiment_output[2]
                self._autosave(row)
            else:
                self.experiment_plot_data.append((experiment_list[i], experiment_output[0], experiment, experiment_output[1:3]))

    def _run_experiment(self, experiment_name, params):
        params["alphabet"] = self.alphabet

        heavy_hitters = {
            "SFP": lambda parameters: PureHHSimulation("SFP", parameters),
            "TreeHist": lambda parameters: PureHHSimulation("TreeHist", parameters),
            "PEM": lambda parameters: PureHHSimulation("PEM", parameters),
        }

        if experiment_name.split(" ")[0] not in heavy_hitters.keys():
            assert "experiment name must be one of: ", heavy_hitters.keys()

        experiment = heavy_hitters.get(experiment_name.split(" ")[0])(params) # Split on space to allow custom names
        data = experiment.run(self.data)
        return (experiment, data)

    def _generate_palette(self, color_palette, x1, x2):
        # Generate colour palette for a graph of heavy hitter data
        # We color bars of words that were discovered by the algo but were not in our original dataset as red
        # We maintain the original coloring of the words that were correctly discovered

        palette = []
        for data in list(x2):
            if data not in list(x1):
                palette.append("#e74c3c")
            else:
                palette.append(color_palette[x1.index(data)])
        return palette

    def _compute_metrics(self, experiment_name, original_data, heavy_hitter_data, experiment_params):
        row = OrderedDict()
        heavy_hitter_data = dict(heavy_hitter_data)

        if self.prune_negative:
            heavy_hitter_data = dict((k, v) for k, v in heavy_hitter_data.items() if v >= 0)

        padded_data = [self._pad_string(x) for x in self.data]
        data_counter = Counter(padded_data)

        k_tp = 0
        avg_freq = 0
        if experiment_params["method"] == "threshold":
            top_k = list(filter(lambda x: x[1]/len(self.data) >= experiment_params.get("threshold", 100), data_counter.items()))
        else:
            try:
                top_k = data_counter.most_common(self.metric_k)
            except AttributeError:
                top_k = data_counter.most_common(experiment_params.get("k", 10))

        errors = []
        fp_errors = []
        for item in top_k:
            if item[0] in heavy_hitter_data.keys():
                k_tp += 1
                errors.append(abs(heavy_hitter_data[item[0]] - item[1]))
            else:
                fp_errors.append(item[1])

        # Calculate FP based on URLs found that are not in the dataset - However, if we want the top 10 urls and it returns a correct URL but is say 11th/12th then we do not count that as either a TP or FP...
        k_fp = 0
        for item in heavy_hitter_data.keys():
            if item not in data_counter.keys():
                k_fp += 1

        errors = np.array(errors)
        fp_errors = np.array(fp_errors)

        if isinstance(experiment_name, tuple):
            experiment_name = experiment_name[0]

        row["heavy_hitter"] = experiment_name

        alt_fo_name = experiment_params.get("alt_fo_name", None)
        if alt_fo_name:
            row["freq_oracle"] = alt_fo_name
        else:
            freq_oracle = experiment_params.get("freq_oracle", None)
            row["freq_oracle"] = freq_oracle if freq_oracle is not None else "default"

        row["info"] = experiment_params.get("info", "")
        row["method"] = experiment_params.get("method", "")

        if len(heavy_hitter_data.keys()) == 0:
            row["recall"] = 0
            row["precision"] = 0
            row["f1"] = 0
            row["k_avg_err"] = 0
            row["k_mse"] = 0
            row["false_positives"] = 0
            row["average_freq_of_fp"] = "NA"
        else:
            row["precision"] = k_tp / (k_tp + k_fp)
            row["recall"] = k_tp / len(top_k)

            if (row["recall"] + row["precision"]) != 0:
                row["f1"] = (2 * row["recall"]*row["precision"]) / (row["recall"] + row["precision"])
            else:
                row["f1"] = 0

            row["true_positives"] = k_tp
            row["false_positives"] = k_fp
            row["k_avg_err"] = errors.mean()
            row["k_mse"] = np.square(errors).mean()
            row["average_freq_of_fp"] = fp_errors.sum()/k_fp if k_fp != 0 else "NA"

        return row

    def _generate_stats(self):
        row_list = []
        for i, ldp_plot_data in enumerate(self.experiment_plot_data):
            row = self.stat_cache.get(i, None)

            if row is None:
                experiment_name = ldp_plot_data[0][0]
                experiment_params = ldp_plot_data[0][1]
                heavy_hitter_data = ldp_plot_data[1]
                experiment = ldp_plot_data[2]

                row = self._compute_metrics(experiment_name, Counter(self.data), heavy_hitter_data, experiment_params)
                row["client_time"] = ldp_plot_data[3][0]
                row["server_time"] = ldp_plot_data[3][1]
                row["total_time"] = ldp_plot_data[3][0] + ldp_plot_data[3][1]
                self.stat_cache[i] = row

            row_list.append(row)
        return row_list

    def _pad_string(self, string):
        # Pad strings that are smaller than some arbitrary max value
        if len(string) < self.word_length:
            string += (self.word_length - len(string)) * self.padding_char
        elif len(string) > self.word_length:
            string = string[0:self.word_length]
        return string

    def _save_experiment_stats(self, stats=None):
        if stats is None:
            row_list = self._generate_stats()
            stats = pd.DataFrame(row_list)

        pd.set_option('display.max_rows', 0)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.4f}'.format)

        print("Experiment Metrics are saved under:", self.stats_filename)
        print("\n", stats, "\n")

        stats.to_csv(self.stats_filename)

    def _autosave(self, row):
        if not os.path.isfile(self.stats_filename):
            stats = pd.DataFrame([row])
        else:
            stats = pd.read_csv(self.stats_filename, index_col=0)
            stats = stats.append(row, ignore_index=True)

        self._save_experiment_stats(stats)
