from pure_ldp.core import FreqOracleServer, generate_hash_funcs
from pure_ldp.frequency_oracles.local_hashing import FastLHServer

import copy
import numpy as np

from collections import Counter, defaultdict


class SketchResponseServer(FreqOracleServer):
    def __init__(self, epsilon, k, m, index_mapper=None, fo_server=None, lh_k=100, estimator_norm=0, sketch_method=0,
                 count_sketch=False):
        """
        Args:
            epsilon (float): Privacy budget
            k (integer): The number of hash functions used in the sketch
            m (integer): Size of the sketch vector to privatise
            index_mapper (optional function): Index mapper function
            fo_server (FreqOracleServer): The FO server used for estimation. Needs to be the same as the FO client that is being used. Default is FastLH Server
            lh_k (Optional int): If no FO server is passed, this can be used to set the k parameter for the FastLH server that is used
            estimator_norm (Optional int): Normalisation performed when estimated sketch rows
                           0 - No Norm
                           1 - Additive Norm
                           2 - Prob Simplex
                           3 (or otherwise) - Threshold cut
            sketch_method (Optional int): The sketch method used in the estimation -
                            0 - Takes the minimum of sketch entries (Count-Min Sketch)
                            1 - Takes the median of sketch entries (Count-Median Sketch)
                            2 - Takes the mean (no debiasing)
                            Anything else - Takes the Unbiased Mean (Count-Mean Sketch)
            count_sketch (optional - boolean): If True, will use count-sketch for estimation (instead of count-mean sketch)
        """
        self.sketch_based = True
        self.k = k
        self.lh_k = lh_k
        self.m = m
        self.hash_funcs = generate_hash_funcs(self.k, self.m)
        self.count_sketch = count_sketch
        self.sketch_method = sketch_method
        self.estimator_norm = estimator_norm
        self.set_name("Sketch Response")

        d = self.m
        self.cs_map = None

        if self.count_sketch:
            self.h_funcs = generate_hash_funcs(k, m)
            self.g_funcs = generate_hash_funcs(k, 2)
            d = 2 * self.m

            def cs_map(x):
                if x > 0:
                    return x - 1
                else:
                    return 2 * self.m - abs(x)

            self.cs_map = cs_map
            self.hash_funcs = list(zip(self.h_funcs, self.g_funcs))

        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.aggregated_data = defaultdict(list)
        self.estimator_list = []

        if isinstance(fo_server, FreqOracleServer) and not isinstance(fo_server, FastLHServer):
            fo_server.update_params(index_mapper=lambda x: x, d=d)
            for i in range(0, self.k):
                self.estimator_list.append(copy.deepcopy(fo_server))
        else:
            try:
                lh_k = fo_server.k
            except AttributeError:
                lh_k = self.lh_k

            # All FLH estimators will use the same hash funcs so we only need to generate the hash matrix across the domain once to save time
            for i in range(0, self.k):
                if i >= 1:
                    self.estimator_list.append(
                        FastLHServer(self.epsilon, d, lh_k, hash_matrix=self.estimator_list[0].hash_matrix,
                                     index_mapper=lambda x: x))
                else:
                    self.estimator_list.append(FastLHServer(self.epsilon, d, lh_k, index_mapper=lambda x: x))
        try:
            self.server_fo_hash_funcs = self.estimator_list[0].get_hash_funcs()
        except AttributeError:
            self.server_fo_hash_funcs = None

        self.reset() # This should be reworked to just initialise the sketch_matrix here...

    def update_params(self, epsilon=None, d=None, index_mapper=None):
        super().update_params(epsilon, d, index_mapper)

        if epsilon is not None:
            try:
                # If the frequency oracle is FLH, then updating epsilon will also update the hash matrix - We only want to do this once and propagate the hashed matrix to all FLH estimators
                    # This saves quite a lot of time if k is large!
                a = self.estimator_list[0].hash_matrix
                self.estimator_list[0].update_params(epsilon=epsilon)
                for i in range(1, len(self.estimator_list)):
                    self.estimator_list[i].update_params(epsilon=epsilon, update_hash_matrix=False)
                    self.estimator_list[i].hash_matrix = self.estimator_list[0].hash_matrix
            except AttributeError:
                for estimator in self.estimator_list:
                    estimator.update_params(epsilon=epsilon)

    def get_hash_funcs(self):
        """
        Returns the hash functions used by the sketch

        Returns:
            List of hash functions
        """
        return self.hash_funcs

    def reset(self):
        """
        Resets aggregated/estimated data to allow for new collection/aggregation
        """
        self.sketch_matrix = np.zeros((self.k, self.m))

    def aggregate(self, data):
        """
        Used to aggregate privatised data to the server
        Args:
            data: privatised data item to aggregate
        """

        k = data[1]
        hash_val = data[0]
        self.estimator_list[k].aggregate(hash_val)
        self.n += 1

    def _update_estimates(self):
        """
        Used internally to update estimates
        Returns: estimated data

        """
        if self.count_sketch:
            for i in range(0, self.k):
                for j in range(0, self.m):
                    pos_val = self.cs_map(j + 1)
                    neg_val = self.cs_map((j + 1) * -1)
                    self.sketch_matrix[i][j] = self.estimator_list[i].estimate(pos_val, suppress_warnings=True) - \
                                               self.estimator_list[i].estimate(neg_val, suppress_warnings=True)
        else:
            for i in range(0, self.k):
                norm = self.estimator_norm
                if self.estimator_norm == "None":
                    norm = 0  # We want to estimate with no normalisation, but NOT round negative to 0 with estimator_norm="None"
                self.sketch_matrix[i] = self.estimator_list[i].estimate_all(range(0, self.m), normalization=norm,
                                                                            suppress_warnings=True)

            if self.estimator_norm == 0:
                pass
                #self.sketch_matrix = self.sketch_matrix.clip(min=0)  # Clip sketch matrix estimates if estimator norm = 0

    def estimate(self, data, suppress_warnings=False):
        """
        Estimates frequency of given data item
        Args:
            data: data item to estimate
            suppress_warnings: Optional boolean - If True, estimation warnings will not be displayed

        Returns: frequency estimate

        """
        self.check_warnings(suppress_warnings)
        index = self.index_mapper(data)
        self.check_and_update_estimates()
        est = []

        for i in range(0, self.k):
            if self.count_sketch:
                hash_index = self.hash_funcs[i][0](index)
                g_val = 2 * self.hash_funcs[i][1](index) - 1
                est.append(g_val * self.sketch_matrix[i][hash_index])
            else:
                hash_index = self.hash_funcs[i](index)
                est.append(self.sketch_matrix[i][hash_index])

        if self.sketch_method == 0:
            est = np.amin(est) * self.k
        elif self.sketch_method == 1:
            est = np.median(est) * self.k
        elif self.sketch_method == 2:
            est = sum(est)
        else:
            est = (self.m / (self.m-1)) * (sum(est) - (self.n/self.m))

        return est