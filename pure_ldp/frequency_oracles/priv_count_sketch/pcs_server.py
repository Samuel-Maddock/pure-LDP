import numpy as np
import warnings

from pure_ldp.core import FreqOracleServer, generate_hash_funcs

class PCSServer(FreqOracleServer):
    def __init__(self, epsilon, l, w, use_median=True, index_mapper=None):
        """
        Private Count Sketch (PCS) Algorithm
        Args:
            epsilon (float): Privacy Budget Epsilon
            l (integer): Number of hash functions for the sketch
            w (integer): Size of sketch  vector
            use_median (optional - boolean): If True, uses median in the count-sketch estimation
            index_mapper (optional function): Index mapper function
        """
        super().__init__(epsilon, None, index_mapper)
        self.l = l
        self.w = w
        self.sketch_matrix = np.zeros((self.l, self.w))
        self.use_median = use_median
        self.name = "PCSServer"
        self.h_funcs = generate_hash_funcs(l, w)
        self.g_funcs = generate_hash_funcs(l, 2)
        self.hash_funcs = list(zip(self.h_funcs, self.g_funcs))

    def update_params(self, epsilon=None, d=None, index_mapper=None, l=None, w=None, use_median=None):
        super().update_params(epsilon, d, index_mapper)
        self.l = l if l is not None else self.l
        self.w = w if w is not None else self.w
        self.use_median = use_median if use_median is not None else self.use_median

        # if l or w is updated we need to reset the sketch matrix and generate new hash functions..
        if l is not None or w is not None:
            self.sketch_matrix = np.zeros((self.l, self.w))
            self.h_funcs = generate_hash_funcs(self.l, self.w)
            self.g_funcs = generate_hash_funcs(self.l, 2)
            self.hash_funcs = list(zip(self.h_funcs, self.g_funcs))

    def get_hash_funcs(self):
        """
        Returns the hash functions used by the sketch

        Returns:
            List of hash functions
        """
        return self.hash_funcs

    def _set_sketch_element(self, data, hash_id):
        self.sketch_matrix[hash_id] += (data * self.l)

    def reset(self):
        """
            Resets the sketch matrix stored
        """
        super().reset()
        self.sketch_matrix = np.zeros((self.l, self.w))

    def aggregate(self, data):
        """
        Aggregates privatised data

        Args:
            data: Data privatised by CMS/HCMS
        """
        self._set_sketch_element(*data)
        self.n += 1

    def _update_estimates(self):
        pass

    def estimate(self, data, suppress_warnings=False):
        """
        Estimates the frequency of the data item

        Args:
            data: item to be estimated
            suppress_warnings (optional bool): If True, will suppress estimation warnings

        Returns: Frequency Estimate

        """
        self.check_warnings(suppress_warnings)
        data = str(data)

        weak_freq_estimates = []
        for hash_id in range(0, self.l):
            h_loc = self.hash_funcs[hash_id][0](data)
            g_val = 2* self.hash_funcs[hash_id][1](data) - 1

            weak_freq_estimates.append(g_val * self.sketch_matrix[hash_id, h_loc])

        if self.use_median:
             return np.median(weak_freq_estimates)
        else:
            return np.mean(weak_freq_estimates)
