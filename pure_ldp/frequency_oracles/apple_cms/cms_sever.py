from pure_ldp.core import FreqOracleServer
import math
import numpy as np
from scipy.linalg import hadamard

from pure_ldp.core import generate_hash_funcs

class CMSServer(FreqOracleServer):
    def __init__(self, epsilon, k, m, is_hadamard=False, index_mapper=None):
        """
        Server frequency oracle for Apple's Count Mean Sketch (CMS)

        Args:
            epsilon (float): Privacy Budget
            k (int): Number of hash functions
            m (int): Size of the hash domain
            is_hadamard (optional bool): If True, uses Hadamard Count Mean Sketch (HCMS)
            index_mapper (optional func): Index map function
        """
        super().__init__(epsilon, None, index_mapper)
        self.sketch_based = True
        self.is_hadamard = is_hadamard
        self.update_params(k,m, epsilon, index_mapper=None)
        self.hash_funcs = generate_hash_funcs(k,m)
        self.sketch_matrix = np.zeros((self.k, self.m))
        self.transformed_matrix = np.zeros((self.k, self.m))

        self.last_estimated = self.n
        self.ones = np.ones(self.m)

        if self.is_hadamard:
            self.had = hadamard(self.m)

    def update_params(self, k=None, m=None, epsilon=None, index_mapper=None):
        """
        Updated internal parameters
        Args:
            k (optional int): Number of hash functions
            m (optional int): Size of hash domain
            epsilon (optional float): Privacy Budget
            d (optional int): Size of domain
            index_mapper (optional func): Index map function
        """
        self.k = k if k is not None else self.k
        self.m = m if m is not None else self.m
        self.hash_funcs = generate_hash_funcs(self.k,self.m)
        super().update_params(epsilon=epsilon, index_mapper=index_mapper) # This also calls reset() to reset sketch size
        if epsilon is not None:
            if self.is_hadamard:
                self.c = (math.pow(math.e, epsilon) + 1) / (math.pow(math.e, epsilon) - 1)
            else:
                self.c = (math.pow(math.e, epsilon / 2) + 1) / (math.pow(math.e, epsilon / 2) - 1)

    def _add_to_cms_sketch(self, data):
        """
        Given privatised data, adds it to the sketch matrics (CMS algorithm)
        Args:
            data: privatised data by CMS
        """
        item, hash_index = data
        self.sketch_matrix[hash_index] = self.sketch_matrix[hash_index] + self.k * ((self.c / 2) * item + 0.5 * self.ones)

    def _add_to_hcms_sketch(self, data):
        """
        Given privatised data, adds it to the sketch matrix (HCMS algorithm)

        Args:
            data: privatisd data by HCMS
        """
        bit_value, j, l = data
        self.sketch_matrix[j][l] = self.sketch_matrix[j][l] + self.k * self.c * bit_value

    def _transform_sketch_matrix(self):
        """
        Transforms the sketch matrix using inverse hadamard (HCMS)
        Returns: Transformed sketch matrix

        """
        return np.matmul(self.sketch_matrix, np.transpose(self.had))

    def _update_estimates(self):
        """
        If using HCMS, transforms the sketch matrix using inverse hadamard
        """
        if self.is_hadamard:
            self.last_estimated = self.n # TODO: Is this needed?
            self.transformed_matrix = self._transform_sketch_matrix()

    def get_hash_funcs(self):
        """
        Returns hash functions for CMSClient

        Returns: list of k hash_funcs

        """
        return self.hash_funcs

    def reset(self):
        """
        Resets sketch matrix (i.e resets all aggregated data)
        """
        super().reset()
        self.sketch_matrix = np.zeros((self.k, self.m))
        self.transformed_matrix = np.zeros((self.k, self.m))

    def aggregate(self, data):
        """
        Aggregates privatised data

        Args:
            data: Data privatised by CMS/HCMS
        """
        if self.is_hadamard:
            self._add_to_hcms_sketch(data)
        else:
            self._add_to_cms_sketch(data)
        self.n += 1

    def estimate(self, data, suppress_warnings=False):
        """
        Estimates the frequency of the data item

        Args:
            data: item to be estimated
            suppress_warnings (optional bool): If True, will suppress estimation warnings

        Returns: Frequency Estimate

        """
        self.check_warnings(suppress_warnings)
        self.check_and_update_estimates()

        # If it's hadamard we need to transform the sketch matrix
            # To prevent this being performance intensive, we only transform if new data has been aggregated since it was last transformed

        sketch = self.sketch_matrix if not self.is_hadamard else self.transformed_matrix

        data = str(data)
        k, m = sketch.shape
        freq_sum = 0
        for i in range(0, k):
            freq_sum += sketch[i][self.hash_funcs[i](data)]

        return (m / (m - 1)) * ((1 / k) * freq_sum - (self.n / m))

