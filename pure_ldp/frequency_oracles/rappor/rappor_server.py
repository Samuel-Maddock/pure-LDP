from pure_ldp.core import FreqOracleServer
from pure_ldp.core import generate_hash

import numpy as np
import random
import sys
import math
import copy
from sklearn.linear_model import Lasso, LinearRegression, ElasticNet
from sklearn.feature_selection import *
from scipy.optimize import nnls
import statsmodels.api as sm


class RAPPORServer(FreqOracleServer):
    def __init__(self, f, m, k, d, num_of_cohorts=8, index_mapper=None, reg_const=None, lasso=False):
        """
        Server frequency oracle for RAPPOR

        Args:
            f (float): RAPPOR's Privacy Parameter
            m (int): Size of the bloom filter
            k (int): Number of hash functions per cohort
            d (int): Domain size d
            num_of_cohorts (optional int): Number of cohorts to split users into
            index_mapper (optional func): Index map function
            reg_const (optional - float): Regularisation constant to use in the regression estimation
            lasso (optional boolean): If True, will use LASSO to select elements in the domain,
                    automatically enabled for large  domains (> 1000). Not recommended for small domains.
        """
        super().__init__(epsilon=None, d=d, index_mapper=index_mapper)
        self.f = f  # Probability used to peturb bloom filters
        self.k = k  # Num of hash funcs
        self.m = m  # Size of bloom filter

        self.epsilon = 2 * self.k * math.log((1 - 0.5 * f) / (0.5 * f))

        self.num_of_cohorts = num_of_cohorts

        self.bloom_filters = [np.zeros(self.m) for i in range(0, self.num_of_cohorts)]
        self.cohort_count = np.zeros(self.num_of_cohorts)
        self.hash_family = self._generate_hash_funcs()

        self.estimated_data = np.zeros(self.d)
        self.normalised_data = []
        self.lasso = lasso

        self.reg_const = reg_const
        if reg_const is None:
            self.reg_const = 0.025 * self.f

    def update_params(self, epsilon=None, d=None, index_mapper=None, f=None, m=None, k=None, num_of_cohorts=None):
        super().update_params(epsilon, d, index_mapper)
        self.f = f if f is not None else self.f
        self.m = m if m is not None else self.m
        self.num_of_cohorts = num_of_cohorts if num_of_cohorts is not None else self.num_of_cohorts

        if f is not None:
            self.epsilon = 2 * self.k * math.log((1 - 0.5 * f) / (0.5 * f))
            self.reg_const = 0.025 * self.f
        if m is not None or num_of_cohorts is not None:
            # If the bloom filter size or number of cohorts changes then reset bloom filters and cohort counts
            self.bloom_filters = [np.zeros(self.m) for i in range(0, self.num_of_cohorts)]
            self.cohort_count = np.zeros(self.num_of_cohorts)

        if k is not None or num_of_cohorts is not None:
            self.hash_family = self._generate_hash_funcs()
            raise RuntimeWarning(
                "RAPPORServer hash functions were reset due to a change in num_of_cohorts or k. RAPPORClient hash functions should be updated manually otherwise you will face inconsistencies")

        self.reset()  # Changing parameters will reset RAPPORs state

    def reset(self):
        super().reset()
        self.normalised_data = []

    def _generate_hash_funcs(self):
        """
        Generates hash functions for RAPPOR instance

        Returns: list of (num_of_cohorts x k) hash_funcs

        """
        hash_family = []
        for i in range(0, self.num_of_cohorts):
            hash_family.append([generate_hash(self.m, random.randint(0, sys.maxsize)) for i in range(0, self.k)])

        return hash_family

    def get_hash_funcs(self):
        """
        Returns hash functions for CMSClient

        Returns: list of (num_of_cohorts x k) hash_funcs

        """
        return self.hash_family

    def aggregate(self, data):
        """
        Aggregates privatised data from LHClient to be used to calculate frequency estimates.

        Args:
            data: Privatised data of the form returned from UEClient.privatise
        """
        bloom_filter = data[0]
        cohort_num = data[1]
        self.cohort_count[cohort_num] += 1
        self.bloom_filters[cohort_num] += bloom_filter
        self.n += 1

    def _update_estimates(self):
        y = self._create_y()
        X = self._create_X()

        if self.reg_const == 0:
            model = LinearRegression(positive=True, fit_intercept=False)
        else:
            model = ElasticNet(positive=True, alpha=self.reg_const,
                               l1_ratio=0, fit_intercept=False,
                               max_iter=10000)  # non-negative least-squares with L2 regularisation to prevent overfitting

        if self.d > 1000 or self.lasso:  # If d is large, we perform feature selection to reduce computation time
            # print("d is large, fitting LASSO to reduce d")
            lasso_model = Lasso(alpha=0.8, positive=True)
            lasso_model.fit(X, y)
            indexes = np.nonzero(lasso_model.coef_)[0]
            # print("LASSO fit,", str(len(indexes)), "features selected")
            X_red = X[:, indexes]
            model.fit(X_red, y)
            self.estimated_data[indexes] = model.coef_ * self.num_of_cohorts
        else:
            model.fit(X, y)
            self.estimated_data = model.coef_ * self.num_of_cohorts

    def _create_X(self):
        X = np.empty((self.m * self.num_of_cohorts, self.d))

        for i in range(0, self.d):
            col = np.zeros((self.num_of_cohorts, self.m))
            for index, funcs in enumerate(self.hash_family):
                for hash in funcs:
                    col[index][hash(str(i))] = 1

            X[:, i] = col.flatten()
        return X

    def _create_y(self):
        y = np.array([])

        for i, bloom_filter in enumerate(self.bloom_filters):
            scaled_bloom = (bloom_filter - (0.5 * self.f) * self.cohort_count[i]) / (1 - self.f)
            y = np.concatenate((y, scaled_bloom))

        return y

    def estimate(self, data, suppress_warnings=False):
        """
        Calculates a frequency estimate of the given data item using the aggregated data.

        Args:
            data: data item
            suppress_warnings: Optional boolean - Suppresses warnings about possible inaccurate estimations

        Returns: float - frequency estimate of the data item

        """

        self.check_warnings(suppress_warnings)
        self.check_and_update_estimates()

        return self.estimated_data[self.index_mapper(data)]

    def convert_eps_to_f(self, epsilon):
        """
        Can be used to convert a privacy budget epsilon to the privacy parameter f for RAPPOR

        Args:
            epsilon: Privacy budget

        Returns: The equivalent privacy parameter f

        """
        return round(1 / (0.5 * math.exp(epsilon / 2) + 0.5), 2)

    def convert_f_to_eps(self, f, k):
        """
        Can be used to a certain RAPPORs epsilon value

        Args:
            f: Privacy parameter
            k: The number of hash functions used by RAPPOR

        Returns: Epsilon value

        """
        return math.log(((1 - 0.5 * f) / (0.5 * f)) ** (2 * k))
