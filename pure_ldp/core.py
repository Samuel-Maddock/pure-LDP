
import numpy as np
import warnings

# Contains base classes for client/server frequency oracles

class FreqOracleClient:
    def __init__(self, epsilon, d, index_mapper=None):
        self.epsilon = epsilon
        self.d = d

        if index_mapper is None:
            self.index_mapper = lambda x: x - 1
        else:
            self.index_mapper = index_mapper

    def _peturb(self, data):
        assert("Must Implement")

    def privatise(self, data, **kwargs):
        assert("Must Implement")


class FreqOracleServer:
    def __init__(self, epsilon, d, index_mapper=None):
        self.epsilon = epsilon
        self.d = d
        self.aggregated_data = np.zeros(self.d)
        self.estimated_data = np.zeros(self.d)
        self.n = 0
        self.name = "FrequencyOracle"

        if index_mapper is None:
            self.index_mapper = lambda x: x - 1
        else:
            self.index_mapper = index_mapper

    def set_name(self, name):
        self.name = name

    def reset(self):
        self.aggregated_data = np.zeros(self.d)
        self.estimated_data = np.zeros(self.d)
        self.n = 0

    def update_params(self, epsilon=None, d=None, index_mapper=None):
        self.epsilon = epsilon if epsilon is not None else self.epsilon
        self.d = d if d is not None else self.d
        self.index_mapper = index_mapper if index_mapper is not None else self.index_mapper

    def check_warnings(self, suppress_warnings=False):
        if not suppress_warnings:
            if self.n < 10000:
                warnings.warn(self.name + " has only aggregated small amounts of data (n=" + str(self.n) +
                              ") estimations may be highly inaccurate", RuntimeWarning)
            if self.epsilon < 1:
                warnings.warn("High privacy has been detected (epsilon = " + str(self.epsilon) +
                              "), estimations may be highly inaccurate on small datasets", RuntimeWarning)

    def aggregate(self, data):
        assert ("Must implement")

    def estimate(self, data, suppress_warnings=False):
        assert("Must implement")

    def estimate_all(self):
        assert("Must implement")

    def get_estimates(self):
        return self.estimated_data
