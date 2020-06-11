import warnings
import numpy as np

class FreqOracleServer:
    def __init__(self, epsilon, d, index_mapper=None):
        """

        Args:
            epsilon: privacy budget
            d: domain size - not all freq oracles need this so can be None
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain

        """
        self.epsilon = epsilon
        self.d = d

        self.aggregated_data = np.zeros(self.d) # Some freq oracle servers keep track of aggregated data to generate estimated_data
        self.estimated_data = [] # Keep track of estimated data for quick access
        self.n = 0 # The number of data items aggregated

        self.name = "FrequencyOracle" # Name of the frequency oracle for warning messages, set using .set_name(name)
        self.last_estimated = 0

        if index_mapper is None:
            self.index_mapper = lambda x: x - 1
        else:
            self.index_mapper = index_mapper

    def set_name(self, name):
        """
        Set's freq servers name
        Args:
            name: string - name of frequency oracle
        """
        self.name = name

    def reset(self):
        """
        This method resets the server's aggregated/estimated data and sets n = 0.
        This should be overridden if other parameters need to be reset.
        """
        self.aggregated_data = np.zeros(self.d)
        self.estimated_data = []
        self.last_estimated = 0
        self.n = 0

    def update_params(self, epsilon=None, d=None, index_mapper=None):
        """
        Method to update params of freq oracle server, should be overridden if more options needed.
        This will reset aggregated/estimated data.
        Args:
            epsilon: Optional - privacy budget
            d: Optional - domain size
            index_mapper: Optional - function
        """
        self.epsilon = epsilon if epsilon is not None else self.epsilon # Updating epsilon here will not update any internal probabilities
        # Any class that implements FreqOracleServer, needs to override update_params to update epsilon properly

        self.d = d if d is not None else self.d
        self.index_mapper = index_mapper if index_mapper is not None else self.index_mapper
        self.reset()

    def check_warnings(self, suppress_warnings=False):
        """
        Used during estimation to check warnings
        Args:
            suppress_warnings: Optional boolean - If True suppresses warnings from being output
        """
        if not suppress_warnings:
            if self.n < 10000:
                warnings.warn(self.name + " has only aggregated small amounts of data (n=" + str(self.n) +
                              ") estimations may be highly inaccurate", RuntimeWarning)
            if self.epsilon < 1:
                warnings.warn("High privacy has been detected (epsilon = " + str(self.epsilon) +
                              "), estimations may be highly inaccurate on small datasets", RuntimeWarning)

    def aggregate(self, data):
        """
        The main method for aggregation, should be implemented by a freq oracle server
        Args:
            data: item to estimate frequency of
        """
        assert ("Must implement")

    def aggregate_all(self, data_list):
        """
        Helper method used to aggregate a list of data
        Args:
            data_list: List of private data to aggregate
        """
        for data in data_list:
            self.aggregate(data)

    def check_and_update_estimates(self):
        """
        Used to check if the "cached" estimated data needs re-estimating, this occurs when new data has been aggregated since last
        """
        if self.last_estimated < self.n:  # If new data has been aggregated since the last estimation, then estimate all
            self.last_estimated = self.n
            self._update_estimates()

    def _update_estimates(self):
        """
        Used internally to update estimates, should be implemented
        """
        assert ("Must implement")

    def estimate(self, data, suppress_warnings=False):
        """
        Calculates frequency estimate of given data item, must be implemented
        Args:
            data: data to estimate the frequency warning of
            suppress_warnings: Optional boolean - if true suppresses warnings
        """
        assert ("Must implement")

    def estimate_all(self, data_list, suppress_warnings=False):
        """
        Helper method, given a list of data items, returns a list of their estimated frequencies
        Args:
            data_list: list of data items to estimate
            suppress_warnings: If True, will suppress estimation warnings

        Returns: list of estimates

        """
        self.check_and_update_estimates()
        return [self.estimate(x, suppress_warnings=suppress_warnings) for x in data_list]

    @property
    def get_estimates(self):
        """
        Returns: Estimated data
        """
        return self.estimated_data
