import numpy as np
import math
import warnings

class UEServer:
    def __init__(self, epsilon, d, use_oue=False, index_mapper=None):
        """
        Args:
            epsilon: float - the privacy budget
            d: integer - the size of the data domain
            use_oue: Optional boolean - If True, will use Optimised Unary Encoding (OUE)
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        self.epsilon = epsilon
        self. d = d
        self.n = 0
        self.aggregated_data = np.zeros(d)

        const = math.pow(math.e, self.epsilon/2)
        self.p = const / (const + 1)
        self.q = 1-self.p

        if use_oue is True:
            self.p = 0.5
            self.q = 1/(math.pow(math.e, self.epsilon) + 1)

        if index_mapper is None:
            self.index_mapper = lambda x: x-1
        else:
            self.index_mapper = index_mapper

    def aggregate(self, priv_data):
        """
        Used to aggregate privatised data by ue_client.privatise

        Args:
            priv_data: privatised data from ue_client.privatise
        """
        self.aggregated_data += priv_data
        self.n += 1

    def estimate(self, data, supress_warnings=False):
        """
        Calculates a frequency estimate of the given data item

        Args:
            data: data item
            supress_warnings: Optional boolean - Supresses warnings about possible inaccurate estimations

        Returns: float - frequency estimate

        """
        if not supress_warnings:
            if self.n < 10000:
                warnings.warn("UEServer has only aggregated small amounts of data (n=" + str(self.n) + ") estimations may be highly inaccurate", RuntimeWarning)
            if self.epsilon < 1:
                warnings.warn("High privacy has been detected (epsilon = " + str(self.epsilon) + "), estimations may be highly inaccurate on small datasets", RuntimeWarning)

        index = self.index_mapper(data)
        return (self.aggregated_data[index] - self.n*self.q)/(self.p-self.q)

