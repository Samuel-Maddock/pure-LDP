import numpy as np
import math
from scipy.optimize import fminbound
from pure_ldp.core import FreqOracleServer

# Client-side for histogram-encoding

class HEServer(FreqOracleServer):
    def __init__(self, epsilon, d, use_the=False, theta=None, index_mapper=None):
        """

        Args:
            epsilon: float - the privacy budget
            d: integer - the size of the data domain
            use_the: Optional boolean - If set to true uses Thresholding Histogram Encoding (THE)
            theta: Optional - If passed, will override the optimal theta value (not recommended)
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """

        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.set_name("HEServer")

        self.is_the = use_the

        if use_the is True:
            self.theta = theta
            if theta is None:
                self.theta = self.__find_optimal_theta()

            self.p = 1 - 0.5 * (math.pow(math.e, self.epsilon / 2 * (self.theta - 1)))
            self.q = 0.5 * (math.pow(math.e, -1 * self.theta * (self.epsilon / 2)))

        self.__find_optimal_theta()

    def __find_optimal_theta(self):
        """
        Used internally to calculate the optimal value of theta if using Threshold Histogram Encoding (THE).

        Find the optimal value of theta to use for thresholding, by minimising
        the variance equation for the fixed value of epsilon.

        Returns: float to 4dp - optimal theta parameter

        """
        # Minimise variance for our fixed epsilon to find the optimal theta value for thresholding
        def var(x):
            num = 2 * math.exp(self.epsilon * x / 2) - 1
            denom = math.pow(1 + math.exp((self.epsilon * (x - 1/2))) - 2 * math.exp((self.epsilon * x / 2)), 2)
            return num/denom

        return round(float(fminbound(var, 0.5, 1)), 4)

    def aggregate(self, priv_data):
        """
        Aggregates HE privatised data, to allow us to calculate estimates.

        Args:
            priv_data: Data privatised via he_client.privatise
        """
        # Threshold rounding
        if self.is_the:
            for index, item in enumerate(priv_data):
                if item <= self.theta:
                    priv_data[index] = 0
                else:
                    priv_data[index] = 1

        self.aggregated_data += priv_data
        self.n += 1

    def estimate(self, data, suppress_warnings=False):
        """
        Calculates a frequency estimate of the given data item

        Args:
            data: data item
            suppress_warnings: Optional boolean - Supresses warnings about possible inaccurate estimations

        Returns: float - frequency estimate

        """
        self.check_warnings(suppress_warnings=suppress_warnings)

        index = self.index_mapper(data)

        if self.is_the:
            return (self.aggregated_data[index] - self.n * self.q) / (self.p - self.q)
        else:
            return self.aggregated_data[index]
