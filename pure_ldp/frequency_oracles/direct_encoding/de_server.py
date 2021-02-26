from pure_ldp.core import FreqOracleServer
import math

class DEServer(FreqOracleServer):
    def __init__(self, epsilon, d, index_mapper=None):
        """
        Args:
            epsilon: float - the privacy budget
            d: integer - the size of the data domain
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.update_params(epsilon, d, index_mapper)


    def update_params(self, epsilon=None, d=None, index_mapper=None):
        """
        Updates DEServer parameters. This will reset any aggregated/estimated data
        Args:
            epsilon: optional - privacy budget
            d: optional - domain size
            index_mapper: optional - index_mapper
        """
        super().update_params(epsilon, d, index_mapper)
        if epsilon is not None or d is not None:
            self.const = math.pow(math.e, self.epsilon) + self.d - 1
            self.p = (self.const-self.d+1) / (self.const)
            self.q = 1/self.const

    def aggregate(self, priv_data):
        """
        Used to aggregate privatised data by DEClient.privatise

        Args:
            priv_data: privatised data from DEClient.privatise
        """
        self.aggregated_data[priv_data] += 1
        self.n += 1

    def _update_estimates(self):
        self.estimated_data = (self.const * self.aggregated_data - self.n) / (self.const - self.d)
        return self.estimated_data

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
        self.check_and_update_estimates()
        return self.estimated_data[index]
