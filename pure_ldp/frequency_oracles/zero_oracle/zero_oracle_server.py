from pure_ldp.core import FreqOracleServer
import numpy as np

class ZeroOracleServer(FreqOracleServer):
    def __init__(self, epsilon, d, index_mapper=None):
        """
        Arguments passed to zero oracle are not used. This estimator just predicts 0 frequency no matter the element.
        Args:
            epsilon: float - The privacy budget (Not used)
            d: integer - Size of the data domain (Not used)
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        self.d = d
        self.epsilon = epsilon
        self.index_mapper = index_mapper
        self.estimated_data = np.zeros(self.d)

    def aggregate(self, data):
        """
        Any data passed is ignored.

        Args:
            data: Data to perturb
        """
        pass

    def estimate(self, data, suppress_warnings=False):
        """
        Returns 0 frequency no matter the item passed.

        Args:
            data: data item
            suppress_warnings: Optional boolean - Suppresses warnings about possible inaccurate estimations

        Returns: float - frequency estimate of the data item

        """
        return 0