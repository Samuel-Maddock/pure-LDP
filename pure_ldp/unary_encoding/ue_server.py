import numpy as np
import math

from pure_ldp.core import FreqOracleServer


class UEServer(FreqOracleServer):
    def __init__(self, epsilon, d, use_oue=False, index_mapper=None):
        """
        Args:
            epsilon: float - the privacy budget
            d: integer - the size of the data domain
            use_oue: Optional boolean - If True, will use Optimised Unary Encoding (OUE)
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.set_name("UEServer")

        const = math.pow(math.e, self.epsilon/2)
        self.p = const / (const + 1)
        self.q = 1-self.p

        if use_oue is True:
            self.p = 0.5
            self.q = 1/(math.pow(math.e, self.epsilon) + 1)


    def aggregate(self, priv_data, **kwargs):
        """
        Used to aggregate privatised data by ue_client.privatise

        Args:
            priv_data: privatised data from ue_client.privatise
        """
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
        return (self.aggregated_data[index] - self.n*self.q)/(self.p-self.q)

