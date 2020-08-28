from pure_ldp.core import FreqOracleServer
from pure_ldp.frequency_oracles.hadamard_response.internal import k2k_hadamard
import math


class HadamardResponseServer(FreqOracleServer):
    def __init__(self, epsilon, d, index_mapper=None, normalization=0):
        """

        Args:
            epsilon (float): Privacy Budget
            d (int): Domain size
            index_mapper (Optional function): A function that maps domain elements to {0, ... d-1}
            normalisation (Optional int): 0 (default) - No normalisation
                           1 - Normalisation (+ clip to 0)
                           2 - Projects estimates onto the probability simplex
        """
        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.aggregated_data = []
        self.update_params(epsilon, d, index_mapper)
        self.set_name("Hadamard Response")
        self.normalization = normalization

    def get_hash_funcs(self):
        if self.epsilon > 1:
            return self.hr.permute
        else:
            return

    def reset(self):
        """
        Resets aggregated/estimated data to allow for new collection/aggregation
        """
        super().reset()
        self.aggregated_data = [] # For Hadamard response, aggregated data is stored in a list not numpy array

    def update_params(self, epsilon=None, d=None, index_mapper=None):
        """
        Updates HR Server parameters, will reset aggregated/estimated data.
        Args:
            epsilon: optional - privacy budget
            d: optional - domain size
            index_mapper: optional - function
        """
        super().update_params(epsilon, d, index_mapper)
        if d is not None or epsilon is not None:
            if self.epsilon <= 1:
                self.hr = k2k_hadamard.Hadamard_Rand_high_priv(d, self.epsilon, encode_acc=1) # hadamard_response
            else:
                self.hr = k2k_hadamard.Hadamard_Rand_general_original(d, self.epsilon, encode_acc=1)

    def aggregate(self, data):
        """
        Used to aggregate privatised data to the server
        Args:
            data: privatised data item to aggregate
        """
        self.aggregated_data.append(data)
        self.n += 1

    def _update_estimates(self):
        """
        Used internally to update estimates
        Returns: estimated data

        """
        self.estimated_data = self.hr.decode_string(self.aggregated_data, normalization=self.normalization-1) * self.n # k2khadamard using norm=0 for normalisation, 1 for simplex and anything else for none
        return self.estimated_data

    def estimate(self, data, suppress_warnings=False):
        """
        Estimates frequency of given data item
        Args:
            data: data item to estimate
            suppress_warnings: Optional boolean - If True, estimation warnings will not be displayed

        Returns: frequency estimate

        """
        self.check_warnings(suppress_warnings)
        index = self.index_mapper(data)
        self.check_and_update_estimates()
        return self.estimated_data[index]
