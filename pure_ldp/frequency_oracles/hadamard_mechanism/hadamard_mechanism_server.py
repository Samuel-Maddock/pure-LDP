from pure_ldp.core import FreqOracleServer
import math
import numpy as np

class HadamardMechServer(FreqOracleServer):
    def __init__(self, epsilon, d, t, use_optimal_t=False, index_mapper=None):
        super().__init__(epsilon, d, index_mapper)
        if use_optimal_t is True:
            self.t = math.ceil(math.log((math.e**self.epsilon + 1), 2))
        else:
            self.t = t

        self.p = (math.exp(self.epsilon) + 2**(self.t-1)-1) / (math.exp(self.epsilon)+2**(self.t) - 1)

    def update_params(self, epsilon=None, d=None, index_mapper=None):
        super().update_params(epsilon, d, index_mapper)

        if epsilon is not None:
            self.p = (math.exp(self.epsilon) + 2 ** (self.t - 1) - 1) / (math.exp(self.epsilon) + 2 ** (self.t) - 1)

    def fwht(self, a) -> None:
        """In-place Fast Walsh–Hadamard Transform of array a."""
        h = 1
        while h < len(a):
            for i in range(0, len(a), h * 2):
                for j in range(i, i + h):
                    x = a[j]
                    y = a[j + h]
                    a[j] = x + y
                    a[j + h] = x - y
            h *= 2

        return a

    def _calculate_p(self, epsilon):
        return math.exp(epsilon) / (1 + math.exp(epsilon))

    def aggregate(self, data):
        """
        Used to aggregate privatised data to the server
        Args:
            data: privatised data item to aggregate
        """
        for item in data:
            self.aggregated_data[item[0]] += item[1]

        self.n += 1

    def _update_estimates(self):
        """
        Used internally to update estimates
        Returns: estimated data

        """
        #had_coeffs = np.array([[self._hadamard_coefficient(i,j) for j in range(0,self.d)] for i in range(0,self.d)])
        #self.estimated_data = np.matmul(had_coeffs, self.aggregated_data / (2 * self.p - 1))
        self.estimated_data = np.array(self.fwht(self.aggregated_data / (2 * self.p - 1)))

    def _hadamard_coefficient(self, x,y):
        return ((-1)**self._bin_dot_product(x,y))

    def _bin_dot_product(self, x, y):
        size =  len(bin(self.d)[2:]) # Get max length of the binary string
        bin_x = bin(x)[2:].zfill(size)
        bin_y = bin(y)[2:].zfill(size)
        return self._count_bin_ones(bin_x, bin_y)

    def _count_bin_ones(self, bin1, bin2):
        count  = 0
        for i, bit in enumerate(bin1):
            if bit=="1" and bin2[i] == "1":
                count += 1
        return count

    def estimate(self, data, suppress_warnings=False):
        """
        Estimates frequency of given data item
        Args:
            data: data item to estimate
            suppress_warnings: Optional boolean - If True, estimation warnings will not be displayed

        Returns: frequency estimate

        """
        self.check_warnings(suppress_warnings)
        self.check_and_update_estimates()

        index = self.index_mapper(data)
        return self.estimated_data[index] / self.t