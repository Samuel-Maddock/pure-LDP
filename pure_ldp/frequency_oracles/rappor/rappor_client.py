from pure_ldp.core import FreqOracleClient
import random
import math

# Implementation of Google's RAPPOR LDP Protocol
    # This RAPPOR implementation is without the instantaneous RR step
    # This does not guarantee longitudinal privacy

class RAPPORClient(FreqOracleClient):
    def __init__(self, f, m, hash_funcs, num_of_cohorts=8, index_mapper=None):
        """

        The RAPPOR Client

        NOTE: This protocol uses f to control the randomisation and not epsilon
            You can use the methods convert_eps_to_f or convert_f_to_eps to set f based on an epsilon value

        Args:
            f: float - Controls the probability of randomisation. Directly influences epsilon (privacy budget)
            m: integer - The size of the bloom filter that is perturbed by RAPPOR
            hash_funcs: list of funcs - Hash functions used by the bloom filter. The number of hash funcs (per cohort) determines the parameter k
            num_of_cohorts: optional - The number of groups to split users into
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        super().__init__(epsilon=None, d=None, index_mapper=index_mapper)
        self.m = m
        self.f = f
        self.hash_family = hash_funcs
        self.num_of_cohorts = num_of_cohorts

    def update_params(self, epsilon=None, d=None, index_mapper=None, f=None, m=None, num_of_cohorts=None, hash_funcs=None):
        super().update_params(epsilon, d, index_mapper)
        self.f = f if f is not None else self.f
        self.m = m if m is not None else self.m
        self.num_of_cohorts = num_of_cohorts if num_of_cohorts is not None else self.num_of_cohorts
        self.hash_family = hash_funcs if hash_funcs is not None else hash_funcs

    def _perturb(self, data):
        """
        Used internally to perturb data using RAPPOR

        Args:
            data: Bloom filter to perturb

        Returns: peturbed data (bloom filter)

        """
        for i,bit in enumerate(data):
            u = random.random()
            if (bit == 1 and u < (1-0.5*self.f)) or (bit == 0 and u < 1/2*self.f):
                data[i] = 1

        return data

    def privatise(self, data):
        """
        Privatises a user's data using RAPPOR

        Args:
            data: The data to be privatised

        Returns:
            A perturbed bloom filter and the user's cohort number
        """
        index = self.index_mapper(data)
        cohort_num = random.randint(0, self.num_of_cohorts-1)
        b = [0]*self.m
        hash_funcs = self.hash_family[cohort_num]
        for func in hash_funcs:
            hash_index = func(str(index))
            b[hash_index] = 1

        return self._perturb(b), cohort_num


    def convert_eps_to_f(self, epsilon):
        """
        Can be used to convert a privacy budget epsilon to the privacy parameter f for RAPPOR

        Args:
            epsilon: Privacy budget

        Returns: The equivalent privacy parameter f

        """
        return round(1/(0.5*math.exp(epsilon/2)+0.5), 2)

    def convert_f_to_eps(self, f, k):
        """
        Can be used to a certain RAPPORs epsilon value

        Args:
            f: Privacy parameter
            k: The number of hash functions used by RAPPOR

        Returns: Epsilon value

        """
        return math.log(((1-0.5*f) / (0.5*f))**(2*k))