import numpy as np
import math

# Client-side for unary-encoding
    # By default parameters are set for Symmetric Unary Encoding (SUE)
    # If is_oue=True is passed to the constructor then it uses Optimised Unary Encoding (OUE)

class UEClient:
    def __init__(self, epsilon, d, use_oue=False, index_mapper=None):
        """

        Args:
            epsilon: float - privacy budget
            d: integer - the size of the data domain
            use_oue: Optional boolean - if True, will use Optimised Unary Encoding (OUE)
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        self.epsilon = epsilon
        self.d = d

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

    def __perturb(self, index):
        """
        Used internally to peturb data using unary encoding

        Args:
            index: the index corresponding the data item

        Returns: privatised data vector

        """
        oh_vec = np.random.choice([1, 0], size=self.d, p=[self.q, 1-self.q])  # If entry is 0, flip with prob q
        oh_vec[index] = np.random.choice([1, 0], p=[self.p, 1-self.p]) # If entry is 1, keep as 1 with prob p
        return oh_vec

    def privatise(self, data):
        """
        Privatises a user's data item using unary encoding.

        Args:
            data: data item

        Returns: privatised data vector

        """
        index = self.index_mapper(data)
        return self.__perturb(index)
