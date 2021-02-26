from pure_ldp.core import FreqOracleClient
from pure_ldp.heavy_hitters._hh_client import HeavyHitterClient
import math
import random

class PEMClient(HeavyHitterClient):
    def __init__(self, epsilon, start_length, max_string_length, fragment_length, alphabet=None, index_mapper=None, fo_client=None, padding_char="*"):
        """

        Args:
            epsilon (float): Privacy Budget
            start_length: The starting fragment length
            max_string_length: max size of the strings to find
            fragment_length: the length to increase the fragment by on each iteration
            alphabet (optional list): The alphabet over which we are privatising strings
            index_mapper (optional func): Index map function
            fo_client (FreqOracleClient): a FreqOracleClient instance, used to privatise the data
            padding_char (optional str): The character used to pad strings to a fixed length
        """
        super().__init__(epsilon, start_length, max_string_length, fragment_length, alphabet, index_mapper, fo_client, padding_char)

        self.g = math.ceil((self.max_string_length-self.start_length) / self.fragment_length)
        self.client.update_params(index_mapper=self.index_mapper)

    def privatise(self, user_string):
        """
        This method is used to privatise a bit string using PEM
        Args:
            user_string: The string to be privatised

        Returns: Privatised string and the group number

        """

        padded_string = self._pad_string(user_string)
        group = random.randint(0,self.g-1)

        # Must calculate domain size based on the user's group and update client params
        d = len(self.alphabet) ** (self.start_length + (group + 1) * self.fragment_length)
        try:
            self.client.update_params(d=d)
        except TypeError: # Oracles like CMS don't have a d parameter to update
            pass

        fragment_size = self.start_length + (group+1)*self.fragment_length
        fragment = padded_string[0:min(fragment_size, len(padded_string))]

        return self.client.privatise(fragment), group
