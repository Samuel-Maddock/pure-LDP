import itertools

from bitstring import BitArray

class HeavyHitterServer():
    def __init__(self, epsilon, start_length, max_string_length, fragment_length, alphabet=None, index_mapper=None, padding_char="*", estimator_norm=0):
        """
        Args:
            epsilon (float): Privacy Budget
            start_length  (int): The starting fragment length
            max_string_length (int): maximum size of the strings to find
            fragment_length: The length to increase the fragment by on each iteration
            alphabet (optional list): The alphabet over which we are privatising strings
            index_mapper (optional func): Index map function
            padding_char (optional str): The character used to pad strings to a fixed length
            estimator_norm (optional int): The normalisation type for the server estimators
                           0 - No Norm
                           1 - Additive Norm
                           2 - Prob Simplex
                           3 (or otherwise) - Threshold cut
        """

        self.epsilon = epsilon
        self.start_length = start_length
        self.max_string_length = max_string_length
        self.fragment_length = fragment_length
        self.padding_char = padding_char
        self.estimator_norm = estimator_norm

        self.n = 0

        if alphabet is not None:
            self.alphabet = alphabet
        else:
            self.alphabet = ["0","1"]

        if index_mapper is not None:
            self.index_mapper = lambda x: index_mapper(x.split(self.padding_char)[0])
        else:
            self.index_mapper = lambda x: BitArray(bin=x.split(self.padding_char)[0]).uint # By default, we assume a binary alphabet with our index mapping bitstrings to ints

    def _generate_fragments(self, fragment_length=None):
        """
        Generates all possible string fragments of a certain length based on the alphabet thats been set
        Returns:
            List of all possible fragments of size fragment_length
        """
        if fragment_length is None:
            fragment_length = self.fragment_length

        return [''.join(comb) for comb in itertools.product(self.alphabet, repeat=fragment_length)]

    def _pad_string(self, string):
        """
        Will truncate or pad a string to the correct max_string_length using the padding_char

        Returns:
            Truncated/Padded string  of size self.max_string_length
        """
        # Pad strings that are smaller than some arbitrary max value
        if len(string) < self.max_string_length:
            string += (self.max_string_length - len(string)) * self.padding_char
        elif len(string) > self.max_string_length:
            string = string[0:self.max_string_length]

        return string

    def aggregate(self, privatised_hh_data):
        """
        Aggregate privatised data for HH server. Need to be implemented by subclass

        """
        raise NotImplementedError("Must implement")

    def find_heavy_hitters(self, k=None, threshold=None):
        """
        Find heavy hitters based off of top-k or a threshold. Needs to be implemented by subclass

        """
        raise NotImplementedError("Must implement")
