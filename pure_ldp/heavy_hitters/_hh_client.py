from pure_ldp.frequency_oracles.local_hashing import LHClient
from pure_ldp.core import FreqOracleClient

from bitstring import BitArray

class HeavyHitterClient():
    def __init__(self, epsilon, start_length, max_string_length, fragment_length, alphabet=None, index_mapper=None, fo_client=None, padding_char="*"):
        """
        Args:
            epsilon (float): Privacy Budget
            start_length  (int): The starting fragment length
            max_string_length (int): maximum size of the strings to find
            fragment_length (int): The length to increase the fragment by on each iteration
            alphabet (optional list): The alphabet over which we are privatising strings
            index_mapper (optional func): Index map function
            fo_client (optional FreqOracleClient): Frequency oracle client used in the heavy hitter. If none is provided will use fast local hashing (FLH).
            padding_char (optional str): The character used to pad strings to a fixed length
        """
        self.epsilon = epsilon
        self.start_length = start_length
        self.max_string_length = max_string_length
        self.fragment_length = fragment_length
        self.padding_char = padding_char

        if alphabet is not None:
            self.alphabet = alphabet
        else:
            self.alphabet = ["0","1"]

        if index_mapper is not None:
            self.index_mapper = lambda x: index_mapper(x.split(self.padding_char)[0])
        else:
            self.index_mapper = lambda x: BitArray(bin=x.split(self.padding_char)[0]).uint # By default, we assume a binary alphabet with our index mapping bitstrings to ints

        if isinstance(fo_client, FreqOracleClient):
            self.client = fo_client
        else:
            self.client = LHClient(self.epsilon, d=None, use_olh=True, index_mapper=self.index_mapper)
            # self.client = create_fo_client_instance("LH", {"epsilon": self.epsilon, "d":None, "use_olh":True, "index_mapper":self.index_mapper})

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

    def privatise(self, user_string):
        """
        Public facing method to privatise user's string
        Args:
            user_string: user's string
        """
        raise NotImplementedError("Must implement")