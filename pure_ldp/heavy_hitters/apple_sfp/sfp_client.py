import numpy as np
import copy

from pure_ldp.core import generate_256_hash
from pure_ldp.heavy_hitters._hh_client import HeavyHitterClient


class SFPClient(HeavyHitterClient):
    def __init__(self, epsilon, fragment_length, max_string_length, alphabet=None, index_mapper=None, fo_client=None,
                 frag_client=None, padding_char="*"):
        """

        Args:
            epsilon: float privacy budget
            fragment_length (int): The length to increase the fragment by on each iteration
            max_string_length (int): maximum size of the strings to find
            alphabet (optional list): The alphabet over which we are privatising strings
            index_mapper (optional func): Index map function
            fo_client (FreqOracleClient): a FreqOracleClient instance, used to privatise the fragments
            frag_client (optional FreqOracleClient): Additionally specify a different frequency oracle client for privatising fragments
            padding_char (optional str): The character used to pad strings to a fixed length
        """

        super().__init__(epsilon, 0, max_string_length, fragment_length, alphabet, index_mapper, fo_client,
                         padding_char)

        self.hash_length = 256
        self.hash_256 = lambda x: generate_256_hash()(x) % self.hash_length

        self.word_client = self.client

        if frag_client is not None:
            self.fragment_client = copy.deepcopy(self.word_client)
        else:
            self.fragment_client = copy.deepcopy(self.word_client)

        word_d = len(self.alphabet) ** self.max_string_length
        fragment_d = len(self.alphabet) ** self.fragment_length * self.hash_length

        # For non-sketching based FOs and also if the domain is binary
        if fo_client.sketch_based:
            fragment_map = self.index_mapper
        else:
            def fragment_map(x):
                split = x.split("_")
                hash_num, frag = split[0], split[1]
                return int(hash_num) * (self.index_mapper(frag) + 1)

        self.word_client.update_params(epsilon=epsilon / 2, index_mapper=self.index_mapper)
        self.fragment_client.update_params(epsilon=epsilon / 2, index_mapper=fragment_map)

        try:
            # TODO: This is slow for freq oracles that scale with d...
            self.word_client.update_params(d=word_d)
            self.fragment_client.update_params(d=fragment_d)
        except TypeError:
            pass

    def _create_fragment(self, string):
        """
        Used to create SFP fragments which are formed from hashes of the whole word and a random substring of the word
            of lengeth self.fragment_length

        Args:
            string: The string to create a fragment from

        Returns: (r, string,l) - r is the fragment, string is the word itself and l is the index of the substring chosen to form the fragment
        """

        fragment_indices = np.arange(0, len(string), step=self.fragment_length)
        l = np.random.choice(fragment_indices)  # Starting index of the fragment
        r = str(self.hash_256(string)) + "_" + string[l: l + (self.fragment_length)]

        return r, string, l

    def privatise(self, user_string):
        """
        Privatises a user's string for SFP

        Args:
            string: User's string to privatise for SFP

        Returns: (priv_fragment, priv_world, l) Privatised fragment, Privatised String and the index of the substring used to form the fragment

        """
        padded_string = self._pad_string(user_string)
        r, string, l = self._create_fragment(padded_string)
        priv_fragment = self.fragment_client.privatise(r)
        priv_word = self.word_client.privatise(string)
        return priv_fragment, priv_word, l
