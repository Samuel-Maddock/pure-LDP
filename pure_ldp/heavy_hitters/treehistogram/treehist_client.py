import numpy as np
import copy

from pure_ldp.heavy_hitters._hh_client import HeavyHitterClient

class TreeHistClient(HeavyHitterClient):
    def __init__(self, epsilon, fragment_length, max_string_length, alphabet=None, index_mapper=None, fo_client=None, padding_char="*"):
        """

        Args:
            epsilon (float): Privacy Budget
            fragment_length: the length to increase the fragment by on each iteration
            max_string_length: max size of the strings to find
            alphabet (optional list): The alphabet over which we are privatising strings
            index_mapper (optional func): Index map function
            fo_client (FreqOracleClient): a FreqOracleClient instance, used to privatise the data
            padding_char (optional str): The character used to pad strings to a fixed length
        """
        super().__init__(epsilon, 0, max_string_length, fragment_length, alphabet, index_mapper, fo_client, padding_char)

        self.num_n_grams = int(max_string_length / fragment_length)  # Number of N-grams

        if (alphabet is not None) and (padding_char in alphabet):
            raise RuntimeError("TreeHistClient was passed a padding character that is in the provided alphabet. The padding character must not be in the alphabet.")

        self.client.update_params(epsilon=epsilon/2, index_mapper=self.index_mapper)

        self.word_estimator = copy.deepcopy(self.client)
        self.fragment_estimator = copy.deepcopy(self.client)

        try:
            # TODO: Rework this - This is slow for freq oracles that scale with d...
            self.word_estimator.update_params(d=len(self.alphabet) ** self.max_string_length)
            self.fragment_estimator.update_params(d=len(self.alphabet) ** self.max_string_length)
        except TypeError:
            pass

    def _choose_random_n_gram_prefix(self, word, N):
        """
        This method is used to choose a random n-gram (n-length prefix) from a word

        Args:
            word (str): The string to choose a prefix from
            N (int): Length of the prefix

        Returns: Random N-length prefix

        """
        assert len(word) % N == 0, 'Word = ' + word + ' is not of correct length'
        random_start_index = np.random.randint(0, len(word) / N) * N
        random_prefix_word = word[0:random_start_index + N] + self.padding_char * (self.fragment_length * self.num_n_grams - len(word[0:random_start_index + N]))
        return random_prefix_word

    def privatise(self, user_string):
        """
        This method is used to privatise a bit string using TreeHistogram

        Args:
            user_string: The string to be privatised

        Returns: Privatised fragment and privatised word

        """
        padded_string = self._pad_string(user_string)
        fragment = self._choose_random_n_gram_prefix(padded_string, self.fragment_length)

        priv_frag = self.fragment_estimator.privatise(fragment)
        priv_word = self.word_estimator.privatise(padded_string)

        return priv_frag, priv_word

