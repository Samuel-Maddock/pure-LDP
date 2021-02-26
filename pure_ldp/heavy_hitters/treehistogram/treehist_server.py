import copy
import itertools

from collections import deque
import heapq
from pure_ldp.core import FreqOracleServer
from pure_ldp.heavy_hitters._hh_server import HeavyHitterServer

from pure_ldp.frequency_oracles import LHServer


class TreeHistServer(HeavyHitterServer):
    def __init__(self, epsilon, max_string_length, fragment_length, alphabet=None, index_mapper=None, fo_server=None, padding_char="*", estimator_norm=0):
        """

        Args:
            epsilon (float): Privacy Budget
            max_string_length: max size of the strings to find
            fragment_length: the length to increase the fragment by on each iteration
            alphabet (optional list): The alphabet over which we are privatising strings
            index_mapper (optional func): Index map function
            fo_client (FreqOracleClient): a FreqOracleClient instance, used to privatise the data
            padding_char (optional str): The character used to pad strings to a fixed length
            estimator_norm (optional int): The normalisation type for the server estimators
                   0 - No Norm
                   1 - Additive Norm
                   2 - Prob Simplex
                   3 (or otherwise) - Threshold cut
        """
        super().__init__(epsilon, 0, max_string_length, fragment_length, alphabet, index_mapper, padding_char, estimator_norm)
        self.num_n_grams = int(max_string_length / fragment_length)  # Number of N-grams

        if alphabet is not None and padding_char in alphabet:
            raise RuntimeError("TreeHistClient was passed a padding character that is in the provided alphabet. The padding character must not be in the alphabet.")

        if isinstance(fo_server, FreqOracleServer):
            server = fo_server
        else:
            server = LHServer(self.epsilon / 2, d=None, use_olh=True)

        server.update_params(epsilon=self.epsilon/2, index_mapper=self.index_mapper)

        self.word_estimator = copy.deepcopy(server)
        self.fragment_estimator = copy.deepcopy(server)

        try:
            # TODO: This is slow for freq oracles that scale with d...
            self.word_estimator.update_params(d=len(self.alphabet) ** self.max_string_length)
            self.fragment_estimator.update_params(d=len(self.alphabet) ** self.max_string_length)
        except TypeError:
            pass

    def aggregate(self, privatised_hh_data):
        """
        Aggregate data privatised by TreeHistClient

        Args:
            privatised_hh_data (tuple): a privatised string from TreeHistClient of the form (privatised_fragment, privatised_word)
        """

        self.fragment_estimator.aggregate(privatised_hh_data[0])
        self.word_estimator.aggregate(privatised_hh_data[1])
        self.n += 1

    def find_heavy_hitters(self, k=None, threshold=None):
        """
        Finds the heavy hitters either based on top-k or a threshold

        Args:
            k: int - used to find the top k most frequent strings
            threshold (float): Threshold value (as a decimal)

        Returns: list of top-k frequent candidates (or > threshold candidates),
                    and a list of their estimated frequencies
        """

        if threshold is None:
            threshold = self.n / 100

        word_length = self.num_n_grams * self.fragment_length
        scaling_factor = self.num_n_grams
        n_gram_set = self._generate_fragments()
        list_n_grams = [s + self.padding_char * (word_length - len(s)) for s in n_gram_set]
        word_queue = deque(list_n_grams)
        candidate_strings = {}

        if k is not None:
            for i in range(0, self.num_n_grams):
                fragments = zip(self.fragment_estimator.estimate_all(word_queue, normalization=self.estimator_norm)*scaling_factor, word_queue)
                top_k = heapq.nlargest(k, fragments)
                word_queue = []
                for item in top_k:
                    current_prefix_after_stripping_empty = item[1].replace(self.padding_char, '')
                    for gram in n_gram_set:
                        toAdd = current_prefix_after_stripping_empty + gram + self.padding_char * (word_length - (len(current_prefix_after_stripping_empty) + self.fragment_length))
                        word_queue.append(toAdd)

            candidate_strings = dict(map(lambda x: (x[1], x[0]), top_k))

        else:
            while word_queue.__len__() != 0:
                current_prefix = word_queue.popleft()
                current_prefix_after_stripping_empty = current_prefix.replace(self.padding_char, '')

                freq_for_current_prefix = int(self.fragment_estimator.estimate(current_prefix) * scaling_factor)

                if freq_for_current_prefix/self.n < threshold:
                    continue

                if len(current_prefix_after_stripping_empty) == word_length:
                    candidate_strings[current_prefix_after_stripping_empty] = freq_for_current_prefix
                    continue

                for gram in n_gram_set:
                    toAdd = current_prefix_after_stripping_empty + gram + self.padding_char * (word_length - (len(current_prefix_after_stripping_empty) + self.fragment_length))
                    word_queue.append(toAdd)

        return list(candidate_strings.keys()), self.word_estimator.estimate_all(candidate_strings.keys(), normalization=self.estimator_norm)
