from collections import defaultdict
from collections import Counter

import numpy as np
import itertools
import copy

from pure_ldp.core import FreqOracleServer
from pure_ldp.heavy_hitters._hh_server import HeavyHitterServer
from pure_ldp.frequency_oracles.local_hashing import LHServer

from bitstring import BitArray

class SFPServer(HeavyHitterServer):
    def __init__(self, epsilon, fragment_length, max_string_length, alphabet=None, index_mapper=None, fo_server=None, frag_server=None, padding_char="*", estimator_norm=0):
        """

        Args:
            epsilon: float privacy budget
            fragment_length (int): The length to increase the fragment by on each iteration
            max_string_length (int): maximum size of the strings to find
            alphabet (optional list): The alphabet over which we are privatising strings
            index_mapper (optional func): Index map function
            fo_server (FreqOracleServer): a FreqOracleServer instance, used to estimate the frequency of words (and fragments if frag_server is not defined)
            frag_server (optional FreqOracleServer): Additionally specify a different frequency oracle client for estimating the frequency of fragments
            padding_char (optional str): The character used to pad strings to a fixed length
            estimator_norm (optional int): The normalisation type for the server estimators
                   0 - No Norm
                   1 - Additive Norm
                   2 - Prob Simplex
                   3 (or otherwise) - Threshold cut
        """

        super().__init__(epsilon, 0, max_string_length, fragment_length, alphabet, index_mapper, estimator_norm)

        self.hash_length = 256
        self.padding_char = padding_char
        self.group_size = int(self.max_string_length / self.fragment_length)

        self.fragment_fo_list = []

        if not isinstance(fo_server, FreqOracleServer):
            fo_server = LHServer(epsilon / 2, d=None, use_olh=True)

        fo_server.update_params(epsilon=epsilon / 2)

        word_d = len(self.alphabet)**self.max_string_length
        fragment_d = len(self.alphabet)**self.fragment_length * self.hash_length

        self.word_fo = copy.deepcopy(fo_server)
        self.word_fo.update_params(index_mapper=self.index_mapper)

        if frag_server is not None:
            fo_server = frag_server # TODO: Check this works - Allows the option to use a different FO for fragments vs words

        if fo_server.sketch_based:
            fragment_map = self.index_mapper
        else:
            def fragment_map(x):
                split = x.split("_")
                hash_num, frag = split[0], split[1]
                return int(hash_num) * (self.index_mapper(frag) + 1)

        try:
            self.word_fo.update_params(d=fragment_d, index_mapper=self.index_mapper)
            fo_server.update_params(d=fragment_d, index_mapper=fragment_map)
        except TypeError:
            pass

        for i in range(0, self.group_size):
            oracle = copy.deepcopy(fo_server)
            self.fragment_fo_list.append(oracle)

    def aggregate(self, privatised_hh_data):
        """
        Aggregate data privatised by SFPClient

        Args:
            privatised_hh_data (tuple): a privatised string from SFPClient of the form (privatised_fragment, privatised_word, fragment location)
        """
        priv_fragment, priv_word, l = privatised_hh_data
        self.word_fo.aggregate(priv_word)
        index = int(l/self.fragment_length)
        self.fragment_fo_list[index].aggregate(priv_fragment)
        self.n += 1

    def _split_fragment(self, fragment):
        fragment_split = fragment.split("_", 1)
        return fragment_split[0], fragment_split[1]

    def _generate_fragments(self, alphabet):
        fragment_arr = itertools.product(alphabet, repeat=self.fragment_length)
        fragment_arr = map(lambda x: "".join(x), fragment_arr)
        fragment_arr = itertools.product(map(str, range(0, self.hash_length)), "_", fragment_arr)
        return list(map(lambda x: "".join(x), fragment_arr))

    def find_heavy_hitters(self, k=None, threshold=None):
        """
        Finds the heavy hitters either based on top-k or a threshold

        Args:
            k: int - used to find the top k most frequent strings
            threshold (float): Threshold value (as a decimal)

        Returns: list of top-k frequent candidates (or > threshold candidates),
                    and a list of their estimated frequencies
        """
        if threshold is None and k is None:
            k = 10

        freq_oracle = self.word_fo
        fragment_estimators = self.fragment_fo_list

        D = []

        alphabet = copy.deepcopy(self.alphabet)
        alphabet.add(self.padding_char)

        fragments = self._generate_fragments(alphabet)

        frequency_dict = defaultdict(lambda: Counter())

        def estimate_fragments(key, frag_estimator):
            frag_dict = dict()

            for frag in fragments:
                frag_dict[frag] = frag_estimator.estimate(frag, suppress_warnings=True)

            return key, Counter(frag_dict)

        pool_map = map(estimate_fragments, range(0,self.max_string_length, self.fragment_length), fragment_estimators)

        for item in pool_map:
            frequency_dict[item[0]] = item[1]

        hash_table = defaultdict(lambda: defaultdict(list))

        fragment_indices = np.arange(0, self.max_string_length, step=self.fragment_length)

        for l in fragment_indices:
            if k is not None: # If k is present then find the top-k fragments to build up the heavy hitters
                fragments = frequency_dict.get(l).most_common(k)
            else: # Otherwise we use a threshold to build up heavy hitters
                fragments = filter(lambda x: x[1]/self.n >= threshold, frequency_dict.get(l).items())

            for fragment in fragments:
                key, value = self._split_fragment(fragment[0])
                hash_table[key][l].append(value)

        for dictionary in hash_table.values():
            fragment_list = list(dictionary.values())

            if len(dictionary.keys()) == int(self.max_string_length / self.fragment_length):
                D += list(map(lambda x: str().join(x), itertools.product(*fragment_list)))

        heavy_hitters = D
        frequencies = freq_oracle.estimate_all(heavy_hitters, normalization=self.estimator_norm)

        return heavy_hitters, frequencies
