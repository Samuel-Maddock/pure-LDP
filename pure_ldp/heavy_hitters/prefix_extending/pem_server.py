from pure_ldp.frequency_oracles.local_hashing import LHServer
from pure_ldp.core import FreqOracleServer
from pure_ldp.heavy_hitters._hh_server import HeavyHitterServer

import math
import copy
import numpy as np

from collections import Counter

class PEMServer(HeavyHitterServer):
    def __init__(self, epsilon, start_length, max_string_length, fragment_length, alphabet=None, index_mapper=None, fo_server=None, padding_char="*", estimator_norm=0):
        """
        Args:
            epsilon: float privacy budget
            start_length: starting size of the fragment
            max_string_length (int): maximum size of the strings to find
            fragment_length (int): The length to increase the fragment by on each iteration
            alphabet (optional list): The alphabet over which we are privatising strings
            index_mapper (optional func): Index map function
            fo_server: instance of FreqOracleServer to aggregate and estimate the heavy hitters
            padding_char (optional str): The character used to pad strings to a fixed length
            estimator_norm (optional int): The normalisation type for the server estimators
                   0 - No Norm
                   1 - Additive Norm
                   2 - Prob Simplex
                   3 (or otherwise) - Threshold cut
        """

        super(PEMServer, self).__init__(epsilon, start_length, max_string_length, fragment_length, alphabet, index_mapper, padding_char, estimator_norm)
        self.g = math.ceil((self.max_string_length - self.start_length) / self.fragment_length)
        self.oracles = []

        if isinstance(fo_server, FreqOracleServer):
            for i in range(0, self.g):
                oracle = copy.deepcopy(fo_server)
                d = len(self.alphabet) ** (self.start_length + (i + 1) * self.fragment_length)
                try:
                    oracle.update_params(d=d, index_mapper=self.index_mapper) # Some oracles need a domain size
                except TypeError:
                    oracle.update_params(index_mapper=self.index_mapper)

                oracle.reset()
                self.oracles.append(oracle)
        else:
            for i in range(0, self.g):
                self.oracles.append(
                    LHServer(self.epsilon, len(self.alphabet) ** (self.start_length + (i + 1) * self.fragment_length), use_olh=True, index_mapper=self.index_mapper))

    def aggregate(self, privatised_hh_data):
        """
        Aggregate data privatised by PEMClient

        Args:
            privatised_hh_data (tuple): a privatised string from PEMClient of the form (privatised_fragment, group)
        """
        privatised_fragment, group = privatised_hh_data
        self.oracles[group].aggregate(privatised_fragment)
        self.n += 1

    def _estimate_freq_fragments(self, oracle, candidates, k=None, threshold=None):
        """

        Estimate the frequency of a set of candidate fragments using the oracle that is passed

        Args:
            oracle: frequency oracle (FreqOracleServer instance)
            candidates: a list of candidate strings
            k: int - used to find the top k most frequent strings
            threshold (float): Used to find heavy hitters based on threshold frequency

        Returns: Iterable of (candidate, frequency) pairs containing

        """
        estimates = dict(zip(candidates, self.g* np.array(oracle.estimate_all(candidates, suppress_warnings=True, normalization=self.estimator_norm))))

        if k is not None:
            return Counter(estimates).most_common(k) # Find top-k frequent fragments
        else:
            return filter(lambda item: (item[1])/self.n >= threshold, estimates.items()) # Find possible fragments that are greater frequency than a certain threshold

    def find_heavy_hitters(self, k=None, threshold=None):
        """
        Finds the heavy hitters either based on top-k or a threshold

        Args:
            k: int - used to find the top k most frequent strings
            threshold (float): Threshold value (as a decimal)

        Returns: list of top-k frequent candidates (or > threshold candidates),
                    and a list of their estimated frequencies
        """

        if k is None and threshold is None:
            k = 10

        # First group estimation
        fragment_size = self.start_length + self.fragment_length
        starting_fragments = self._generate_fragments(fragment_size)

        freq_candidates = list(self._estimate_freq_fragments(self.oracles[0], starting_fragments, k, threshold))
        # Set of possible fragments to be added at each stage
        inc_frags = self._generate_fragments()

        for i in range(1, self.g):
            # Form new fragments
            candidates = []
            for frag in inc_frags:
                new_frags = [bs[0] + frag for bs in freq_candidates]
                candidates.extend(new_frags)
            freq_candidates = self._estimate_freq_fragments(self.oracles[i], candidates, k, threshold) # Estimate top k (or threshold) of the new fragments

        if threshold:
            freq_candidates = sorted(freq_candidates, key=lambda item: item[1], reverse=True)
        try:
            heavy_hitters, frequencies = zip(*list(freq_candidates))
        except ValueError:
            heavy_hitters = []
            frequencies  = []

        return heavy_hitters, frequencies

