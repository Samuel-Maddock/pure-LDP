from pure_ldp.local_hashing.lh_server import LHServer
from pure_ldp.core import FreqOracleServer

import math
import itertools
import copy
import numpy as np

from bitstring import BitArray
from collections import Counter


class PEMServer:
    def __init__(self, epsilon, domain_size, start_length, segment_length, FOServer=None):
        self.epsilon = epsilon
        self.domain_size = domain_size
        self.segment_length = segment_length
        self.start_length = start_length

        self.g = math.ceil((self.domain_size - self.start_length) / self.segment_length)
        self.oracles = []

        if isinstance(FOServer, FreqOracleServer):
            for i in range(0, self.g):
                oracle = copy.deepcopy(FOServer)
                d = 2 ** (self.start_length + (i + 1) * self.segment_length)
                oracle.update_params(d=d,
                                     index_mapper=lambda x: x) # Some oracles need a domain size
                oracle.reset()
                self.oracles.append(oracle)
        else:
            for i in range(0, self.g):
                self.oracles.append(
                    LHServer(self.epsilon, 2 ** (self.start_length + (i + 1) * self.segment_length), use_olh=True, index_mapper= lambda x:x))

    def aggregate(self, privatised_fragment, group):
        self.oracles[group].aggregate(privatised_fragment)

    def _estimate_top_k(self, oracle, candidates, k):
        # TODO: Faster/nicer way to do this?
        top_k, _ = zip(*Counter(dict(zip(candidates, oracle.estimate_all(candidates, suppress_warnings=True)))).most_common(k))
        return top_k

    def find_top_k(self, k):
        fragment_size = self.start_length + (0 + 1) * self.segment_length
        candidates = range(0, 2 ** fragment_size)
        top_k = self._estimate_top_k(self.oracles[0], candidates, k)

        freq_candidates = list(map(lambda x: BitArray(uint=x, length=fragment_size).bin, top_k))

        for i in range(1, self.g):
            fragment_size = self.start_length + (i + 1) * self.segment_length

            frags = [''.join(comb) for comb in itertools.product(["0", "1"], repeat=self.segment_length)]

            candidates = []
            for frag in frags:
                candidates.extend([BitArray(bin=bs + frag).uint for bs in freq_candidates])

            top_k = self._estimate_top_k(self.oracles[i], candidates, k)

            freq_candidates = list(map(lambda x: BitArray(uint=x, length=fragment_size).bin, top_k))

        freqs = self.g * np.array(self.oracles[self.g-1].estimate_all([BitArray(bin=x).uint for x in freq_candidates], suppress_warnings=True))
        return freq_candidates, freqs
