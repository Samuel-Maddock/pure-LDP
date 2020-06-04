from pure_ldp.local_hashing.lh_server import LHServer

import math
import itertools

from bitstring import BitArray
from collections import Counter


class PEMServer:
    def __init__(self, epsilon, domain_size, start_length, segment_length):
        self.epsilon = epsilon
        self.domain_size = domain_size
        self.segment_length = segment_length
        self.start_length = start_length

        self.g = math.ceil((self.domain_size - self.start_length) / self.segment_length)
        self.oracles = []
        for i in range(0, self.g):
            self.oracles.append(
                LHServer(self.epsilon, 2 ** (self.start_length + (i + 1) * self.segment_length), use_olh=True))

    def aggregate(self, privatised_fragment, group, seed):
        self.oracles[group].aggregate(privatised_fragment, seed)

    def find_top_k(self, k):
        fragment_size = self.start_length + (0 + 1) * self.segment_length
        candidates = range(0, 2 ** fragment_size)
        top_k = dict(Counter(dict(map(lambda x: (x, self.oracles[0].estimate(x)), candidates))).most_common(4))

        freq_candidates = list(map(lambda x: BitArray(uint=x, length=fragment_size).bin, top_k.keys()))

        for i in range(1, self.g):
            fragment_size = self.start_length + (i + 1) * self.segment_length

            frags = [''.join(comb) for comb in itertools.product(["0", "1"], repeat=self.segment_length)]

            candidates = []
            for frag in frags:
                candidates.extend([BitArray(bin=bs + frag).uint for bs in freq_candidates])

            top_k = dict(Counter(dict(map(lambda x: (x, self.oracles[i].estimate(x)), candidates))).most_common(k))

            freq_candidates = list(map(lambda x: BitArray(uint=x, length=fragment_size).bin, top_k.keys()))

        return freq_candidates
