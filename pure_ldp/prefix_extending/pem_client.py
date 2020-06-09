from pure_ldp.local_hashing.lh_client import LHClient

import math
import random

from bitstring import BitArray

class PEMClient:
    def __init__(self, epsilon, domain_size, start_length, segment_length):
        self.epsilon = epsilon
        self.domain_size = domain_size
        self.segment_length = segment_length
        self.start_length = start_length

        self.g = math.ceil((self.domain_size-self.start_length)/self.segment_length)
        self.client = LHClient(self.epsilon, d=None, use_olh=True)

    def privatise(self, bit_string):
        group = random.randint(0,self.g-1)
        fragment_size = self.start_length + (group+1)*self.segment_length
        fragment = bit_string[0:min(fragment_size, len(bit_string))]
        num = BitArray(bin=fragment).uint
        return self.client.privatise(num), group
