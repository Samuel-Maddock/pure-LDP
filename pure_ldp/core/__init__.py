
import xxhash

# Base classes for frequency oracles
from ._freq_oracle_client import FreqOracleClient
from ._freq_oracle_server import FreqOracleServer

# Helper functions for generating hash funcs

def generate_hash_funcs(k, m):
    """
    Generates k hash functions that map data to the range {0, 1,..., m-1}
    Args:
        k: The number of hash functions
        m: The domain {0,1,...,m-1} that hash func maps too
    Returns: List of k hash functions
    """
    hash_funcs = []
    for i in range(0, k):
        hash_funcs.append(generate_hash(m, i))
    return hash_funcs


def generate_256_hash():
    """

    Returns: A hash function that maps data to {0,1,... 255}

    """
    return lambda data: xxhash.xxh64(data, seed=10).intdigest() % 256


def generate_hash(m, seed):
    """
    Generate a single hash function that maps data to {0, ... ,m-1}
    Args:
        m: int domain to map too
        seed: int the seed for the hash function

    Returns: Hash function

    """
    return lambda data: xxhash.xxh64(str(data), seed=seed).intdigest() % m