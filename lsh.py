"""
lsh.py
======
STEP 6b - Minhashing   : every product -> signature of n numbers
STEP 6c - LSH banding  : signatures -> candidate pairs
"""

import zlib
from collections import defaultdict
from itertools import combinations

import numpy as np

# A large prime for the hash functions h(x) = (alpha*x + beta) mod P.
# 2^31 - 1 is prime and small enough that alpha*x (< 2^62) fits in a 64-bit integer.
PRIME = 2**31 - 1


def element_to_int(element):
    """
    Minhash needs numbers, but our elements are strings like 'abc#2'.
    zlib.crc32 -->  function that turns a string into an
    integer and gives the same number every run.
    """
    return zlib.crc32(element.encode("utf-8")) % PRIME


# -------------------------------------------------------------------------
# STEP 6b - Minhashing
# -------------------------------------------------------------------------
class MinHasher:
    """
    Simulates n random permutations with n random hash functions
        h_k(x) = (alpha_k * x + beta_k) mod P .
    The minhash of a set under h_k is the smallest hash value among its elements
    (element that comes first in the random order from lecture).
    Property used by LSH:  P[minhash_k(A) == minhash_k(B)] = Jaccard(A, B).
    """

    def __init__(self, n_hashes, seed=0):
        rng = np.random.default_rng(seed)
        self.n = n_hashes
        # alpha and beta are the random elements in the hash function --> need to take random draws
        self.alpha = rng.integers(1, PRIME, size=n_hashes, dtype=np.int64) # list
        self.beta = rng.integers(0, PRIME, size=n_hashes, dtype=np.int64) # list

    def signature(self, elements):
        """Signature of one product= vector of n minhash values."""
        # convert each element of the ste to an inetger so we can have single a vector length E
        x = np.array([element_to_int(e) for e in elements], dtype=np.int64)
        # rows = hash functions, columns = elements.
        # (n x 1) * (1 x |E|) -> (n x |E|) matrix of hash values; min over columns.
        # apply hashed to every random draw of alpha and beta
        hashed = (self.alpha[:, None] * x[None, :] + self.beta[:, None]) % PRIME
        # alpha[:, None] turns the list into a column
        # x[None, :] turns the list into a row
        # : means “keep this axis as it is”, and None means “insert a new axis of length 1 here"
        return hashed.min(axis=1)

    def signature_matrix(self, element_lists):
        """Signature matrix: matrix of one row per product and one column per minhash value"""
        sig = np.empty((len(element_lists), self.n), dtype=np.int64)
        for i, elements in enumerate(element_lists):
            if elements:
                sig[i] = self.signature(elements)
            else:
                # A product with no tokens left: give it a unique negative
                # signature so it never collides with anything.
                sig[i] = -(i + 1)
        return sig


# -------------------------------------------------------------------------
# STEP 6c - LSH banding
# -------------------------------------------------------------------------
def lsh_candidate_pairs(signatures, bands, rows):
    """
    Split each signature into  bands of rows
    Within each band, products whose band-part is identical land in the same bucket;
    every pair inside a bucket becomes a candidate pair.
    P[candidate] = 1 - (1 - s^r)^b  (the S-curve).
    Returns a set of pairs (i, j) with i < j.
    """
    assert bands * rows <= signatures.shape[1], "not enough minhashes for b*r"
    candidates = set()
    for band in range(bands):
        # take cols of current band for every item
        part = signatures[:, band * rows:(band + 1) * rows]
        buckets = defaultdict(list)
        for i, row in enumerate(part):
            # row is item $i$’s $r$ numbers in this band.
            # row.tobytes() turns them into one byte string, which can be a dictionary key.
            # Two items get the same key exactly when all r numbers match
            # each key is a bucket which holds indices of items with that band content
            # buckets are separate for each band (band k of prod A must be compared with
            # band k of prod B) --> Cnt mix bands then
            buckets[row.tobytes()].append(i)
        for members in buckets.values():
            if len(members) > 1:
                candidates.update(combinations(members, 2))
                # create every pair (i,j) with i<j
                # which go into the candidates set.
    return candidates


def s_curve_threshold(bands, rows):
    """Approximate similarity at which P[candidate] = 1/2:  t ~ (1/b)^(1/r)."""
    return (1.0 / bands) ** (1.0 / rows)
