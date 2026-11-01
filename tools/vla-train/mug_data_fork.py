"""Recollected mug data: the parent's train-only normalization carries over byte-for-byte to the new dataset."""
import hashlib,json

SCHEMA='hsr-mug-data-fork-v1'

def normalization_sha256(normalization):return hashlib.sha256(json.dumps(normalization,sort_keys=True).encode()).hexdigest()

def verify_inherited_normalization(fork,normalization):
    """The new train episodes did not produce these statistics; they must be exactly the parent's."""
    if fork.get('schema')!=SCHEMA or normalization_sha256(normalization)!=fork.get('parentNormalizationSha256'):
        raise ValueError('Mug data fork must keep the parent normalization byte-for-byte')
