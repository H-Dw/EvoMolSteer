"""Bounded-memory checks for optional, regeneratable node/edge details."""
import hashlib
import json
import numpy as np
from .io import write_json


class ArrayAudit:
    """Retain exact calculation fingerprints without writing the dense matrix."""
    def __init__(self, columns):
        self.columns = list(columns)
        self.rows = 0
        self.finite = np.zeros(len(columns), dtype=np.int64)
        self.maximum = np.zeros(len(columns))
        self.hash = hashlib.sha256()
        self.partitions = []

    def update(self, label, values):
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != len(self.columns):
            raise ValueError('Unexpected diagnostic array shape')
        finite = np.isfinite(values)
        self.rows += len(values)
        self.finite += finite.sum(0)
        if len(values):
            self.maximum = np.maximum(self.maximum, np.where(finite,np.abs(values),0).max(0))
        h = hashlib.sha256(json.dumps([values.dtype.str,list(values.shape)],separators=(',',':')).encode())
        h.update(values.tobytes(order='C'))
        sha = h.hexdigest()
        self.hash.update(sha.encode())
        self.partitions.append({'label':label,'rows':len(values),'sha256':sha})

    def save(self, path):
        write_json(path, {'columns':self.columns,'rows':self.rows,'finite_counts':self.finite,
            'max_abs_finite':self.maximum,'partition_hash':self.hash.hexdigest(),
            'partitions':self.partitions,'purpose':'All rates computed; detail can be regenerated from the retained structural source'})
