import logging
logger = logging.getLogger(__name__)

class QIHSE:
    def __init__(self, lib_path=None):
        pass
    def store_atom(self, atom_id, features, meta):
        pass
    def create_table(self, table_name, schema):
        pass

class QihseVectorDBBackend:
    FAISS = 1
    HNSW = 2

class QihseQueryMode:
    pass

class QihseDistanceMetric:
    pass

class QihseOpenFlags:
    pass

class QihseMemoryTier:
    pass
