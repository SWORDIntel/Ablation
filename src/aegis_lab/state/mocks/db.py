import os
import json
import logging
import uuid
import threading
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone

class QihseKVStore:
    def __init__(self, table_name, path):
        self.table_name = table_name
        self.path = path
        self.store = {}

    def upsert(self, pk_field, data, vector=None):
        pk = data.get(pk_field)
        self.store[pk] = data
        return data

    def query(self, filters=None, pk_field="id"):
        res = list(self.store.values())
        if filters:
            res = [r for r in res if all(r.get(k) == v for k, v in filters.items())]
        return res
