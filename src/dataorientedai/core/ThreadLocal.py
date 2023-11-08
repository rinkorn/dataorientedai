import abc
import threading
from collections import defaultdict

from dataorientedai.core.interfaces.IThreadLocal import IThreadLocal


# %%
class ThreadLocal(IThreadLocal):
    def __init__(self):
        self._store = threading.local()

    def __getitem__(self, key: str):
        value = None
        if key in self:
            value = self._store.__getattribute__(key)
        return value

    def __setitem__(self, key: str, value: any):
        self._store.__setattr__(key, value)

    def __contains__(self, key):
        return key in self._store.__dict__.keys()


class ThreadLocal2(IThreadLocal):
    def __init__(self):
        self._store = defaultdict(dict)

    def __getitem__(self, key: str):
        thread_id = ThreadLocal._get_thread_id()
        if key not in self._store[thread_id]:
            return None
        return self._store[thread_id][key]

    def __setitem__(self, key: str, value: any):
        thread_id = ThreadLocal._get_thread_id()
        self._store[thread_id][key] = value

    def __contains__(self, key):
        thread_id = ThreadLocal._get_thread_id()
        return key in self._store[thread_id]

    @staticmethod
    def _get_thread_id():
        return threading.get_native_id()
