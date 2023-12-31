import threading

from dataorientedai.core.interfaces.IDictionary import IDictionary


class Dictionary(IDictionary):
    def __init__(self):
        self._store = {}

    def __getitem__(self, key: str):
        return self._store[key]

    def __setitem__(self, key: str, value: any):
        self._store[key] = value

    def __contains__(self, key: str):
        return key in self._store.keys()

    def __delitem__(self, key: str):
        del self._store[key]


class ThreadSafeDictionary(IDictionary):
    def __init__(self):
        self._store = {}

    def __getitem__(self, key: str):
        with threading.Lock():
            return self._store.get(key)

    def __setitem__(self, key: str, value: any):
        with threading.Lock():
            self._store[key] = value

    def __contains__(self, key: str):
        with threading.Lock():
            return key in self._store

    def __delitem__(self, key: str):
        with threading.Lock():
            del self._store[key]


ConcurrentDictionary = ThreadSafeDictionary
