import abc

from dataorientedai.core.interfaces.IUObject import IUObject


class UObject(IUObject):
    def __init__(self):
        self.store = {}

    def __getitem__(self, key: str):
        return self.store[key]

    def __setitem__(self, key: str, value: any):
        self.store[key] = value

    def __repr__(self):
        return self.store.__repr__()
