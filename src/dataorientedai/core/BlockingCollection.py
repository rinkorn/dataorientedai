from queue import Queue

from dataorientedai.core.interfaces.IQueue import IQueue


class BlockingCollection(IQueue):
    """Должна быть потокобезопасной!!!"""

    def __init__(self, maxsize=0):
        self._queue = Queue(maxsize)

    def put(self, item, block=True, timeout=1):
        # self._queue.put(item, block=block, timeout=timeout)
        self._queue.put(item)

    def get(self):
        return self._queue.get()

    def empty(self):
        return self._queue.empty()

    def qsize(self):
        return self._queue.qsize()
