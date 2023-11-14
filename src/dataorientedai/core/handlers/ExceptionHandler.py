from collections import defaultdict

from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IExceptionHandler import IExceptionHandler


class ExceptionHandler(IExceptionHandler):
    def __init__(self):
        self.store = defaultdict(dict)

    def setup(self, cmd: ICommand, exc: Exception, lambda_func: callable):
        # cmd: Move,
        # exc: ValueError,
        # (cmd, exc) => queue.put(LogPrinterCmd(cmd, exc))
        cmd_key = cmd.__name__
        exc_key = exc.__name__
        self.store[cmd_key][exc_key] = lambda_func

    def handle(self, cmd: ICommand, exc: Exception):
        cmd_key = cmd.__class__.__name__
        exc_key = exc.__name__
        lambda_func = self.store[cmd_key][exc_key]
        lambda_func(cmd, exc)
