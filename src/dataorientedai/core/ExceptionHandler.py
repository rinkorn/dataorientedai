import datetime as dt
from collections import defaultdict
from pathlib import Path

from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IExceptionHandler import IExceptionHandler
from dataorientedai.core.interfaces.IQueue import IQueue


# %%
class LogPrintCmd(ICommand):
    def __init__(self, cmd, exc):
        self.cmd = cmd
        self.exc = exc

    def execute(self):
        print(
            f"Time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Command: {self.cmd.__class__.__name__}.\n"
            f"Exception: {self.exc.__name__}.\n"
            f"\n"
        )


class LogWriteCmd(ICommand):
    def __init__(self, cmd, exc):
        self.cmd = cmd
        self.exc = exc

    def execute(self):
        with open(Path("log.txt"), "a") as file:
            file.write(
                f"Time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
                f"Command: {self.cmd.__class__.__name__}.\n"
                f"Exception: {self.exc.__name__}.\n"
                f"\n"
            )


class RepeateCmd(ICommand):
    def __init__(self, cmd: ICommand):
        self.cmd = cmd

    def execute(self):
        self.cmd.execute()


class DoubleRepeateCmd(ICommand):
    def __init__(self, cmd: ICommand):
        self.cmd = cmd

    def execute(self):
        self.cmd.execute()


# %%
class LogPrinterExcHandler(IExceptionHandler):
    def __init__(self, queue: IQueue):
        self.queue = queue

    def handle(self, cmd: ICommand, exc: Exception):
        self.queue.put(LogPrintCmd(cmd, exc))


class LogWriterExcHandler(IExceptionHandler):
    def __init__(self, queue: IQueue):
        self.queue = queue

    def handle(self, cmd: ICommand, exc: Exception):
        self.queue.put(LogWriteCmd(exc))


class RepeaterExcHandler(IExceptionHandler):
    def __init__(self, queue: IQueue):
        self.queue = queue

    def handle(self, cmd: ICommand, exc: Exception):
        if not isinstance(cmd, RepeateCmd):
            self.queue.put(RepeateCmd(cmd))
        else:
            self.queue.put(LogWriteCmd(cmd, exc))


class DoubleRepeaterExcHandler(IExceptionHandler):
    def __init__(self, queue):
        self.queue = queue
        self.store = {}

    def handle(self, cmd: ICommand, exc: Exception):
        key = (cmd.__class__.__name__, exc.__name__)

        if not isinstance(cmd, RepeateCmd | DoubleRepeateCmd):
            self.queue.put(RepeateCmd(cmd))
            return

        if not isinstance(cmd, DoubleRepeateCmd):
            self.queue.put(DoubleRepeateCmd(cmd))
            return


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
