import datetime as dt

from dataorientedai.core.interfaces.ICommand import ICommand


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
