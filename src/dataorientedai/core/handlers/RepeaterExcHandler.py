from dataorientedai.core.commands.LogWriteCmd import LogWriteCmd
from dataorientedai.core.commands.RepeateCmd import RepeateCmd
from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IExceptionHandler import IExceptionHandler
from dataorientedai.core.interfaces.IQueue import IQueue


class RepeaterExcHandler(IExceptionHandler):
    def __init__(self, queue: IQueue):
        self.queue = queue

    def handle(self, cmd: ICommand, exc: Exception):
        if not isinstance(cmd, RepeateCmd):
            self.queue.put(RepeateCmd(cmd))
        else:
            self.queue.put(LogWriteCmd(cmd, exc))
