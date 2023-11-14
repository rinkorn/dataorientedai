from dataorientedai.core.commands.DoubleRepeateCmd import DoubleRepeateCmd
from dataorientedai.core.commands.RepeateCmd import RepeateCmd
from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IExceptionHandler import IExceptionHandler
from dataorientedai.core.interfaces.IQueue import IQueue


class DoubleRepeaterExcHandler(IExceptionHandler):
    def __init__(self, queue: IQueue):
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
