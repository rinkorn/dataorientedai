from dataorientedai.core.commands.LogPrintCmd import LogPrintCmd
from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IExceptionHandler import IExceptionHandler
from dataorientedai.core.interfaces.IQueue import IQueue


class LogPrinterExcHandler(IExceptionHandler):
    def __init__(self, queue: IQueue):
        self.queue = queue

    def handle(self, cmd: ICommand, exc: Exception):
        self.queue.put(LogPrintCmd(cmd, exc))
