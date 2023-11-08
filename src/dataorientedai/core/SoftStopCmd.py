from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IDictionary import IDictionary


class SoftStopCmd(ICommand):
    def __init__(self, context: IDictionary):
        self.context = context

    def execute(self):
        previous_process = self.context["process"]

        def process():
            previous_process()
            queue = self.context["queue"]
            if queue.qsize() == 0:
                self.context["can_continue"] = False

        self.context["process"] = process
