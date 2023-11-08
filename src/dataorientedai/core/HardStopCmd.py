from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IDictionary import IDictionary


class HardStopCmd(ICommand):
    def __init__(self, context: IDictionary):
        self.context = context

    def execute(self):
        self.context["can_continue"] = False
