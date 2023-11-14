from dataorientedai.core.interfaces.ICommand import ICommand


class EmptyCmd(ICommand):
    def execute(self):
        pass
