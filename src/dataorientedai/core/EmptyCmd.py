from dataorientedai.core import ICommand


class EmptyCmd(ICommand):
    def execute(self):
        pass
