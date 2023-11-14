import abc

from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IUObject import IUObject


class IHardStoppable(abc.ABC):
    @abc.abstractmethod
    def set_can_continue(self, value):
        pass


class HardStoppableAdapter(IHardStoppable):
    def __init__(self, uobject: IUObject):
        self.o = uobject

    def set_can_continue(self, value: bool):
        self.o.__setitem__("can_continue", value)


class HardStopCmd(ICommand):
    def __init__(self, context: IHardStoppable):
        self.o = context

    def execute(self):
        self.o.set_can_continue(False)


if __name__ == "__main__":
    from dataorientedai.core.BlockingCollection import BlockingCollection
    from dataorientedai.core.commands.EmptyCmd import EmptyCmd
    from dataorientedai.core.Dictionary import Dictionary
    from dataorientedai.core.UObject import UObject

    queue = BlockingCollection()

    def process():
        cmd = queue.get()
        try:
            cmd.execute()
            print(f"Executed: {cmd.__class__.__name__}")
        except Exception as e:
            exc = type(e)
            try:
                print(f"Error! {exc} in {type(cmd)}")
            except Exception as e:
                print(f"Fatal error! {exc} in {type(cmd)}")

    # obj = Dictionary()
    obj = UObject()

    obj["can_continue"] = True
    obj["queue"] = queue
    obj["process"] = process
    obj["thread_timeout"] = 10

    obj["queue"].put(EmptyCmd())
    obj["queue"].put(EmptyCmd())
    obj["queue"].put(HardStopCmd(HardStoppableAdapter(obj)))
    obj["queue"].put(EmptyCmd())

    while obj["can_continue"]:
        obj["process"]()
        print(
            f"Queue.qsize: {obj['queue'].qsize()}.\n"
            f"Can continue? {obj['can_continue']}.\n"
        )
