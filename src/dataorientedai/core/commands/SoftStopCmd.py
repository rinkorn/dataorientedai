import abc

from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IUObject import IUObject


class ISoftStoppable(abc.ABC):
    @abc.abstractmethod
    def get_process(self):
        pass

    @abc.abstractmethod
    def set_process(self, value):
        pass

    @abc.abstractmethod
    def get_queue(self):
        pass

    @abc.abstractmethod
    def set_can_continue(self, value):
        pass


class SoftStoppableAdapter(ISoftStoppable):
    def __init__(self, context: IUObject):
        self.o = context

    def get_process(self):
        return self.o.__getitem__("process")

    def set_process(self, value):
        self.o.__setitem__("process", value)

    def get_queue(self):
        return self.o.__getitem__("queue")

    def set_can_continue(self, value):
        self.o.__setitem__("can_continue", value)


class SoftStopCmd(ICommand):
    def __init__(self, context: ISoftStoppable):
        self.o = context

    def execute(self):
        previous_process = self.o.get_process()

        def process():
            previous_process()
            queue = self.o.get_queue()
            if queue.qsize() == 0:
                self.o.set_can_continue(False)

        self.o.set_process(process)


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

    obj = Dictionary()
    # obj = UObject()
    obj["can_continue"] = True
    obj["queue"] = queue
    obj["process"] = process
    obj["thread_timeout"] = 10

    obj["queue"].put(EmptyCmd())
    obj["queue"].put(EmptyCmd())
    obj["queue"].put(SoftStopCmd(SoftStoppableAdapter(obj)))
    obj["queue"].put(EmptyCmd())

    while obj["can_continue"]:
        obj["process"]()
        print(
            f"Queue.qsize: {obj['queue'].qsize()}.\n"
            f"Can continue? {obj['can_continue']}.\n"
        )
