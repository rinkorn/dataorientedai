import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class MyHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        return super().on_any_event(event)

    def on_closed(self, event):
        return super().on_closed(event)

    def on_opened(self, event):
        return super().on_opened(event)

    def on_moved(self, event):
        return super().on_moved(event)

    def on_deleted(self, event):
        return super().on_deleted(event)

    def on_created(self, event):
        return super().on_created(event)

    def on_modified(self, event):
        if event.is_directory:
            return
        # Реагируйте на изменения в файлах
        print(f"File {event.src_path} has been modified.")


if __name__ == "__main__":
    path = Path().cwd()  # Путь к директории, которую вы хотите отслеживать
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)

    print(f"Отслеживание изменений в директории: {path}")

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
