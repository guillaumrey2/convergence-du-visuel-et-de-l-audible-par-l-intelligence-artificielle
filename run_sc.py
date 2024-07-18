import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

class Watcher:
    DIRECTORY_TO_WATCH = "~/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/results"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=False)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Observer Stopped")

        self.observer.join()

class Handler(FileSystemEventHandler):

    @staticmethod
    def on_created(event):
        if event.is_directory:
            return None

        elif event.src_path.endswith(".json"):
            print(f"New JSON file detected: {event.src_path}")
            
            # Escape backslashes in the file path
            escaped_path = event.src_path.replace("\\", "\\\\")
            
            # Write the escaped path to a temporary file
            temp_file_path = "~/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/temp.txt"
            with open(temp_file_path, 'w') as f:
                f.write(escaped_path)
            
            # Call SuperCollider script
            sclang_path = "/usr/local/bin/sclang"
            sc_script_path = "~/scscript/sound_synthesis.scd"
            subprocess.run([sclang_path, sc_script_path], check=True)

if __name__ == '__main__':
    w = Watcher()
    w.run()
