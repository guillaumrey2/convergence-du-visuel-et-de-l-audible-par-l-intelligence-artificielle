import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

class Watcher:
    DIRECTORY_TO_WATCH = os.path.expanduser("~/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/results")

    def __init__(self):
        self.observer = Observer()
        self.event_handler = Handler()
        self.observer.schedule(self.event_handler, self.DIRECTORY_TO_WATCH, recursive=False)
        
    def run(self):
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
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
            
            # Write the path to a temporary file
            temp_file_path = os.path.expanduser("~/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/temp.txt")
            with open(temp_file_path, 'w') as f:
                f.write(event.src_path)
            
            # Call SuperCollider script
            sclang_path = "/usr/local/bin/sclang"
            sc_script_path = os.path.expanduser("~/scscript/sound_synthesis.scd")
            # Set environment variables to avoid display issues
            env = os.environ.copy()
            env['QT_QPA_PLATFORM'] = 'offscreen'
            subprocess.run([sclang_path, sc_script_path], check=True, env=env)

if __name__ == '__main__':
    w = Watcher()
    w.run()
