import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Watcher:
    DIRECTORY_TO_WATCH = os.path.expanduser("~/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/results")

    def __init__(self):
        self.observer = Observer()
        self.event_handler = Handler()
        self.observer.schedule(self.event_handler, self.DIRECTORY_TO_WATCH, recursive=False)
        logging.info(f"Initialized watcher for directory: {self.DIRECTORY_TO_WATCH}")

    def run(self):
        self.observer.start()
        logging.info("Observer started")
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            self.observer.stop()
            logging.info("Observer stopped by keyboard interrupt")
        self.observer.join()
        logging.info("Observer joined")

class Handler(FileSystemEventHandler):
    @staticmethod
    def on_created(event):
        if event.is_directory:
            logging.info(f"Ignored directory creation: {event.src_path}")
            return None

        elif event.src_path.endswith(".json"):
            logging.info(f"New JSON file detected: {event.src_path}")
            
            # Write the path to a temporary file
            temp_file_path = os.path.expanduser("~/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/temp.txt")
            with open(temp_file_path, 'w') as f:
                f.write(event.src_path)
            logging.info(f"Path written to temp file: {temp_file_path}")
            
            # Call SuperCollider script
            sclang_path = "/usr/local/bin/sclang"
            sc_script_path = os.path.expanduser("~/scscript/sound_synthesis.scd")
            # Set environment variables to avoid display issues
            env = os.environ.copy()
            env['QT_QPA_PLATFORM'] = 'offscreen'
            try:
                subprocess.run([sclang_path, sc_script_path], check=True, env=env)
                logging.info(f"SuperCollider script executed: {sc_script_path}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Error executing SuperCollider script: {e}")

if __name__ == '__main__':
    logging.info("Starting run_sc.py")
    w = Watcher()
    w.run()
    logging.info("run_sc.py completed")