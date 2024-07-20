#!/bin/bash

WATCHED_DIR=~/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/results
LOG_FILE=~/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/watch_results.log

# Create or clear the log file
> "$LOG_FILE"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

log_message "Starting watch_results.sh"

# Monitor the results directory for new JSON files
while true; do
    log_message "Starting inotifywait loop"
    inotifywait -m -e create --format '%w%f' "$WATCHED_DIR" | while read NEWFILE
    do
        log_message "Detected new file: $NEWFILE"

        if [[ $NEWFILE == *.json ]]; then  # Process only .json files
            log_message "New JSON file detected: $NEWFILE"
        else
            log_message "Ignored file: $NEWFILE"
        fi

        log_message "Loop continues..."
    done
    log_message "inotifywait loop ended, restarting..."
    sleep 1
done
