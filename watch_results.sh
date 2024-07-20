#!/bin/bash

WATCHED_DIR=~/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/results
SCLANG_PATH=/usr/local/bin/sclang
SC_SCRIPT_PATH=~/scscript/sound_synthesis.scd
LOG_FILE=~/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/watch_results.log

# Create or clear the log file
> "$LOG_FILE"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

log_message "Starting watch_results.sh"

# Monitor the results directory for new JSON files
inotifywait -m -e create --format '%w%f' "$WATCHED_DIR" | while read NEWFILE
do
    log_message "Detected new file: $NEWFILE"

    if [[ $NEWFILE == *.json ]]; then  # Process only .json files
        log_message "New JSON file detected: $NEWFILE"
        
        # Call SuperCollider script
        QT_QPA_PLATFORM=offscreen "$SCLANG_PATH" "$SC_SCRIPT_PATH" >> "$LOG_FILE" 2>&1
        
        if [[ $? -ne 0 ]]; then
            log_message "Error executing SuperCollider script for file: $NEWFILE"
        else
            log_message "SuperCollider script executed successfully for file: $NEWFILE"
        fi
    else
        log_message "Ignored file: $NEWFILE"
    fi
done
