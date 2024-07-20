#!/bin/bash

WATCHED_DIR=~/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/results
SCLANG_PATH=/usr/local/bin/sclang
SC_SCRIPT_PATH=~/scscript/sound_synthesis.scd
LOG_FILE=~/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/watch_results.log
TEMP_FILE=~/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/temp.txt

# Create or clear the log file
> "$LOG_FILE"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

log_message "Starting watch_results.sh"

# Function to restart SuperCollider server
restart_supercollider() {
    log_message "Restarting SuperCollider server..."
    pkill -f sclang
    sleep 1
    QT_QPA_PLATFORM=offscreen "$SCLANG_PATH" "$SC_SCRIPT_PATH" >> "$LOG_FILE" 2>&1
    log_message "SuperCollider server restarted."
}

# Monitor the results directory for new JSON files
inotifywait -m -e create --format '%w%f' "$WATCHED_DIR" | while read NEWFILE
do
    log_message "Detected new file: $NEWFILE"

    if [[ $NEWFILE == *.json ]]; then  # Process only .json files
        log_message "New JSON file detected: $NEWFILE"
        
        # Write the path to a temporary file
        echo "$NEWFILE" > "$TEMP_FILE"
        log_message "Path written to temp file: $TEMP_FILE"
        
        # Restart SuperCollider server
        restart_supercollider
        
        if [[ $? -ne 0 ]]; then
            log_message "Error executing SuperCollider script for file: $NEWFILE"
        else
            log_message "SuperCollider script executed successfully for file: $NEWFILE"
        fi

        log_message "Finished processing file: $NEWFILE"
    else
        log_message "Ignored file: $NEWFILE"
    fi

    log_message "Loop continues..."
done