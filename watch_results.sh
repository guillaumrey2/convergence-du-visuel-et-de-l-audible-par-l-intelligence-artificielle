#!/bin/bash

WATCHED_DIR=~/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/results
SCLANG_PATH=/usr/local/bin/sclang
SC_SCRIPT_PATH=~/scscript/sound_synthesis.scd

# Monitor the results directory for new JSON files
inotifywait -m -e create --format '%w%f' "$WATCHED_DIR" | while read NEWFILE
do
    if [[ $NEWFILE == *.json ]]; then  # Process only .json files
        echo "New JSON file detected: $NEWFILE"
        
        # Call SuperCollider script
        QT_QPA_PLATFORM=offscreen "$SCLANG_PATH" "$SC_SCRIPT_PATH"
        
        if [[ $? -ne 0 ]]; then
            echo "Error executing SuperCollider script for file: $NEWFILE"
        else
            echo "SuperCollider script executed successfully for file: $NEWFILE"
        fi
    fi
done
