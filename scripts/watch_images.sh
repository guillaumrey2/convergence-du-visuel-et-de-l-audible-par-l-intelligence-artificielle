#!/bin/bash

WATCHED_DIR="./images"
OUTPUT_DIR="./results"

inotifywait -m -e close_write --format '%w%f' "$WATCHED_DIR" | while read NEWFILE
do
    if [[ $NEWFILE == *.jpg || $NEWFILE == *.png ]]; then  # Process only .jpg and .png files
        echo "Processing new image: $NEWFILE"
        python image_analysis.py "$NEWFILE" "$OUTPUT_DIR"
    fi
done
