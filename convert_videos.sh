#!/bin/bash

# Ensure output directory exists
mkdir -p train-videos
mkdir -p video-downloads

# Iterate over all mp4 files in the downloads directory
for file in video-downloads/*.mp4; do
    # Skip if no files found
    [ -e "$file" ] || continue
    
    basename=$(basename "$file")
    target="train-videos/$basename"
    
    # Only convert if the file doesn't already exist in the target directory
    if [ ! -f "$target" ]; then
        echo "Converting: $basename"
        ffmpeg -i "$file" \
            -vf "crop='trunc(min(iw,ih*2)/2)*2':'trunc(min(iw/2,ih)/2)*2',scale=64:32,fps=30" \
            -c:v libx265 -preset slow -crf 18 -an "$target"
    else
        echo "Skipping: $basename (already converted)"
    fi
done

echo "Done converting videos!"