#!/bin/bash

directory="data/NFCorpus/articles"

# Check if the directory exists
if [ ! -d "$directory" ]; then
    echo "Directory $directory does not exist."
    exit 1
fi

total=0

# Loop through each subdirectory in the given directory
for folder in "$directory"/*/; do
    if [ -d "$folder" ]; then
        count=$(find "$folder" -type f | wc -l)
        echo "$(basename "$folder"): $count files"
        total=$((total + count))
    fi
done

echo "Total: $total files"
