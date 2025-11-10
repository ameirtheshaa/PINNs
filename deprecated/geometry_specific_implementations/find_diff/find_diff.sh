#!/bin/bash

# Assign the two directories to variables
dir1="/Users/ameirshaa/Dropbox/School/Graduate/CERN/Scripts/PINNs/old/old_cylindersphere/26012024"
dir2="/Users/ameirshaa/Dropbox/School/Graduate/CERN/Scripts/PINNs/current"

# Loop through files in the first directory
for file in "$dir1"/*; do
  # Extract just the filename, no path
  filename=$(basename "$file")
  # Check if the file exists in the second directory
  if [ -f "$dir2/$filename" ]; then
    # If it exists, compare the files and save the output
    echo "Comparing $filename..."
    diff_output="diff_${filename}.txt" # Define the output file name
    # Execute diff and save the output
    diff "$file" "$dir2/$filename" > "$diff_output" || echo "Differences found in $filename. See $diff_output for details."
  else
    echo "$filename does not exist in $dir2"
  fi
done