#!/bin/bash

TARGET_DIR="."

# find "$TARGET_DIR": starts the search from the specified directory.
# -type f: only matches regular files.
# -name "*.db": matches files ending with .db.
# -exec rm -f {} \;: deletes each matched file.
find "$TARGET_DIR" -type f -name "*.db" -print -exec rm -f {} \;
