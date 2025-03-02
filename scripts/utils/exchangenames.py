#!/usr/bin/env python3

import os

def replace_strings_in_files(root_dir):
    """
    Recursively goes through every file in the given directory (root_dir),
    replacing:
        "CLEAN" with "NEUTRAL"
        "clean" with "neutral"
        "TYPO"  with "JAILBREAK"
        "typo"  with "jb"
    Modifies and saves files only if at least one replacement occurs.
    Prints "Modified <filename>" each time a file is changed.
    """
    
    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            
            # Attempt to read the file as text with error ignoring
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    original_content = f.read()
            except OSError:
                # If for some reason the file can't be read, skip it
                continue
            
            # Perform the replacements
            updated_content = original_content
            updated_content = updated_content.replace("CLEAN", "NEUTRAL")
            updated_content = updated_content.replace("clean", "neutral")
            updated_content = updated_content.replace("TYPO", "JAILBREAK")
            updated_content = updated_content.replace("typo", "jb")
            
            # Only write back if there's a change
            if updated_content != original_content:
                try:
                    with open(filepath, "w", encoding="utf-8", errors="ignore") as f:
                        f.write(updated_content)
                    print(f"Modified {filepath}")
                except OSError:
                    # If for some reason the file can't be written, skip it
                    continue

if __name__ == "__main__":
    # Specify the directory you want to process here
    # or modify to accept an argument from sys.argv if desired.
    root_dir = "scripts/analyses"
    
    replace_strings_in_files(root_dir)
