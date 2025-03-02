import sys
import os
import shutil

def copy_and_replace(file1, file2):
    if not file1.endswith(".pbs") or not file2.endswith(".pbs"):
        print("Error: Both files must have a .pbs extension.")
        sys.exit(1)
    
    if not os.path.exists(file1):
        print(f"Error: File '{file1}' not found.")
        sys.exit(1)
    
    # Extract pure filenames and relative paths after 'hpc/'
    name1 = os.path.splitext(os.path.basename(file1))[0]
    name2 = os.path.splitext(os.path.basename(file2))[0]
    
    hpc_index1 = file1.find("hpc/")
    hpc_index2 = file2.find("hpc/")
    
    if hpc_index1 != -1 and hpc_index2 != -1:
        relative_path1 = file1[hpc_index1 + len("hpc/"):]  # Path after 'hpc/'
        relative_path2 = file2[hpc_index2 + len("hpc/"):]  # Path after 'hpc/'
    else:
        relative_path1 = ""
        relative_path2 = ""
    
    # Copy the original file to the new location with the new filename
    shutil.copy(file1, file2)
    
    # Read and modify content
    with open(file2, 'r') as file:
        content = file.read()
    
    # FIRST: Replace all occurrences of the full relative path (excluding 'hpc/')
    if relative_path1 and relative_path2:
        content = content.replace(relative_path1, relative_path2)
    
    # SECOND: Replace all occurrences of name1 with name2
    content = content.replace(name1, name2)
    
    # THIRD: Handle cases where the relative path might be prefixed by other directories
    # Example: $PBS_O_WORKDIR/scripts/analyses/2_pca_repeat.py
    dir1 = os.path.dirname(relative_path1)
    dir2 = os.path.dirname(relative_path2)
    if dir1 and dir2:
        content = content.replace(dir1, dir2)
    
    # Write the modified content to the new file
    with open(file2, 'w') as file:
        file.write(content)
    
    print(f"Created '{file2}' with necessary replacements.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/utils/swapfile.py <file1.pbs> <file2.pbs>")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    copy_and_replace(file1, file2)
