
import os

# ========================================
# File Operations Utility Module
# ========================================
def read_file(file_path):
    """Read the contents of a file."""
    with open(file_path, 'r') as file:
        return file.read()

def write_file(file_path, data):
    """Write data to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        file.write(data)

