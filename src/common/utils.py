# -*- coding: utf-8 -*-
def save_file(file_path: str, content: str):
    """Save a file."""
    with open(file_path, "w") as f:
        f.write(content)
