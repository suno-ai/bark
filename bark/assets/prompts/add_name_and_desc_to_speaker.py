import argparse
import numpy as np
import os

def load_npz_file(filepath):
    with np.load(filepath) as data:
        return dict(data)

def save_npz_file(filepath, data):
    np.savez(filepath, **data)

def update_metadata(filepath, metadata):
    data = load_npz_file(filepath)

    for key, value in metadata.items():
        data[key] = value

    save_npz_file(filepath, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add or update metadata in a .npz file. Shows up in --list_speakers output I found myself just using filenames for simplicity.")
    parser.add_argument("filepath", help="Path to the .npz file.")
    parser.add_argument("--name", type=str, help="Short Name of the speaker file.")
    parser.add_argument("--desc", type=str, help="Longer Descriptio of the .npz file.")
    args = parser.parse_args()

    metadata = {}
    if args.name:
        metadata["name"] = args.name
    if args.desc:
        metadata["desc"] = args.desc

    update_metadata(args.filepath, metadata)
