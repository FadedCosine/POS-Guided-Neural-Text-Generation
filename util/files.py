import os
import glob
import json
import pickle 
def get_files(path):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)
    return paths


def file_iterator(path):
    files = get_files(path)
    for file in files:
        f = open(file, errors='ignore')
        yield f.read()


def check_file(filename):
    return os.path.exists(filename)

def load_pkl(filename):
    with open(filename, 'rb') as f:
        v = pickle.load(f)
    return v
    
def load_json(filename):
    with open(filename, 'r') as f:
        v = json.load(f)
       
    return v