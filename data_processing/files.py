import os
import glob
import json
import pandas as pd


def get_files(path):
    paths = []
    if os.path.isfile(path):
        # Simple file
        if 'iternums' not in path:
            paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                if 'iternums' not in fname:
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


def lyrics_iterator(path):
    df = pd.read_csv(path)
    for idx, i in df.iterrows():
        title = i['title']
        lyrics = i['lyrics']
        if type(title) == type(lyrics):
            out = title + lyrics
        else:
            out = lyrics
        yield out


def check_file(filename):
    return os.path.exists(filename)


def load_json(filename):
    with open(filename, 'r') as f:
        v = json.load(f)
    return v
    
def load_pkl(filename):
    with open(filename, 'rb') as f:
        v = pickle.load(f)
    return v

def merges(path, chunk=500000):
    fl = get_files(path)
    dirname = os.path.dirname(fl[0]) + '_chunked'
    if not os.path.exists(dirname): os.makedirs(dirname)
    n_saved = 0
    cnt =0
    new = pd.DataFrame()
    for i in fl:
        df = pd.read_pickle(i)
        cnt+= len(df)
        new = new.append(df)
        if cnt > chunk:
            new = new.reset_index(drop=True)
            new.to_pickle(os.path.join(dirname,'indexed_{}.pkl'.format(n_saved)))
            n_saved +=1
            cnt = 0
            new = pd.DataFrame()
    if len(new):
        new = new.reset_index(drop=True)
        new.to_pickle(os.path.join(dirname, 'indexed_{}.pkl'.format(n_saved)))
        n_saved += 1