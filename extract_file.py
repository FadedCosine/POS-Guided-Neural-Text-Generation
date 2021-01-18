import argparse
import json
import os

def get_files(path):
    filename = []
        # Directory
    for (dirpath, _, fnames) in os.walk(path):
        for fname in fnames:
            print(fname)
            if 'iternums' not in path and fname.endswith(".json"):
                filename.append(fname)
    return filename

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderpath',type=str)
    return parser.parse_args()

def main():
    args = get_args()
    print(os.path.basename(args.folderpath))
    
    filenames = get_files(args.folderpath)
    print("filenames is ", filenames)
    for filename in filenames:
        
        write_src_file = open(os.path.join(args.folderpath, filename.replace(".json", "_src.txt")), 'w')
        write_tgt_file = open(os.path.join(args.folderpath, filename.replace(".json", "_tgt.txt")), 'w')
        write_hyp_file = open(os.path.join(args.folderpath, filename.replace(".json", "_hyp.txt")), 'w')

        with open(os.path.join(args.folderpath, filename), "r") as f:
            for line in f:
                o = json.loads(line)
                for sour, pred, gt_ in zip(o['prefix'], o['decoded_predict'], o['decoded_true']):
                    write_src_file.write(" ".join(sour[1:-1]) + '\n')
                    write_tgt_file.write(" ".join(gt_[1:-1]) + '\n')
                    write_hyp_file.write(" ".join(pred) + '\n')
        write_src_file.close()
        write_tgt_file.close()
        write_hyp_file.close()


if __name__ =='__main__':
    main()