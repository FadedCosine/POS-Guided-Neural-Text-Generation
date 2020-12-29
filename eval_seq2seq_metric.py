from util.counter import *
from util.evaluate import *
import argparse
import json

def get_files(path):
    paths = []
    if os.path.isfile(path):
        # Simple file
        if 'iternums' not in path and fname.endswith(".json"):
            paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                if 'iternums' not in path and fname.endswith(".json"):
                    paths.append(os.path.join(dirpath, fname))
    return paths

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str)
    parser.add_argument('--folderpath',type=str)
    return parser.parse_args()


def main():
    args = get_args()
    print(os.path.basename(args.folderpath))

    filenames = get_files(args.folderpath)

    sb = {}
    kd = {}
    msj = {}
    uniq = {}
    dist = {}
    bleu = {}
    repit = {}
    self_bleu_gt = {}

    for filename in filenames:
        source = []
        predict = []
        gt = []
        with open(filename, "r") as f:
            for line in f:
                o = json.loads(line)
                source.append(o['prefix'].split()[1:-1])
                predict.append(o['decoded_predict'].split())
                gt.append(o['decoded_true'].split()[1:-1])

        sb[filename] = bleu_upto(source, predict, 5)
        bleu[filename] = bleu_upto(gt, predict, 5)
        self_bleu_gt[filename] = bleu_upto(source, gt, 5)
        dist[filename] = distinct_upto(predict, 5)
        kd[filename] = kld(gt, predict, 1)
        repit[filename] = repetition(predict)
        m = ms_jaccard(gt, predict, 5)
        msj[filename] = m

        s = set()
        for i in predict:
            s.update(i)
        uniq[filename] = len(s)

    print('--------------------self-bleu(Down)----------------------')
    for i in sb.keys():
        print('{:<65}{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *sb[i]))

    print('--------------------bleu(Up)----------------------')
    for i in bleu.keys():
        print('{:<65}{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *bleu[i]))

    print('--------------------self-bleu gt(Down)----------------------')
    for i in self_bleu_gt.keys():
        print('{:<65}{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *self_bleu_gt[i]))

    print('--------------------kl-divergence(Down)----------------------')
    for i in kd.keys():
        print('{:<65}{:.4f}'.format(os.path.basename(i), kd[i]))

    print('--------------------ms_jaccard(Up)----------------------')
    for i in msj.keys():
        print('{:<65}{:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i),  *msj[i]))

    print('--------------------distinct(Up)----------------------')
    for i in dist.keys():
        print('{:<65}{:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *dist[i]))

    print('--------------------repetition(Down)----------------------')
    for i in bleu.keys():
        print('{:<65}{:.6f}'.format(os.path.basename(i), repit[i]))

    print('--------------------uniq_seq(Up)----------------------')
    for i in uniq.keys():
        print('{:<65}'.format(os.path.basename(i)),  uniq[i])

if __name__ =='__main__':
    main()

