from util.counter import *
from util.evaluate import *
import argparse
import json

def get_files(path):
    paths = []
    if os.path.isfile(path):
        # Simple file
        if 'iternums' not in path and path.endswith(".json"):
            paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                print(fname)
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
    wer = {}
    rouge = {}

    for filename in filenames:
        print("filename is ", filename)
        source = []
        predict = []
        gt = []
        with open(filename, "r") as f:
            for line in f:
                o = json.loads(line)
                # source.append(o['prefix'].split()[1:-1])
                # predict.append(o['decoded_predict'].split())
                # gt.append(o['decoded_true'].split()[1:-1])
                for sour, pred, gt_ in zip(o['prefix'], o['decoded_predict'], o['decoded_true']):
                    # print(" ".join(sour[1:-1]))
                    # print(" ".join(pred))
                    # print(" ".join(gt_[1:-1]))
                    # print()
                    source.append(sour[1:-1])
                    predict.append(pred)
                    gt.append(gt_[1:-1])
    
        sb[filename] = selfbleu(predict, 5)
        bleu[filename] = bleu_upto(gt, predict, 5)
        rouge[filename] = rogue(gt, predict, 5)
        self_bleu_gt[filename] = bleu_upto(source, gt, 5)
        dist[filename] = distinct_upto(predict, 5)
        kd[filename] = kld(gt, predict, 1)
        repit[filename] = repetition(predict)
        wer[filename] = self_wer(source, predict)
        m = ms_jaccard(gt, predict, 5)
        msj[filename] = m

        s = set()
        for i in predict:
            s.update(i)
        uniq[filename] = len(s)


    print('--------------------bleu(Up)----------------------')
    for i in bleu.keys():
        print('{:<65}{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *bleu[i]))

    print('--------------------rouge(Up)----------------------')
    for i in bleu.keys():
        print('{:<65}{:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *rouge[i]))

    print('--------------------self-bleu gt(Down)----------------------')
    for i in self_bleu_gt.keys():
        print('{:<65}{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *self_bleu_gt[i]))

    print('--------------------kl-divergence(Down)----------------------')
    for i in kd.keys():
        print('{:<65}{:.4f}'.format(os.path.basename(i), kd[i]))

    print('--------------------ms_jaccard(Up)----------------------')
    for i in msj.keys():
        print('{:<65}{:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i),  *msj[i]))

    print('--------------------self-bleu(Down)----------------------')
    for i in sb.keys():
        print('{:<65}{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *sb[i]))

    print('--------------------distinct(Up)----------------------')
    for i in dist.keys():
        print('{:<65}{:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *dist[i]))

    print('--------------------repetition(Down)----------------------')
    for i in bleu.keys():
        print('{:<65}{:.6f}'.format(os.path.basename(i), repit[i]))

    print('--------------------self-WER(UP)----------------------')
    for i in bleu.keys():
        print('{:<65}{:.6f}'.format(os.path.basename(i), wer[i]))

    print('--------------------uniq_seq(Up)----------------------')
    for i in uniq.keys():
        print('{:<65}'.format(os.path.basename(i)),  uniq[i])

if __name__ =='__main__':
    main()

