from data_processing.files import *
from util.counter import *
from util.evaluate import *
import argparse


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

    for filename in filenames:

        df = pd.read_pickle(filename)
        predict = df['decoded_predict']
        gt = df['decoded_true']

        b = selfbleu(predict, 5)
        sb[filename] = b
        b = bleu_upto(gt, predict, 5)
        repit[filename] = repetition(predict)
        bleu[filename] = b
        dist[filename] = distinct_upto(predict, 5)
        kd[filename] = kld(gt, predict, 1)

        m = ms_jaccard(gt, predict, 5)
        msj[filename] = m

        s = set()
        for i in predict:
            s.update(i)
        uniq[filename] = len(s)

    print('--------------------repetition(Down)----------------------')
    for i in bleu.keys():
        print('{:<65}{:.6f}'.format(os.path.basename(i), repit[i]))


    print('--------------------self-bleu(Down)----------------------')
    for i in sb.keys():
        print('{:<65}{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *sb[i]))

    print('--------------------kl-divergence(Down)----------------------')
    for i in kd.keys():
        print('{:<65}{:.4f}'.format(os.path.basename(i), kd[i]))

    print('--------------------ms_jaccard(Up)----------------------')
    for i in msj.keys():
        print('{:<65}{:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i),  *msj[i]))

    print('--------------------distinct(Up)----------------------')
    for i in dist.keys():
        print('{:<65}{:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *dist[i]))

    print('--------------------Rep(Down)----------------------')
    for i in dist.keys():
        print('{:<65}{:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *rep_1[i]))

    print('--------------------Wrong Rep(Down)----------------------')
    for i in dist.keys():
        print('{:<65}{:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *w_rep_1[i]))

    print('--------------------uniq_seq(Up)----------------------')
    for i in uniq.keys():
        print('{:<65}'.format(os.path.basename(i)),  uniq[i])


if __name__ =='__main__':
    main()

