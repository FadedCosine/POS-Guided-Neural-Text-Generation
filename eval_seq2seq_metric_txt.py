from util.counter import *
from util.evaluate import *
import argparse
import json
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
def get_files(path):
    paths = []
    if os.path.isfile(path):
        # Simple file
        if 'iternums' not in path and path.endswith(".txt"):
            paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                if 'iternums' not in path and fname.endswith(".txt"):
                    paths.append(os.path.join(dirpath, fname))
    return paths

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderpath',type=str)
    parser.add_argument('--reference',type=str, default="data/paraNMT/ref.txt")
    parser.add_argument('--source',type=str, default="data/paraNMT/src.txt")
    return parser.parse_args()


def main():
    args = get_args()
    # folderpath = os.path.join(args.folderpath, "topp-{p}-topk-{k}-{usage}".format(p=args.top_p, k=args.top_k, usage=args.usage))
    folderpath = args.folderpath
    filenames = sorted(get_files(folderpath))
    logger.info(filenames)
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
    source = []
    with open(args.source, 'r') as reader:
        for line in reader:
            source.append(line.strip().split())
    gt = []
    with open(args.reference, 'r') as reader:
        for line in reader:
            gt.append(line.strip().split())
    for filename in filenames:
        logger.info("filename is {}".format(filename))
        
        predict = []
        with open(filename, "r") as f:
            for line in f:
                predict.append(line.strip().split())
    
        # sb[filename] = selfbleu(predict, 5)
        bleu[filename] = bleu_upto(gt, predict, 5)
        sb[filename] = bleu_upto(source, predict, 5)
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


    logger.info('--------------------bleu(Up)----------------------')
    for i in bleu.keys():
        logger.info('{:<65}{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *bleu[i]))

    logger.info('--------------------rouge(Up)----------------------')
    for i in bleu.keys():
        logger.info('{:<65}{:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *rouge[i]))

    logger.info('--------------------self-bleu gt(Down)----------------------')
    for i in self_bleu_gt.keys():
        logger.info('{:<65}{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *self_bleu_gt[i]))

    logger.info('--------------------kl-divergence(Down)----------------------')
    for i in kd.keys():
        logger.info('{:<65}{:.4f}'.format(os.path.basename(i), kd[i]))

    logger.info('--------------------ms_jaccard(Up)----------------------')
    for i in msj.keys():
        logger.info('{:<65}{:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i),  *msj[i]))

    logger.info('--------------------self-bleu(Down)----------------------')
    for i in sb.keys():
        logger.info('{:<65}{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *sb[i]))

    logger.info('--------------------distinct(Up)----------------------')
    for i in dist.keys():
        logger.info('{:<65}{:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *dist[i]))

    logger.info('--------------------repetition(Down)----------------------')
    for i in bleu.keys():
        logger.info('{:<65}{:.6f}'.format(os.path.basename(i), repit[i]))

    logger.info('--------------------self-WER(UP)----------------------')
    for i in bleu.keys():
        logger.info('{:<65}{:.6f}'.format(os.path.basename(i), wer[i]))

    logger.info('--------------------uniq_seq(Up)----------------------')
    for i in uniq.keys():
        logger.info('{:<65}{}'.format(os.path.basename(i), uniq[i]))

if __name__ =='__main__':
    main()

