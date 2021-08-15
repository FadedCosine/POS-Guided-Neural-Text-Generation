from util.counter import *
from util.evaluate import *
import argparse
import json
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
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
                if 'iternums' not in path and fname.endswith(".json"):
                    paths.append(os.path.join(dirpath, fname))
    return paths

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderpath',type=str)
    parser.add_argument('--usage',type=str, default="temp-1")
    parser.add_argument('--top-p',type=float)
    parser.add_argument('--top-k',type=int)
    parser.add_argument('--analyse-pos',action="store_true")
    parser.add_argument('--ctrl-pos',type=str, default="JJ")
    return parser.parse_args()


def main():
    args = get_args()
    # folderpath = os.path.join(args.folderpath, "topp-{p}-topk-{k}-{usage}".format(p=args.top_p, k=args.top_k, usage=args.usage))
    folderpath = args.folderpath
    # logger.info("=" * 20 + "topp-{p}-topk-{k}-{usage}".format(p=args.top_p, k=args.top_k, usage=args.usage) + "=" * 20)
    filenames = sorted(get_files(folderpath))
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
    avg_pos_pred = {}
    avg_pos_src = {}
    avg_pos_gt = {}

    for filename in filenames:
        logger.info("filename is {}".format(filename))
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
                    # logger.info(" ".join(sour[1:-1]))
                    # logger.info(" ".join(pred))
                    # logger.info(" ".join(gt_[1:-1]))
                    # logger.info()
                    source.append(sour[1:-1])
                    predict.append(pred[:-1])
                    gt.append(gt_[1:-1])
    
        # sb[filename] = selfbleu(predict, 5)
        # print("predict is : ", predict)
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
        if args.analyse_pos:
            avg_pos_pred[filename] = count_avg_pos(predict, args.ctrl_pos)
            avg_pos_src[filename] = count_avg_pos(source, args.ctrl_pos)
            avg_pos_gt[filename] = count_avg_pos(gt, args.ctrl_pos)

        s = set()
        for i in predict:
            s.update(i)
        uniq[filename] = len(s)


    logger.info('--------------------bleu(Up)----------------------')
    for i in bleu.keys():
        logger.info('{:<65}{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *bleu[i]))

    logger.info('--------------------self-bleu(Down)----------------------')
    for i in sb.keys():
        logger.info('{:<65}{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *sb[i]))

    logger.info('--------------------self-WER(UP)----------------------')
    for i in bleu.keys():
        logger.info('{:<65}{:.6f}'.format(os.path.basename(i), wer[i]))

    if args.analyse_pos:
        logger.info('--------------------{} average number on Predict (Up)----------------------'.format(args.ctrl_pos))
        for i in avg_pos_pred.keys():
            logger.info('{:<65}{}'.format(os.path.basename(i), avg_pos_pred[i]))
        logger.info('--------------------{} average number on Source (Up)----------------------'.format(args.ctrl_pos))
        for i in avg_pos_src.keys():
            logger.info('{:<65}{}'.format(os.path.basename(i), avg_pos_src[i]))
        logger.info('--------------------{} average number on Ground-Truth (Up)----------------------'.format(args.ctrl_pos))
        for i in avg_pos_gt.keys():
            logger.info('{:<65}{}'.format(os.path.basename(i), avg_pos_gt[i]))

    logger.info('--------------------distinct(Up)----------------------')
    for i in dist.keys():
        logger.info('{:<65}{:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *dist[i]))

    logger.info('--------------------repetition(Down)----------------------')
    for i in bleu.keys():
        logger.info('{:<65}{:.6f}'.format(os.path.basename(i), repit[i]))

    logger.info('--------------------uniq_seq(Up)----------------------')
    for i in uniq.keys():
        logger.info('{:<65}{}'.format(os.path.basename(i), uniq[i]))

    logger.info('--------------------kl-divergence(Down)----------------------')
    for i in kd.keys():
        logger.info('{:<65}{:.4f}'.format(os.path.basename(i), kd[i]))

    logger.info('--------------------rouge(Up)----------------------')
    for i in bleu.keys():
        logger.info('{:<65}{:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *rouge[i]))


    logger.info('--------------------self-bleu gt(Down)----------------------')
    for i in self_bleu_gt.keys():
        logger.info('{:<65}{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *self_bleu_gt[i]))

    logger.info('--------------------ms_jaccard(Up)----------------------')
    for i in msj.keys():
        logger.info('{:<65}{:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i),  *msj[i]))



if __name__ =='__main__':
    main()

