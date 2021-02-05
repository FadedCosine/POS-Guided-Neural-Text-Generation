from data_processing.files import *
from util.counter import *
from util.evaluate import *
import argparse
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_data',type=str)
    return parser.parse_args()

def eval_LM_human(filename):
    
    df = pd.read_pickle(filename)
    predict = df['decoded_predict']
    gt = df['decoded_true']

    sb[filename] = selfbleu(gt, 5)
    repit[filename] = repetition(gt)
    dist[filename] = distinct_upto(gt, 5)
    s = set()
    for i in gt:
        s.update(i)
    uniq[filename] = len(s)
    
    logger.info('--------------------self-bleu(Down)----------------------')
    for i in sb.keys():
        logger.info('{:<65}{:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(os.path.basename(i), *sb[i]))


    logger.info('--------------------repetition(Down)----------------------')
    for i in repit.keys():
        logger.info('{:<65}{:.6f}'.format(os.path.basename(i), repit[i]))

    logger.info('--------------------uniq_seq(Up)----------------------')
    for i in uniq.keys():
        logger.info('{:<65}{}'.format(os.path.basename(i), uniq[i]))

    logger.info('--------------------distinct(Up)----------------------')
    for i in dist.keys():
        logger.info('{:<65}{:.3f}, {:.3f}, {:.3f}'.format(os.path.basename(i), *dist[i]))

def eval_seq2seq_human(gt_filename):
    source = []
    predict = []
    gt = []
    with open(gt_filename, "r") as f:
        for line in f:
            o = json.loads(line)
            for sour, pred, gt_ in zip(o['prefix'], o['decoded_predict'], o['decoded_true']):
                source.append(sour[1:-1])
                predict.append(pred[0])
                gt.append(gt_[1:-1])


    self_bleu_gt[filename] = bleu_upto(source, gt, 5)
    dist[filename] = distinct_upto(gt, 5)

    repit[filename] = repetition(gt)
    wer[filename] = self_wer(source, gt)
    s = set()
    for i in gt:
        s.update(i)
    uniq[filename] = len(s)


    logger.info('--------------------self-bleu gt(Down)----------------------')
    for i in self_bleu_gt.keys():
        logger.info('{:<65}{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(os.path.basename(i), *self_bleu_gt[i]))


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



def main():
    args = get_args()
    eval_seq2seq_human(args.gt_data)



if __name__ =='__main__':
    main()

