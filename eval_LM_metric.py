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
    parser.add_argument('--top-p',type=float)
    parser.add_argument('--top-k',type=int)
    parser.add_argument('--folderpath',type=str)
    return parser.parse_args()


def main():
    args = get_args()

    # folderpath = os.path.join(args.folderpath, "topp-{p}-topk-{k}-temp-1".format(p=args.top_p, k=args.top_k))
    folderpath = args.folderpath
    # logger.info("=" * 20 + "topp-{p}-topk-{k}-temp-1".format(p=args.top_p, k=args.top_k) + "=" * 20)
    filenames = sorted(get_files(folderpath))

    sb = {}
    kd = {}
    msj = {}
    uniq = {}
    dist = {}
    bleu = {}
    repit = {}
    rep_1 = {}
    w_rep_1 = {}
    for filename in filenames:
        logger.info("read_pickle : {}".format(filename))
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


    logger.info('--------------------repetition(Down)----------------------')
    for i in bleu.keys():
        logger.info('{:<65}{:.6f}'.format(os.path.basename(i), repit[i]))


    logger.info('--------------------self-bleu(Down)----------------------')
    for i in sb.keys():
        logger.info('{:<65}{:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(os.path.basename(i), *sb[i]))

    logger.info('--------------------kl-divergence(Down)----------------------')
    for i in kd.keys():
        logger.info('{:<65}{:.3f}'.format(os.path.basename(i), kd[i]))

    logger.info('--------------------ms_jaccard(Up)----------------------')
    for i in msj.keys():
        logger.info('{:<65}{:.3f}, {:.3f}, {:.3f}'.format(os.path.basename(i),  *msj[i]))

    logger.info('--------------------distinct(Up)----------------------')
    for i in dist.keys():
        logger.info('{:<65}{:.3f}, {:.3f}, {:.3f}'.format(os.path.basename(i), *dist[i]))

    logger.info('--------------------uniq_seq(Up)----------------------')
    for i in uniq.keys():
        logger.info('{:<65}{}'.format(os.path.basename(i), uniq[i]))



if __name__ =='__main__':
    main()

