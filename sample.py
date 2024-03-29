from model.transformer_gpt2 import *
from model.transformer import *
from model.rnn import RNNModel
from util.batch_generator import *
from util.files import *
from util.trainer import ExperTrainer
import os
from util.args import Argument
import apex
from pytorch_transformers import WarmupLinearSchedule
from util.sampling import *
from util.beam_search import *
import pandas as pd
import pickle
import time
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def get_model(args):
    with open(args.pos2word_path,'rb') as reader:
        pos2word = pickle.load(reader)
    with open(args.token_in_pos_id_path,'rb') as reader:
        token_in_pos_id = torch.from_numpy(pickle.load(reader)).to(args.device)
    logger.info("vocab_size is {}, padding id is {}".format(args.token_tokenizer.vocab_size, args.token_tokenizer.padding_id))
    if args.rnn_type is None:
        if args.dataset == "wikitext-103":
            model = Transformer_Decoder(args.token_tokenizer.vocab_size, args.batch_seqlen, args.hidden_dim, args.projection_dim, args.n_heads,
                                    args.head_dim, args.n_layers, args.cutoffs, args.dropout_rate, args.dropatt_rate,
                                    args.token_tokenizer.padding_id, rel_att=args.relative_pos, experimental_loss=args.experimental_loss,
                                    pos2word=pos2word, token_in_pos_id=token_in_pos_id)
        elif args.dataset == "paraNMT":
            model = Transformer(args.token_tokenizer.vocab_size, args.batch_seqlen, args.hidden_dim, args.projection_dim, args.n_heads,
                                    args.head_dim, args.n_layers, args.cutoffs, args.dropout_rate, args.dropatt_rate,
                                    args.token_tokenizer.padding_id, rel_att=args.relative_pos, experimental_loss=args.experimental_loss,
                                    pos2word=pos2word, token_in_pos_id=token_in_pos_id)
    else:
        if args.dataset == "wikitext-103":
            model = RNNModel(args.rnn_type, args.token_tokenizer.vocab_size, args.embedding_dim, args.hidden_dim, args.rnn_layer, args.cutoffs,
                                    args.token_tokenizer.padding_id, experimental_loss=args.experimental_loss, dropout=args.dropout_rate,
                                    pos2word=pos2word, token_in_pos_id=token_in_pos_id)
    initializer = Initializer('normal', 0.02, 0.1)
    initializer.initialize(model)
    model = model.to(args.device)
    model.load_state_dict(torch.load(args.saved_path))
    model.eval()
    return model


def get_batchfier(args):
    logger.info("test_path : {}".format(args.test_path))
    if args.dataset == 'bugs':
        test_batchfier = LyricsSampleBatchfier([args.test_path], args.batch_size*32,
                              10000, args.nprefix, args.ngenerate, device=args.device)
    elif args.dataset == 'wikitext-103':
        test_batchfier = BpttSamplingIterator(load_pkl(args.test_path), args.batch_size*32,
                                      args.nprefix, args.ngenerate, device=args.device)
    elif args.dataset =='paraNMT':
        test_batchfier = ParaIteratorWithPOS(load_pkl(args.test_path), load_pkl(args.test_pos_path), args.batch_size, args.token_tokenizer, max_seq_len=args.batch_seqlen, device=args.device)
    return test_batchfier

@torch.no_grad()
def generate_LM_sample(args, model, batchfier):
    def truncate(x,prefix_len):
        return [i[prefix_len:] for i in x]
    prefixs = []
    truths = []
    generated = []
    idx = 0
    tot_len = len(batchfier)
    for inp in batchfier:
        prefix = inp[0][:,:args.nprefix]
        gt = inp[0][:,args.nprefix:]
        
        if gt.size(-1) == 0:
            break
        if args.top_k == 0:
            top_w = args.top_p
        else:
            top_w = args.top_k
        
        if args.pos_top_k == 0:
            pos_top_w = args.pos_top_p
        else:
            pos_top_w = args.pos_top_k
        # if args.beam_size > 0:
        # else:
        res, _ = LM_sampling(model, args.ngenerate, prefix, top_w, args.temperature,
                            args.experimental_loss, args.sampling_mode, pos_top_w)
        generated.extend(truncate(res,args.nprefix))
        truths.extend(gt.tolist())
        prefixs.extend(prefix.tolist())
        prefices = [args.token_tokenizer.convert_ids_to_words(item) for item in prefix.tolist()]

        predicts = [args.token_tokenizer.convert_ids_to_words(item) for item in truncate(res,args.nprefix)]
        # for prefix, decoded_predict in zip(prefices, predicts):
        #     print("prefix is : ", " ".join(prefix))
        #     print("decoded_predict is : "," ".join(decoded_predict))
        idx += 1
        if idx % 1 == 0:
            logging.info("Finish generating {}/{} batch.".format(idx, tot_len))
    return pd.DataFrame({'prefix':prefixs, 'decoded_predict':generated,'decoded_true':truths})

@torch.no_grad()
def generate_seq2seq_sample(args, model, batchfier, max_decoding_len=64):
    prefixs = []
    truths = []
    generated = []
    idx = 0
    tot_len = batchfier.len()
    cache_file = open(args.sampled_savepath + ".json", "w")
    if isinstance(batchfier, IterableDataset):
        batchfier = DataLoader(dataset=batchfier,
                                batch_size=batchfier.size,
                                shuffle=False,
                                collate_fn=batchfier.collate, )
    for inp in batchfier:
        input_x = inp[0]
        # logger.info("input_x is {}".format(input_x[..., :10]))
        gt = inp[3]
        if args.top_k == 0:
            top_w = args.top_p
        else:
            top_w = args.top_k
        if args.pos_top_k == 0:
            pos_top_w = args.pos_top_p
        else:
            pos_top_w = args.pos_top_k
        if args.beam_size > 0:
            beam_sequences = seq2seq_beam_search(model, args.batch_seqlen, args.token_tokenizer, inp, top_w, args.temperature, args.experimental_loss, args.beam_size, args.sampling_mode, pos_top_w)
            res = [sentence.output for sentence in beam_sequences]
            
            # res = []
            # for beam in beam_sequences:
            #     res.append([sentence.output for sentence in beam])

        else:
            # try:
            res = seq2seq_sampling(model, max_decoding_len, args.token_tokenizer, inp, top_w, args.temperature,
                        args.experimental_loss, sampling_mode=args.sampling_mode, pos_top_w=pos_top_w, sampling_num=args.generate_num, control_pos_id=args.pos_tokenizer.convert_tag_to_id(args.control_pos), control_factor=args.control_factor)
            # except RuntimeError:
            #     logger.info("RuntimeError, continue!")
            #     torch.cuda.empty_cache()
            #     continue
        # print(res)
        generated.extend(res)
        truths.extend(gt.tolist())
        prefixs.extend(input_x.tolist())
        idx += 1
        o = {}
        o['prefix'] = [args.token_tokenizer.convert_ids_to_words(item) for item in input_x.tolist()]
        if args.beam_size > 0:
            o['decoded_predict'] = [args.token_tokenizer.convert_ids_to_words(batch_item) for batch_item in res]
        else:
            o['decoded_predict'] = [[args.token_tokenizer.convert_ids_to_words(item) for item in batch_item] for batch_item in res]
        o['decoded_true'] = [args.token_tokenizer.convert_ids_to_words(item) for item in gt.tolist()]
        print(json.dumps(o), file=cache_file, flush=True)
        # for prefix, preds, gt in zip(o['prefix'], o['decoded_predict'], o['decoded_true']):
            # logger.info("Source : " + " ".join(prefix[1:-1]))
            # logger.info("Target : " + " ".join(gt[1:-1]))
            # logger.info("Generated : ")
            # if args.beam_size > 0:
            #     logger.info(preds)
            # else:
            #     for pred in preds:
            #         logger.info(" ".join(pred))
            # logger.info("\n")
        if idx % 1 == 0:
            logger.info("Finish generating {}/{} batch.".format(idx, tot_len))
        torch.cuda.empty_cache()
    return pd.DataFrame({'prefix':prefixs, 'decoded_predict':generated,'decoded_true':truths})

if __name__ == '__main__':
    args = Argument(is_train=False)
    start_time = time.time()
    logger.info('learning_rate {}, experimental : {} cutoffs len : {}'.format(args.learning_rate, 
        args.experimental_loss, len(args.cutoffs)))
    logger.info("cutoffs is {}".format(args.cutoffs))
    logger.info("args is ")
    logger.info(args.__dict__)
    model = get_model(args)
    test_batchfier = get_batchfier(args)
    logger.info("save sampling sentence in {}".format(args.sampled_savepath))
    if not os.path.exists(os.path.dirname(args.sampled_savepath)):
        os.makedirs(os.path.dirname(args.sampled_savepath))
    logger.info("Start to generate sentence")
    logger.info("POS <JJ> id is : {}".format(args.pos_tokenizer.convert_tag_to_id("JJ")))
    if args.dataset == "wikitext-103":
        df = generate_LM_sample(args, model, test_batchfier)
    elif args.dataset == "paraNMT":
        df = generate_seq2seq_sample(args, model, test_batchfier)
    end_time = time.time()
    logger.info("generate {} cost time : {}".format(args.sampled_savepath, end_time-start_time))
    
    df.to_pickle(args.sampled_savepath)
