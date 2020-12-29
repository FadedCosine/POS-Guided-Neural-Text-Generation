from model.transformer_gpt2 import *
from util.batch_generator import *
from util.files import *
from util.trainer import Evaluater
import os
from util.args import EMNLPArgument
import apex
from util.sampling import *
import pickle

def get_model(args):
    pos2word = None
    token_in_pos_id = None
    if args.experimental_loss == 3:
        with open(args.pos2word_path,'rb') as reader:
            pos2word = pickle.load(reader)
        with open(args.token_in_pos_id_path,'rb') as reader:
            token_in_pos_id = torch.from_numpy(pickle.load(reader)).to(args.device)
    logger.info("vocab_size is {}".format(args.vocab_size))
    if args.dataset == "wiki103":
        model = Transformer_Decoder(args.vocab_size, args.batch_seqlen, args.hidden_dim, args.projection_dim, args.n_heads,
                             args.head_dim, args.n_layers, args.cutoffs, args.dropout_rate, args.dropatt_rate,
                             args.token_tokenizer.padding_id, rel_att=args.relative_pos,experimental_loss=args.experimental_loss,
                             pos2word=pos2word, token_in_pos_id=token_in_pos_id)
    initializer = Initializer('normal', 0.02, 0.1)
    initializer.initialize(model)
    model = model.to(args.device)
    model.load_state_dict(torch.load(args.saved_path))
    model.eval()
    return model

def get_batchfier(args):
    if args.dataset =='bugs':
        test_batchfier = Lyrics_Batchfier([args.test_path], args.batch_size, seq_len=args.batch_seqlen,
                                          padding_index=args.token_tokenizer.padding_id, epoch_shuffle=True)
    else:
        if args.loss_type == "experimental3":
            test_batchfier = BpttIteratorWithPOS(load_pkl(args.test_path), load_pkl(args.test_pos_path), args.batch_size, args.batch_seqlen, device=args.device)
        else:
            test_batchfier = BpttIterator(load_pkl(args.test_path), args.batch_size, args.batch_seqlen, device=args.device)

    return test_batchfier


if __name__ == '__main__':
    args = EMNLPArgument(is_train=False)
    print(args.learning_rate, 'experimental : {} cutoffs : {}'.format(
        args.experimental_loss, len(args.cutoffs)))

    print(args.__dict__)
    print(args.sampled_savepath)
    model = get_model(args)
    test_batchfier = get_batchfier(args)
    evaluater = Evaluater(model, test_batchfier, args.token_tokenizer.padding_id, args.experimental_loss, args.experimental_loss==3)

    evaluater.eval()


    # train_lstm(model,batchfier,optimizer)