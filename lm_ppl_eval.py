from model.transformer_gpt2 import *
from model.rnn import RNNModel
from util.batch_generator import *
from util.files import *
from util.trainer import Evaluater
import os
from util.args import Argument
import apex
from util.sampling import *
import pickle
from main import get_model, get_batchfier

if __name__ == '__main__':
    args = Argument(is_train=False)
    print(args.learning_rate, 'experimental : {} cutoffs : {}'.format(
        args.experimental_loss, len(args.cutoffs)))
    print(args.__dict__)
    model = get_model(args)
    _, test_batchfier = get_batchfier(args)
    evaluater = Evaluater(model, test_batchfier, args.token_tokenizer.padding_id,args.dataset, args.experimental_loss)
    evaluater.eval()

    # train_lstm(model,batchfier,optimizer)