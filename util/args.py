import os
import yaml
from .files import *
import argparse
from .counter import *
import indexer.tokenizer as tokenizer
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

class Argument:
    def __init__(self, path='config', is_train=True):
        
        training_path = os.path.join(path, 'training.yaml')
        model_data = os.path.join(path,'model.yaml')
        data = {}
        with open(training_path, "r") as t, open(model_data,'r') as m:
            training_data = yaml.load(t.read(), Loader=yaml.FullLoader)
            model_data = yaml.load(m.read(), Loader=yaml.FullLoader)
        self.is_train = is_train
        args = self.get_args(is_test=not is_train)


       
        if args.dataset =='wikitext-103':
            data.update(model_data['wiki'])
        elif args.dataset =='paraNMT':
            data.update(model_data['paraNMT'])
        data.update(vars(args))
        data.update(training_data)
        self.load_files(data)
        self.__dict__ = data

    def get_args(self, is_test=False):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="wikitext-103",
                            help='dataset_name')
        # parser.add_argument("--task", type=str, default="LM", choices=["LM", "seq2seq"],
        #                     help='task_name')
        parser.add_argument("--root", type=str,
                            help='root directory')
        # parser.add_argument("--encoder-class",type=str,default='SPBPE')
        parser.add_argument("--rnn-type", type=str, default=None)
        parser.add_argument("--vocab-size", type=int, default=270000)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument("--n-cutoffs", type=int)
        parser.add_argument("--division", type=str, default='efficiency')
        parser.add_argument("--tagger", type=str, default='core')
        parser.add_argument("--loss-type", help="choice [ MLE, FACE, UL, F2v1, F2v2, POS]",
                            required=True, type=str)
        parser.add_argument("--model-checkpoint", help="transfer for finetuning model",default="", type=str)
        parser.add_argument("--finetune",action="store_true")
        parser.add_argument("--max-update", help="max update for finetuning phase ", default=1500, type=int)
        parser.add_argument("--report_step", type=int, default=50, help="Report loss every report_step")
        parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
        # if is_test:
        parser.add_argument("--saved-path", type=str)
        parser.add_argument("--nprefix", type=int, default=50)
        parser.add_argument("--ngenerate", type=int, default=100)
        parser.add_argument("--beam-size", type=int, default=0)
        parser.add_argument("--sampling-mode", type=int, default=0)
        parser.add_argument("--top-k", type=int, default=0)
        parser.add_argument("--top-p", type=float, default=0.0)
        parser.add_argument("--pos-top-k", type=int, default=0)
        parser.add_argument("--pos-top-p", type=float, default=0.0)
        parser.add_argument("--analyse-ctrl", action="store_true", help="If is analysing controllability")
        parser.add_argument("--control-pos", type=str, default='JJ')
        parser.add_argument("--control-factor", type=float, default=1.0)
        parser.add_argument("--sample-dirname", type=str, default='SGCP_samples')
        parser.add_argument("--generate-num", type=int, default=1)
        parser.add_argument("--temperature", type=float, default=1)

        return parser.parse_args()

    def load_files(self, data):
        
        if data['loss_type'] == 'F2v1':
            data['experimental_loss'] = 1
        elif data['loss_type'] == 'F2v2':
            data['experimental_loss'] = 2
        elif data['loss_type'] == 'POS':
            data['experimental_loss'] = 3
        elif data['loss_type'] == 'MoS':
            data['experimental_loss'] = 4
        else:
            data['experimental_loss'] = 0
        
        if data['dataset'] == "wikitext-103":
            data['add_special_token'] = False
        elif data['dataset'] == "paraNMT":
            data['add_special_token'] = True

        dataset_fullname = data['dataset']

        dirname = os.path.join(data['root'], dataset_fullname)
     
        logger.info("experimental_loss is {}".format(data['experimental_loss']))
        
        probs_path = os.path.join(dirname, '{tagger}_{vocab_size}_probs.pkl'.format(tagger=data['tagger'], vocab_size=data['vocab_size']))
        
        probs = load_pkl(probs_path)
        if data['division'] == 'efficiency':
            if data['n_cutoffs']:
                data['cutoffs'] = compute_cutoffs(probs, data['n_cutoffs'])
            else:
                print("count ideal cutoffs")
                data['cutoffs'] = ideal_cutoffs(probs)
        elif data['division'] == 'uniform':
            assert data['n_cutoffs'] is not None
            data['cutoffs'] = uniform_cutoffs(probs, data['n_cutoffs'])
        else:
            raise NotImplementedError
        

        
        # if data['experimental_loss'] == 3:
        data['train_pos_path'] = os.path.join(dirname, '{tagger}_train_{vocab_size}.pos'.format(tagger=data['tagger'], vocab_size=data['vocab_size']))
        data['test_pos_path'] = os.path.join(dirname, '{tagger}_test_{vocab_size}.pos'.format(tagger=data['tagger'], vocab_size=data['vocab_size']))
        data['train_path'] = os.path.join(dirname, '{tagger}_train_{vocab_size}.token'.format(tagger=data['tagger'], vocab_size=data['vocab_size']))
        data['test_path'] = os.path.join(dirname, '{tagger}_test_{vocab_size}.token'.format(tagger=data['tagger'], vocab_size=data['vocab_size']))
        # data['train_path'] = os.path.join(dirname, 'encoded_' + basename, 'train')
        # data['test_path'] = os.path.join(dirname, 'encoded_' + basename, 'test')


        data['pos2word_path'] = os.path.join(dirname, '{tagger}_{vocab_size}_pos2word.pkl'.format(tagger=data['tagger'], vocab_size=data['vocab_size']))
        data['token_in_pos_id_path'] = os.path.join(dirname, '{tagger}_{vocab_size}_token_in_pos_id.pkl'.format(tagger=data['tagger'], vocab_size=data['vocab_size']))
        token_vocab_path = os.path.join(dirname, '{tagger}_token.vocab'.format(tagger=data['tagger']))
        pos_vocab_path = os.path.join(dirname, '{tagger}_pos.vocab'.format(tagger=data['tagger']))
        data['token_tokenizer'] = tokenizer.TokenTokenizer(token_vocab_path, data['vocab_size'], add_special_token=data['add_special_token'])
        data['pos_tokenizer'] = tokenizer.POSTokenizer(pos_vocab_path, add_special_token=data['add_special_token'])
        
        savename = '_{}_'.format(data['loss_type'])
        savename += 'layer_{}_lr_{}_cutoffs_{}_{}'.format(data['n_layers'],data['learning_rate'], len(data['cutoffs']), data['tagger'])
        if data['division'] == 'uniform':
            savename += '_uniform'
        data['savename'] = os.path.join('data/checkpoint','{}'.format(data['dataset']), savename)
        # if not self.is_train:
        # sample_dirname = os.path.join('prefix-{}_nsample-{}'.format(data['nprefix'],data['ngenerate']),
                                        # 'topp-{}-topk-{}-temp-{}'.format(data['top_p'], data['top_k'], data['temperature']))
        sample_dirname = data['sample_dirname']                        
        sample_basename = '{}'.format(data['loss_type'])
        if data['beam_size'] > 0:
            sample_basename += "_mode-{}-beam{}".format(data['sampling_mode'], data['beam_size'])
        else:
            sample_basename += '_mode-{}-topp-{}-topk-{}'.format(data['sampling_mode'], data['top_p'], data['top_k'])
        if data['analyse_ctrl']:
            sample_basename += '_ctrl-{}x{}'.format(data['control_pos'], data['control_factor'])
        # if data['experimental_loss'] == 1 or data['experimental_loss'] == 2:
        #     sample_basename += '_mode-{}'.format(data['sampling_mode'])
        # elif data['experimental_loss'] == 3:
        #     sample_basename += '_mode-{}-pos-topp-{}-topk-{}'.format(data['sampling_mode'], data['pos_top_p'], data['pos_top_k'])
           
            
        
            
        data['sampled_savepath'] = os.path.join('data','sampled','{}'.format(data['dataset']), sample_dirname, sample_basename)