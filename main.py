from model.transformer_gpt2 import *
from model.transformer import Transformer
from util.batch_generator import *
from util.files import *
from util.trainer import ExperTrainer
import os
from util.args import Argument
from util.losses import *
import torch
import apex
from pytorch_transformers import WarmupLinearSchedule
import logging
import pickle
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
            
    if args.model_checkpoint=="":
        initializer = Initializer('normal', 0.02, 0.1)
        initializer.initialize(model)
    else:
        state_dict=torch.load(args.model_checkpoint)
        model.load_state_dict(state_dict)

    model = model.to(args.device)
    return model

def get_batchfier(args):
    if args.dataset =='bugs':
        train_batchfier = Lyrics_Batchfier([args.train_path], args.batch_size, seq_len=args.batch_seqlen,
                                           padding_index=args.token_tokenizer.padding_id, epoch_shuffle=True)
        test_batchfier = Lyrics_Batchfier([args.test_path], args.batch_size, seq_len=args.batch_seqlen,
                                          padding_index=args.token_tokenizer.padding_id, epoch_shuffle=True)
    elif args.dataset =='wikitext-103':
        train_batchfier = BpttIteratorWithPOS(load_pkl(args.train_path), load_pkl(args.train_pos_path), args.batch_size, args.batch_seqlen, device=args.device)
        test_batchfier = BpttIteratorWithPOS(load_pkl(args.test_path), load_pkl(args.test_pos_path), args.batch_size, args.batch_seqlen, device=args.device)

    elif args.dataset =='paraNMT':

        train_batchfier = ParaIteratorWithPOS(load_pkl(args.train_path), load_pkl(args.train_pos_path), args.batch_size, args.token_tokenizer, max_seq_len=args.batch_seqlen, device=args.device)
        test_batchfier = ParaIteratorWithPOS(load_pkl(args.test_path), load_pkl(args.test_pos_path), args.batch_size, args.token_tokenizer, max_seq_len=args.batch_seqlen, device=args.device)

    return train_batchfier, test_batchfier

def get_loss(args):
    lt = args.loss_type
    if lt in ('F2v1', 'F2v2', 'POS'):
        loss = FactorizedLoss(args.token_tokenizer.padding_id) # F2-softmax loss
    elif lt == 'MLE':
        loss = PlainLoss(args.token_tokenizer.padding_id) # MLE loss
    elif lt == 'UL':
        loss = CandidateLoss(rank_alpha=1.0, padding_idx=args.token_tokenizer.padding_id) # unlikelihood token loss
    elif lt == 'FACE':
        loss = FACELoss(padding_idx=args.token_tokenizer.padding_id,vocab_size=args.token_tokenizer.vocab_size,ignore_freq_index=[args.token_tokenizer.padding_id],ft="out",wt="pre")
        # if loss.ft=="out" and args.train_phase=="train":
        #     raise NotImplementedError("ft-out only can be used in fine-tune phase")
    elif "-seq" in lt: # unlikelihood token and seq loss
        seq_loss = SequencePenaltyCriterion(4,50,100,"repeat")
        loss = CandidateLoss(rank_alpha=1.0, padding_idx=args.token_tokenizer.padding_id) 
        loss=(seq_loss,loss)
    else:
        raise NotImplementedError
    return loss

def get_trainer(args, model, train_batchfier, test_batchfier):
    if args.dataset == 'bugs':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    if args.mixed_precision:
        print('mixed_precision')
        opt_level = 'O2'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    # # Distributed training (should be after apex fp16 initialization)
    # if args.local_rank != -1:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
    #                                                       output_device=args.local_rank,
    #                                                       find_unused_parameters=True)
    
    decay_step = train_batchfier.len() * args.n_epoch
    scheduler = WarmupLinearSchedule(optimizer, args.warmup_step, decay_step)
    criteria = get_loss(args)
    trainer = ExperTrainer(model, train_batchfier, test_batchfier, optimizer, scheduler, args.update_step, criteria,
                      args.clip_norm, args.mixed_precision, args.dataset)
    return trainer

if __name__ == '__main__':
    args = Argument()
 
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    logger.info('learning_rate {}, experimental : {} cutoffs len : {}'.format(args.learning_rate, 
        args.experimental_loss, len(args.cutoffs)))
    logger.info("cutoffs is {}".format(args.cutoffs))
    # cutoffs [ ) idx左闭右开，且不存0和最后一个，即为隔板
    logger.info("args is ")
    logger.info(args.__dict__)
    model = get_model(args)
    
    train_batchfier, test_batchfier = get_batchfier(args)
    
    
    trainer = get_trainer(args, model, train_batchfier, test_batchfier)
    prev_step = 0
    res = []
    init_epoch = 0
    if args.model_checkpoint != "":
        init_epoch = int(args.model_checkpoint.split('_')[-1]) + 1
    logger.info(init_epoch)
    if args.finetune:
        args.n_epoch=1
    
    for i in range(args.n_epoch):
        print('epoch {}'.format(i + 1))
        if not args.finetune:
            trainer.train_epoch(args)
            test_loss=trainer.test_epoch()
            savepath = os.path.join(args.savename + '_epoch_{}'.format(init_epoch+i))
            if not os.path.exists(os.path.dirname(savepath)):
                os.makedirs(os.path.dirname(savepath))
            logger.info("save model in {}".format(savepath))
            torch.save(model.state_dict(), savepath)

        else:
            if "-seq" in args.loss_type:
                trainer.seq_level_finetune(args.savename,args)
                test_loss = trainer.test_epoch()
                res.append(test_loss)
            if args.loss_type=="face":
                args.nprefix = 50
                args.ngenerate = 100
                args.top_k = 1
                args.temperature = 1.0
                args.experimental_loss = False
                args.sampling_mode = 0 # hyperparams for measuring d-1 metric
                trainer.finetune_face(args)
        torch.cuda.empty_cache()
    print(res)
