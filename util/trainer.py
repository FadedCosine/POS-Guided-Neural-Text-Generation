from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
import math
from model.ops import *
from pytorch_transformers import WarmupLinearSchedule
import apex
from util.losses import *
import os
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class FactorizedCriteria:
    def __init__(self):
        pass

    def __call__(self, x, y ):
        return x

def get_trainer(args, model, train_batchfier, test_batchfier):
    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    if args.mixed_precision:
        print('mixed_precision')
        opt_level = 'O2'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)
    scheduler = WarmupLinearSchedule(optimizer, args.warmup_step, args.decay_step)
    criteria = FactorizedCriteria() if args.experimental_loss else nn.CrossEntropyLoss(ignore_index=args.token_tokenizer.padding_id)
    trainer = Trainer(model, train_batchfier, test_batchfier, optimizer, scheduler, args.update_step, criteria,
                      args.clip_norm, args.mixed_precision)

    return trainer


class Trainer:
    def __init__(self, model, train_batchfier, test_batchfier, optimizers, schedulers,
                 update_step, criteria, clip_norm, mixed_precision, dataset):
        self.model = model
        self.is_rnn_model = hasattr(model, 'model_type') and model.model_type == 'RNN'
        self.train_batchfier = train_batchfier
        self.test_batchfier = test_batchfier
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.criteria = criteria
        self.step = 0
        self.update_step = update_step
        self.mixed_precision = mixed_precision
        self.clip_norm = clip_norm
        self.dataset = dataset

    def get_acc(self, logits, y):
        _, predicted = torch.max(logits.data, 1)
        total = y.size(0)
        correct = (predicted == y).sum().item()
        return correct, total

    def top_k_acc(self, logits, y, top_k):
        total = y.size(0)
        _, indices = torch.topk(logits, top_k, 1)
        indices = indices.t()
        correct = indices.eq(y.view(1, -1).expand_as(indices))
        return correct.sum().item(), total

    def train_epoch(self, args):
        def reset_pbar(pbar, n_bar):
            pbar.close()
            pbar = tqdm(100)
            return pbar, n_bar + 1, 0, 0, 0

        model = self.model
        batchfier = self.train_batchfier
        criteria = self.criteria
        optimizer = self.optimizers
        scheduler = self.schedulers
        total_len = self.train_batchfier.len()
        # train_sampler = RandomSampler(train_dataset)
        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, )

        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0
       
     
        model.zero_grad()
        test_d1_score=0
    
        if self.is_rnn_model:
            hidden = model.init_hidden(batchfier.size)
        for idx, inp in enumerate(batchfier):
            x, x_l, x_pos, y, y_l, y_pos = inp
            if 0 in x_l:
                continue
            
            if self.is_rnn_model:
                hidden = repackage_hidden(hidden)
                
                logits, hidden = model(x, y, hidden, dec_output_POS=y_pos)
            else:
                logits, _ = model(x, x_l, y, y_l, y_pos)
            # print(logits)
            if self.dataset == "wikitext-103":
                tgt = y
            elif self.dataset == "paraNMT":
                tgt = y[..., 1:]
            loss = criteria(logits, tgt.contiguous().view(-1))
            step_loss += loss.item()
            tot_loss += loss.item()
            # correct, total = self.get_acc(logits, tgt.contiguous().view(-1))
            # acc += correct / total
            if self.mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tot_cnt += 1
           
            if not tot_cnt % self.update_step:
                self.step += 1
           
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                optimizer.step()
                model.zero_grad()
                scheduler.step(self.step)
                
               
                if tot_cnt % args.report_step == 0:
                    logger.info(
                        "training loss : %f training ppl : %f, lr : %f, iter : %d / %d" % (
                            step_loss / (args.report_step), math.exp(step_loss / (args.report_step)),
                            scheduler.get_lr()[0], idx , total_len))
                    step_loss = 0
            
                # torch.cuda.empty_cache()

        return math.exp(tot_loss / tot_cnt)
        
    def test_epoch(self):
        model = self.model
        batchfier = self.test_batchfier
        total_len = self.test_batchfier.len()
        if isinstance(self.criteria,tuple):
            _,criteria= self.criteria
        else:
            criteria = self.criteria
        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, )

        model.eval()
        pbar_cnt = 0
        step_loss = 0
        n_samples = 0
        t_correct = 0
        t_total = 0
        t_correct3 = 0
        t_total3 = 0

        
        if self.is_rnn_model:
            hidden = model.init_hidden(batchfier.size)
        for inp in batchfier:
            with torch.no_grad():
                x, x_l, x_pos, y, y_l, y_pos = inp
                if 0 in x_l:
                    continue
                if self.is_rnn_model:
                    hidden = repackage_hidden(hidden)
                    logits, hidden = model(x, y, hidden, dec_output_POS=y_pos)
                else:
                    logits, _ = model(x, x_l, y, y_l, y_pos)
                # print(logits)
                if self.dataset == "wikitext-103":
                    tgt = y
                elif self.dataset == "paraNMT":
                    tgt = y[..., 1:]
                loss = criteria(logits, tgt.contiguous().view(-1))
                n_samples+= tgt.numel()
                step_loss += loss.item()
                # correct, total = self.get_acc(logits, tgt)
                # correct3, total3 = self.top_k_acc(logits, tgt, 3)
                # t_correct += correct
                # t_total += total
                # t_correct3 += correct3
                # t_total3 += total3
                
                pbar_cnt += 1
                logger.info(
                        "test loss : %f training ppl : %f, iter : %d / %d" % (
                            step_loss / pbar_cnt, math.exp(step_loss / pbar_cnt), pbar_cnt, total_len))

                # pbar.set_description(
                #     "test loss : %f training ppl : %f, acc : %f, top_3 : %f" % (
                #         step_loss / pbar_cnt, math.exp(step_loss / pbar_cnt), t_correct/t_total, t_correct3/t_total3))
    
        return math.exp(step_loss / pbar_cnt)


class ExperTrainer(Trainer):
    def __init__(self, model, train_batchfier, test_batchfier, optimizers, schedulers,
                 update_step, criteria, clip_norm, mixed_precision, dataset):
        super(ExperTrainer,self).__init__(model, train_batchfier, test_batchfier, optimizers, schedulers,
                 update_step, criteria, clip_norm, mixed_precision, dataset)
    

    def test_d1_epoch(self,args):
        from sample import generate_seq2seq_sample
        from util.evaluate import distinct_n_corpus_level

        model = self.model
        batchfier = self.test_batchfier

        # if isinstance(batchfier, IterableDataset):
        #     batchfier = DataLoader(dataset=batchfier,
        #                            batch_size=batchfier.size,
        #                            shuffle=False,
        #                            collate_fn=batchfier.collate, )
        model.eval()
        df=generate_seq2seq_sample(args,model,batchfier)
        d_1_score=distinct_n_corpus_level(df["decoded_predict"],1)

        return d_1_score

    def finetune_face(self,args):
        def reset_pbar(pbar, n_bar):
            pbar.close()
            pbar = tqdm(100)
            return pbar, n_bar + 1, 0, 0, 0

        model = self.model
        batchfier = self.train_batchfier
        criteria = self.criteria
        optimizer = self.optimizers
        scheduler = self.schedulers
        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, )

        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0
        pbar = tqdm(100)
        pbar_cnt = 0
        model.zero_grad()
        test_d1_score=0
        
        if self.is_rnn_model:
            hidden = model.init_hidden(batchfier.size)
        for inp in batchfier:
            x, x_l, x_pos, y, y_l, y_pos = inp
            if 0 in x_l:
                continue
            if self.is_rnn_model:
                hidden = repackage_hidden(hidden)
                logits, hidden = model(x, y, hidden, dec_output_pos=y_pos)
            else:
                logits, _ = model(x, x_l, y, y_l, y_pos)
            # print(logits)
            if self.dataset == "wikitext-103":
                tgt = y
            elif self.dataset == "paraNMT":
                tgt = y[..., 1:]
            loss = criteria(logits, tgt.contiguous().view(-1))
            # print(logits)
            step_loss += loss.item()
            tot_loss += loss.item()
            # correct, total = self.get_acc(logits, tgt.contiguous().view(-1))
            # acc += correct / total
            if self.mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tot_cnt += 1
            if not tot_cnt % self.update_step:
                self.step += 1
                pbar_cnt += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                optimizer.step()
                model.zero_grad()
                scheduler.step(self.step)
                pbar.set_description(
                    "training loss : %f training ppl : %f, lr : %f, iter : %d" % (
                        step_loss / (self.update_step *pbar_cnt), math.exp(step_loss / (self.update_step*pbar_cnt)),
                         scheduler.get_lr()[0], n_bar), )
                pbar.update()
                if pbar_cnt == 100:
                    pbar, n_bar, pbar_cnt, step_loss, acc = reset_pbar(pbar, n_bar)
                    step_d1_score=self.test_d1_epoch(args)
                    print("eval d1 score: {0} n_bar: {1}".format(step_d1_score,n_bar))

                    if step_d1_score>test_d1_score:
                        savepath = os.path.join(args.savename + '_n_bar_{}'.format(n_bar))
                        if not os.path.exists(os.path.dirname(savepath)):
                            os.makedirs(os.path.dirname(savepath))
                        torch.save(model.state_dict(), savepath)
                        test_d1_score=step_d1_score
                    else:
                        return test_d1_score

    def seq_level_finetune(self, savename, args):
        def reset_pbar(pbar, n_bar):
            pbar.close()
            pbar = tqdm(100)
            return pbar, n_bar + 1, 0, 0, 0

        test_result = [10000.0]
        model = self.model
        batchfier = self.train_batchfier
        seq_criteria, criteria = self.criteria
        optimizer = self.optimizers
        scheduler = self.schedulers
        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, )
        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0
        pbar = tqdm(100)
        pbar_cnt = 0
        model.zero_grad()
        if self.is_rnn_model:
            hidden = model.init_hidden(batchfier.size)
        for total_iter, inp in enumerate(batchfier):
            x, x_l, x_pos, y, y_l, y_pos = inp
            if 0 in x_l:
                continue
            if self.is_rnn_model:
                hidden = repackage_hidden(hidden)
                logits, hidden = model(x, y, hidden, dec_output_pos=y_pos)
            else:
                logits, _ = model(x, x_l, y, y_l, y_pos)
            # print(logits)
            if self.dataset == "wikitext-103":
                tgt = y
            elif self.dataset == "paraNMT":
                tgt = y[..., 1:]

            # print(logits)
            if torch.rand(1).item() < seq_criteria.sequence_tune_rate:
                if x.size(1) < seq_criteria.sequence_prefix_length + seq_criteria.sequence_completion_length:
                    continue
                loss = seq_criteria(model, x, x_l, y)
            else:
                logits, _ = model(x, x_l, y, y_l, y_pos)
                loss = criteria(logits, tgt.contiguous().view(-1))

            step_loss += loss.item()
            tot_loss += loss.item()
            # correct, total = self.get_acc(logits, tgt.contiguous().view(-1))
            # acc += correct / total
            if self.mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tot_cnt += 1
            if not tot_cnt % self.update_step:
                self.step += 1
                pbar_cnt += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                optimizer.step()
                model.zero_grad()
                scheduler.step(self.step)
                pbar.set_description(
                    "training loss : %f training ppl : %f, lr : %f, iter : %d" % (
                        step_loss / (self.update_step * pbar_cnt), math.exp(step_loss / (self.update_step * pbar_cnt)),
                        scheduler.get_lr()[0], n_bar), )
                pbar.update()

                if pbar_cnt == 100:
                    pbar, n_bar, pbar_cnt, step_loss, acc = reset_pbar(pbar, n_bar)
                    test_loss = self.test_epoch()
                    print("\n evaluation loss : {}".format(test_loss))
                    torch.save(model.state_dict(), savename + "_nbar_{}".format(n_bar) )
                    test_result.append(test_loss)

            if total_iter >= args.max_update:
                break

        pbar.close()
        return math.exp(tot_loss / tot_cnt)


class Evaluater:
    def __init__(self, model, batchfier, padding_idx, dateset, experimental=False):
        self.model = model
        self.is_rnn_model = hasattr(model, 'model_type') and model.model_type == 'RNN'
        self.batchfier = batchfier
        self.padding_idx = padding_idx
        self.macro_criterion=nn.CrossEntropyLoss(ignore_index=self.padding_idx,reduction="none")
        self.criterion=nn.CrossEntropyLoss(ignore_index=self.padding_idx,reduction="none")
        self.experimental = experimental
        self.dataset = dateset
   

    def init_macro_ppl(self, device):
        vocab_size = self.model.word_embedding.num_embeddings
        setattr(self, 'ppls', torch.zeros((vocab_size,)).to(device))
        setattr(self, 'cnts', torch.zeros((vocab_size,)).to(device))

    def init_macro_acc(self, device):
        vocab_size = self.model.word_embedding.num_embeddings
        setattr(self, 'accs', torch.zeros((vocab_size,)).to(device))
        setattr(self, 'acnts', torch.zeros((vocab_size,)).to(device))

    def macro_ppl(self, logits, y):
        if not hasattr(self, 'ppls'):
            self.init_macro_ppl(logits.device)

        vocab_size = self.model.word_embedding.num_embeddings
        ar = torch.arange(vocab_size).to(logits.device)
        loss = self.macro_criterion(logits, y)

        idx = (ar[:, None] == y).to(self.cnts.dtype)
        added_cnt = idx.sum(dim=-1)
        add_loss = (idx * loss[None]).sum(dim=-1)

        self.cnts += added_cnt
        self.ppls += add_loss
        ny = self.cnts.nonzero().numel()

        #delete padding
        self.ppls[self.padding_idx] = 0
        mppl = (self.ppls / (self.cnts + 1e-6)).sum() / ny
        return torch.exp(mppl).item()

    def acc(self, logits, y):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == y).sum().item()
        return correct

    def macro_acc(self, logits, y):
        if not hasattr(self, 'accs'):
            self.init_macro_acc(logits.device)

        vocab_size = self.model.word_embedding.num_embeddings
        ar = torch.arange(vocab_size).to(logits.device)
        idx = (ar[:, None] == y).to(self.cnts.dtype)
        added_cnt = idx.sum(dim=-1)
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == y)

        add_correct = (idx * correct[None]).sum(dim=-1)

        self.acnts += added_cnt
        self.accs += add_correct

        #delete padding
        self.accs[self.padding_idx] = 0
        ny = self.acnts.nonzero().numel()
        macc = (self.accs / (self.acnts + 1e-6)).sum() / ny
        return macc.item()

    def eval(self):
        model = self.model
        model.eval()
        batchfier = self.batchfier
        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, )
        model.eval()
        pbar = tqdm(batchfier)
        step_loss = 0
        n_samples = 0
        t_correct = 0
        if self.is_rnn_model:
            hidden = model.init_hidden(batchfier.size)
        for inp in pbar:
            with torch.no_grad():
                x, x_l, x_pos, y, y_l, y_pos = inp
                if 0 in x_l:
                    continue
                if self.is_rnn_model:
                    hidden = repackage_hidden(hidden)
                    logits, hidden = model(x, y, hidden, dec_output_pos=y_pos)
                else:
                    logits, _ = model(x, x_l, y, y_l, y_pos)
                # print(logits)
                if self.dataset == "wikitext-103":
                    tgt = y
                elif self.dataset == "paraNMT":
                    tgt = y[..., 1:]
            
                if self.experimental:
                    logits, _ = model.sampling(x, x_l, None, 0, 1)
                else:
                    logits, _ = model(x, x_l, y, y_l, y_pos)
                y = tgt.contiguous().view(-1)
                losses = self.criterion(logits, y)
                n_samples+= (tgt != self.padding_idx).sum().item()
                t_correct += self.acc(logits, y)
                step_loss += losses.sum().item()
                mac_ppl = self.macro_ppl(logits, y)
                mac_acc = self.macro_acc(logits, y)
                pbar.set_description(
                    "test loss : %f training ppl : %f acc : %f mac ppl : %f mac acc : %f" % (
                        step_loss / n_samples, math.exp(step_loss / n_samples),
                        t_correct / n_samples, mac_ppl, mac_acc))
                
        pbar.close()
