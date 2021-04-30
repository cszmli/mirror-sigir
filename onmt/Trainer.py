from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.modules
import logging


class MirrorStatistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss_cxz2y=0, loss_cyz2x=0, loss_cz2x=0, loss_cz2y=0, loss_kl_1=0, loss_kl_2=0, n_words=0, n_correct=0):
        self.loss_cxz2y = loss_cxz2y
        self.loss_cyz2x = loss_cyz2x
        self.loss_cz2x = loss_cz2x
        self.loss_cz2y = loss_cz2y
        self.loss_kl = loss_kl_1
        self.loss_kl_2 = loss_kl_2
        self.loss = 0.0
        self.n_src_words = 0.0

        self.n_words_cxz2y = 0.0
        self.n_correct_cxz2y = 0.0

        self.n_words_cyz2x = 0.0
        self.n_correct_cyz2x = 0.0

        self.n_words_cz2y = 0.0
        self.n_correct_cz2y = 0.0

        self.n_words_cz2x = 0.0
        self.n_correct_cz2x = 0.0
        self.start_time = time.time()
        self.counter=0

    def update(self, stat_cxz2y, stat_cyz2x, stat_cz2y, stat_cz2x, kl_loss, back_factor=1.):
        self.counter+=1
        self.loss_cxz2y += stat_cxz2y.loss
        self.n_words_cxz2y += stat_cxz2y.n_words
        self.n_correct_cxz2y += stat_cxz2y.n_correct

        self.loss_cyz2x += stat_cyz2x.loss
        self.n_words_cyz2x += stat_cyz2x.n_words
        self.n_correct_cyz2x += stat_cyz2x.n_correct

        self.loss_cz2x += stat_cz2x.loss
        self.n_words_cz2x += stat_cz2x.n_words
        self.n_correct_cz2x += stat_cz2x.n_correct

        self.loss_cz2y += stat_cz2y.loss
        self.n_words_cz2y += stat_cz2y.n_words
        self.n_correct_cz2y += stat_cz2y.n_correct

        self.loss_kl += kl_loss
        self.loss =1.0 * (self.loss_cxz2y + self.loss_cyz2x  * back_factor + self.loss_cz2x + self.loss_cz2y * back_factor ) 
        self.n_words = self.n_words_cxz2y + self.n_words_cyz2x + self.n_words_cz2x + self.n_words_cz2y
        self.n_correct = self.n_correct_cxz2y + self.n_correct_cyz2x + self.n_correct_cz2x + self.n_correct_cz2y
    
    # def ppl_all(self):


    def accuracy(self, n_correct=None, n_words=1):
        if n_correct is not None:
            return 100 * (1.0 * n_correct / n_words)
        else:
            return 100 * (1.0 * self.n_correct / self.n_words)

    def ppl(self, loss=None, n_words=1):
        if loss is None:
            return math.exp(min(self.loss / self.n_words, 100))
        else:
            return math.exp(min(loss / n_words, 100))


    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()

        print(("Overall: Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f;" +
               " %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
            #    self.loss * 0.5 + self.loss_kl,
               time.time() - start))
        # print("----------")
        # print(1.0 * self.n_correct_cxz2y.item()/self.n_words_cxz2y.item())
        # print(1.0 * self.n_correct_cxz2y.item(), self.n_words_cxz2y.item())
        # print(1.0 * math.exp(self.loss_cxz2y/self.n_words_cxz2y.item()))
        # print(self.loss_cxz2y/self.n_words_cxz2y.item())
        # print(self.loss_cxz2y, self.n_words_cxz2y.item())
        # print(math.exp(self.loss_cyz2x/self.n_words_cyz2x.item()))
        # print(self.loss_cyz2x/self.n_words_cyz2x.item())
        # print(self.loss_cyz2x, self.n_words_cyz2x.item())
        print(("CXZ2Y: acc: %6.3f; ppl: %6.3f ") %(self.accuracy(self.n_correct_cxz2y, self.n_words_cxz2y),self.ppl(self.loss_cxz2y, self.n_words_cxz2y)))
        print(("CYZ2X: acc: %6.3f; ppl: %6.3f ") %(self.accuracy(self.n_correct_cyz2x, self.n_words_cyz2x),self.ppl(self.loss_cyz2x, self.n_words_cyz2x)))
        print(("CZ2Y: acc: %6.3f; ppl: %6.3f ") %(self.accuracy(self.n_correct_cz2y, self.n_words_cz2y),self.ppl(self.loss_cz2y, self.n_words_cz2y)))
        print(("CZ2X: acc: %6.3f; ppl: %6.3f ") %(self.accuracy(self.n_correct_cz2x, self.n_words_cz2x),self.ppl(self.loss_cz2x, self.n_words_cz2x)))
        print(("KL loss: %6.5f ") %(self.loss_kl/batch))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_kl", self.loss_kl/batch)
        # experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (1. * self.n_correct / self.n_words)

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
    """

    def __init__(self, model,
                 train_loss, valid_loss, 
                 optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 normalization="sents", accum_count=1, back_factor=1.0, kl_balance=False, kl_fix=-1):
        # Basic attributes.
        self.model = model
        self.optim = optim
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.back_factor = back_factor
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.accum_count = accum_count
        self.padding_idx = self.train_loss.loss_cxz2y.padding_idx
        self.normalization = normalization
        self.kl_knealling = 1.
        self.kl_balance = kl_balance
        self.best_model = {'model_name':'ooo', 'model_ppl':9999}
        self.checkpoint_list=[]
        assert(accum_count > 0)
        if accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""
        self.kl_fix = kl_fix
        # Set model in training mode.
        self.model.train()


    def get_kl_knealling(self, epoch, batch_id, batch_num):
        # kl_ft = min(1.0 * (epoch * batch_num + batch_id) / (4.0 * batch_num),1.)
        if self.kl_fix>=0:
            return self.kl_fix
        else:
            return min(1.0 * (epoch * batch_num + batch_id) / (self.kl_balance * batch_num),1.)


    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = MirrorStatistics()
        report_stats = MirrorStatistics()
        idx = 0
        truebatch = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        # print("accum count: {}, num_batches: {}".format(self.accum_count, num_batches))
        

        def gradient_accumulation(truebatch_, total_stats_,
                                  report_stats_, nt_):
            if self.accum_count > 1:
                self.model.zero_grad()

            for batch in truebatch_:
                target_size = max(batch.tgt.size(0), batch.tgt_back.size(0))
                # print("target size: {}, target_back size: {}".format(batch.tgt.size(), batch.tgt_back.size()))
                # 42, 256
                # Truncated BPTT (we discard this function here)
                # if self.trunc_size:
                #     trunc_size = self.trunc_size
                # else:
                trunc_size = target_size

                dec_state = None
                src = onmt.io.make_features(batch, 'src', self.data_type)
                src_back = onmt.io.make_features(batch, 'src_back', self.data_type)
                ctx = onmt.io.make_features(batch, 'ctx', self.data_type)
                if self.data_type == 'text':
                    _, src_lengths = batch.src
                    _, ctx_lengths = batch.ctx
                    _, src_back_lengths = batch.src_back
                    report_stats.n_src_words += src_lengths.sum()
                else:
                    src_lengths = None

                tgt_outer = onmt.io.make_features(batch, 'tgt')
                tgt_back_outer = onmt.io.make_features(batch, 'tgt_back')

                for j in range(0, target_size-1, trunc_size):
                    # 1. Create truncated target.
                    tgt = tgt_outer[j: j + trunc_size]
                    tgt_back = tgt_back_outer[j: j + trunc_size]

                    # 2. F-prop all but generator.
                    # if self.accum_count == 1:
                    # self.model.zero_grad()
                    self.optim._zero_grad()
                    # outputs, attns, dec_state = \
                        # self.model(src, tgt, ctx, src_lengths, src_back, tgt_back, src_back_lengths, ctx_lengths, dec_state)
                    results = self.model(src, tgt, ctx, src_lengths, src_back, tgt_back, src_back_lengths, ctx_lengths)

                    out_cxz2y, dec_state_cxz2y, attns_cxz2y=results.out_cxz2y, results.dec_state_cxz2y, results.attns_cxz2y,
                    out_cyz2x, dec_state_cyz2x, attns_cyz2x=results.out_cyz2x, results.dec_state_cyz2x, results.attns_cyz2x,
                    out_cz2x, dec_state_cz2x, attns_cz2x=results.out_cz2x, results.dec_state_cz2x, results.attns_cz2x,
                    out_cz2y, dec_state_cz2y, attns_cz2y =results.out_cz2y, results.dec_state_cz2y, results.attns_cz2y
                    kl_loss = results.kl_loss

                    # 3. Compute loss in shards for memory efficiency.
                    batch_stats_cxz2y, loss_cxz2y = self.train_loss.loss_cxz2y.mirror_compute_loss(
                            batch, out_cxz2y, attns_cxz2y, back=False, ctx_src=False)
                    batch_stats_cyz2x, loss_cyz2x = self.train_loss.loss_cyz2x.mirror_compute_loss(
                            batch, out_cyz2x, attns_cyz2x, back=True, ctx_src=False) 
                    batch_stats_cz2x, loss_cz2x = self.train_loss.loss_cz2x.mirror_compute_loss(
                            batch, out_cz2x, attns_cz2x, back=True, ctx_src=True)
                    batch_stats_cz2y, loss_cz2y = self.train_loss.loss_cz2y.mirror_compute_loss(
                            batch, out_cz2y, attns_cz2y, back=False, ctx_src=True)
                   
                    loss_all = self.kl_knealling * kl_loss.div(nt_) + 0.5 * (loss_cxz2y + loss_cyz2x * self.back_factor + loss_cz2x + loss_cz2y * self.back_factor).div(nt_)
                    # loss_all += kl_loss.div(nt_)
                    loss_all.backward()

                    # 4. Update the parameters and statistics.
                    # if self.accum_count == 1:
                    self.optim.step()
                    total_stats_.update(batch_stats_cxz2y, batch_stats_cyz2x, batch_stats_cz2y, batch_stats_cz2x, self.kl_knealling*kl_loss.div(nt_))
                    report_stats_.update(batch_stats_cxz2y, batch_stats_cyz2x, batch_stats_cz2y, batch_stats_cz2x, self.kl_knealling*kl_loss.div(nt_))
 
                    # print(report_stats_.loss_cxz2y, report_stats_.n_words_cxz2y, report_stats_.loss_cyz2x, report_stats_.n_words_cyz2x)
                    # print(report_stats_.n_correct_cxz2y, report_stats_.n_words_cxz2y, report_stats_.n_correct_cyz2x, report_stats_.n_words_cyz2x)

                    # If truncated, don't backprop fully.
                    # if dec_state is not None:
                        # dec_state.detach()

            # if self.accum_count > 1:
                # self.optim.step()

        for i, batch_ in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.loss_cxz2y.cur_dataset = cur_dataset
            self.train_loss.loss_cz2x.cur_dataset = cur_dataset
            self.train_loss.loss_cyz2x.cur_dataset = cur_dataset
            self.train_loss.loss_cz2y.cur_dataset = cur_dataset
            # if self.kl_balance:
            self.kl_knealling = self.get_kl_knealling(epoch-1, i, len(train_iter))
            truebatch.append(batch_)
            accum += 1
            if self.normalization is "tokens":
                normalization += batch_.tgt[1:].data.view(-1) \
                                       .ne(self.padding_idx)
            else:
                normalization += batch_.batch_size

            if accum == self.accum_count:
                gradient_accumulation(
                        truebatch, total_stats,
                        report_stats, normalization)

                if report_func is not None:
                    report_stats = report_func(
                            epoch, idx, num_batches,
                            total_stats.start_time, self.optim.lr,
                            report_stats)

                truebatch = []
                accum = 0
                normalization = 0
                idx += 1
            # if i>100:
            #     break

        if len(truebatch) > 0:
            gradient_accumulation(
                    truebatch, total_stats,
                    report_stats, normalization)
            truebatch = []

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        total_stats = MirrorStatistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.loss_cxz2y.cur_dataset = cur_dataset
            self.valid_loss.loss_cz2x.cur_dataset = cur_dataset
            self.valid_loss.loss_cyz2x.cur_dataset = cur_dataset
            self.valid_loss.loss_cz2y.cur_dataset = cur_dataset


            src = onmt.io.make_features(batch, 'src', self.data_type)                
            src_back = onmt.io.make_features(batch, 'src_back', self.data_type)
            ctx = onmt.io.make_features(batch, 'ctx', self.data_type)

            if self.data_type == 'text':
                _, src_lengths = batch.src
                _, src_back_lengths = batch.src_back
                _, ctx_lengths = batch.ctx
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')
            tgt_back = onmt.io.make_features(batch, 'tgt_back')

            # F-prop through the model.
            results = self.model(src, tgt, ctx, src_lengths, src_back, tgt_back, src_back_lengths, ctx_lengths)

            out_cxz2y, attns_cxz2y = results.out_cxz2y, results.attns_cxz2y,
            out_cyz2x, attns_cyz2x = results.out_cyz2x, results.attns_cyz2x,
            out_cz2x,  attns_cz2x = results.out_cz2x, results.attns_cz2x,
            out_cz2y,  attns_cz2y = results.out_cz2y, results.attns_cz2y
            kl_loss = results.kl_loss

            # 3. Compute loss in shards for memory efficiency.
            batch_stats_cxz2y = self.valid_loss.loss_cxz2y.mirror_monolithic_compute_loss(
                    batch, out_cxz2y, attns_cxz2y, back=False, ctx_src=False)
            batch_stats_cyz2x = self.valid_loss.loss_cyz2x.mirror_monolithic_compute_loss(
                    batch, out_cyz2x, attns_cyz2x, back=True, ctx_src=False)
            batch_stats_cz2y = self.valid_loss.loss_cz2y.mirror_monolithic_compute_loss(
                    batch, out_cz2y, attns_cz2y, back=False, ctx_src=True)
            batch_stats_cz2x = self.valid_loss.loss_cz2x.mirror_monolithic_compute_loss(
                    batch, out_cz2x, attns_cz2x, back=True, ctx_src=True)

            # total_stats.update(batch_stats_cxz2y, batch_stats_cyz2x, batch_stats_cz2y, batch_stats_cz2x, kl_loss.div(batch.batch_size))
            total_stats.update(batch_stats_cxz2y, batch_stats_cyz2x, batch_stats_cz2y, batch_stats_cz2x, self.kl_knealling * kl_loss)


        # Set model back to training mode.
        self.model.train()

        return total_stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats, time_str='', total_loss=0):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        # real_generator_cxz2y = (real_model.generator_cxz2y.module
        #                   if isinstance(real_model.generator_cxz2y, nn.DataParallel)
        #                   else real_model.generator_cxz2y)
        # real_generator_cz2x = (real_model.generator_cz2x.module
        #                   if isinstance(real_model.generator_cz2x, nn.DataParallel)
        #                   else real_model.generator_cz2x)
        # real_generator_cyz2x = (real_model.generator_cyz2x.module
        #                   if isinstance(real_model.generator_cyz2x, nn.DataParallel)
        #                   else real_model.generator_cyz2x)
        # real_generator_cz2y = (real_model.generator_cz2y.module
        #                   if isinstance(real_model.generator_cz2y, nn.DataParallel)
        #                   else real_model.generator_cz2y)

        model_state_dict = real_model.state_dict()
        # model_state_dict = {k: v for k, v in model_state_dict.items()
        #                     if 'generator' not in k}        
        model_state_dict = {k: v for k, v in model_state_dict.items()}
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }

        # total_loss = (valid_stats.loss * 0.5 + valid_stats.loss_kl)
        total_loss = total_loss
        torch.save(checkpoint,
                   '%s_time_%s_acc_%.2f_ppl_%.2f_lossall_%.2f_e%d.pt'
                   % (opt.save_model, time_str, valid_stats.accuracy(),
                      valid_stats.ppl(), total_loss, epoch))

        file_path='%s_time_%s_acc_%.2f_ppl_%.2f_lossall_%.2f_e%d.pt'% (opt.save_model, time_str, valid_stats.accuracy(),
                      valid_stats.ppl(), total_loss, epoch)
        self.checkpoint_list.append(file_path)
        # if valid_stats.ppl()<=self.best_model['model_ppl']:
        #     self.best_model['model_ppl']=valid_stats.ppl()
        #     self.best_model['model_name']=file_path        
        if total_loss<=self.best_model['model_ppl']:
            self.best_model['model_ppl']=total_loss
            self.best_model['model_name']=file_path
