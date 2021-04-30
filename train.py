#!/usr/bin/env python

from __future__ import division

import argparse
import glob
import os
import sys
import random
from itertools import chain
import logging
logging.basicConfig(level = logging.INFO)
import torch
from datetime import datetime
# try:
#     torch._utils._rebuild_tensor_v2
# except AttributeError:
#     def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
#         tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
#         tensor.requires_grad = requires_grad
#         tensor._backward_hooks = backward_hooks
#         return tensor
#     torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torch.nn as nn
from torch import cuda

import onmt
import onmt.io
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import use_gpu, Pack
import opts


parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)

opt = parser.parse_args()
if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn = (opt.encoder_type == "brnn")
if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

if opt.rnn_type == "SRU" and not opt.gpuid:
    raise AssertionError("Using SRU requires -gpuid set.")

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)

if len(opt.gpuid) > 1:
    sys.stderr.write("Sorry, multigpu isn't supported yet, coming soon!\n")
    sys.exit(1)


# Set up the Crayon logging server.
if opt.exp_host != "":
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=opt.exp_host)

    experiments = cc.get_experiment_names()
    print(experiments)
    if opt.exp in experiments:
        cc.remove_experiment(opt.exp)
    experiment = cc.create_experiment(opt.exp)


# print(opt)

def report_func(epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch+1, num_batches, start_time)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr)
        report_stats = onmt.MirrorStatistics()

    return report_stats


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """
    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def get_cur_dataset(self):
        return self.cur_dataset

    def _next_dataset_iterator(self, dataset_iter):
        try:
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return onmt.io.OrderedIterator(
                dataset=self.cur_dataset, batch_size=self.batch_size,
                batch_size_fn=self.batch_size_fn,
                device=self.device, train=self.is_train,
                sort=False, sort_within_batch=True,
                repeat=False)


def make_dataset_iter(datasets, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_size_fn = None
    if is_train and opt.batch_type == "tokens":
        def batch_size_fn(new, count, sofar):
            return sofar + max(len(new.tgt), len(new.src)) + 1

    # device = opt.gpuid[0] if opt.gpuid else -1
    device = torch.device('cuda') if opt.gpuid else -1
    

    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train)


def make_loss_compute(model, tgt_vocab, opt, generator):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    if opt.copy_attn:
        compute = onmt.modules.MirrorCopyGeneratorLossCompute(
            generator, tgt_vocab, opt.copy_attn_force)
    else:
        compute = onmt.Loss.MirrorNMTLossCompute(
            generator, tgt_vocab,
            label_smoothing=opt.label_smoothing)
    if use_gpu(opt):
        compute.cuda()

    return compute

def make_mirror_loss(model, fields, opt):
    # loss_id=='cxz2y':
    generator1 = model.generator_cxz2y
    loss_cxz2y = make_loss_compute(model, fields['tgt'].vocab, opt, generator1)
    # loss_id=='cz2x':
    generator2 = model.generator_cz2x
    loss_cz2x = make_loss_compute(model, fields['tgt_back'].vocab, opt, generator2)
    # loss_id=='cyz2x':
    generator3 = model.generator_cyz2x
    loss_cyz2x = make_loss_compute(model, fields['tgt_back'].vocab, opt, generator3)
    # loss_id=='cz2y':
    generator4 = model.generator_cz2y
    loss_cz2y = make_loss_compute(model, fields['tgt'].vocab, opt, generator4)
    loss_mirror = Pack(loss_cxz2y=loss_cxz2y,
                        loss_cz2x=loss_cz2x,
                        loss_cyz2x=loss_cyz2x,
                        loss_cz2y=loss_cz2y)
    return loss_mirror


def train_model(model, fields, optim, data_type, model_opt, time_str=''):

    train_loss_dict = make_mirror_loss(model, fields, opt)
    valid_loss_dict = make_mirror_loss(model, fields, opt)


    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches

    trainer = onmt.Trainer(model, train_loss_dict, valid_loss_dict, optim,
                           trunc_size, shard_size, data_type,
                           opt.normalization, opt.accum_count, opt.back_factor,
                           opt.kl_balance, opt.kl_fix)

    # train_datasets = lazily_load_dataset("train")
    # valid_datasets = lazily_load_dataset("valid")
    # train_iter = make_dataset_iter(train_datasets, fields, opt)
    # valid_iter = make_dataset_iter(valid_datasets, fields, opt, is_train=False)
    val_value = []
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        # print('')
        # train_iter = make_dataset_iter(train_datasets, fields, opt)
        # valid_iter = make_dataset_iter(valid_datasets, fields, opt, is_train=False)
        # 1. Train for one epoch on the training set.
        train_datasets = lazily_load_dataset("train")
        train_iter = make_dataset_iter(train_datasets, fields, opt)
        print("train iter size: {}".format(len(train_iter)))
        train_stats = trainer.train(train_iter, epoch, report_func)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())
        del train_datasets, train_iter
        # 2. Validate on the validation set.
        valid_iter = make_dataset_iter(lazily_load_dataset("valid"),
                                       fields, opt,
                                       is_train=False)
        valid_size = valid_iter.__len__()
        with torch.no_grad():
            valid_stats = trainer.validate(valid_iter)
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation loss_total: %g' % (valid_stats.loss * 0.5))
        print('Validation accuracy: %g' % valid_stats.accuracy())
        # print('Validation kl: %g' % (valid_stats.loss_kl/valid_stats.counter))
        print('Validation kl: %g' % (valid_stats.loss_kl/valid_size))
        print("current kl_factor: {}".format(str(trainer.kl_knealling)))
        val_value.append((valid_stats.loss * 0.5 + valid_stats.loss_kl)/valid_size)
        del valid_iter
        # 3. Log to remote server.
        if opt.exp_host:
            train_stats.log("train", experiment, optim.lr)
            valid_stats.log("valid", experiment, optim.lr)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats.ppl(), epoch)
        # 5. Drop a checkpoint if needed.
        # if epoch >= opt.start_checkpoint_at:
        if epoch >= opt.kl_balance:
            trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats, time_str, val_value[-1])

        if len(val_value)>(opt.kl_balance+3) and val_value[-1]>val_value[-2] and val_value[-2]>val_value[-3]:
            print("early stop due to the val loss")
            break
    print("**** Training Finished *****")

    assert trainer.best_model['model_name'] in trainer.checkpoint_list
    for cp in trainer.checkpoint_list:
        if cp!=trainer.best_model['model_name']:
            print(cp)
            os.remove(cp)
    print("Cleaning redundant checkpoints")

    print("the best model path: {}".format(trainer.best_model['model_name']))
    print("the best model ppl: {}".format(trainer.best_model['model_ppl']))


def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def lazily_load_dataset(corpus_type):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        print('Loading %s dataset from %s, number of examples: %d' %
              (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one onmt.io.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        yield lazy_dataset_loader(pt, corpus_type)


def load_fields(dataset, data_type, checkpoint):

    fields = onmt.io.load_fields_from_vocab(
                torch.load(opt.data + '.vocab.pt'), data_type)
    print("field keys: {}".format(fields.keys()))
    fields = dict([(k, f) for (k, f) in fields.items()
                  if k in dataset.examples[0].__dict__])
    print("field keys: {}".format(fields.keys()))

    if checkpoint is not None:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = onmt.io.load_fields_from_vocab(
                    checkpoint['vocab'], data_type)

    if data_type == 'text':
        print(' * vocabulary size. source = %d; target = %d source_back = %d; target_back = %d ctx = %d' %
              (len(fields['src'].vocab), len(fields['tgt'].vocab), len(fields['src_back'].vocab), len(fields['tgt_back'].vocab), len(fields['ctx'].vocab)))
    else:
        print(' * vocabulary size. target = %d' %
              (len(fields['tgt'].vocab)))

    return fields


def collect_report_features(fields):
    src_features = onmt.io.collect_features(fields, side='src')
    tgt_features = onmt.io.collect_features(fields, side='tgt')

    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        print(' * tgt feature %d size = %d' % (j, len(fields[feat].vocab)))


def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')

    model = onmt.ModelConstructor.make_mirror_model(model_opt, fields,
                                                  use_gpu(opt), checkpoint, opt.mirror_type)

    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)

    return model


def build_optim(model, checkpoint):
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)

    optim.set_parameters(model.parameters())

    return optim
def get_time():
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")



def main():

    # Lazily load a list of train/validate dataset.
    print("Lazily loading train/validate datasets from '%s'" % opt.data)
    train_datasets = lazily_load_dataset("train")
    print(' * maximum batch size: %d' % opt.batch_size)
     
    # Peek the fisrt dataset to determine the data_type.
    # (This will load the first dataset.)
    first_dataset = next(train_datasets)
    train_datasets = chain([first_dataset], train_datasets)
    data_type = first_dataset.data_type

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear.
        opt.start_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        model_opt = opt

    # Load fields generated from preprocess phase.
    fields = load_fields(first_dataset, data_type, checkpoint)

    # Report src/tgt features.
    collect_report_features(fields)

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    tally_parameters(model)
    check_save_model_path()

    # Build optimizer.
    optim = build_optim(model, checkpoint)

    # Do training.
    time_str = get_time()
    print("Start training at: {}".format(time_str))
    train_model(model, fields, optim, data_type, model_opt, time_str)


if __name__ == "__main__":
    main()
