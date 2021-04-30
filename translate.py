#!/usr/bin/env python

from __future__ import division, unicode_literals
import os
import argparse
import math
import codecs
import torch

from itertools import count

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import opts

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()


def _report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total / words_total)))


def _report_bleu():
    import subprocess
    print()
    res = subprocess.check_output(
        "perl tools/multi-bleu.perl %s < %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(">> " + res.strip())


def _report_rouge():
    import subprocess
    res = subprocess.check_output(
        "python tools/test_rouge.py -r %s -c %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(res.strip())


def get_sent_string_from_indices(sent, vocab):
    string = " ".join([vocab.itos[word_id] for word_id in sent])
    return string

def print_oracle_targets(oracle_targets, vocab):
    targets, bleu_scores = oracle_targets
    for i, oracle_target in enumerate(targets):
        print(get_sent_string_from_indices(oracle_target, vocab), bleu_scores[i])

def write_source_and_m_diverse_targets(writer, sources, diverse_targets, vocab):
    batch_size = len(sources)
    for i, source in enumerate(sources):
        # Write the source sentence first
        writer.write("Src:\t")
        src_string = get_sent_string_from_indices(source, vocab)
        writer.write(src_string + "\n=======================\n")
        # Write the targets one by one
        targets = diverse_targets[i]
        # Print the targets of M iterations
        print(len(targets))
        for j in range(len(targets)):
            # each element is a n-best list of indices
            for k in range(len(targets[j])):
                word_ids = targets[j][k]
                tgt_string = get_sent_string_from_indices(word_ids, vocab)
                writer.write(tgt_string + "\n")
            writer.write("$$$$$$$$$$$$$$$$$$$$$$$$\n")
        writer.write("\n")


def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
        device = torch.device('cuda')

    # Load the model.
    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)

    # Load the MMI model (if given)
    if(opt.mmi_model != ""):
        mmi_fields, mmi_model, mmi_model_opt = \
            onmt.ModelConstructor.load_mmi_model(opt, dummy_opt.__dict__)
    # Initialize the MMI scorer
    if(opt.mmi_model != ""):
        scorer = onmt.translate.MMIGlobalScorer(mmi_model, mmi_fields, cuda=opt.cuda)
    else:
        scorer = None

    # File to write sentences to.
    out_file = codecs.open(opt.output, 'w', 'utf-8')

    # Test data

    data = onmt.io.build_dataset(fields, opt.data_type,
                                 opt.src, opt.tgt, opt.ctx,
                                 src_dir=opt.src_dir,
                                 sample_rate=opt.sample_rate,
                                 window_size=opt.window_size,
                                 window_stride=opt.window_stride,
                                 window=opt.window,
                                 use_filter_pred=False)


    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
    data_iter = onmt.io.OrderedIterator(
        dataset=data, device=torch.device('cuda') if opt.cuda else opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        sort_within_batch=True, shuffle=False)

    # Translator
    # scorer = onmt.translate.GNMTGlobalScorer(opt.alpha, opt.beta)
    translator = onmt.translate.Translator(model, fields,
                                           beam_size=opt.beam_size,
                                           n_best=opt.n_best,
                                           global_scorer=scorer,
                                           max_length=opt.max_length,
                                           copy_attn=model_opt.copy_attn,
                                           cuda=opt.cuda,
                                           beam_trace=opt.dump_beam != "",
                                           min_length=opt.min_length,
                                           use_dc=False if opt.use_dc<0 else True)
    builder = onmt.translate.TranslationBuilder(
        data, translator.fields,
        opt.n_best, opt.replace_unk, opt.tgt)

    # Statistics
    counter = count(1)
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0

    ##NOTE: Inserting custom code to manually save the M-diverse n-best list
    # writer = None
    # if opt.diverse_targets:
    #     print("The parameter passed is: \n", opt.diverse_targets)
    #     writer = codecs.open(opt.diverse_targets, "w", "utf-8")

    batch_id = 0

    for batch in data_iter:
        print("batch count: {}".format(batch_id))
        batch_id+=1
        with torch.no_grad():
            batch_data = translator.translate_batch(batch, data)
            # print("^^^^^^^^^^^\nThe orcale target for the current batch are:")
            # print_oracle_targets(batch_oracle_targets, fields["tgt"].vocab)
            # if writer:
            #     write_source_and_m_diverse_targets(writer, sources, diverse_targets, fields["tgt"].vocab)
            # print(batch_data.tgt.data)
            translations = builder.from_batch(batch_data)

        for trans in translations:
            pred_score_total += trans.pred_scores[0]
            pred_words_total += len(trans.pred_sents[0])
            if opt.tgt:
                gold_score_total += trans.gold_score
                gold_words_total += len(trans.gold_sent)

            n_best_preds = [" ".join(pred)
                            for pred in trans.pred_sents[:opt.n_best]]
            out_file.write('\n'.join(n_best_preds))
            out_file.write('\n')
            out_file.flush()

            if opt.verbose:
                sent_number = next(counter)
                output = trans.log(sent_number)
                os.write(1, output.encode('utf-8'))

    _report_score('PRED', pred_score_total, pred_words_total)
    if opt.tgt:
        _report_score('GOLD', gold_score_total, gold_words_total)
        if opt.report_bleu:
            _report_bleu()
        if opt.report_rouge:
            _report_rouge()

    if opt.dump_beam:
        import json
        json.dump(translator.beam_accum,
                  codecs.open(opt.dump_beam, 'w', 'utf-8'))


if __name__ == "__main__":
    main()
