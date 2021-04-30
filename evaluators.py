import numpy as np
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import logging
from collections import defaultdict
from embedding_metrics import embedding_score, eval_embedding
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import os
import nltk


def get_tokenize():
    return nltk.RegexpTokenizer(r'\w+|#\w+|<\w+>|%\w+|[^\w\s]+').tokenize


class EvaluatorBase(object):
    def initialize(self):
        raise NotImplementedError

    def add_example(self, ref, hyp, domain='default'):
        raise NotImplementedError

    def get_report(self, include_error=False):
        raise NotImplementedError

    @staticmethod
    def _get_prec_recall(tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1

    @staticmethod
    def _get_tp_fp_fn(label_list, pred_list):
        tp = len([t for t in pred_list if t in label_list])
        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(label_list) - tp)
        return tp, fp, fn


class BleuEvaluator(EvaluatorBase):
    """
    Use string matching to find the F-1 score of slots
    Use logistic regression to find F-1 score of acts
    Use string matching to find F-1 score of KB_SEARCH
    """
    # logger = logging.getLogger(__name__)

    def __init__(self, data_name, embedding_path):
        self.data_name = data_name
        self.domain_labels = []
        self.domain_hyps = []
        self.w2v = None
        if os.path.exists(embedding_path):
            self.w2v = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
        else:
            raise ValueError("cannot find the embedding file")

    def initialize(self):
        self.domain_labels = []
        self.domain_hyps = []

    def add_example(self, ref, hyp, domain='default'):
        self.domain_labels.append(ref)
        self.domain_hyps.append(hyp)

    def embedding_report(self,):
        refs = self.domain_labels
        hyps = self.domain_hyps
        eval_embedding(refs, hyps, self.w2v)

    def calc_diversity(self):
        tokens = [0.0,0.0]
        tokenize = get_tokenize()
        types = [defaultdict(int),defaultdict(int)]
        predictions = self.domain_hyps
        for line in predictions:
            words = tokenize(line)[:]
            for n in range(2):
                for idx in range(len(words)-n):
                    ngram = ' '.join(words[idx:idx+n+1])
                    types[n][ngram] = 1
                    tokens[n] += 1
        div1 = len(types[0].keys())/tokens[0]
        div2 = len(types[1].keys())/tokens[1]
        # print(types[1].keys())
        print("Distinct-1: {}, words: {}; Distinct-2: {}, words: {}".format(div1, str(len(types[0].keys()))+'/' + str(tokens[0]),  div2, str(len(types[1].keys()))+'/' + str(tokens[1])))


    def get_report(self, include_error=False):
        reports = []
        tokenize = get_tokenize()
        len_list = []

        labels = self.domain_labels
        predictions = self.domain_hyps
        # print("Generate report for {} samples".format(len(predictions)))
        refs, hyps = [], []
        for label, hyp in zip(labels, predictions):
            ref_tokens = tokenize(label)[:]
            hyp_tokens = tokenize(hyp)[:]

            refs.append([ref_tokens])
            hyps.append(hyp_tokens)
            len_list.append(len(hyp_tokens))
        # compute corpus level scores
        bleu1 = bleu_score.corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1, weights=(1, 0, 0, 0))
        bleu2 = bleu_score.corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1, weights=(0.5, 0.5, 0, 0))
        bleu3 = bleu_score.corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = bleu_score.corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)

        report = "\nBLEU-1 %f BLEU-2 %f BLEU-3 %f BLEU-4 %f Avg-len %f" % ( bleu1, bleu2, bleu3, bleu4, np.mean(len_list))
        print(report)


def load_file(ref, pred):
    with open(ref, 'r') as f:
        data_ref = f.readlines()
    with open(pred, 'r') as f:
        data_pred = f.readlines()
    num = min(len(data_ref), len(data_pred))
    return data_ref[:num], data_pred[:num]


def daily_eval(evaluator):
    ref_path = '/your/folder/mirror-cvae/data/daily_dialog/test/mmi_dialogues_y.txt'
    pred_path_list = []

    pred_path_list.append('/your/folder/folder_mirror_nmt/new_mirror_output_nodc10_r235_b10_real.txt')
    pred_path_list.append('/your/folder/folder_mirror_nmt/new_mirror_output_nodc10_r236_b10_real.txt')
    pred_path_list.append('/your/folder/folder_mirror_nmt/new_mirror_output_nodc10_r237_b10_real.txt')

    pred_path_list.append('/your/folder/folder_mirror_nmt/new_mirror_output_nodc10_r257_b10_real.txt')
    pred_path_list.append('/your/folder/folder_mirror_nmt/new_mirror_output_nodc10_r258_b10_real.txt')
    pred_path_list.append('/your/folder/folder_mirror_nmt/new_mirror_output_nodc10_r259_b10_real.txt')





    pred_path_list.append('/your/folder/folder_dc_mmi/new_DC_10_decoding_dailydialog_dev_test_predictions_r1.txt')
    # pred_path_list.append('/your/folder/folder_dc_mmi/new_DCMMI_10_decoding_dailydialog_dev_test_predictions_r1.txt')
    pred_path_list.append('/your/folder/folder_dc_mmi/new_DCMMI_10_decoding_dailydialog_dev_test_predictions_r2.txt')
    pred_path_list.append('/your/folder/VHRED/Output/daily_hred.txt')
    pred_path_list.append('/your/folder/VHRED/Output/daily_vhred.txt')
    pred_path_list.append('/your/folder/folder_dc_mmi/new_Seq2Seq_decoding_dailydialog_dev_test_predictions_r1.txt')
    pred_path_list.append('/your/folder/folder_dc_mmi/new_Seq2Seq_decoding_dailydialog_dev_test_predictions_r2.txt') 
    pred_path_list.append('/your/folder/folder_dc_mmi/new_MMI_decoding_dailydialog_dev_test_predictions_r1_b10.txt')  #MMI B10
    pred_path_list.append('/your/folder/folder_dc_mmi/new_MMI_decoding_dailydialog_dev_test_predictions_r1.txt')  #MMI B50
    pred_path_list.append(ref_path)

    print("\n\n\n#####################")
    print("**** DailyDialog ****")

    for pred_path in pred_path_list:
        print("================================")
        print("\ninitialize evaluator")
        print('current file: ' + pred_path)
        evaluator.initialize()    
        data_ref, data_pred = load_file(ref_path, pred_path)
        print("line number: {}".format(len(data_pred)))
        evaluator.domain_labels = data_ref
        evaluator.domain_hyps = data_pred
        evaluator.get_report()
        evaluator.calc_diversity()
        evaluator.embedding_report()


def movie_eval(evaluator):
    ref_path = '/your/folder/movietriple/test/pair_mmi_dialogues_y.txt'
    pred_path_list = []
    pred_path_list.append('/your/folder/folder_mirror_nmt/new_mirror_output_movienodc10_r1_b10_real.txt')

    pred_path_list.append('/your/folder/folder_dc_mmi/DC_10_decoding_pair_movie_dev_test_predictions.txt')
    # pred_path_list.append('/your/folder/folder_dc_mmi/DCMMI_10_decoding_pair_movie_dev_test_predictions.txt')
    pred_path_list.append('/your/folder/folder_dc_mmi/DCMMI_10_decoding_pair_movie_dev_test_predictions_r2.txt')
    pred_path_list.append('/your/folder/VHRED/Output/movie_hred.txt')
    pred_path_list.append('/your/folder/VHRED/Output/movie_vhred.txt')
    pred_path_list.append('/your/folder/folder_dc_mmi/seq2seq_10_decoding_pair_movie_dev_test_predictions.txt')
    pred_path_list.append('/your/folder/folder_dc_mmi/new_Seq2Seq_decoding_movie_dev_test_predictions_r2.txt')
    pred_path_list.append('/your/folder/folder_dc_mmi/MMI_10_decoding_pair_movie_dev_test_predictions_b10.txt')
    pred_path_list.append('/your/folder/folder_dc_mmi/MMI_10_decoding_pair_movie_dev_test_predictions_b50.txt')
    pred_path_list.append(ref_path)

    print("\n\n\n#####################")
    print('Movie')

    for pred_path in pred_path_list:
        print("\ninitialize evaluator")
        print('current file: ' + pred_path)
        evaluator.initialize()    
        data_ref, data_pred = load_file(ref_path, pred_path)
        print("line number: {}".format(len(data_pred)))
        evaluator.domain_labels = data_ref
        evaluator.domain_hyps = data_pred
        evaluator.get_report()
        evaluator.calc_diversity()
        evaluator.embedding_report()
if __name__ == "__main__":
    evaluator = BleuEvaluator('Test', '/your/folder/GoogleNews-vectors-negative300.bin')
    daily_eval(evaluator)
    movie_eval(evaluator)
    # reddit_eval(evaluator)
    # persona_eval(evaluator)




    



    





