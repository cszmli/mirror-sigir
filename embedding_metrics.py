"""
Word embedding based evaluation metrics for dialogue.

This method implements three evaluation metrics based on Word2Vec word embeddings, which compare a target utterance with a model utterance:
1) Computing cosine-similarity between the mean word embeddings of the target utterance and of the model utterance
2) Computing greedy meatching between word embeddings of target utterance and model utterance (Rus et al., 2012)
3) Computing word embedding extrema scores (Forgues et al., 2014)

We believe that these metrics are suitable for evaluating dialogue systems.

Example run:

    python embedding_metrics.py path_to_ground_truth.txt path_to_predictions.txt path_to_embeddings.bin

The script assumes one example per line (e.g. one dialogue or one sentence per line), where line n in 'path_to_ground_truth.txt' matches that of line n in 'path_to_predictions.txt'.

NOTE: The metrics are not symmetric w.r.t. the input sequences. 
      Therefore, DO NOT swap the ground truths with the predicted responses.

References:

A Comparison of Greedy and Optimal Assessment of Natural Language Student Input Word Similarity Metrics Using Word to Word Similarity Metrics. Vasile Rus, Mihai Lintean. 2012. Proceedings of the Seventh Workshop on Building Educational Applications Using NLP, NAACL 2012.

Bootstrapping Dialog Systems with Word Embeddings. G. Forgues, J. Pineau, J. Larcheveque, R. Tremblay. 2014. Workshop on Modern Machine Learning and Natural Language Processing, NIPS 2014.


"""
from random import randint
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import argparse
import scipy.stats as stats


def cosine_similarity(s, g):
    similarity = np.sum(s * g, axis=1) / np.sqrt((np.sum(s * s, axis=1) * np.sum(g * g, axis=1)))

    # return np.sum(similarity)
    return similarity

def eval_embedding(ground_truth, samples, word2vec):
    samples = [[word2vec[s] for s in sent.split() if s in word2vec] for sent in samples]
    ground_truth = [[word2vec[s] for s in sent.split() if s in word2vec] for sent in ground_truth]

    indices = [i for i, s, g in zip(range(len(samples)), samples, ground_truth) if s != [] and g != []]
    samples = [samples[i] for i in indices]
    ground_truth = [ground_truth[i] for i in indices]
    n = len(samples)

    metric_average = embedding_metric(samples, ground_truth, word2vec, 'average')
    metric_extrema = embedding_metric(samples, ground_truth, word2vec, 'extrema')
    metric_greedy = embedding_metric(samples, ground_truth, word2vec, 'greedy')

    confidence_avg, confidence_extrema, confidence_greedy = 1.96*np.std(metric_average)/np.sqrt(len(metric_average)), 1.96*np.std(metric_extrema)/np.sqrt(len(metric_extrema)), 1.96*np.std(metric_greedy)/np.sqrt(len(metric_greedy))
    print_str = 'Metrics - Average: {} (-+{}), Extrema: {} (-+{}), Greedy: {} (-+{})'.format(metric_average.mean(), confidence_avg, metric_extrema.mean(), confidence_extrema, metric_greedy.mean(), confidence_greedy)
    print(print_str)

def embedding_metric(samples, ground_truth, word2vec, method='average'):

    if method == 'average':
        # s, g: [n_samples, word_dim]
        s = [np.mean(sample, axis=0) for sample in samples]
        g = [np.mean(gt, axis=0) for gt in ground_truth]
        return cosine_similarity(np.array(s), np.array(g))
    elif method == 'extrema':
        s_list = []
        g_list = []
        for sample, gt in zip(samples, ground_truth):
            s_max = np.max(sample, axis=0)
            s_min = np.min(sample, axis=0)
            s_plus = np.absolute(s_min) <= s_max
            s_abs = np.max(np.absolute(sample), axis=0)
            s = s_max * s_plus + s_min * np.logical_not(s_plus)
            s_list.append(s)

            g_max = np.max(gt, axis=0)
            g_min = np.min(gt, axis=0)
            g_plus = np.absolute(g_min) <= g_max
            g_abs = np.max(np.absolute(gt), axis=0)
            g = g_max * g_plus + g_min * np.logical_not(g_plus)
            g_list.append(g)

        return cosine_similarity(np.array(s_list), np.array(g_list))
    elif method == 'greedy':
        sim_list = []
        for s, g in zip(samples, ground_truth):
            s = np.array(s)
            g = np.array(g).T
            sim = (np.matmul(s, g)
                   / np.sqrt(np.matmul(np.sum(s * s, axis=1, keepdims=True), np.sum(g * g, axis=0, keepdims=True))))
            sim = np.max(sim, axis=0)
            sim_list.append(np.mean(sim))

        # return np.sum(sim_list)
        return np.array(sim_list)
    else:
        raise NotImplementedError

def embedding_score(refs, preds, w2v):
    r = average(refs, preds, w2v)
    print(" Embedding Average Score: %f +/- %f ( %f )" %(r[0], r[1], r[2]))
    r = greedy_match(refs, preds, w2v)
    print(" Greedy Matching Score: %f +/- %f ( %f )" %(r[0], r[1], r[2]))
    r = extrema_score(refs, preds, w2v)
    print(" Extrema Score: %f +/- %f ( %f )" %(r[0], r[1], r[2]))
    
def greedy_match(fileone, filetwo, w2v):
    res1 = greedy_score(fileone, filetwo, w2v)
    res2 = greedy_score(filetwo, fileone, w2v)
    res_sum = (res1 + res2)/2.0

    return np.mean(res_sum), 1.96*np.std(res_sum)/np.sqrt(float(len(res_sum))), np.std(res_sum),res_sum.tolist()


def greedy_score(fileone, filetwo, w2v):
    r1 = fileone
    r2 = filetwo
    # dim = w2v.layer_size # embedding dimensions
    dim =300
    scores = []

    for i in range(len(r1)):
        tokens1 = r1[i].strip().split()
        tokens2 = r2[i].strip().split()
        X= np.zeros((dim,))
        y_count = 0
        x_count = 0
        o = 0.0
        Y = np.zeros((dim,1))
        for tok in tokens2:
            if tok in w2v:
                Y = np.hstack((Y,(w2v[tok].reshape((dim,1)))))
                y_count += 1

        for tok in tokens1:
            if tok in w2v:
                w_vec = w2v[tok].reshape((1,dim))
                tmp = np.dot(w_vec, Y)/ np.linalg.norm(w_vec)/np.linalg.norm(Y)
                # tmp  = w2v[tok].reshape((1,dim)).dot(Y)
                o += np.max(tmp)
                x_count += 1

        # if none of the words in response or ground truth have embeddings, count result as zero
        if x_count < 1 or y_count < 1:
            scores.append(0)
            continue

        o /= float(x_count)
        scores.append(o)


    return np.asarray(scores)


def extrema_score(fileone, filetwo, w2v):
    r1 = fileone
    r2 = filetwo
    scores = []

    for i in range(len(r1)):
        tokens1 = r1[i].strip().split()
        tokens2 = r2[i].strip().split()
        X= []
        for tok in tokens1:
            if tok in w2v:
                X.append(w2v[tok])
        Y = []
        for tok in tokens2:
            if tok in w2v:
                Y.append(w2v[tok])

        # if none of the words have embeddings in ground truth, skip
        if np.linalg.norm(X) < 0.00000000001:
            continue

        # if none of the words have embeddings in response, count result as zero
        if np.linalg.norm(Y) < 0.00000000001:
            scores.append(0)
            continue

        xmax = np.max(X, 0)  # get positive max
        xmin = np.min(X,0)  # get abs of min
        xtrema = []
        for i in range(len(xmax)):
            if np.abs(xmin[i]) > xmax[i]:
                xtrema.append(xmin[i])
            else:
                xtrema.append(xmax[i])
        X = np.array(xtrema)   # get extrema

        ymax = np.max(Y, 0)
        ymin = np.min(Y,0)
        ytrema = []
        for i in range(len(ymax)):
            if np.abs(ymin[i]) > ymax[i]:
                ytrema.append(ymin[i])
            else:
                ytrema.append(ymax[i])
        Y = np.array(ytrema)

        o = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)

        scores.append(o)

    scores = np.asarray(scores)
    return np.mean(scores), 1.96*np.std(scores)/np.sqrt(float(len(scores))), np.std(scores),scores.tolist()


def average(fileone, filetwo, w2v):
    r1 = fileone
    r2 = filetwo
    # dim = w2v.layer_size # dimension of embeddings
    dim =300
    scores = []

    for i in range(len(r1)):
        tokens1 = r1[i].strip().split()
        tokens2 = r2[i].strip().split()
        X= np.zeros((dim,))
        for tok in tokens1:
            if tok in w2v:
                X+=w2v[tok]
        Y = np.zeros((dim,))
        for tok in tokens2:
            if tok in w2v:
                Y += w2v[tok]

        # if none of the words in ground truth have embeddings, skip
        if np.linalg.norm(X) < 0.00000000001:
            continue

        # if none of the words have embeddings in response, count result as zero
        if np.linalg.norm(Y) < 0.00000000001:
            scores.append(0)
            continue

        X = np.array(X)/np.linalg.norm(X)
        Y = np.array(Y)/np.linalg.norm(Y)
        o = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)

        scores.append(o)

    scores = np.asarray(scores)
    return np.mean(scores), 1.96*np.std(scores)/np.sqrt(float(len(scores))), np.std(scores),scores.tolist()


