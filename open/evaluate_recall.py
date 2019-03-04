""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function

import os
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions, k, no_f1=False):
    count = 0
    f1 = exact_match = total = 0
    ranks = {}
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    count += 1
                    # print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']][:k]
                rank, cur_exact_match = max(enumerate(metric_max_over_ground_truths(
                    exact_match_score, each, ground_truths) for each in prediction), key=lambda item: item[1])
                exact_match += cur_exact_match
                ranks[qa['id']] = rank + 1 if cur_exact_match == 1.0 else 1e9
                if not no_f1:
                    f1 += max(metric_max_over_ground_truths(
                        f1_score, each, ground_truths) for each in prediction)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    if count:
        print('There are %d unanswered question(s)' % count)

    return {'exact_match': exact_match, 'f1': f1, 'ranks': ranks}


def get_args():
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('data_path', help='Dataset file')
    parser.add_argument('od_out_path', help='Prediction File')
    parser.add_argument('--do_f1', default=False, action='store_true')
    parser.add_argument('--k_start', default=1, type=int)
    parser.add_argument('--scores_dir', default='scores')
    args = parser.parse_args()

    args.scores_path = os.path.join(args.scores_dir, 'scores_%s' % os.path.basename(args.od_out_path))
    return args


def evaluate_recall(args):
    if not os.path.exists(args.scores_dir):
        os.makedirs(args.scores_dir)

    with open(args.data_path) as data_path:
        dataset_json = json.load(data_path)
        dataset = dataset_json['data']
    with open(args.od_out_path) as od_out_path:
        predictions = json.load(od_out_path)
    num_answers = len(next(iter(predictions.values())))
    if not args.do_f1:
        e = evaluate(dataset, predictions, num_answers + 1)
        ranks = e['ranks']
        scores = []
        for k in range(1, num_answers + 1):
            b = [float(rank <= k) for rank in ranks.values()]
            scores.append([k, sum(b) / len(b)])
        with open(args.scores_path, 'w') as fp:
            json.dump(scores, fp)

        print('Top-k results for %s:' % args.od_out_path)
        for k, score in scores:
            print('%d: %.2f' % (k, score * 100))
    else:
        for k in range(args.k_start, num_answers + 1):
            e = evaluate(dataset, predictions, k)
            print('%d: f1=%.2f, em=%.2f' % (k, e['f1'], e['exact_match']))


def main():
    args = get_args()
    evaluate_recall(args)


if __name__ == '__main__':
    main()
