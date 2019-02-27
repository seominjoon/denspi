""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
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


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    parser.add_argument('--no_f1', default=False, action='store_true')
    parser.add_argument('--k_start', default=1, type=int)
    parser.add_argument('--scores_path', default='scores.json')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    num_answers = len(next(iter(predictions.values())))
    if args.no_f1:
        e = evaluate(dataset, predictions, num_answers + 1)
        ranks = e['ranks']
        scores = {}
        for k in range(1, num_answers + 1):
            b = [float(rank <= k) for rank in ranks.values()]
            scores[k] = sum(b) / len(b)
        with open(args.scores_path, 'w') as fp:
            json.dump(scores, fp)
        print(json.dumps(scores))
    else:
        for k in range(args.k_start, num_answers + 1):
            print('%d:' % k, json.dumps(evaluate(dataset, predictions, k)))