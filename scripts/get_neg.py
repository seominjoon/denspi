import json
import os
import argparse
import random

from tqdm import tqdm


def add_dummy(args):
    random.seed(args.seed)
    with open(args.from_path, 'r') as fp:
        in_ = json.load(fp)

    article = {'title': 'dummy', 'paragraphs': []}
    for _ in range(args.num_samples):
        sampled_article = random.choice(in_['data'])
        sampled_para = random.choice(sampled_article['paragraphs'])
        sampled_qa = random.choice(sampled_para['qas'])
        if len(sampled_qa['answers']) == 0:
            point = random.choice(range(len(sampled_para['context'])))
            left_offset = random.choice(range(args.min_dist, args.max_dist))
            right_offset = random.choice(range(args.min_dist, args.max_dist))
            context = sampled_para['context'][max(0, point - left_offset):min(len(sampled_para['context']), point + right_offset)]
        elif len(sampled_qa['answers']) == 1:
            continue
        else:
            raise Exception()
        new_para = {'context': context, 'qas': [sampled_qa]}

        article['paragraphs'].append(new_para)

    print(article)
    out = {'data': [article] + in_['data']}

    with open(args.to_path, 'w') as fp:
        json.dump(out, fp)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('from_path')
    parser.add_argument('to_path')
    parser.add_argument('--num_samples', default=10000, type=int)
    parser.add_argument('--min_dist', default=10, type=int)
    parser.add_argument('--max_dist', default=100, type=int)
    parser.add_argument('--seed', default=29, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    add_dummy(args)


if __name__ == '__main__':
    main()
