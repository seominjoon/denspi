import json
import os
import argparse
import random

from tqdm import tqdm


def normalize(text):
    return text.lower().replace('_', ' ')


def downsize_and_append(args):
    # TODO : split into several files
    dev_data_path = os.path.join(args.data_dir, 'dev-v1.1.json')
    with open(dev_data_path, 'r') as fp:
        dev_data = json.load(fp)

    names = os.listdir(args.from_dir)
    from_paths = [os.path.join(args.from_dir, name) for name in names]

    for from_path in tqdm(from_paths):
        with open(from_path, 'r') as fp:
            from_ = json.load(fp)
        articles = random.sample(from_['data'], args.sample_per_file)
        for article in articles:
            to_article = {'paragraphs': [], 'title': article['title']}
            context = ""
            for para in article['paragraphs']:
                if args.concat:
                    if len(para['context']) > args.max_num_chars:
                        continue
                    context = context + " " + para['context']
                    if args.min_num_chars <= len(context):
                        to_article['paragraphs'].append({'context': context})
                        context = ""
                else:
                    if args.min_num_chars <= len(para['context']) < args.max_num_chars:
                        to_article['paragraphs'].append(para)
            dev_data['data'].append(to_article)

    if not os.path.exists(args.to_dir):
        os.makedirs(args.to_dir)
    for start_idx in range(0, len(dev_data['data']), args.docs_per_file):
        to_path = os.path.join(args.to_dir, str(int(start_idx / args.docs_per_file)).zfill(4))
        cur_data = {'data': dev_data['data'][start_idx:start_idx + args.docs_per_file]}
        with open(to_path, 'w') as fp:
            json.dump(cur_data, fp)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('from_dir')
    parser.add_argument('to_dir')
    parser.add_argument('--min_num_chars', default=500, type=int)
    parser.add_argument('--max_num_chars', default=2000, type=int)
    parser.add_argument('--sample_per_file', default=1, type=int)
    parser.add_argument('--docs_per_file', default=1000, type=int)
    parser.add_argument('--concat', default=False, action='store_true')

    return parser.parse_args()


def main():
    args = get_args()
    downsize_and_append(args)


if __name__ == '__main__':
    main()
