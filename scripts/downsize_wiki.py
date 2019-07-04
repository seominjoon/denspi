import json
import os
import argparse
import random

from tqdm import tqdm


def normalize(text):
    return '_'.join(text.lower().split())


def downsize_wiki(args):
    dev_data_path = os.path.join(args.data_dir, 'dev-v1.1.json')
    with open(dev_data_path, 'r') as fp:
        dev_data = json.load(fp)
    titles = [article['title'] for article in dev_data['data']]
    titles = set(map(normalize, titles))

    if not os.path.exists(args.to_dir):
        os.makedirs(args.to_dir)

    names = os.listdir(args.from_dir)
    from_paths = [os.path.join(args.from_dir, name) for name in names]
    to_paths = [os.path.join(args.to_dir, name) for name in names]

    included_titles = []

    for from_path, to_path in zip(tqdm(from_paths), to_paths):
        with open(from_path, 'r') as fp:
            from_ = json.load(fp)
        to = {'data': []}
        for article in from_['data']:
            if normalize(article['title']) not in titles and random.random() > args.sample_ratio:
                continue
            if normalize(article['title']) in titles:
                included_titles.append(article['title'])
            to_article = {'paragraphs': [], 'title': article['title']}
            for para in article['paragraphs']:
                if args.min_num_chars <= len(para['context']) < args.max_num_chars:
                    to_article['paragraphs'].append(para)
            to['data'].append(to_article)

        with open(to_path, 'w') as fp:
            json.dump(to, fp)

    print('included titles:', included_titles)
    print('%d/%d' % (len(included_titles), len(titles)))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('from_dir')
    parser.add_argument('to_dir')
    parser.add_argument('--min_num_chars', default=50, type=int)
    parser.add_argument('--max_num_chars', default=5000, type=int)
    parser.add_argument('--sample_per_file', default=1000, type=int)
    parser.add_argument('--docs_per_file', default=1000, type=int)

    return parser.parse_args()


def main():
    args = get_args()
    downsize_wiki(args)


if __name__ == '__main__':
    main()
