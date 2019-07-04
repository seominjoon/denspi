import json
import os
import argparse
import random

from tqdm import tqdm


def normalize(text):
    return text.lower().replace('_', ' ')


def downsize_and_append(args):
    random.seed(args.seed)
    # TODO : split into several files
    with open(args.data_path, 'r') as fp:
        dev_data = json.load(fp)
    titles = set(normalize(article['title']) for article in dev_data['data'])

    names = os.listdir(args.from_dir)

    included_titles = set()
    data = {'data': []}
    for name in tqdm(names):
        from_path = os.path.join(args.from_dir, name)
        with open(from_path, 'r') as fp:
            from_ = json.load(fp)

        for ai, article in enumerate(from_['data']):
            article['id'] = int(name) * 1000 + ai

        articles = []
        for article in from_['data']:
            title = normalize(article['title'])
            if args.sample_ratio < 1.0:
                r = random.random()
                if title in titles:
                    articles.append(article)
                    included_titles.add(title)
                elif r <= args.sample_ratio:
                    articles.append(article)
            else:
                articles.append(article)
                if title in titles:
                    included_titles.add(title)

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
            data['data'].append(to_article)

    print('%d/%d' % (len(included_titles), len(titles)))

    # Shuffle for load balance
    random.shuffle(data['data'])

    if not os.path.exists(args.to_dir):
        os.makedirs(args.to_dir)
    for start_idx in range(0, len(data['data']), args.docs_per_file):
        to_path = os.path.join(args.to_dir, str(int(start_idx / args.docs_per_file)).zfill(4))
        cur_data = {'data': data['data'][start_idx:start_idx + args.docs_per_file]}
        with open(to_path, 'w') as fp:
            json.dump(cur_data, fp)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('from_dir')
    parser.add_argument('to_dir')
    parser.add_argument('--min_num_chars', default=500, type=int)
    parser.add_argument('--max_num_chars', default=2000, type=int)
    parser.add_argument('--sample_ratio', default=1.0, type=float)
    parser.add_argument('--docs_per_file', default=1000, type=int)
    parser.add_argument('--seed', default=29, type=int)
    parser.add_argument('--concat', default=False, action='store_true')

    return parser.parse_args()


def main():
    args = get_args()
    downsize_and_append(args)


if __name__ == '__main__':
    main()
