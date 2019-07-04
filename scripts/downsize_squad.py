import json
import argparse
import random


def downsize_squad(args):
    random.seed(args.seed)

    with open(args.from_path, 'r') as fp:
        dev_data = json.load(fp)

        for article in dev_data['data']:
            for para in article['paragraphs']:
                new_qas = []
                for qa in para['qas']:
                    if random.random() < args.ratio:
                        new_qas.append(qa)
                para['qas'] = new_qas

        with open(args.to_path, 'w') as fp:
            json.dump(dev_data, fp)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('from_path')
    parser.add_argument('to_path')
    parser.add_argument('--ratio', default=0.1, type=float)
    parser.add_argument('--seed', default=29, type=int)

    return parser.parse_args()


def main():
    args = get_args()
    downsize_squad(args)


if __name__ == '__main__':
    main()
