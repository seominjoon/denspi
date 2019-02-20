import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('from_path')
    parser.add_argument('to_path')
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.from_path, 'r') as fp:
        in_ = json.load(fp)

    text = "This is just a short sentence for test."
    paragraph = {'context': text, 'qas': []}
    article = {'paragraphs': [paragraph], 'title': 'dummy'}
    in_['data'].append(article)

    with open(args.to_path, 'w') as fp:
        json.dump(in_, fp)


if __name__ == "__main__":
    main()
