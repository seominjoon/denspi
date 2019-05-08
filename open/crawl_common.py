import argparse
import json

from tqdm import tqdm

import requests


def ask(query, url='http://despi.ngrok.io/api'):
    res = requests.get(url, {'query': query})
    out = res.json()
    return out


def get_questions(path):
    questions = []
    with open(path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            each = json.loads(line.strip())
            questions.append(each)
    return questions


def crawl_common(common_path, out_path, draft=False):
    questions = get_questions(common_path)
    answers = {}
    if draft:
        questions = questions[:2]
    for question in tqdm(questions):
        answer = ask(question['question']["stem"])
        answers[question['id']] = answer
    with open(out_path, 'w') as fp:
        json.dump(answers, fp)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('common_path')
    parser.add_argument('out_path')
    parser.add_argument('--draft', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = get_args()
    crawl_common(args.common_path, args.out_path, draft=args.draft)


if __name__ == "__main__":
    main()
