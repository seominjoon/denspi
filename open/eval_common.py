import argparse
import json

import spacy
from tqdm import tqdm

nlp = spacy.load('en_core_web_lg')  # make sure to use larger model!


def get_sim(sent1, sent2):
    t1 = nlp(sent1)
    t2 = nlp(sent2)
    return t1.similarity(t2)


def eval_common(common_path, crawl_path, draft=False):
    common = []
    with open(common_path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        common.append(json.loads(line.strip()))

    with open(crawl_path, 'r') as fp:
        crawl = json.load(fp)

    if draft:
        common = common[:10]

    corrects = []
    for each_common in tqdm(common):
        id_ = each_common['id']
        each_crawl = crawl[id_]
        answers = [e['answer'] for e in each_crawl['ret']]
        question = each_common['question']
        max_score = -9999
        max_label = None
        for choice in question['choices']:
            choice_label = choice['label']
            choice_text = choice['text']
            score = max(get_sim(choice_text, answer) for answer in answers)
            if score > max_score:
                max_score = score
                max_label = choice_label
        correct = max_label == each_common['answerKey']
        corrects.append(correct)

    print("%d/%d = %.2f" % (sum(corrects), len(corrects), sum(corrects)/len(corrects)))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('common_path')
    parser.add_argument('crawl_path')
    parser.add_argument('--draft', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    eval_common(args.common_path, args.crawl_path, draft=args.draft)


if __name__ == "__main__":
    main()
