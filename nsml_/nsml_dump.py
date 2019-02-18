import argparse
import math
import subprocess


def run_dump(args):
    if args.model == 'base':
        model_option = "--bert_model_option base_uncased"
    elif args.model == 'large':
        model_option = "--parallel"
    else:
        raise ValueError(args.model)

    def get_cmd(start_doc, end_doc):
        return ["nsml",
                "run",
                "-d",
                "piqa-nfs",
                "-g",
                "1",
                "-e",
                "run_piqa.py",
                "--memory",
                "32G",
                "--nfs-output",
                "-a",
                "--fs nfs --do_index --data_dir %s --predict_file %d:%d  --filter_threshold %.2f "
                "--output_dir %s --index_file %d-%d.hdf5 --phrase_size %s "
                "--load_dir %s --iteration 1 %s" % (args.data_dir, start_doc, end_doc, args.filter_threshold,
                                                    args.output_dir, start_doc, end_doc, args.phrase_size,
                                                    args.load_dir, model_option)]

    num_docs = 5076
    num_gpus = args.num_gpus
    num_docs_per_gpu = math.ceil(num_docs / num_gpus)
    start_docs = list(range(0, 5076, num_docs_per_gpu))
    end_docs = start_docs[1:] + [num_docs - 1]

    print(start_docs)
    print(end_docs)

    for start_doc, end_doc in zip(start_docs, end_docs):
        subprocess.run(get_cmd(start_doc, end_doc))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='index/wiki/large-qna')
    parser.add_argument('--phrase_size', default=961, type=int)
    parser.add_argument('--load_dir', default='piqateam/piqa-nfs/76')
    parser.add_argument('--data_dir', default='data/docs_100_5000')
    parser.add_argument('--model', default='large')
    parser.add_argument('--filter_threshold', default=-2, type=float)
    parser.add_argument('--num_gpus', default=30, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    run_dump(args)


if __name__ == '__main__':
    main()
