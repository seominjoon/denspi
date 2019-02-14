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
                "--fs nfs --do_index --data_dir data/docs --predict_file %d:%d  "
                "--output_dir %s --index_file %d-%d.hdf5 --phrase_size %s "
                "--load_dir %s --iteration 1 %s" % (start_doc, end_doc,
                                                    args.output_dir, start_doc, end_doc, args.phrase_size,
                                                    args.load_dir, model_option)]

    num_docs = 5076
    num_gpus = 20
    num_docs_per_gpu = math.ceil(num_docs / num_gpus)
    start_docs = list(range(0, 5076, num_docs_per_gpu))
    end_docs = start_docs[1:] + [num_docs - 1]

    print(start_docs)
    print(end_docs)

    for start_doc, end_doc in zip(start_docs, end_docs):
        subprocess.run(get_cmd(start_doc, end_doc))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='index/wiki/large')
    parser.add_argument('--phrase_size', default=511, type=int)
    parser.add_argument('--load_dir', default='piqa-nfs/132')
    parser.add_argument('--model', default='large')
    return parser.parse_args()


def main():
    args = get_args()
    run_dump(args)


if __name__ == '__main__':
    main()
