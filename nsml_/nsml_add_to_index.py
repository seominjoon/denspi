import argparse
import math
import os
import subprocess


def run_add_to_index(args):
    def get_cmd(start_doc, end_doc):
        return ["nsml",
                "run",
                "-d",
                "piqa-nfs",
                "-g",
                "0",
                "-c",
                "4",
                "-e",
                "mips.py",
                "--memory",
                "%dG" % args.mem_size,
                "--nfs-output",
                "-a",
                "%s add --index_name %s --dump_path %d-%d.hdf5" % (args.dump_dir, args.index_name, start_doc, end_doc)]

    num_docs = args.end - args.start
    num_gpus = args.num_gpus
    if num_gpus > num_docs:
        print('You are requesting more GPUs than the number of docs; adjusting num gpus to %d' % num_docs)
        num_gpus = num_docs
    num_docs_per_gpu = int(math.ceil(num_docs / num_gpus))
    start_docs = list(range(args.start, args.end, num_docs_per_gpu))
    end_docs = start_docs[1:] + [args.end]

    print(start_docs)
    print(end_docs)

    for start_doc, end_doc in zip(start_docs, end_docs):
        subprocess.run(get_cmd(start_doc, end_doc))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_dir', default=None)
    parser.add_argument('--phrase_size', default=961, type=int)
    parser.add_argument('--load_dir', default='piqateam/piqa-nfs/76')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--data_name', default='500-2000')
    parser.add_argument('--model', default='large')
    parser.add_argument('--filter_threshold', default=-2, type=float)
    parser.add_argument('--num_gpus', default=10, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=5076, type=int)
    parser.add_argument('--mem_size', default=32, type=int, help='mem size in GB')

    parser.add_argument('--index_name', default='default_index')
    args = parser.parse_args()

    if args.dump_dir is None:
        args.dump_dir = os.path.join('dump/%s_%s' % (os.path.basename(args.load_dir),
                                                     os.path.basename(args.data_name)))
    if not os.path.exists(args.dump_dir):
        os.makedirs(args.dump_dir)

    args.phrase_data_dir = os.path.join(args.data_dir, args.data_name)
    args.phrase_dump_dir = os.path.join(args.dump_dir, 'phrase')
    if not os.path.exists(args.phrase_dump_dir):
        os.makedirs(args.phrase_dump_dir)

    return args


def main():
    args = get_args()
    run_add_to_index(args)


if __name__ == '__main__':
    main()
