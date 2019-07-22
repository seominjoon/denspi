import argparse
import math
import os
import subprocess


def run_dump_question(args):
    if args.model == 'base':
        model_option = "--bert_model_option base_uncased"
    elif args.model == 'large':
        model_option = "--parallel"
    else:
        raise ValueError(args.model)

    para = '--para' if args.para else ''

    def get_cmd():
        return ["nsml",
                "run",
                "-d",
                "piqa-nfs",
                "-g",
                "1",
                "-e",
                "run_piqa.py",
                "--memory",
                "%dG" % args.mem_size,
                "--nfs-output",
                "-a",
                "--fs nsml_nfs --do_embed_question --data_dir %s %s"
                "--output_dir %s --phrase_size %s "
                "--load_dir %s --iteration 1 %s" % (args.data_dir, para,
                                                    args.dump_dir, args.phrase_size,
                                                    args.load_dir, model_option)]

    subprocess.run(get_cmd())


def run_dump_phrase(args):
    if args.model == 'base':
        model_option = "--bert_model_option base_uncased"
    elif args.model == 'large':
        model_option = "--parallel"
    else:
        raise ValueError(args.model)

    para = '--para' if args.para else ''

    def get_cmd(start_doc, end_doc):
        return ["nsml",
                "run",
                "-d",
                "piqa-nfs",
                "-g",
                "1",
                "-c",
                str(args.num_cpus),
                "-e",
                "run_piqa.py",
                "--memory",
                f"{args.mem_size}G",
                "--nfs-output",
                "-a",
                f"--fs nsml_nfs --do_index --data_dir {args.phrase_data_dir} "
                f"--predict_file {start_doc}:{end_doc}  --filter_threshold {args.filter_threshold} "
                f"--output_dir {args.phrase_dump_dir} --index_file {start_doc}-{end_doc}.hdf5 "
                f"--phrase_size {args.phrase_size} --load_dir {args.load_dir} --iteration 1 {model_option} {para}"]

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
        if args.no_block:
            subprocess.Popen(get_cmd(start_doc, end_doc))
        else:
            subprocess.run(get_cmd(start_doc, end_doc))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_dir', default=None)
    parser.add_argument('--phrase_size', default=961, type=int)
    parser.add_argument('--load_dir', default='piqateam/piqa-nfs/3021')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--data_name', default='wikipedia_filtered')
    parser.add_argument('--model', default='large')
    parser.add_argument('--filter_threshold', default=-2, type=float)
    parser.add_argument('--num_gpus', default=20, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=5076, type=int)
    parser.add_argument('--mem_size', default=32, type=int, help='mem size in GB')
    parser.add_argument('--num_cpus', default=4, type=int, help='num cpus per gpu')
    parser.add_argument('--no_block', default=False, action='store_true')
    parser.add_argument('--para', default=False, action='store_true')
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
    # run_dump_question(args)
    run_dump_phrase(args)


if __name__ == '__main__':
    main()
