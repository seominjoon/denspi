import argparse
import os
from time import time

from evaluate_recall import evaluate_recall
from run_index import run_index
from run_pred import run_pred


def get_args():
    parser = argparse.ArgumentParser()

    ##### run_index.py
    parser.add_argument('dump_dir')
    # parser.add_argument('stage')

    # moved from run_pred.py
    parser.add_argument('data_path')

    parser.add_argument('--dump_paths', default=None,
                        help='Relative to `dump_dir/phrase`. '
                             'If specified, creates subindex dir and save there with same name')
    parser.add_argument('--subindex_name', default='index', help='used only if dump_path is specified.')
    parser.add_argument('--offset', default=0, type=int)

    # relative paths in dump_dir/index_name
    parser.add_argument('--quantizer_path', default='quantizer.faiss')
    parser.add_argument('--max_norm_path', default='max_norm.json')
    parser.add_argument('--trained_index_path', default='trained.faiss')
    parser.add_argument('--index_path', default='index.faiss')
    parser.add_argument('--idx2id_path', default='idx2id.hdf5')

    # Adding options
    parser.add_argument('--add_all', default=False, action='store_true')

    # coarse, fine, add
    parser.add_argument('--num_clusters', type=int, default=256)
    parser.add_argument('--hnsw', default=False, action='store_true')
    parser.add_argument('--fine_quant', default='SQ8',
                        help='SQ8|SQ4|PQ# where # is number of bytes per vector (for SQ it would be 480 Bytes)')
    # stable params
    parser.add_argument('--max_norm', default=None, type=float)
    parser.add_argument('--max_norm_cf', default=1.0, type=float)
    parser.add_argument('--norm_th', default=999, type=float)
    parser.add_argument('--para', default=False, action='store_true')
    parser.add_argument('--doc_sample_ratio', default=0.2, type=float)
    parser.add_argument('--vec_sample_ratio', default=0.2, type=float)

    parser.add_argument('--fs', default='local')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--num_dummy_zeros', default=0, type=int)
    parser.add_argument('--replace', default=False, action='store_true')
    parser.add_argument('--num_docs_per_add', default=1000, type=int)

    ##### run_pred.py
    # parser.add_argument('data_path')
    parser.add_argument('--question_dump_path', default='question.hdf5')

    # parser.add_argument('--index_name', default='default_index')
    # parser.add_argument('--index_path', default='index.hdf5')
    # parser.add_argument('--idx2id_path', default='idx2id.hdf5')

    parser.add_argument('--pred_dir', default='pred')

    # MIPS params
    parser.add_argument('--sparse_weight', default=3, type=float)
    parser.add_argument('--start_top_k', default=100, type=int)
    parser.add_argument('--nprobe', default=16, type=int)

    # stable MIPS params
    parser.add_argument('--max_answer_length', default=30, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    # parser.add_argument('--para', default=False, action='store_true')
    parser.add_argument('--sparse', default=False, action='store_true')

    parser.add_argument('--no_od', default=False, action='store_true')
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--step_size', default=10, type=int)
    # parser.add_argument('--fs', default='local')

    #### evaluate_recall.py
    # parser.add_argument('data_path', help='Dataset file')
    # parser.add_argument('od_out_path', help='Prediction File')
    parser.add_argument('--do_f1', default=False, action='store_true')
    parser.add_argument('--k_start', default=1, type=int)
    parser.add_argument('--scores_dir', default='scores')

    args = parser.parse_args()

    args.stage = 'all'

    #### run_index.py
    coarse = 'hnsw' if args.hnsw else 'flat'
    args.index_name = '%d_%s_%s' % (args.num_clusters, coarse, args.fine_quant)

    if args.fs == 'nfs':
        from nsml import NSML_NFS_OUTPUT
        args.dump_dir = os.path.join(NSML_NFS_OUTPUT, args.dump_dir)

    args.index_dir = os.path.join(args.dump_dir, args.index_name)

    args.quantizer_path = os.path.join(args.index_dir, args.quantizer_path)
    args.max_norm_path = os.path.join(args.index_dir, args.max_norm_path)
    args.trained_index_path = os.path.join(args.index_dir, args.trained_index_path)
    args.index_path = os.path.join(args.index_dir, args.index_path)
    args.idx2id_path = os.path.join(args.index_dir, args.idx2id_path)

    #### run_pred.py
    if args.fs == 'nfs':
        from nsml import NSML_NFS_OUTPUT
        args.data_path = os.path.join(NSML_NFS_OUTPUT, args.data_path)
        # args.dump_dir = os.path.join(NSML_NFS_OUTPUT, args.dump_dir)
    phrase_dump_path = os.path.join(args.dump_dir, 'phrase.hdf5')
    args.phrase_dump_dir = phrase_dump_path if os.path.exists(phrase_dump_path) else os.path.join(args.dump_dir,
                                                                                                  'phrase')

    args.question_dump_path = os.path.join(args.dump_dir, args.question_dump_path)
    # args.index_path = os.path.join(args.index_dir, args.index_path)
    # args.idx2id_path = os.path.join(args.index_dir, args.idx2id_path)

    args.pred_dir = os.path.join(args.dump_dir, args.pred_dir)
    out_name = '%s_%d_%.1f_%d_%d' % (args.index_name, args.max_answer_length, args.sparse_weight, args.start_top_k,
                                     args.nprobe)
    args.od_out_path = os.path.join(args.pred_dir, 'od_%s.json' % out_name)
    args.cd_out_path = os.path.join(args.pred_dir, 'cd_%s.json' % out_name)

    #### evaluate_recall.py
    args.scores_path = os.path.join(args.scores_dir, 'scores_%s' % os.path.basename(args.od_out_path))

    args.counter_path = os.path.join(args.pred_dir, 'counter.json')

    return args


def run_index_pred_eval(args):
    t0 = time()
    run_index(args)
    t1 = time()
    run_pred(args)
    t2 = time()
    evaluate_recall(args)

    print('run_index: %.1f mins, run_pred: %.1f mins' % ((t1 - t0) / 60, (t2 - t1) / 60))


def main():
    args = get_args()
    run_index_pred_eval(args)


if __name__ == '__main__':
    main()
