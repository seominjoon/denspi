import math
import subprocess


def run_piqa_dump():
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
                "--output_dir index/wiki/large --index_file demo_%d-%d.hdf5 "
                "--load_dir KR18816/piqa-nfs/132 --iteration 1 --parallel" % (start_doc, end_doc, start_doc, end_doc)]

    num_docs = 5076
    num_gpus = 20
    num_docs_per_gpu = math.ceil(num_docs / num_gpus)
    start_docs = list(range(0, 5076, num_docs_per_gpu))
    end_docs = start_docs[1:] + [num_docs - 1]

    start_docs = [2079, 4385]
    end_docs = [2286, 4572]

    print(start_docs)
    print(end_docs)

    for start_doc, end_doc in zip(start_docs, end_docs):
        subprocess.run(get_cmd(start_doc, end_doc))


def run_bert_eval():
    def get_cmd(file):
        return ["nsml",
                "run",
                "-d",
                "piqa-nfs",
                "-g",
                "1",
                "-e",
                "run_squad.py",
                "--memory",
                "32G",
                "--nfs-output",
                "-a",
                "--fs nfs --data_dir data/docs_eval --do_predict --predict_file %s  "
                "--output_dir eval/bert --load_path pytorch_model_squad_finetuned.bin "
                "--parallel" % file]

    num_docs = 21146
    num_per_doc = 1000
    start_docs = list(range(0, num_docs, num_per_doc))
    end_docs = [min(start + num_per_doc, num_docs) for start in start_docs]
    files = ['%d-%d.json' % (start, end) for start, end in zip(start_docs, end_docs)]

    files = files[1:]

    print(files)

    for file in files:
        subprocess.run(get_cmd(file))


if __name__ == '__main__':
    run_bert_eval()
