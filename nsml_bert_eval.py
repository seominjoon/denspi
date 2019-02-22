import subprocess


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
