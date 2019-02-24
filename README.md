# hiqa
Question Answering through Hierarchical Document-Phrase Index

## 0. Prerequisites
See `requirements.txt`.

## 1. Training & "Unofficial" Evaluation

For large model (you need four P40s), see `train_and_eval` in `Makefile` for running on NSML.

## 2. Train Filters
We are training filter independently of the main training phase.
We load the model from #1. 
See `train_filter` in `Makefile` for running on NSML.

## 3. Dump Phrase Vectors
See `dump_phrases` in `Makefile`

## 4. Dump Question Vectors
See `dump_questions` in `Makefile`

## 5. "Official" Evaluation
Get help from `run_eval.py -h`

## 6. Run Demo
Get help from `run_demo.py -h`


## Dump Directory
```
model-num_data-name
|-- phrase
    |-- 0-500.hdf5
    ...
    `-- 5000-5049.hdf5
|-- question.hdf5
|-- index_0
    |-- quantizer.faiss
    |-- index
        |-- 0-500.faiss
        ...
        `-- 5000-5049.faiss
    |-- index.faiss
    `-- misc_file
|-- index_1
    |-- quantizer.faiss
    |-- index
        |-- 0-500.faiss
        ...
        `-- 5000-5049.faiss
    |-- index.faiss
    `-- misc_file
...
```