# Real-Time Open-Domain QA with Dense-Sparse Phrase Index

![teaser](figs/teaser.png)

- [Paper](https://arxiv.org/abs/1906.05807), to appear at [ACL 2019](http://www.acl2019.org)
- [Live Demo][demo]
- BibTeX:
```
@inproceedings{denspi,
  title={Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index},
  author={Seo, Minjoon and Lee, Jinhyuk and Kwiatkowski, Tom and Parikh, Ankur P and Farhadi, Ali and Hajishirzi, Hannaneh},
  booktitle={ACL},
  year={2019}
}
```

We enumerate, embed and index every phrase in Wikipedia (60 Billion) so that open-domain QA can be formulated as a pure phrase retrieval problem. Our model is able to read the entire Wikpedia in 0.5s with CPUs, allowing it to reach long-tail answers with much faster inference speed than "retrieve & read" models (at least 58x). Feel free to check it out in our [demo][demo].


## Demo
This section will let you host the demo that looks like

![demo](figs/demo.png)
 
on your machine.
You can also try it out [here][demo].
You will need to download ~1.5 TB of files, but once you have them, it will take less than a minute to start serving.

### Prerequisites

#### A. Hardware
- CPUs: at least 4 cores recommended.
- RAM: at least 30GB needed.
- Storage: at least 1.5TB of SSD needed.
- GPUs: not needed.

If you are using Google Cloud 
(our demo is also being hosted on Google Cloud, with 8 vCPUs, 30 GB RAM, and 6 local SSDs),
we highly recommend using [local SSD](https://cloud.google.com/compute/docs/disks/local-ssd), 
which is not only cheaper but also offers lower disk access latency (at the cost of persistency).



#### B. Environment
We highly recommend Conda environment, since `faiss` cannot be installed with pip.
Note that we have two `requirements.txt` files: one in this directory, and one in `open` subfolder.
This directory's file is for hosting a (PyTorch-based) server that maps the input question to a vector.
`open`'s file is for hosting the search server and the demo itself.
In this tutorial, we will simply install both in the same environment.

1. Make sure you are using `python=3.6` through Conda. For instance, once you have Conda, create the environment via
```
conda create -n denspi python=3.6
```
then activate:
```
conda activate denspi
```

2. First, manually install `faiss` with `conda`:
```
conda install faiss-cpu=1.5.2 -c pytorch
```
3. Before installing with pip, make sure that you have installed `DrQA`. 
Visit [here](https://github.com/facebookresearch/DrQA) for instructions.
If you are using conda, you will probably need to install java-jdk in order to install DrQA:
```
conda install -c cyclus java-jdk
```
4. Then install both requirement files:
```
pip install -r requirements.txt
pip install -r open/requirements.txt
```
Note that this will give you an error if you don't have `faiss` and `DrQA` already installed.


#### C. Download
Model and dump files are currently provided through Google Cloud Storage under bucket `denspi`,
 so first make sure that you have installed `gsutil` ([link](https://cloud.google.com/storage/docs/gsutil_install)).
You will then need to download four directories.
0. Create `$ROOT_DIR` and cd to it:
```
mkdir $ROOT_DIR; cd $ROOT_DIR
```
1. You will need BERT-related files. 
```
gsutil cp -r gs://denspi/v1-0/bert .
```
2. You will need tfidf-related information from DrQA. 
```
gsutil cp -r gs://denspi/v1-0/wikipedia .
```
3. You will need training and eval data. Skip this if you will not be training the model yourself.
```
gsutil cp -r gs://denspi/v1-0/data .
```
4. You will need the model files. Skip this if you will train yourself (see "Train" below).
```
gsutil cp -r gs://denspi/v1-0/model .
``` 
5. You will need to download the entire phrase index dump. **Warning**: this will take up 1.5 TB!
```
gsutil cp -r gs://denspi/v1-0/dump .
```

You can also choose to download all at once via
```
gsutil cp -r gs://denspi/v1-0 $ROOT_DIR
```


### Run Demo

Serve API on port `$API_PORT`:
```
python run_piqa.py --do_serve --load_dir $ROOT_DIR/model --metadata_dir $ROOT_DIR/bert --do_load --parallel --port $API_PORT
```
This lets you to perform GET request on `$API_PORT` to obtain the embedding of the question in json (list) format.


Serve the demo on `$DEMO_PORT`:
```
cd open/
python run_demo.py $ROOT_DIR/dump $ROOT_DIR/wikipedia --api_port $API_PORT --port $DEMO_PORT
```

Demo will be served in ~1 minute.


## Train
Note that we provide pretrained model at Google Coud Storage `gs://denspi/v1-0/model` and you can simply download it.
This section will be only applicable if you want to train on your own.

Note that you will need 4 x P40 GPUs (24 GB) for the specified batch size.

1. Train on SQuAD v1.1: this will train the model on vanilla `train-v1.1.json` without negative examples
at `$SAVE1_DIR` of your choice. This will take approximately 16 hours (~5 hours per epoch, 3 epochs).
```
python run_piqa.py --train_batch_size 12 --do_train --freeze_word_emb --save_dir $SAVE1_DIR
```

2. Train with negative samples: this will finetune the model once more with negative examples sampled 
from other documents so that the model can better behave in open-domain environment.
This also takes approximately 16 hours with 3 epochs.
```
python run_piqa.py --train_batch_size  9 --do_embed_question --do_train_neg --freeze_word_emb --load_dir $SAVE1_DIR --iteration 3 --save_dir $SAVE2_DIR

``` 

3. Finally train a phrase classifier, 
which freezes every parameter except the linear layer at the end for classification.
This will be much faster, less than 4 hours.
```
python run_piqa.py --train_batch_size 12 --do_train_filter --num_train_epochs 1 --load_dir $SAVE2_DIR --iteration 3 --save_dir $SAVE3_DIR
```


## Create a Custom Phrase Index
For sanity check, we will first test with a small phrase index that corresponds to the entire dev data of SQuAD,
which contains approximately 300k words. Note that Wikipedia has 3 Billion words,
so this will be approximately 1/10k-th size experiment.

For a fast sanity check, we provide `gs://denspi/v1-0/data/dev-v1.1/` that contains two small json files, `0000` and `0001`, each of which corresponds to approximately half of 
`dev-v1.1.json` (i.e. identical when concatenated).
Of course, as long as you follow the same data format, you can use your own documents.
Note that you only need to provide `title` field of each article and `context` of the paragraphs in the article.
Other fields (such as `qas` of each paragraph) are all ignored.

Given the trained model from the above, we first dump the dense phrase vectors:
```
python run_piqa.py --do_dump --filter_threshold -2 --save_dir $SAVE3_DIR --metadata_dir $ROOT_DIR/bert --data_dir $ROOT_DIR/data/dev-v1.1 --predict_file 0:1 --output_dir $ROOT_DIR/your_dump/phrase --dump_file 0-1.hdf5
```
Note that this only dumps the first file, `0000`. You can basically dump in a distributed way by controlling `--predict_file` range.
Of course, you can simply give `predict_file 0:2` to dump everything into a single hdf5.
But when the number of documents get big, you will need to make this distributed.
`--dump_file` must exactly correspond to the range of input file names that you use.

Now move to the `./open` folder (`cd open/`).
Then create a faiss index:
```
python run_index.py $ROOT_DIR/your_dump all
```

You also need to create (paragraph-level) tf-idf vectors corresponding to your custom documents:
```
python dump_tfidf.py $ROOT_DIR/your_dump/phrase/ $ROOT_DIR/your_dump/tfidf $ROOT_DIR/wikipedia --start 0 --end 2 
```
Here `--end` indicates the range of your original input files.


Now you are ready to run your demo on your own dump! Just change the dump directory from the old `$ROOT_DIR/dump`
to your new `$ROOT_DIR/your_dump` with two little changes:

```
python run_demo.py $ROOT_DIR/dump $ROOT_DIR/wikipedia --api_port $API_PORT --port $DEMO_PORT --index_name 64_hnsw_SQ8 --sparse_type p
```

`--index_name` is different because we use a different parameters (no hnsw, 64 clusters) in `python run_index.py` (since we have a small corpus).
We will explain what it means in the below section.
`--sparse_type` can be either `p` or `dp`, meaning whether we use just paragraph tf-idf vectors or also use document-level tf-idf vectors.
Current version only supports `p`. Using the document vectors as well is trivially easy and will be updated soon.


## Create a Large Phrase Index
You can follow a similar procedure from the above to extend this small phrase index to a large one, as large as Wikipedia.
We just outline a few things that you should be aware of when the scale grows:

1. As mentioned above, you might want to distribute dumping because you will need 500 GPU hours for 3 Billion words to be processed.
2. You will need a large `--num_clusters` during `run_index.py`, e.g. 1M for Wikipedia. You also want to use `--hnsw` flag for faster inference (not necessary though if speed is not your concern).
3. Running `run_index.py` will consume a lot of CPU RAM memory. Basically, your RAM needs to be able to store the entire faiss index if not built in a distributed fashion (on-disk distribution).
It is possible to distribute it so that we can do it faster and also consume less memory in each machine. We will add procedures for this soon.



## Questions?
Please use Github Issues.

## Acknowledgment
Our code makes a heavy use of [faiss](https://github.com/facebookresearch/faiss), 
[DrQA](https://github.com/facebookresearch/DrQA) and [BERT](https://github.com/google-research/bert), in particular,
Huggingface's [PyTorch implementation](https://github.com/huggingface/pytorch-pretrained-BERT).
We thank them for open-sourcing these projects!

[demo]: http://allgood.cs.washington.edu:15001/
