PIQA_NFS = /data_nfs/camist002/user/piqa-nfs
DUMP_DIR = $(PIQA_NFS)/index/squad/sparse
# DUMP_DIR = ~/github/hiqa-private/index/
# DUMP_DIR = $(PIQA_NFS)/index/squad/2382_m
# DUMP_DIR = $(PIQA_NFS)/index/squad/1365_m
# DUMP_DIR = $(PIQA_NFS)/index/squad/2392_m
INDEX_DIR = $(DUMP_DIR)/default_index
DATA_PATH = $(PIQA_NFS)/data/dev-v1.1.json


coarse:
	python run_index.py $(DUMP_DIR) coarse --para --num_clusters 128

fine:
	python run_index.py $(DUMP_DIR) fine --para

add:
	python run_index.py $(DUMP_DIR) add --para

pred:
	python run_eval.py $(DATA_PATH) $(DUMP_DIR) \
	--index_path default_index/index/dump.faiss \
	--idx2id_path default_index/index/dump.hdf5 \
	--num_clusters 128 \
	--para --sparse

eval:
	python ../eval/evaluate-v1.1.py $(DATA_PATH) $(DUMP_DIR)/pred_cd.json
	python ../eval/evaluate-recall.py $(DATA_PATH) $(DUMP_DIR)/pred_od.json


# nsml

nsml_coarse:
	python nsml_cluster.py


nsml_add:
	python nsml_add_to_index.py

# You need to serve with the main bert model.
demo:
	python run_demo.py
