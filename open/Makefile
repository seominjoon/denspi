PIQA_NFS = /data_nfs/camist002/user/piqa-nfs
LOCAL_DUMP_DIR = dump/76_dev-100M
DUMP_DIR = $(PIQA_NFS)/$(LOCAL_DUMP_DIR)
# DUMP_DIR = $(PIQA_NFS)/dump/76_dev-10M
# DUMP_DIR = $(PIQA_NFS)/index/squad/large-qna-1Md
# DUMP_DIR = $(PIQA_NFS)/index/squad/154_m_long
# DUMP_DIR = $(PIQA_NFS)/index/squad/2382_m
# DUMP_DIR = $(PIQA_NFS)/index/squad/1365_m
# DUMP_DIR = $(PIQA_NFS)/index/squad/2392_m
INDEX_DIR = $(DUMP_DIR)/default_index
LOCAL_DATA_PATH = data/dev-v1.1.json
DATA_PATH = $(PIQA_NFS)/$(LOCAL_DATA_PATH)
PARA = --para
PARA =
ADD_ALL = --add_all
NUM_CLUSTERS = 32786
HNSW = --hnsw
NPROBE = 64

# local
coarse:
	python run_index.py $(DUMP_DIR) coarse $(PARA) --num_clusters $(NUM_CLUSTERS) $(HNSW)

fine:
	python run_index.py $(DUMP_DIR) fine $(PARA)

add:
	python run_index.py $(DUMP_DIR) add $(PARA) $(ADD_ALL)

pred:
	python run_eval.py $(DATA_PATH) $(DUMP_DIR) \
	--index_path default_index/index/dump.faiss \
	--idx2id_path default_index/index/dump.hdf5 \
	--nprobe $(NPROBE) \
	--top_k 10 \
	$(PARA)

eval_cd:
	python ../eval/evaluate-v1.1.py $(DATA_PATH) $(DUMP_DIR)/pred_cd.json

eval_od:
	python ../eval/evaluate-recall.py $(DATA_PATH) $(DUMP_DIR)/pred_od.json

# You need to serve with the main bert model.
demo:
	python run_demo.py $(DUMP_DIR) --nprobe 64 --top_k 10 --index_path index/dump.faiss --idx2id_path index/dump.hdf5

# nsml
nsml_add:
	nsml run -d piqa-nfs -g 0 -c 16 -e run_index.py --memory 32G --nfs-output -a " \
	$(LOCAL_DUMP_DIR) add --fs nfs $(PARA) $(ADD_ALL)"

nsml_pred:
	nsml run -d piqa-nfs -g 0 -c 16 -e run_eval.py --memory 32G --nfs-output -a " \
	--fs nfs \
	$(LOCAL_DATA_PATH) $(LOCAL_DUMP_DIR) \
	--index_path default_index/index/dump.faiss \
	--idx2id_path default_index/index/dump.hdf5 \
	--nprobe $(NPROBE) \
	--top_k 10 \
	$(PARA)"
