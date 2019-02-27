PIQA_NFS = /data_nfs/camist002/user/piqa-nfs
# DUMP_DIR = $(PIQA_NFS)/index/squad/sparse
# DUMP_DIR = ~/github/hiqa-private/index/
LOCAL_DUMP_DIR = dump/2494_dev-100M
LOCAL_INDEX_DIR = dump/76_dev-100M
# LOCAL_DUMP_DIR = dump/2494_dev-10M
DUMP_DIR = $(PIQA_NFS)/$(LOCAL_DUMP_DIR)
# DUMP_DIR = $(PIQA_NFS)/dump/76_dev-10M
# DUMP_DIR = $(PIQA_NFS)/index/squad/large-qna-1Md
# DUMP_DIR = $(PIQA_NFS)/index/squad/154_m_long
# DUMP_DIR = $(PIQA_NFS)/index/squad/2382_m
# DUMP_DIR = $(PIQA_NFS)/index/squad/1365_m
# DUMP_DIR = $(PIQA_NFS)/index/squad/2392_m
# INDEX_DIR = $(DUMP_DIR)/default_index
INDEX_DIR = $(PIQA_NFS)/$(LOCAL_INDEX_DIR)
LOCAL_DATA_PATH = data/dev-3.json
DATA_PATH = $(PIQA_NFS)/$(LOCAL_DATA_PATH)
PARA = --para
ADD_ALL = --add_all
NUM_CLUSTERS = 4096
HNSW =
NPROBE = 64
START_TOP_K = 1000
TOP_K = 1000
STEP_SIZE = 1
SPARSE_WEIGHT = 3e+0

# local
coarse:
	python run_index.py $(DUMP_DIR) coarse $(PARA) --num_clusters $(NUM_CLUSTERS) $(HNSW)

fine:
	python run_index.py $(DUMP_DIR) fine $(PARA)

add:
	python run_index.py $(DUMP_DIR) add $(PARA) $(ADD_ALL)

pred:
	python run_pred.py $(DATA_PATH) $(DUMP_DIR) \
	--index_path $(INDEX_DIR)/default_index/index/dump.faiss \
	--idx2id_path default_index/index/dump.hdf5 \
	--nprobe $(NPROBE) \
	--top_k $(TOP_K) \
	--step_size $(STEP_SIZE) \
	$(PARA) \
	--sparse \
	--start_top_k $(START_TOP_K) \
	--sparse_weight $(SPARSE_WEIGHT)

eval_cd:
	python ../eval/evaluate-v1.1.py $(DATA_PATH) $(DUMP_DIR)/pred_cd.json

eval_od:
	python ../eval/evaluate-recall.py $(DATA_PATH) $(DUMP_DIR)/pred_od.json --no_f1 --scores_path ~/scores_1k.json

# You need to serve with the main bert model.
demo:
	python run_demo.py $(DUMP_DIR) --nprobe $(NPROBE) --top_k $(TOP_K) --index_path index/dump.faiss --idx2id_path index/dump.hdf5

# nsml
nsml_add:
	nsml run -d piqa-nfs -g 0 -c 16 -e run_index.py --memory 32G --nfs-output -a " \
	$(LOCAL_DUMP_DIR) add --fs nfs $(PARA) $(ADD_ALL)"

nsml_pred:
	nsml run -d piqa-nfs -g 0 -c 4 -e run_pred.py --memory 32G --nfs-output -a " \
	--fs nfs \
	$(LOCAL_DATA_PATH) $(LOCAL_DUMP_DIR) $(LOCAL_INDEX_DIR) \
	--index_path default_index/index/dump.faiss \
	--idx2id_path default_index/index/dump.hdf5 \
	--nprobe $(NPROBE) \
	--top_k $(TOP_K) \
	--sparse \
	--start_top_k $(START_TOP_K) \
	--sparse_weight $(SPARSE_WEIGHT) \
	--step_size $(STEP_SIZE) \
	$(PARA)"

nsml_all_0:
	nsml run -d piqa-nfs -g 1 -c 16 -e run_index_pred_eval.py --memory 32G --nfs-output -a " \
	$(LOCAL_DUMP_DIR) $(LOCAL_DATA_PATH) \
	--fs nfs \
	--sparse \
	--para \
	--add_all \
	--step_size $(STEP_SIZE) \
	--top_k $(TOP_K) \
	--num_clusters $(NUM_CLUSTERS) \
	$(HNSW) \
	--fine_quant $(FINE_QUANT) \
	--sparse_weight $(SPARSE_WEIGHT) \
	--start_top_k $(START_TOP_K) \
	--nprobe $(NPROBE) \
	"
