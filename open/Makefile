PIQA_NFS = /data_nfs/camist002/user/piqa-nfs
# DUMP_DIR = $(PIQA_NFS)/index/squad/sparse
# DUMP_DIR = ~/github/despi/index/
LOCAL_DUMP_DIR = dump/76_dev-100M-cc
LOCAL_INDEX_DIR = dump/76_dev-100M-cc
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
PARA =
SPARSE = --sparse
ADD_ALL = --add_all
NUM_CLUSTERS = 4096
HNSW =
NPROBE = 64
START_TOP_K = 1000
TOP_K = 1000
STEP_SIZE = 1
SPARSE_WEIGHT = 1e-1
SPARSE_TYPE = dp
FINE_QUANT = SQ8

# local
coarse:
	python run_index.py $(DUMP_DIR) coarse $(PARA) --num_clusters $(NUM_CLUSTERS) $(HNSW)

fine:
	python run_index.py $(DUMP_DIR) fine $(PARA)

add:
	python run_index.py $(DUMP_DIR) add $(PARA) $(ADD_ALL)

pred:
	python run_pred.py $(DATA_PATH) $(DUMP_DIR) \
	--index_name 32768_flat_SQ8 \
	--nprobe $(NPROBE) \
	--top_k $(TOP_K) \
	--step_size $(STEP_SIZE) \
	$(SPARSE) \
	$(PARA) \
	--start_top_k $(START_TOP_K) \
	--sparse_weight $(SPARSE_WEIGHT) \
	--sparse_type $(SPARSE_TYPE)

eval_cd:
	python ../eval/evaluate-v1.1.py $(DATA_PATH) $(DUMP_DIR)/pred_cd.json

eval_od:
	python ../eval/evaluate-recall.py $(DATA_PATH) $(DUMP_DIR)/pred_od.json --no_f1 --scores_path ~/scores_1k.json

all_0:
	python run_index_pred_eval.py \
	$(DUMP_DIR) $(DATA_PATH) \
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
	--nprobe $(NPROBE)

all_1:
	python run_index_pred_eval.py \
	$(DUMP_DIR) $(DATA_PATH) \
	--add_all \
	--step_size $(STEP_SIZE) \
	--top_k $(TOP_K) \
	--num_clusters 128 \
	$(HNSW) \
	--fine_quant $(FINE_QUANT) \
	--start_top_k $(START_TOP_K) \
	--nprobe $(NPROBE)



# You need to serve with the main bert model.
demo:
	python run_demo.py $(DUMP_DIR) --nprobe $(NPROBE) --top_k $(TOP_K) --index_path index/dump.faiss --idx2id_path index/dump.hdf5

# nsml
nsml_add:
	nsml run -d piqa-nfs -g 0 -c 16 -e run_index.py --memory 32G --nfs-output -a " \
	$(LOCAL_DUMP_DIR) add --fs nfs $(PARA) $(ADD_ALL)"

nsml_pred:
	nsml run -d piqa-nfs -g 0 -c 4 -e run_pred.py --memory 48G --nfs-output -a " \
	--fs nfs \
	$(LOCAL_DATA_PATH) $(LOCAL_DUMP_DIR) \
	--index_name 32768_flat_SQ8 \
	--nprobe $(NPROBE) \
	--top_k $(TOP_K) \
	$(SPARSE) \
	--start_top_k $(START_TOP_K) \
	--sparse_weight $(SPARSE_WEIGHT) \
	--step_size $(STEP_SIZE) \
	$(PARA) \
	--sparse_type $(SPARSE_TYPE)"

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

nsml_all_1:
	nsml run -d piqa-nfs -g 0 -c 4 -e run_index_pred_eval.py --memory 16G --nfs-output -a " \
	$(LOCAL_DUMP_DIR) $(LOCAL_DATA_PATH) \
	--fs nfs \
	--add_all \
	--step_size $(STEP_SIZE) \
	--top_k $(TOP_K) \
	--num_clusters $(NUM_CLUSTERS) \
	--hnsw \
	--fine_quant $(FINE_QUANT) \
	--start_top_k $(START_TOP_K) \
	--nprobe $(NPROBE) \
	"

nsml_all_2:
	nsml run -d piqa-nfs -g 0 -c 4 -e run_index_pred_eval.py --memory 16G --nfs-output -a " \
	$(LOCAL_DUMP_DIR) $(LOCAL_DATA_PATH) \
	--fs nfs \
	--add_all \
	--step_size $(STEP_SIZE) \
	--top_k $(TOP_K) \
	--num_clusters $(NUM_CLUSTERS) \
	--fine_quant $(FINE_QUANT) \
	--start_top_k $(START_TOP_K) \
	--nprobe 128 \
	"

nsml_all_3:
	nsml run -d piqa-nfs -g 0 -c 4 -e run_index_pred_eval.py --memory 16G --nfs-output -a " \
	$(LOCAL_DUMP_DIR) $(LOCAL_DATA_PATH) \
	--fs nfs \
	--add_all \
	--step_size $(STEP_SIZE) \
	--top_k $(TOP_K) \
	--num_clusters $(NUM_CLUSTERS) \
	$(HNSW) \
	--fine_quant PQ121 \
	--num_dummy_zeros 3 \
	--start_top_k $(START_TOP_K) \
	--nprobe $(NPROBE) \
	"

nsml_all_4:
	nsml run -d piqa-nfs -g 1 -c 4 -e run_index_pred_eval.py --memory 16G --nfs-output -a " \
	$(LOCAL_DUMP_DIR) $(LOCAL_DATA_PATH) \
	--fs nfs \
	--add_all \
	--step_size $(STEP_SIZE) \
	--top_k $(TOP_K) \
	--num_clusters $(NUM_CLUSTERS) \
	$(HNSW) \
	--fine_quant PQ241 \
	--num_dummy_zeros 1 \
	--start_top_k $(START_TOP_K) \
	--replace \
	--cuda \
	--nprobe $(NPROBE) \
	"

nsml_all_5:
	nsml run -d piqa-nfs -g 0 -c 4 -e run_index_pred_eval.py --memory 16G --nfs-output -a " \
	$(LOCAL_DUMP_DIR) $(LOCAL_DATA_PATH) \
	--fs nfs \
	--add_all \
	--step_size $(STEP_SIZE) \
	--top_k $(TOP_K) \
	--num_clusters $(NUM_CLUSTERS) \
	$(HNSW) \
	--fine_quant PQ481 \
	--start_top_k $(START_TOP_K) \
	--nprobe $(NPROBE) \
	"

nsml_add_10M:
	python nsml_add_to_index.py dump/76_dev-10M-c --num_clusters 4096 --num_gpus 2


nsml_merge_10M:
	nsml run -d piqa-nfs -g 0 -c 4 -e run_index.py --memory 24G --nfs-output -a " \
	dump/76_dev-10M-c merge \
	--fs nfs \
	--replace \
	"

nsml_pred_10M:
	nsml run -d piqa-nfs -g 0 -c 4 -e run_pred.py --memory 24G --nfs-output -a " \
	data/dev-3.json dump/76_dev-10M-c \
	--index_name 4096_flat_SQ8 \
	--fs nfs \
	"

nsml_add_1B:
	python nsml_add_to_index.py dump/76_dev-1B-c --num_clusters 262144

nsml_add_1B_18:
	nsml run -d piqa-nfs -g 1 -c 2 -e run_index.py --memory 24G --nfs-output -a " \
	dump/76_dev-1B-c add \
	--dump_paths 2720-2800.hdf5,3974-4053.hdf5,4132-4211.hdf5,4300-4310.hdf5,2040-2045.hdf5 \
	--offset 18000000000 \
	--num_clusters 262144 \
	--fs nfs \
	--cuda \
	"

nsml_merge_1B:
	nsml run -d piqa-nfs -g 0 -c 4 -e run_index.py --memory 40G --nfs-output -a " \
	dump/76_dev-1B-c merge \
	--fs nfs \
	--num_clusters 262144 \
	"

nsml_pred_1B_10:
	nsml run -d piqa-nfs -g 0 -c 8 -e run_pred.py --memory 24G --nfs-output -a " \
	data/dev-3.json dump/76_dev-1B-c \
	--index_name 262144_flat_SQ8 \
	--fs nfs \
	--top_k 10 \
	"

nsml_pred_1B_100:
	nsml run -d piqa-nfs -g 0 -c 8 -e run_pred.py --memory 24G --nfs-output -a " \
	data/dev-3.json dump/76_dev-1B-c \
	--index_name 262144_flat_SQ8 \
	--fs nfs \
	--top_k 100 \
	"

nsml_pred_1B_1000:
	nsml run -d piqa-nfs -g 0 -c 8 -e run_pred.py --memory 24G --nfs-output -a " \
	data/dev-3.json dump/76_dev-1B-c \
	--index_name 262144_flat_SQ8 \
	--fs nfs \
	--top_k 1000 \
	"

t10_all:
	python run_index_pred_eval.py \
	$(PIQA_NFS)/dump/76_t10 $(DATA_PATH) \
	--fs local \
	--add_all \
	--step_size 10 \
	--top_k 100 \
	--num_clusters 4096 \
	--fine_quant SQ8 \
	--start_top_k 100 \
	--nprobe 64 \
	--cuda

nsml_t10_all:
	nsml run -d piqa-nfs -g 1 -c 4 -e run_index_pred_eval.py --memory 24G --nfs-output -a " \
	dump/76_t10 $(LOCAL_DATA_PATH) \
	--fs nfs \
	--add_all \
	--step_size 10 \
	--top_k 100 \
	--num_clusters 4096 \
	--fine_quant SQ8 \
	--start_top_k 100 \
	--nprobe 64 \
	--cuda \
	"

nsml_t10_add:
	python nsml_add_to_index.py dump/76_t10 --num_clusters 4096 --num_gpus 5

nsml_t10_merge:
	nsml run -d piqa-nfs -g 0 -c 4 -e run_index.py --memory 24G --nfs-output -a " \
	dump/76_t10 merge \
	--fs nfs \
	--num_clusters 4096 \
	"

nsml_t10_pred:
	nsml run -d piqa-nfs -g 0 -c 4 -e run_pred.py --memory 24G --nfs-output -a " \
	data/dev-3.json dump/76_t10 \
	--index_name 4096_flat_SQ8 \
	--fs nfs \
	--top_k 100 \
	"

t10_merge:
	python run_index.py \
	/home/minjoon/dump/76_t10 merge \
	--fs local \
	--num_clusters 4096

t10_pred:
	python run_pred.py \
	$(PIQA_NFS)/data/dev-3.json /home/minjoon/dump/76_t10 \
	--index_name 4096_flat_SQ8 \
	--fs local \
	--top_k 100

t10_demo:
	python run_demo.py /home/minjoon/dump/76_t10 --nprobe 64 --top_k 1 --index_name 4096_flat_SQ8

t100_merge:
	python run_index.py \
	$(PIQA_NFS)/dump/76_t100 merge \
	--fs local \
	--num_clusters 32768 \
	--replace

t100_pred:
	python run_pred.py \
	$(PIQA_NFS)/data/dev-3.json $(PIQA_NFS)/dump/76_t100 \
	--index_name 32768_flat_SQ8 \
	--fs local \
	--top_k 100 \
	--cuda

nsml_t100_all:
	nsml run -d piqa-nfs -g 1 -c 4 -e run_index_pred_eval.py --memory 32G --nfs-output -a " \
	dump/76_t100 $(LOCAL_DATA_PATH) \
	--fs nfs \
	--add_all \
	--step_size 10 \
	--top_k 100 \
	--num_clusters 32768 \
	--fine_quant SQ8 \
	--start_top_k 100 \
	--nprobe 64 \
	--cuda \
	"

nsml_t100_add:
	python nsml_add_to_index.py dump/76_t100 --num_clusters 32768 --num_gpus 10

nsml_t100_merge:
	nsml run -d piqa-nfs -g 0 -c 4 -e run_index.py --memory 24G --nfs-output -a " \
	dump/76_t100 merge \
	--fs nfs \
	--num_clusters 32768 \
	"

t100_demo:
	python run_demo.py $(PIQA_NFS)/dump/76_t100 --nprobe 128 --top_k 100 --index_name 32768_flat_SQ8

nsml_t100_pred:
	nsml run -d piqa-nfs -g 0 -c 4 -e run_pred.py --memory 24G --nfs-output -a " \
	data/dev-3.json dump/76_t100 \
	--index_name 32768_flat_SQ8 \
	--fs nfs \
	--top_k 100 \
	"

nsml_1B_all:
	nsml run -d piqa-nfs -g 1 -c 8 -e run_index_pred_eval.py --memory 64G --nfs-output -a " \
	dump/76_dev-1B-c $(LOCAL_DATA_PATH) \
	--fs nfs \
	--add_all \
	--step_size 10 \
	--top_k 100 \
	--hnsw \
	--num_clusters 262144 \
	--fine_quant SQ8 \
	--start_top_k 100 \
	--nprobe 64 \
	--cuda \
	--hnsw \
	"

nsml_1B_all:
	nsml run -d piqa-nfs -g 1 -c 8 -e run_index_pred_eval.py --memory 64G --nfs-output -a " \
	dump/76_dev-1B-c $(LOCAL_DATA_PATH) \
	--fs nfs \
	--add_all \
	--step_size 10 \
	--top_k 100 \
	--hnsw \
	--num_clusters 524288 \
	--fine_quant SQ8 \
	--start_top_k 100 \
	--nprobe 64 \
	--cuda \
	--vec_sample_ratio 0.1 \
	"

1B_all:
	python run_index_pred_eval.py \
	/home/minjoon/dump/76_dev-1B-c /home/data/dev-3.json \
	--fs local \
	--add_all \
	--step_size 10 \
	--top_k 100 \
	--hnsw \
	--num_clusters 524288 \
	--fine_quant SQ8 \
	--start_top_k 100 \
	--nprobe 64 \
	--cuda \
	--vec_sample_ratio 0.1

1B_demo:
	python run_demo.py $(PIQA_NFS)/dump/76_dev-1B-c --nprobe 128 --top_k 100 --index_name 262144_flat_SQ8



nsml_1B_merge:
	nsml run -d piqa-nfs -g 0 -c 4 -e run_index.py --memory 24G --nfs-output -a " \
	dump/76_dev-1B-c merge \
	--fs nfs \
	--num_clusters 262144 \
	"

nsml_1B_pred:
	nsml run -d piqa-nfs -g 0 -c 4 -e run_pred.py --memory 24G --nfs-output -a " \
	data/dev-3.json dump/76_dev-1B-c \
	--index_name 262144_flat_SQ8 \
	--fs nfs \
	--top_k 100 \
	--index_path index.faiss_ \
	"

ROOT = /home/minjoon

1B_move:
	python run_index.py \
	$(ROOT)/dump/76_dev-1B-c move \
	--fs local \
	--num_clusters 262144

1B_remove:
	python remove_doc_id.py \
	$(ROOT)/dump/76_dev-1B-c/262144_flat_SQ8/index $(ROOT)/dump/76_dev-1B-c/pred/counter.json \
	--ratio 0.003 \

1B_merge:
	python run_index.py \
	$(ROOT)/dump/76_dev-1B-c merge \
	--fs local \
	--num_clusters 262144 \
	--replace

1B_pred:
	python run_pred.py \
	$(ROOT)/data/dev-3.json $(ROOT)/dump/76_dev-1B-c \
	--index_name 262144_flat_SQ8 \
	--fs local \
	--top_k 100 \
	--nprobe 256 \
	--cuda

1B_pred_p:
	python run_pred.py \
	$(ROOT)/data/dev-3.json $(ROOT)/dump/76_dev-1B-c \
	--index_name 262144_flat_SQ8 \
	--nprobe 64 \
	--start_top_k 1000 \
	--top_k 100 \
	--step_size $(STEP_SIZE) \
	--sparse \
	--sparse_weight $(SPARSE_WEIGHT) \
	--sparse_type p \
	--cuda

1B_eval:
	python evaluate_recall.py \
	$(ROOT)/data/dev-3.json $(ROOT)/dump/76_dev-1B-c/pred/od_262144_flat_SQ8_3.0_100_100_256.json \
	--scores_dir $(ROOT)/dump/76_dev-1B-c/pred/scores

1B_pred_dp:
	python run_pred.py \
	$(ROOT)/data/dev-3.json $(ROOT)/dump/76_dev-1B-c \
	--index_name 262144_flat_SQ8 \
	--nprobe 64 \
	--start_top_k 1000 \
	--top_k 100 \
	--step_size $(STEP_SIZE) \
	--sparse \
	--sparse_weight $(SPARSE_WEIGHT) \
	--sparse_type dp \
	--cuda

1B_pred_dp_full:
	python run_pred.py \
	$(ROOT)/data/dev-v1.1.json $(ROOT)/dump/76_dev-1B-c \
	--index_name 262144_flat_SQ8 \
	--nprobe 64 \
	--start_top_k 1000 \
	--top_k 100 \
	--step_size $(STEP_SIZE) \
	--sparse \
	--sparse_weight $(SPARSE_WEIGHT) \
	--sparse_type dp \
	--cuda

1B_pred_demo:
	python run_pred.py \
	$(ROOT)/data/dev-3.json $(ROOT)/dump/76_dev-1B-c \
	--index_name 262144_flat_SQ8 \
	--nprobe 32 \
	--start_top_k 100 \
	--top_k 10 \
	--step_size $(STEP_SIZE) \
	--sparse \
	--sparse_weight $(SPARSE_WEIGHT) \
	--sparse_type p \
	--cuda

1B_demo:
	python run_demo.py $(ROOT)/dump/76_dev-1B-c --nprobe 256 --start_top_k 1000 --top_k 10 \
	--index_name 262144_flat_SQ8 \
	--cuda \
	--sparse \
	--sparse_type dp \
	--port 8100 \

1B_demo_fast:
	python run_demo.py $(ROOT)/dump/76_dev-1B-c --nprobe 32 --start_top_k 100 --top_k 10 \
	--index_name 262144_flat_SQ8 \
	--sparse \
	--cuda \
	--sparse_type p \
	--port 8101 \

1B_pred_dp_128:
	python run_pred.py \
	$(ROOT)/data/dev-3.json $(ROOT)/dump/76_dev-1B-c \
	--index_name 262144_flat_SQ8 \
	--nprobe 128 \
	--start_top_k 1000 \
	--top_k 100 \
	--step_size $(STEP_SIZE) \
	--sparse \
	--sparse_weight $(SPARSE_WEIGHT) \
	--sparse_type dp \
	--cuda

1B_pred_dp_256:
	python run_pred.py \
	$(ROOT)/data/dev-3.json $(ROOT)/dump/76_dev-1B-c \
	--index_name 262144_flat_SQ8 \
	--nprobe 256 \
	--start_top_k 1000 \
	--top_k 100 \
	--step_size $(STEP_SIZE) \
	--sparse \
	--sparse_weight $(SPARSE_WEIGHT) \
	--sparse_type dp \
	--cuda

1B_pred_dp_512:
	python run_pred.py \
	$(ROOT)/data/dev-3.json $(ROOT)/dump/76_dev-1B-c \
	--index_name 262144_flat_SQ8 \
	--nprobe 512 \
	--start_top_k 1000 \
	--top_k 100 \
	--step_size $(STEP_SIZE) \
	--sparse \
	--sparse_weight $(SPARSE_WEIGHT) \
	--sparse_type dp \
	--cuda

nsml_1B_add:
	python nsml_add_to_index.py dump/76_dev-1B-c --num_clusters 524288 --num_cpus 2 --num_gpus 20 --mem_size 30  --hnsw --cuda
