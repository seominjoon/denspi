API_PORT = 8104
DEMO_PORT = 8105

serve:
	CUDA_VISIBLE_DEVICES= python ../run_piqa.py \
	    --do_serve \
	    --load_dir ../data/model \
	    --metadata_dir ../data/bert \
	    --do_load \
	    --parallel \
	    --port $(API_PORT)

demo:
	python run_demo.py \
	    /home/minjoon/dump/76_dev-1B-c \
	    ../data/wikipedia \
	    --api_port $(API_PORT) \
	    --port $(DEMO_PORT)

pred:
	python run_pred.py \
	    /home/minjoon/dump/76_dev-1B-c \
	    ../data/wikipedia \
	    --data_path /home/jinhyuk/data/dev-v1.1.json \
	    --search_strategy hybrid

eval:
	python evaluate_recall.py \
	    ~/data/dev-v1.1.json \
	    /home/minjoon/dump/76_dev-1B-c/predictions/od_1048576_hnsw_SQ8_dp_0.05_1000_10_64.json # Hybrid
	    
# /home/minjoon/dump/76_dev-1B-c/predictions/od_1048576_hnsw_SQ8_dp_0.05_1000_10_64.json (DSF)
