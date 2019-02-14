dl:
	python run_piqa.py \
	--bert_model_option 'base_uncased' \
	--do_train \
	--do_train_filter \
	--do_predict \
	--do_eval \
	--do_embed_question \
	--do_index \
	--do_serve \
	--num_train_epochs 1 \
	--draft_num_examples 1 \
	--train_batch_size 1 \
	--predict_batch_size 1 \
	--draft \
	--compression_offset 0.0 \
	--compression_scale 20.0 \
	--split_by_para \
	--no_cuda

train_base:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--train_file train-v1.1-qna-1-1.json \
	--train_batch_size 18 \
	--phrase_size 127 \
	--do_train \
	--do_predict \
	--do_eval"

eval_base:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--load_dir KR18816/piqa-nfs/249 \
	--iteration 2 \
	--phrase_size 127 \
	--do_predict \
	--do_eval"

train_filter_base:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--train_file train-v1.1-qna-1-1.json \
	--train_batch_size 18 \
	--phrase_size 127 \
	--do_train_filter \
	--do_predict \
	--do_eval \
	--num_train_epochs 1 \
	--load_dir KR18816/piqa-nfs/56 \
	--iteration 3"

train:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 24G --nfs-output -a " \
	--fs nfs \
	--train_file train-v1.1-qna-1-1.json \
	--train_batch_size 18 \
	--phrase_size 511 \
	--do_train \
	--do_predict \
	--do_eval"

train_filter:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--train_file train-v1.1-qna-1-1.json \
	--train_batch_size 18 \
	--phrase_size 511 \
	--do_train_filter \
	--do_predict \
	--do_eval \
	--num_train_epochs 1 \
	--load_dir KR18816/squad_bert_2/56 \
	--iteration 3"

dump_phrases:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--output_dir index/squad/large \
	--index_file phrase.hdf5 \
	--load_dir KR18816/piqa-nfs/132 \
	--iteration 1 \
	--parallel"

dump_questions:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_embed_question \
	--output_dir index/squad/large \
	--question_emb_file question.hdf5 \
	--load_dir KR18816/piqa-nfs/132 \
	--iteration 1 \
	--parallel"
