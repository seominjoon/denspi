dl:
	python run_piqa.py \
	--bert_model_option 'base_uncased' \
	--train_file train-v1.1-qnad.json \
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
	--phrase_size 127 \
	--split_by_para

train_base_511:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--train_file train-v1.1.json \
	--train_batch_size 18 \
	--phrase_size 511 \
	--do_train \
	--do_predict \
	--do_eval"


train_base_na_127:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--train_file train-v1.1-na-1-1.json \
	--train_batch_size 18 \
	--phrase_size 127 \
	--do_train \
	--do_predict \
	--do_eval"

train_base_na_511:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--train_file train-v1.1-na-1-1.json \
	--train_batch_size 18 \
	--phrase_size 511 \
	--do_train \
	--do_predict \
	--do_eval"

train_base_qna_127:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--train_file train-v1.1-qna-1-1.json \
	--train_batch_size 18 \
	--phrase_size 127 \
	--do_train \
	--do_predict \
	--do_eval"

train_base_qna_511:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--train_file train-v1.1-qna-1-1.json \
	--train_batch_size 18 \
	--phrase_size 511 \
	--do_train \
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
	--num_train_epochs 1 \
	--load_dir piqateam/piqa-nfs/27 \
	--iteration 3 \
	--filter_threshold -2 \
	--do_eval"

eval_base:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--load_dir piqateam/piqa-nfs/27 \
	--iteration 3 \
	--phrase_size 127 \
	--do_predict \
	--do_eval"

dump_base:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--do_embed_question \
	--do_index \
	--output_dir index/squad/base \
	--index_file phrase.hdf5 \
	--phrase_size 511 \
	--question_emb_file question.hdf5 \
	--load_dir piqateam/piqa-nfs/60 \
	--filter_threshold -999999 \
	--split_by_para \
	--iteration 1"

dump_base_qna:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--do_embed_question \
	--do_index \
	--output_dir index/squad/base-qna \
	--index_file phrase.hdf5 \
	--phrase_size 127 \
	--question_emb_file question.hdf5 \
	--load_dir piqateam/piqa-nfs/27 \
	--filter_threshold -999999 \
	--split_by_para \
	--iteration 3"

dump_base_na:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--do_embed_question \
	--do_index \
	--output_dir index/squad/base-na \
	--phrase_size 127 \
	--index_file phrase.hdf5 \
	--question_emb_file question.hdf5 \
	--load_dir piqateam/piqa-nfs/31 \
	--filter_threshold -999999 \
	--split_by_para \
	--iteration 3"

dump_base_qna_511:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--do_embed_question \
	--do_index \
	--output_dir index/squad/base-qna-511 \
	--index_file phrase.hdf5 \
	--phrase_size 511 \
	--question_emb_file question.hdf5 \
	--load_dir piqateam/piqa-nfs/59 \
	--filter_threshold -999999 \
	--split_by_para \
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

train_na_961:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 24G --nfs-output -a " \
	--fs nfs \
	--train_file train-v1.1-na-1-1.json \
	--train_batch_size 18 \
	--phrase_size 961 \
	--do_train \
	--do_predict \
	--do_eval"

train_qna_511:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 24G --nfs-output -a " \
	--fs nfs \
	--train_file train-v1.1-qna-1-1.json \
	--train_batch_size 18 \
	--phrase_size 511 \
	--do_train \
	--do_predict \
	--do_eval"

train_qna_961:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 24G --nfs-output -a " \
	--fs nfs \
	--train_file train-v1.1-qna-1-1.json \
	--train_batch_size 18 \
	--phrase_size 961 \
	--do_train \
	--do_predict \
	--do_eval"

train_qnad_961:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 24G --nfs-output -a " \
	--fs nfs \
	--train_file train-v1.1-qnad.json \
	--train_batch_size 18 \
	--phrase_size 961 \
	--do_train \
	--do_predict \
	--do_eval \
	--num_train_epochs 100"

train_base_qnad_961:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 24G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--train_file train-v1.1-qnad.json \
	--train_batch_size 18 \
	--phrase_size 961 \
	--do_train \
	--do_predict \
	--do_eval \
	--num_train_epochs 10"

dump_na_961:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 24G --nfs-output -a " \
	--fs nfs \
	--do_embed_question \
	--do_index \
	--output_dir index/squad/large-na-3 \
	--phrase_size 961 \
	--index_file phrase.hdf5 \
	--question_emb_file question.hdf5 \
	--load_dir piqateam/piqa-nfs/61 \
	--filter_threshold -999999 \
	--split_by_para \
	--parallel \
	--iteration 3"

dump_qna_961:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_embed_question \
	--do_index \
	--output_dir index/squad/large-qna \
	--phrase_size 961 \
	--index_file phrase.hdf5 \
	--question_emb_file question.hdf5 \
	--load_dir piqateam/piqa-nfs/76 \
	--filter_threshold -2 \
	--split_by_para \
	--parallel \
	--iteration 1"

dump_qna_961_m:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--output_dir index/squad/large-qna-m \
	--predict_file dev-m.json \
	--phrase_size 961 \
	--index_file phrase.hdf5 \
	--question_emb_file question.hdf5 \
	--load_dir piqateam/piqa-nfs/76 \
	--filter_threshold -2 \
	--split_by_para \
	--parallel \
	--iteration 1"

train_filter:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--train_file train-v1.1-qna-1-1.json \
	--train_batch_size 18 \
	--phrase_size 961 \
	--do_train_filter \
	--do_predict \
	--do_eval \
	--num_train_epochs 1 \
	--load_dir piqateam/piqa-nfs/64 \
	--filter_threshold -2 \
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

train_and_eval_base:
	nsml run -d piqa-nfs -g 2 -e run_piqa.py --memory 16G --nfs-output -a " \
	--bert_model_option 'base_uncased' \
	--train_file train-v1.1-na-1-1.json \
	--train_batch_size 18 \
	--phrase_size 127 \
	--fs nfs \
	--do_train \
	--do_predict \
	--do_eval \
	--num_train_epochs 3"
