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
	--split_by_para \
	--do_train_sparse \
	--load_dir piqateam_piqa-nfs_76 \
	--iteration 1 \
	--use_sparse

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
	--phrase_size 961 \
	--do_train \
	--do_predict \
	--do_eval"

train_sparse:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 24G --nfs-output -a " \
	--fs nfs \
	--train_file train-v1.1-long.json \
	--train_batch_size 18 \
	--phrase_size 961 \
	--use_sparse \
	--do_train_sparse \
	--do_predict \
	--do_eval \
	--load_dir piqateam/piqa-nfs/76 \
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
	--load_dir piqateam/piqa-nfs/2438 \
	--iteration 3"

train_filter_sparse:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--train_file train-v1.1-qna-1-1.json \
	--train_batch_size 18 \
	--phrase_size 961 \
	--use_sparse \
	--do_train_filter \
	--do_predict \
	--do_eval \
	--draft \
	--draft_num_examples 50000 \
	--load_dir piqateam/piqa-nfs/186 \
	--iteration 3"

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

train_short_961:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 12G --nfs-output -a " \
	--fs nfs \
	--train_file train-v1.1-short.json \
	--train_batch_size 18 \
	--phrase_size 961 \
	--do_load \
	--load_dir piqateam/piqa-nfs/76 \
	--iteration 1 \
	--do_train \
	--do_predict \
	--num_train_epochs 100 \
	--do_eval"

eval_short_961:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 12G --nfs-output -a " \
	--fs nfs \
	--phrase_size 961 \
	--load_dir piqadump/piqa-nfs/154 \
	--iteration 1 \
	--do_predict \
	--do_eval"

eval_961:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 12G --nfs-output -a " \
	--fs nfs \
	--phrase_size 961 \
	--do_load \
	--load_dir piqateam/piqa-nfs/76 \
	--iteration 1 \
	--do_predict \
	--do_eval"

train_long_961:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 24G --nfs-output -a " \
	--fs nfs \
	--train_file train-v1.1-long.json \
	--train_batch_size 18 \
	--phrase_size 961 \
	--do_load \
	--load_dir piqateam/piqa-nfs/76 \
	--iteration 1 \
	--do_train \
	--do_predict \
	--num_train_epochs 100 \
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

train_qnad2_961:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 24G --nfs-output -a " \
	--fs nfs \
	--train_file train-v1.1-qnad2.json \
	--train_batch_size 18 \
	--phrase_size 961 \
	--do_train \
	--do_predict \
	--do_eval \
	--num_train_epochs 100"

train_base_qnad_705:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 24G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--train_file train-v1.1-qnad.json \
	--train_batch_size 18 \
	--phrase_size 705 \
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

dump_qnad_961:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_embed_question \
	--do_index \
	--output_dir index/squad/1365 \
	--phrase_size 961 \
	--index_file phrase.hdf5 \
	--question_emb_file question.hdf5 \
	--load_dir piqateam/piqa-nfs/1365 \
	--filter_threshold -999999 \
	--split_by_para \
	--parallel \
	--iteration 3"


# 2382,2392
dump_qnad_961_m:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_embed_question \
	--do_index \
	--predict_file dev-m.json \
	--output_dir index/squad/2382_m \
	--phrase_size 961 \
	--index_file phrase.hdf5 \
	--question_emb_file question.hdf5 \
	--load_dir piqateam/piqa-nfs/2382 \
	--filter_threshold -999999 \
	--split_by_para \
	--parallel \
	--iteration 1"

dump_qnad_961_m_2:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_embed_question \
	--do_index \
	--predict_file dev-m.json \
	--output_dir index/squad/154_m_2 \
	--phrase_size 961 \
	--index_file phrase.hdf5 \
	--question_emb_file question.hdf5 \
	--load_dir piqadump/piqa-nfs/154 \
	--filter_threshold -999999 \
	--split_by_para \
	--parallel \
	--iteration 2"

dump_qnad_961_m_long:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_embed_question \
	--do_index \
	--predict_file dev-m.json \
	--output_dir index/squad/154_m_long \
	--phrase_size 961 \
	--index_file phrase.hdf5 \
	--question_emb_file question.hdf5 \
	--load_dir piqadump/piqa-nfs/175 \
	--filter_threshold -999999 \
	--split_by_para \
	--parallel \
	--iteration 1"


train_filter_qnad_961:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 24G --nfs-output -a " \
	--fs nfs \
	--train_file train-v1.1-qnad.json \
	--train_batch_size 18 \
	--phrase_size 961 \
	--do_train_filter \
	--do_predict \
	--do_eval \
	--num_train_epochs 1 \
	--load_dir piqateam/piqa-nfs/1365 \
	--filter_threshold -2 \
	--iteration 3"


train_filter_long:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 24G --nfs-output -a " \
	--fs nfs \
	--train_file train-v1.1-long.json \
	--train_batch_size 18 \
	--phrase_size 961 \
	--do_train_filter \
	--do_predict \
	--do_eval \
	--num_train_epochs 1 \
	--load_dir piqadump/piqa-nfs/175 \
	--filter_threshold -2 \
	--draft \
	--draft_num_examples 50000 \
	--iteration 1"


dump_qna_961_1M:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 32G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--output_dir index/squad/large-qna-1M \
	--predict_file dev-1M.json \
	--phrase_size 961 \
	--index_file phrase.hdf5 \
	--question_emb_file question.hdf5 \
	--load_dir piqateam/piqa-nfs/76 \
	--filter_threshold -2 \
	--split_by_para \
	--parallel \
	--iteration 1"


dump_qna_961_1Md:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 32G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--output_dir index/squad/large-qna-1Md \
	--predict_file dev-1M.json \
	--phrase_size 961 \
	--index_file phrase.hdf5 \
	--question_emb_file question.hdf5 \
	--load_dir piqateam/piqa-nfs/76 \
	--filter_threshold -2 \
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
	--output_dir index/squad/sparse \
	--index_file phrase.hdf5 \
	--load_dir KR18816/piqa-nfs/132 \
	--phrase_size 961 \
	--split_by_para \
	--iteration 1 \
	--parallel"

dump_questions:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_embed_question \
	--output_dir index/squad/sparse \
	--question_emb_file question.hdf5 \
	--load_dir KR18816/piqa-nfs/132 \
	--phrase_size 961 \
	--split_by_para \
	--iteration 1 \
	--parallel"

dump_phrases_sp:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--output_dir index/squad/sparse \
	--index_file phrase_sp.hdf5 \
	--load_dir piqateam/piqa-nfs/2438 \
	--phrase_size 961 \
	--split_by_para \
	--iteration 3 \
	--use_sparse \
	--parallel"

dump_questions_sp:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_embed_question \
	--output_dir index/squad/sparse \
	--question_emb_file question_sp.hdf5 \
	--load_dir piqateam/piqa-nfs/2438 \
	--phrase_size 961 \
	--split_by_para \
	--iteration 3 \
	--use_sparse \
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

# Main routine
train:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 24G --nfs-output -a " \
	--fs nfs \
	--train_file train-v1.1-qnad.json \
	--train_batch_size 18 \
	--phrase_size 961 \
	--do_train \
	--do_predict \
	--do_eval \
	--num_train_epochs 100"

train_filter:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 24G --nfs-output -a " \
	--fs nfs \
	--train_file train-v1.1-qnad.json \
	--train_batch_size 18 \
	--phrase_size 961 \
	--do_train_filter \
	--do_predict \
	--do_eval \
	--num_train_epochs 1 \
	--load_dir piqateam/piqa-nfs/64 \
	--filter_threshold -2 \
	--iteration 3"

dump_lod:
	python nsml_dump.py --data_name down_20 --load_dir piqateam/piqa-nfs/2440 --num_gpus 10 --mem_size 16


dump_10M:
	python nsml_dump.py --data_name dev-10M --load_dir piqateam/piqa-nfs/76 --num_gpus 5 --mem_size 24 --end 51 --no_block

dump_100M:
	python nsml_dump.py --data_name dev-100M --load_dir piqateam/piqa-nfs/76 --num_gpus 20 --mem_size 24 --end 508 --no_block

dump_100M_:
	python nsml_dump.py --data_name dev-100M --load_dir piqateam/piqa-nfs/76 --num_gpus 1 --mem_size 24 --start 130 --end 156
