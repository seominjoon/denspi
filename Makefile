t084:
	nsml run -d squad_bert_2 -g 1 -e run_piqa.py --memory 12G -a " \
	--bert_model_option 'base_uncased' \
	--do_train \
	--do_predict \
	--do_eval \
	--num_train_epochs 1"

t085:
	nsml run -d squad_bert_2 -g 4 -e run_piqa.py --memory 16G -a " \
	--do_train \
	--do_predict \
	--do_eval \
	--num_train_epochs 1"

t040:
	nsml run -d squad_bert_2 -g 4 -e run_piqa.py --memory 16G -a " \
	--do_train \
	--do_predict \
	--do_eval \
	--num_train_epochs 3"

t051:
	nsml run -d squad_bert_2 -g 1 -e run_piqa.py --memory 12G -a " \
	--bert_model_option 'base_uncased' \
	--do_train \
	--do_predict \
	--do_eval \
	--num_train_epochs 3"

t056:
	nsml run -d squad_bert_2 -g 4 -e run_piqa.py --memory 16G -a " \
	--train_file train-v1.1-na-1-1.json \
	--do_train \
	--do_predict \
	--do_eval \
	--train_batch_size 18 \
	--num_train_epochs 3"

t058:
	nsml run -d squad_bert_2 -g 2 -e run_piqa.py --memory 12G -a " \
	--train_file train-v1.1-na-1-1.json \
	--bert_model_option 'base_uncased' \
	--do_train \
	--do_predict \
	--do_eval \
	--train_batch_size 18 \
	--num_train_epochs 3"

t082:
	nsml run -d squad_bert_2 -g 4 -e run_piqa.py --memory 16G -a " \
	--train_file train-v2.0.json \
	--do_train \
	--do_predict \
	--do_eval \
	--train_batch_size 18 \
	--num_train_epochs 5"

t083:
	nsml run -d squad_bert_2 -g 2 -e run_piqa.py --memory 12G -a " \
	--train_file train-v2.0.json \
	--bert_model_option 'base_uncased' \
	--do_train \
	--do_predict \
	--do_eval \
	--train_batch_size 18 \
	--num_train_epochs 5"

dl:
	python run_piqa.py \
	--bert_model_option 'base_uncased' \
	--do_train \
	--do_predict \
	--do_eval \
	--num_train_epochs 1 \
	--draft_num_examples 1 \
	--train_batch_size 1 \
	--predict_batch_size 1 \
	--draft \
	--no_cuda

kor_dl:
	python run_piqa_korquad.py \
	--bert_model_option 'base_cased' \
	--data_dir ~/data/korquad_bert/train \
	--eval_script evaluate-v1.0.py \
	--train_file KorQuAD_v1.0_train.json \
	--gt_file KorQuAD_v1.0_dev.json \
	--predict_file KorQuAD_v1.0_dev.json \
	--do_train \
	--do_predict \
	--do_eval \
	--num_train_epochs 1 \
	--draft_num_examples 1 \
	--train_batch_size 1 \
	--predict_batch_size 1 \
	--draft \
	--no_cuda

k018:
	nsml run -d korquad_bert -g 1 -e run_piqa_korquad.py --memory 16G -a " \
	--bert_model_option 'base_cased' \
	--eval_script evaluate-v1.0.py \
	--train_file KorQuAD_v1.0_train.json \
	--gt_file KorQuAD_v1.0_dev.json \
	--predict_file KorQuAD_v1.0_dev.json \
	--do_train \
	--do_predict \
	--num_train_epochs 1 \
	--do_case \
	--do_eval"


k019:
	nsml run -d korquad_bert -g 1 -e run_piqa_korquad.py --memory 16G -a " \
	--bert_model_option 'base_cased' \
	--eval_script evaluate-v1.0.py \
	--train_file KorQuAD_v1.0_train.json \
	--gt_file KorQuAD_v1.0_dev.json \
	--predict_file KorQuAD_v1.0_dev.json \
	--do_train \
	--do_predict \
	--num_train_epochs 3 \
	--do_case \
	--do_eval"

k020:
	nsml run -d korquad_bert -g 1 -e run_piqa_korquad.py --memory 16G -a " \
	--bert_model_option 'base_cased' \
	--eval_script evaluate-v1.0.py \
	--train_file KorQuAD_v1.0_train.json \
	--gt_file KorQuAD_v1.0_dev.json \
	--predict_file KorQuAD_v1.0_dev.json \
	--do_train \
	--do_predict \
	--seed 29 \
	--num_train_epochs 3 \
	--do_case \
	--do_eval"

k021:
	nsml run -d korquad_bert -g 2 -e run_piqa_korquad.py --memory 16G -a " \
	--bert_model_option 'base_cased' \
	--eval_script evaluate-v1.0.py \
	--train_file KorQuAD_v1.0_train.json \
	--gt_file KorQuAD_v1.0_dev.json \
	--predict_file KorQuAD_v1.0_dev.json \
	--do_train \
	--do_predict \
	--num_train_epochs 3 \
	--do_case \
	--max_seq_length 512 \
	--do_eval"

k021:
	nsml run -d korquad_bert -g 2 -e run_piqa_korquad.py --memory 16G -a " \
	--bert_model_option 'base_cased' \
	--eval_script evaluate-v1.0.py \
	--train_file KorQuAD_v1.0_train.json \
	--gt_file KorQuAD_v1.0_dev.json \
	--predict_file KorQuAD_v1.0_dev.json \
	--do_train \
	--do_predict \
	--num_train_epochs 3 \
	--do_case \
	--max_seq_length 512 \
	--do_eval"

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

dl_load:
	python run_piqa.py \
	--bert_model_option 'base_uncased' \
	--iteration 1 \
	--do_index \
	--num_train_epochs 1 \
	--draft_num_examples 1 \
	--train_batch_size 1 \
	--predict_batch_size 1 \
	--draft \
	--no_cuda

p088_t051:
	nsml run -d squad_bert_2 -g 1 -e run_piqa.py --memory 12G -a " \
	--fs nsml \
	--bert_model_option 'base_uncased' \
	--do_index \
	--index_file 0000.hdf5 \
	--load_dir KR18816/squad_bert_2/51 \
	--iteration 3"

p090_t051:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 12G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--do_index \
	--output_dir out/base \
	--index_file 0000.hdf5 \
	--load_dir KR18816/squad_bert_2/51 \
	--iteration 3"

p017_t051:
	nsml run -d piqa-nfs -g 2 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--output_dir index/squad/large \
	--index_file 0000.hdf5 \
	--load_dir KR18816/squad_bert_2/49 \
	--iteration 3"

p020_t051:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--output_dir index/squad/large \
	--index_file 0001.hdf5 \
	--load_dir KR18816/squad_bert_2/49 \
	--iteration 3 \
	--parallel"

p021_t051:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--output_dir index/squad/large \
	--index_file 0002.hdf5 \
	--load_dir KR18816/squad_bert_2/49 \
	--iteration 3 \
	--predict_batch_size 12 \
	--parallel"

p022_t051:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--output_dir index/squad/large \
	--index_file 0002.hdf5 \
	--load_dir KR18816/squad_bert_2/49 \
	--iteration 3 \
	--predict_batch_size 20 \
	--parallel"

p023_t051:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--do_embed_question \
	--output_dir index/squad/large \
	--index_file index3.hdf5 \
	--load_dir KR18816/squad_bert_2/49 \
	--iteration 3 \
	--parallel"

p050_t051:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 12G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--do_train_filter \
	--do_predict \
	--filter_threshold 0.0 \
	--do_eval \
	--load_dir KR18816/squad_bert_2/51 \
	--iteration 3"

p030_t051:
	nsml run -d piqa-nfs -g 0 -e run_piqa.py --memory 12G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--do_train_filter \
	--do_predict \
	--filter_threshold 0.0 \
	--do_eval \
	--draft \
	--draft_num_examples 1000 \
	--load_dir KR18816/piqa-nfs/35 \
	--iteration 3"

p038_t035:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 12G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--do_predict \
	--filter_threshold -3 \
	--do_eval \
	--load_dir KR18816/piqa-nfs/35 \
	--iteration 1"

p039_t035:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 12G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--do_predict \
	--filter_threshold -2 \
	--do_eval \
	--load_dir KR18816/piqa-nfs/35 \
	--iteration 1"

p040_t035:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 12G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--do_predict \
	--filter_threshold -1 \
	--do_eval \
	--load_dir KR18816/piqa-nfs/35 \
	--iteration 1"

p051_t050:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 12G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--do_predict \
	--filter_threshold -1 \
	--do_eval \
	--load_dir KR18816/piqa-nfs/50 \
	--iteration 1"

p052_t050:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 12G --nfs-output -a " \
	--fs nfs \
	--bert_model_option 'base_uncased' \
	--do_predict \
	--filter_threshold -2 \
	--do_eval \
	--load_dir KR18816/piqa-nfs/50 \
	--iteration 1"

p059_t049:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_train_filter \
	--do_predict \
	--filter_threshold -2 \
	--do_eval \
	--load_dir KR18816/squad_bert_2/49 \
	--iteration 3"

debug:
	python run_piqa.py \
	--bert_model_option 'base_uncased' \
	--predict_file /Users/user/data/squad/debug.json \
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
	--no_cuda

p076_t059:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_predict \
	--filter_threshold -2 \
	--do_eval \
	--load_dir KR18816/piqa-nfs/59 \
	--parallel \
	--iteration 1"

p086_t059:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--do_embed_question \
	--output_dir index/squad/large \
	--index_file index.hdf5 \
	--load_dir KR18816/piqa-nfs/59 \
	--filter_threshold -2 \
	--compression_offset 0.0 \
	--compression_scale 20.0 \
	--iteration 1 \
	--parallel"

p091_t059:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--do_embed_question \
	--output_dir index/squad/large \
	--index_file index.hdf5 \
	--load_dir KR18816/piqa-nfs/59 \
	--compression_offset 0.0 \
	--compression_scale 20.0 \
	--iteration 1 \
	--parallel"


p092_t059:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--do_embed_question \
	--output_dir index/squad/large \
	--index_file index_2_20.hdf5 \
	--load_dir KR18816/piqa-nfs/59 \
	--compression_offset 2.0 \
	--compression_scale 20.0 \
	--iteration 1 \
	--parallel"

p093_t059:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--do_embed_question \
	--output_dir index/squad/large \
	--index_file index_2_20_s.hdf5 \
	--load_dir KR18816/piqa-nfs/59 \
	--compression_offset 2.0 \
	--compression_scale 20.0 \
	--iteration 1 \
	--split_by_para \
	--parallel"

p097_t059:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--do_embed_question \
	--output_dir index/squad/large \
	--index_file index_s.hdf5 \
	--load_dir KR18816/piqa-nfs/59 \
	--iteration 1 \
	--split_by_para \
	--parallel"

p098_t059:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--do_embed_question \
	--output_dir index/squad/large \
	--index_file index_2_20_s_.hdf5 \
	--compression_offset 2.0 \
	--compression_scale 20.0 \
	--load_dir KR18816/piqa-nfs/59 \
	--iteration 1 \
	--split_by_para \
	--parallel"

p112_t059:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--output_dir index/squad/large \
	--index_file index_f-2.hdf5 \
	--load_dir KR18816/piqa-nfs/59 \
	--filter_threshold -2 \
	--iteration 1 \
	--split_by_para \
	--parallel"

p113_t059:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--output_dir index/squad/large \
	--index_file index_o2_s20_f-2.hdf5 \
	--load_dir KR18816/piqa-nfs/59 \
	--compression_offset 2.0 \
	--compression_scale 20.0 \
	--filter_threshold -2 \
	--iteration 1 \
	--split_by_para \
	--parallel"

p114_t059:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--output_dir index/squad/large \
	--index_file index_o2_s20.hdf5 \
	--load_dir KR18816/piqa-nfs/59 \
	--compression_offset 2.0 \
	--compression_scale 20.0 \
	--iteration 1 \
	--split_by_para \
	--parallel"

p124_t059:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--output_dir index/squad/large \
	--index_file demo.hdf5 \
	--load_dir KR18816/piqa-nfs/59 \
	--iteration 1 \
	--parallel"


p128_t059:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--data_dir data/docs \
	--predict_file 0:10 \
	--output_dir index/squad/large \
	--index_file demo_0-10.hdf5 \
	--load_dir KR18816/piqa-nfs/59 \
	--iteration 1 \
	--parallel"

p130_t059:
	nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--data_dir data/docs \
	--predict_file 2000:2002 \
	--output_dir index/wiki/large \
	--index_file demo_2000-2002.hdf5 \
	--load_dir KR18816/piqa-nfs/59 \
	--iteration 1 \
	--parallel"
