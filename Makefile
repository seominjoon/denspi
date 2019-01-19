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
	--do_predict \
	--do_eval \
	--do_index \
	--num_train_epochs 1 \
	--draft_num_examples 1 \
	--train_batch_size 1 \
	--predict_batch_size 1 \
	--draft \
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

p007_t051:
	nsml run -d piqa-nfs -g 4 -e run_piqa.py --memory 16G --nfs-output -a " \
	--fs nfs \
	--do_index \
	--output_dir out/large \
	--index_file 0000.hdf5 \
	--load_dir KR18816/squad_bert_2/49 \
	--iteration 3"
