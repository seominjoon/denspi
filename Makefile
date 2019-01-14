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
	--do_embed_question \
	--do_embed_context \
	--filter_cf 0.1 \
	--num_train_epochs 1 \
	--draft_num_examples 1 \
	--train_batch_size 1 \
	--predict_batch_size 1 \
	--draft \
	--no_cuda
