# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import os
import random
from tqdm import tqdm as tqdm_

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import tokenization
from bert import BertConfig
from optimization import BERTAdam
from phrase import BertPhraseModel
from post import write_predictions, write_context, write_question, get_questions, write_hdf5
from pre import convert_examples_to_features, read_squad_examples, convert_documents_to_features, \
    convert_questions_to_features, SquadExample

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult", ["unique_id", "all_logits"])
ContextResult = collections.namedtuple("ContextResult",
                                       ['unique_id', 'start', 'end', 'span_logits', 'filter_logits'])
QuestionResult = collections.namedtuple("QuestionResult",
                                        ['unique_id', 'start', 'end'])


def tqdm(*args, mininterval=5.0, **kwargs):
    return tqdm_(*args, mininterval=mininterval, **kwargs)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--fs', type=str, default='local',
                        help='local|nsml|nfs|nfs_nsml. `nfs_nsml` uses nfs as input and nsml as output')

    # Data paths
    parser.add_argument('--data_dir', default='ckpt/', type=str)
    parser.add_argument("--train_file", default='train-v1.1.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default='dev-v1.1.json', type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument('--gt_file', default='dev-v1.1.json', type=str, help='ground truth file needed for evaluation.')

    # Metadata paths
    parser.add_argument('--metadata_dir', default='ckpt/', type=str)
    parser.add_argument("--vocab_file", default='vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--bert_model_option", default='large_uncased', type=str,
                        help="model architecture option. [large_uncased] or [base_uncased]")
    parser.add_argument("--bert_config_file", default='bert_config.json', type=str,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--init_checkpoint", default='pytorch_model.bin', type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")

    # Output and load paths
    parser.add_argument("--output_dir", default='out', type=str,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument('--load_dir', default='out', type=str)

    # Local paths (if we want to run cmd)
    parser.add_argument('--eval_script', default='evaluate-v1.1.py', type=str)

    # Do
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument('--do_eval', default=False, action='store_true')
    parser.add_argument('--do_embed_context', default=False, action='store_true')
    parser.add_argument('--do_embed_question', default=False, action='store_true')
    parser.add_argument('--do_merge_eval', default=False, action='store_true')
    parser.add_argument('--do_benchmark', default=False, action='store_true')
    parser.add_argument('--do_serve', default=False, action='store_true')
    parser.add_argument('--do_index', default=False, action='store_true')

    # Model options
    parser.add_argument("--do_case", default=False, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--train_batch_size", default=12, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=16, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--save_checkpoints_steps", default=1000, type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--iterations_per_loop", default=1000, type=int,
                        help="How many steps to make in each estimator call.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--draft_num_examples', type=int, default=12)
    parser.add_argument('--span_vec_size', default=64, type=int)
    parser.add_argument('--span_threshold', default=-1e9, type=float)
    parser.add_argument('--filter_cf', default=0.0, type=float)
    parser.add_argument('--port', default=9003, type=int)
    parser.add_argument('--parallel', default=False, action='store_true')

    args = parser.parse_args()

    # Filesystem routines
    if args.fs == 'local':
        class Processor(object):
            def __init__(self, path):
                self._save = None
                self._load = None
                self._path = path

            def bind(self, save, load):
                self._save = save
                self._load = load

            def save(self, checkpoint=None, save_fn=None, **kwargs):
                path = os.path.join(self._path, str(checkpoint))
                if save_fn is None:
                    self._save(path, **kwargs)
                else:
                    save_fn(path, **kwargs)

            def load(self, checkpoint, load_fn=None, session=None, **kwargs):
                assert self._path == session
                path = os.path.join(self._path, str(checkpoint), 'model.pt')
                if load_fn is None:
                    self._load(path, **kwargs)
                else:
                    load_fn(path, **kwargs)

        processor = Processor(args.load_dir)
    elif args.fs == 'nfs':
        import nsml
        from nsml import NSML_NFS_OUTPUT
        args.data_dir = os.path.join(NSML_NFS_OUTPUT, args.data_dir)
        args.metadata_dir = os.path.join(NSML_NFS_OUTPUT, args.metadata_dir)
        # args.load_dir should be the session name
        processor = nsml
        args.output_dir = os.path.join(NSML_NFS_OUTPUT, args.output_dir)
    elif args.fs == 'nsml':
        import nsml
        from nsml import DATASET_PATH
        args.data_dir = os.path.join(DATASET_PATH, 'train')
        args.metadata_dir = os.path.join(DATASET_PATH, 'train')
        # args.load_dir should be the session name
        processor = nsml
        # args.output_dir is local, so no change
    elif args.fs == 'nsml_nfs':
        import nsml
        from nsml import NSML_NFS_OUTPUT
        args.data_dir = os.path.join(NSML_NFS_OUTPUT, args.data_dir)
        args.metadata_dir = os.path.join(NSML_NFS_OUTPUT, args.metadata_dir)
        # args.load_dir should be the session name
        processor = nsml
        # args.output_dir is local, so no change
    else:
        raise ValueError(args.fs)

    # Configure paths
    args.train_file = os.path.join(args.data_dir, args.train_file)
    args.predict_file = os.path.join(args.data_dir, args.predict_file)
    args.gt_file = os.path.join(args.data_dir, args.gt_file)

    args.bert_config_file = os.path.join(args.metadata_dir, args.bert_config_file.replace(".json", "") +
                                         "_" + args.bert_model_option + ".json")
    args.init_checkpoint = os.path.join(args.metadata_dir, args.init_checkpoint.replace(".bin", "") +
                                        "_" + args.bert_model_option + ".bin")
    args.vocab_file = os.path.join(args.metadata_dir, args.vocab_file)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        pass
        # raise ValueError("Output directory () already exists and is not empty.")
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=not args.do_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = read_squad_examples(
            input_file=args.train_file, is_training=True, draft=args.draft, draft_num_examples=args.draft_num_examples)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    model = BertPhraseModel(bert_config,
                            span_vec_size=args.span_vec_size)
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))

    if args.do_train and args.init_checkpoint is not None:
        state_dict = torch.load(args.init_checkpoint, map_location='cpu')
        if next(iter(state_dict)).startswith('bert.'):
            state_dict = {key[len('bert.'):]: val for key, val in state_dict.items()}
            state_dict = {key: val for key, val in state_dict.items() if key in model.encoder.bert_model.state_dict()}
        model.encoder.bert_model.load_state_dict(state_dict)
    if args.fp16:
        model.half()

    if not args.optimize_on_cpu:
        model.to(device)
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}
    ]
    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif args.parallel or n_gpu > 1:
        model = torch.nn.DataParallel(model)

    #     bind_model(model, args)
    #     if args.pause:
    #         nsml.paused(scope=locals())
    bind_model(processor, model, optimizer)

    global_step = 0
    if args.do_train:
        train_features, train_features_ = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True)
        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

        all_input_ids_ = torch.tensor([f.input_ids for f in train_features_], dtype=torch.long)
        all_input_mask_ = torch.tensor([f.input_mask for f in train_features_], dtype=torch.long)

        if args.fp16:
            (all_input_ids, all_input_mask,
             all_start_positions,
             all_end_positions) = tuple(t.half() for t in (all_input_ids, all_input_mask,
                                                           all_start_positions, all_end_positions))
            all_input_ids_, all_input_mask_ = tuple(t.half() for t in (all_input_ids_, all_input_mask_))

        train_data = TensorDataset(all_input_ids, all_input_mask,
                                   all_input_ids_, all_input_mask_,
                                   all_start_positions, all_end_positions)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for epoch in range(int(args.num_train_epochs)):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Epoch %d" % (epoch + 1))):
                batch = tuple(t.to(device) for t in batch)
                (input_ids, input_mask,
                 input_ids_, input_mask_,
                 start_positions, end_positions) = batch
                loss = model(input_ids, input_mask,
                             input_ids_, input_mask_,
                             start_positions, end_positions)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.optimize_on_cpu:
                        model.to('cpu')
                    optimizer.step()  # We have accumulated enought gradients
                    model.zero_grad()
                    if args.optimize_on_cpu:
                        model.to(device)
                    global_step += 1

            processor.save(epoch + 1)
    else:
        assert args.load_dir is not None, 'You need to provide --load_dir'
        processor.load(args.iteration, session=args.load_dir)

    if args.do_predict:

        eval_examples = read_squad_examples(
            input_file=args.predict_file, is_training=False, draft=args.draft,
            draft_num_examples=args.draft_num_examples)
        eval_features, query_eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_input_ids_ = torch.tensor([f.input_ids for f in query_eval_features], dtype=torch.long)
        all_input_mask_ = torch.tensor([f.input_mask for f in query_eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        if args.fp16:
            (all_input_ids, all_input_mask, all_example_index) = tuple(t.half() for t in (all_input_ids, all_input_mask,
                                                                                          all_example_index))
            all_input_ids_, all_input_mask_ = tuple(t.half() for t in (all_input_ids_, all_input_mask_))

        eval_data = TensorDataset(all_input_ids, all_input_mask,
                                  all_input_ids_, all_input_mask_,
                                  all_example_index)
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        model.eval()
        logger.info("Start evaluating")

        def get_results():
            for (input_ids, input_mask, input_ids_, input_mask_, example_indices) in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                input_ids_ = input_ids_.to(device)
                input_mask_ = input_mask_.to(device)
                with torch.no_grad():
                    batch_all_logits = model(input_ids, input_mask, input_ids_, input_mask_)
                for i, example_index in enumerate(example_indices):
                    all_logits = batch_all_logits[i].detach().cpu().numpy()
                    eval_feature = eval_features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    yield RawResult(unique_id=unique_id,
                                    all_logits=all_logits)

        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        write_predictions(eval_examples, eval_features, get_results(),
                          args.max_answer_length,
                          not args.do_case, output_prediction_file, args.verbose_logging,
                          args.span_threshold)

        if args.do_eval:
            command = "python %s %s %s" % (args.eval_script, args.gt_file, output_prediction_file)
            import subprocess
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(output)

    if args.do_embed_context:
        context_examples = read_squad_examples(
            context_only=True,
            input_file=args.predict_file, is_training=False, draft=args.draft,
            draft_num_examples=args.draft_num_examples)
        context_features = convert_documents_to_features(
            examples=context_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride)

        logger.info("***** Running embedding *****")
        logger.info("  Num orig examples = %d", len(context_examples))
        logger.info("  Num split examples = %d", len(context_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in context_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in context_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        if args.fp16:
            (all_input_ids, all_input_mask, all_example_index) = tuple(t.half() for t in (all_input_ids, all_input_mask,
                                                                                          all_example_index))
        context_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
        if args.local_rank == -1:
            context_sampler = SequentialSampler(context_data)
        else:
            context_sampler = DistributedSampler(context_data)
        context_dataloader = DataLoader(context_data, sampler=context_sampler, batch_size=args.predict_batch_size)

        model.eval()
        logger.info("Start embedding")

        def get_context_results():
            for (input_ids, input_mask, example_indices) in context_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                with torch.no_grad():
                    batch_start, batch_end, batch_span_logits, batch_filter_logits = model(input_ids, input_mask)
                for i, example_index in enumerate(example_indices):
                    start = batch_start[i].detach().cpu().numpy()
                    end = batch_end[i].detach().cpu().numpy()
                    span_logits = batch_span_logits[i].detach().cpu().numpy()
                    filter_logits = batch_filter_logits[i].detach().cpu().numpy()
                    context_feature = context_features[example_index.item()]
                    unique_id = int(context_feature.unique_id)
                    yield ContextResult(unique_id=unique_id,
                                        start=start,
                                        end=end,
                                        span_logits=span_logits,
                                        filter_logits=filter_logits)

        def context_save(filename, **kwargs):
            write_context(context_examples, context_features, get_context_results(),
                          args.span_threshold, args.max_answer_length,
                          not args.do_case, filename, args.verbose_logging)

        iteration = epoch + 1 if args.do_train else args.iteration

        if args.nfs:
            from nsml import NSML_NFS_OUTPUT
            context_save(os.path.join(NSML_NFS_OUTPUT, '%s_embed_%s' % (iteration, args.context_embed_dir)))
        else:
            processor.save('%s_embed_%s' % (iteration, args.context_embed_dir), save_fn=context_save)

    if args.do_embed_question:
        question_examples = read_squad_examples(
            question_only=True,
            input_file=args.predict_file, is_training=False, draft=args.draft,
            draft_num_examples=args.draft_num_examples)
        query_eval_features = convert_questions_to_features(
            examples=question_examples,
            tokenizer=tokenizer,
            max_query_length=args.max_query_length)

        all_input_ids_ = torch.tensor([f.input_ids for f in query_eval_features], dtype=torch.long)
        all_input_mask_ = torch.tensor([f.input_mask for f in query_eval_features], dtype=torch.long)
        all_example_index_ = torch.arange(all_input_ids_.size(0), dtype=torch.long)
        if args.fp16:
            all_input_ids_, all_input_mask_ = tuple(t.half() for t in (all_input_ids_, all_input_mask_))

        question_data = TensorDataset(all_input_ids_, all_input_mask_, all_example_index_)

        if args.local_rank == -1:
            question_sampler = SequentialSampler(question_data)
        else:
            question_sampler = DistributedSampler(question_data)
        question_dataloader = DataLoader(question_data, sampler=question_sampler, batch_size=args.predict_batch_size)

        model.eval()
        logger.info("Start embedding")

        def get_question_results():
            for (input_ids_, input_mask_, example_indices) in question_dataloader:
                input_ids_ = input_ids_.to(device)
                input_mask_ = input_mask_.to(device)
                with torch.no_grad():
                    batch_start, batch_end = model(query_ids=input_ids_,
                                                   query_mask=input_mask_)
                for i, example_index in enumerate(example_indices):
                    start = batch_start[i].detach().cpu().numpy()
                    end = batch_end[i].detach().cpu().numpy()
                    query_eval_feature = query_eval_features[example_index.item()]
                    unique_id = int(query_eval_feature.unique_id)
                    yield QuestionResult(unique_id=unique_id,
                                         start=start,
                                         end=end)

        def question_save(filename, **kwargs):
            write_question(question_examples, query_eval_features, get_question_results(), filename)

        iteration = epoch + 1 if args.do_train else args.iteration

        if args.nfs:
            from nsml import NSML_NFS_OUTPUT
            question_save(os.path.join(NSML_NFS_OUTPUT, '%s_embed_%s' % (iteration, args.question_embed_dir)))
        else:
            processor.save('%s_embed_%s' % (iteration, args.question_embed_dir), save_fn=question_save)

    if args.do_merge_eval:
        assert args.load_dir is not None, 'You need to provide --load_dir'

        local_question_embed_dir = 'local_question_embed_dir'

        def copy(filename, **kwargs):
            import shutil
            shutil.copytree(filename, local_question_embed_dir)

        processor.load('%s_embed_%s' % (iteration, args.question_embed_dir), load_fn=copy)

        def merge_load(filename, **kwargs):
            output_prediction_file = os.path.join(args.output_dir, "merge_predictions.json")
            merge_command = "python merge.py %s %s %s %s --progress --q_mat" % (args.gt_file, filename,
                                                                                local_question_embed_dir,
                                                                                output_prediction_file)
            eval_command = "python evaluate-v1.1.py %s %s" % (args.gt_file, output_prediction_file)
            import subprocess
            process = subprocess.Popen(merge_command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(output)
            print(error)
            process = subprocess.Popen(eval_command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(output)
            print(error)

        iteration = epoch + 1 if args.do_train else args.iteration
        processor.load('%s_embed_%s' % (iteration, args.context_embed_dir), load_fn=merge_load)

    if args.do_benchmark:
        question_examples = read_squad_examples(
            question_only=True,
            input_file=args.predict_file, is_training=False, draft=args.draft,
            draft_num_examples=args.draft_num_examples)
        query_eval_features = convert_questions_to_features(
            examples=question_examples,
            tokenizer=tokenizer,
            max_query_length=args.max_query_length)

        logger.info("***** Running embedding *****")
        logger.info("  Num orig examples = %d", len(question_examples))
        logger.info("  Num split examples = %d", len(query_eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids_ = torch.tensor([f.input_ids for f in query_eval_features], dtype=torch.long)
        all_input_mask_ = torch.tensor([f.input_mask for f in query_eval_features], dtype=torch.long)
        all_example_index_ = torch.arange(all_input_ids_.size(0), dtype=torch.long)

        question_data = TensorDataset(all_input_ids_, all_input_mask_, all_example_index_)

        if args.local_rank == -1:
            question_sampler = SequentialSampler(question_data)
        else:
            question_sampler = DistributedSampler(question_data)
        question_dataloader = DataLoader(question_data, sampler=question_sampler, batch_size=args.predict_batch_size)

        model.eval()
        logger.info("Start benchmarking")

        def get_question_results():
            for (input_ids_, input_mask_, example_indices) in question_dataloader:
                input_ids_ = input_ids_.to(device)
                input_mask_ = input_mask_.to(device)
                with torch.no_grad():
                    batch_start, batch_end = model(query_ids=input_ids_, query_mask=input_mask_)
                for i, example_index in enumerate(example_indices):
                    start = batch_start[i].detach().cpu().numpy()
                    end = batch_end[i].detach().cpu().numpy()
                    query_eval_feature = query_eval_features[example_index.item()]
                    unique_id = int(query_eval_feature.unique_id)
                    yield QuestionResult(unique_id=unique_id,
                                         start=start,
                                         end=end)

        import time
        start = time.time()
        count = 0
        for _, _ in tqdm(get_questions(question_examples, query_eval_features, get_question_results()),
                         total=len(query_eval_features)):
            count += 1
        end = time.time()

        print('latency per query: %dms' % ((end - start) * 1000 / count))

    if args.do_index:
        if ':' not in args.predict_file:
            predict_files = [args.predict_file]
            offsets = [0]
        else:
            dirname = os.path.dirname(args.predict_file)
            basename = os.path.basename(args.predict_file)
            start, end = list(map(int, basename.split(':')))
            predict_files = [os.path.join(dirname, str(i).zfill(4)) for i in range(start, end)]
            offsets = [int(each) * 1000 for each in predict_files]

        for offset, predict_file in zip(offsets, predict_files):
            context_examples = read_squad_examples(
                context_only=True,
                input_file=predict_file, is_training=False, draft=args.draft,
                draft_num_examples=args.draft_num_examples)

            for example in context_examples:
                example.doc_idx += offset

            context_features = convert_documents_to_features(
                examples=context_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride)

            logger.info("***** Running indexing on %s *****" % predict_file)
            logger.info("  Num orig examples = %d", len(context_examples))
            logger.info("  Num split examples = %d", len(context_features))
            logger.info("  Batch size = %d", args.predict_batch_size)

            all_input_ids = torch.tensor([f.input_ids for f in context_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in context_features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            if args.fp16:
                all_input_ids, all_input_mask, all_example_index = tuple(
                    t.half() for t in (all_input_ids, all_input_mask, all_example_index))

            context_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)

            if args.local_rank == -1:
                context_sampler = SequentialSampler(context_data)
            else:
                context_sampler = DistributedSampler(context_data)
            context_dataloader = DataLoader(context_data, sampler=context_sampler,
                                            batch_size=args.predict_batch_size)

            model.eval()
            logger.info("Start indexing")

            def get_context_results():
                for (input_ids, input_mask, example_indices) in context_dataloader:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    with torch.no_grad():
                        batch_start, batch_end, batch_span_logits = model(input_ids,
                                                                          input_mask)
                    for i, example_index in enumerate(example_indices):
                        start = batch_start[i].detach().cpu().numpy()
                        end = batch_end[i].detach().cpu().numpy()
                        span_logits = batch_span_logits[i].detach().cpu().numpy()
                        context_feature = context_features[example_index.item()]
                        unique_id = int(context_feature.unique_id)
                        yield ContextResult(unique_id=unique_id,
                                            start=start,
                                            end=end,
                                            span_logits=span_logits,
                                            filter_logits=None)

            hdf5_path = os.path.join(args.output_dir, 'index.hdf5')
            write_hdf5(context_examples, context_features, get_context_results(),
                       args.max_answer_length, not args.do_case, hdf5_path, args.verbose_logging)

    if args.do_serve:
        def get_vec(text):
            question_examples = [SquadExample(qas_id='serve', question_text=text)]
            query_eval_features = convert_questions_to_features(
                examples=question_examples,
                tokenizer=tokenizer)

            all_input_ids_ = torch.tensor([f.input_ids for f in query_eval_features], dtype=torch.long)
            all_input_mask_ = torch.tensor([f.input_mask for f in query_eval_features], dtype=torch.long)
            all_example_index_ = torch.arange(all_input_ids_.size(0), dtype=torch.long)

            question_data = TensorDataset(all_input_ids_, all_input_mask_, all_example_index_)

            if args.local_rank == -1:
                question_sampler = SequentialSampler(question_data)
            else:
                question_sampler = DistributedSampler(question_data)
            question_dataloader = DataLoader(question_data, sampler=question_sampler,
                                             batch_size=args.predict_batch_size)

            model.eval()
            logger.info("Start benchmarking")

            def get_question_results():
                for (input_ids_, input_mask_, example_indices) in question_dataloader:
                    input_ids_ = input_ids_.to(device)
                    input_mask_ = input_mask_.to(device)
                    with torch.no_grad():
                        batch_start, batch_end = model(query_ids=input_ids_,
                                                       query_mask=input_mask_)
                    for i, example_index in enumerate(example_indices):
                        start = batch_start[i].detach().cpu().numpy()
                        end = batch_end[i].detach().cpu().numpy()
                        query_eval_feature = query_eval_features[example_index.item()]
                        unique_id = int(query_eval_feature.unique_id)
                        yield QuestionResult(unique_id=unique_id,
                                             start=start,
                                             end=end)

            _, matrix = next(iter(get_questions(question_examples, query_eval_features, get_question_results())))
            vec = matrix[0].tolist()
            return vec

        from flask import Flask, request, jsonify
        from flask_cors import CORS

        from tornado.wsgi import WSGIContainer
        from tornado.httpserver import HTTPServer
        from tornado.ioloop import IOLoop

        from time import time

        app = Flask(__name__)

        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
        CORS(app)

        @app.route('/api', methods=['GET'])
        def api():
            query = request.args['query']
            start = time()
            out = get_vec(query)
            end = time()
            print('latency: %dms' % int((end - start) * 1000))
            return jsonify(out)

        print('Starting server at %d' % args.port)
        http_server = HTTPServer(WSGIContainer(app))
        http_server.listen(args.port)
        IOLoop.instance().start()


def bind_model(processor, model, optimizer):
    def save(filename, save_model=True, saver=None, **kwargs):
        if not os.path.exists(filename):
            os.makedirs(filename)
        if save_model:
            state = {'model': model.state_dict(), 'optimizer': optimizer}
            model_path = os.path.join(filename, 'model.pt')
            dummy_path = os.path.join(filename, 'dummy')
            torch.save(state, model_path)
            with open(dummy_path, 'w') as fp:
                json.dump([], fp)
            print('Model saved at %s' % model_path)
        if saver is not None:
            saver(filename)

    def load(filename, load_model=True, loader=None, **kwargs):
        if load_model:
            # print('%s: %s' % (filename, os.listdir(filename)))
            model_path = os.path.join(filename, 'model.pt')
            if not os.path.exists(model_path):  # for compatibility
                model_path = filename
            state = torch.load(model_path, map_location='cpu')
            state_dict = state['model']
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(state['optimizer'])
            print('Model loaded from %s' % model_path)
        if loader is not None:
            loader(filename)

    processor.bind(save=save, load=load)


if __name__ == "__main__":
    main()
