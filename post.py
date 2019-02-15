import collections
import io
import json
import logging
from multiprocessing import Queue
from tempfile import TemporaryFile

import numpy as np
import six
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm as tqdm_
import torch

import tokenization

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

QuestionResult = collections.namedtuple("QuestionResult",
                                        ['qas_id', 'start', 'end', 'span_logit'])


def tqdm(*args, min_interval=5.0, **kwargs):
    return tqdm_(*args, mininterval=min_interval, **kwargs)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def write_predictions(all_examples, all_features, all_results,
                      max_answer_length, do_lower_case, output_prediction_file, verbose_logging,
                      threshold):
    id2feature = {feature.unique_id: feature for feature in all_features}
    id2example = {id_: all_examples[id2feature[id_].example_index] for id_ in id2feature}

    word_count = 0
    start_count = 0
    end_count = 0

    predictions = {}
    scores = {}

    for result in tqdm(all_results, total=len(all_features)):
        feature = id2feature[result.unique_id]
        example = id2example[result.unique_id]
        id_ = example.qas_id

        word_count += len(feature.tokens)
        for start_index in range(len(feature.tokens)):
            if result.filter_start_logits[start_index] >= threshold:
                start_count += 1
        for end_index in range(len(feature.tokens)):
            if result.filter_end_logits[end_index] >= threshold:
                end_count += 1

        for start_index in range(len(feature.tokens)):
            if result.filter_start_logits[start_index] < threshold:
                continue
            for end_index in range(start_index, min(len(feature.tokens), start_index + max_answer_length - 1)):
                if result.filter_end_logits[end_index] < threshold:
                    continue
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue

                score = result.all_logits[start_index, end_index]
                if id_ not in scores or score > scores[id_]:
                    orig_text, start_pos, end_pos = get_final_text_(example, feature, start_index, end_index,
                                                                    do_lower_case, verbose_logging)
                    phrase = orig_text[start_pos:end_pos]
                    predictions[id_] = phrase
                    scores[id_] = score

    print('num_start_vecs=%d, num_words=%d, nspw=%.4f' % (start_count, word_count, start_count / word_count))
    print('num_end_vecs=%d, num_words=%d, nepw=%.4f' % (end_count, word_count, end_count / word_count))

    with open(output_prediction_file, 'w') as fp:
        json.dump(predictions, fp)


def get_metadata(id2example, features, results, max_answer_length, do_lower_case, verbose_logging, split_by_para):
    start = np.concatenate([result.start[1:len(feature.tokens) - 1] for feature, result in zip(features, results)],
                           axis=0)
    end = np.concatenate([result.end[1:len(feature.tokens) - 1] for feature, result in zip(features, results)], axis=0)

    # sparse = np.concatenate([result.sparse[1:len(feature.tokens)-1,1:len(feature.tokens)-1] for feature, result in zip(features, results)], axis=0)

    fs = np.concatenate([result.filter_start_logits[1:len(feature.tokens) - 1]
                         for feature, result in zip(features, results)],
                        axis=0)
    fe = np.concatenate([result.filter_end_logits[1:len(feature.tokens) - 1]
                         for feature, result in zip(features, results)],
                        axis=0)

    span_logits = np.zeros([np.shape(start)[0], max_answer_length], dtype=start.dtype)
    start2end = -1 * np.ones([np.shape(start)[0], max_answer_length], dtype=np.int32)
    idx = 0
    for feature, result in zip(features, results):
        for i in range(1, len(feature.tokens) - 1):
            for j in range(i, min(i + max_answer_length, len(feature.tokens) - 1)):
                span_logits[idx, j - i] = result.span_logits[i, j]
                start2end[idx, j - i] = idx + j - i
            idx += 1

    word2char_start = np.zeros([start.shape[0]], dtype=np.int32)
    word2char_end = np.zeros([start.shape[0]], dtype=np.int32)

    sep = ' [PAR] '
    full_text = ""
    prev_example = None
    word_pos = 0
    for feature in features:
        example = id2example[feature.unique_id]
        if prev_example is not None and feature.doc_span_index == 0:
            full_text = full_text + ' '.join(prev_example.doc_tokens) + sep

        for i in range(1, len(feature.tokens) - 1):
            _, start_pos, _ = get_final_text_(example, feature, i, min(len(feature.tokens) - 2, i + 1), do_lower_case,
                                              verbose_logging)
            _, _, end_pos = get_final_text_(example, feature, max(1, i - 1), i, do_lower_case,
                                            verbose_logging)
            start_pos += len(full_text)
            end_pos += len(full_text)
            word2char_start[word_pos] = start_pos
            word2char_end[word_pos] = end_pos
            word_pos += 1
        prev_example = example
    full_text = full_text + ' '.join(prev_example.doc_tokens)

    metadata = {'did': prev_example.doc_idx, 'context': full_text, 'title': prev_example.title,
                'start': start, 'end': end, 'span_logits': span_logits,
                'start2end': start2end,
                'word2char_start': word2char_start, 'word2char_end': word2char_end,
                'filter_start': fs, 'filter_end': fe}
    if split_by_para:
        metadata['pid'] = prev_example.pid

    return metadata


def filter_metadata(metadata, threshold):
    start_idxs, = np.where(metadata['filter_start'] > threshold)
    end_idxs, = np.where(metadata['filter_end'] > threshold)
    end_long2short = {long: short for short, long in enumerate(end_idxs)}

    metadata['word2char_start'] = metadata['word2char_start'][start_idxs]
    metadata['word2char_end'] = metadata['word2char_end'][end_idxs]
    metadata['start'] = metadata['start'][start_idxs]
    metadata['end'] = metadata['end'][end_idxs]
    metadata['span_logits'] = metadata['span_logits'][start_idxs]
    metadata['start2end'] = metadata['start2end'][start_idxs]
    for i, each in enumerate(metadata['start2end']):
        for j, long in enumerate(each.tolist()):
            metadata['start2end'][i, j] = end_long2short[long] if long in end_long2short else -1

    return metadata


def compress_metadata(metadata, offset, scale):
    for key in ['start', 'end']:
        metadata[key] = float_to_int8(metadata[key], offset, scale)
    return metadata


def write_hdf5(all_examples, all_features, all_results,
               max_answer_length, do_lower_case, hdf5_path, filter_threshold, verbose_logging, offset=None, scale=None,
               split_by_para=False, use_sparse=False):
    assert len(all_examples) > 0

    import h5py
    from multiprocessing import Process
    from time import time

    id2feature = {feature.unique_id: feature for feature in all_features}
    id2example = {id_: all_examples[id2feature[id_].example_index] for id_ in id2feature}

    # Separating writing part for potentially multi-processed dumping
    def add(inqueue_, outqueue_):
        for id2example_, features_, results_, split_by_para in iter(inqueue_.get, None):
            metadata = get_metadata(id2example_, features_, results_, max_answer_length, do_lower_case, verbose_logging,
                                    split_by_para)
            metadata = filter_metadata(metadata, filter_threshold)
            outqueue_.put(metadata)

    def write(outqueue_):
        with h5py.File(hdf5_path) as f:
            while True:
                metadata = outqueue_.get()
                if metadata:
                    did = str(metadata['did'])
                    if did in f:
                        if not split_by_para:
                            print('%s exists; skipping' % did)
                            continue
                        dg = f[did]
                    else:
                        dg = f.create_group(did)
                    if split_by_para:
                        pid = str(metadata['pid'])
                        if pid in dg:
                            print('%s %s exists; skipping' % (did, pid))
                            continue
                        dg = dg.create_group(pid)

                    dg.attrs['context'] = metadata['context']
                    dg.attrs['title'] = metadata['title']
                    if offset is not None:
                        metadata = compress_metadata(metadata, offset, scale)
                        dg.attrs['offset'] = offset
                        dg.attrs['scale'] = scale
                    dg.create_dataset('start', data=metadata['start'])
                    dg.create_dataset('end', data=metadata['end'])
                    # dg.create_dataset('sparse', data=metadata['sparse'])
                    dg.create_dataset('span_logits', data=metadata['span_logits'])
                    dg.create_dataset('start2end', data=metadata['start2end'])
                    dg.create_dataset('word2char_start', data=metadata['word2char_start'])
                    dg.create_dataset('word2char_end', data=metadata['word2char_end'])

                else:
                    break

    features = []
    results = []
    inqueue = Queue()
    outqueue = Queue()
    write_p = Process(target=write, args=(outqueue, ))
    p = Process(target=add, args=(inqueue, outqueue))
    write_p.start()
    p.start()

    start_time = time()
    for count, result in enumerate(tqdm(all_results, total=len(all_features))):
        example = id2example[result.unique_id]
        feature = id2feature[result.unique_id]
        if split_by_para:
            condition = len(features) > 0 and feature.doc_span_index == 0
        else:
            condition = len(features) > 0 and example.pid == 0 and feature.doc_span_index == 0

        if condition:
            in_ = (id2example, features, results, split_by_para)
            inqueue.put(in_)
            # add(id2example, features, results, split_by_para)
            features = [feature]
            results = [result]
        else:
            features.append(feature)
            results.append(result)
        if count % 500 == 0:
            print('%d/%d at %.1f' % (count + 1, len(all_features), time() - start_time))
    in_ = (id2example, features, results, split_by_para)
    inqueue.put(in_)
    inqueue.put(None)
    p.join()
    outqueue.put(None)
    write_p.join()


def get_question_results(question_examples, query_eval_features, question_dataloader, device, model):
    id2feature = {feature.unique_id: feature for feature in query_eval_features}
    id2example = {id_: question_examples[id2feature[id_].example_index] for id_ in id2feature}
    for (input_ids_, input_mask_, example_indices) in question_dataloader:
        input_ids_ = input_ids_.to(device)
        input_mask_ = input_mask_.to(device)
        with torch.no_grad():
            batch_start, batch_end, batch_span_logits, batch_sparse = model(query_ids=input_ids_,
                                                              query_mask=input_mask_)
        for i, example_index in enumerate(example_indices):
            start = batch_start[i].detach().cpu().numpy().astype(np.float16)
            end = batch_end[i].detach().cpu().numpy().astype(np.float16)
            span_logit = batch_span_logits[i].detach().cpu().numpy().astype(np.float16)
            query_eval_feature = query_eval_features[example_index.item()]
            unique_id = int(query_eval_feature.unique_id)
            qas_id = id2example[unique_id].qas_id
            yield QuestionResult(qas_id=qas_id,
                                 start=start,
                                 end=end,
                                 span_logit=span_logit)


def write_question_results(question_results, path):
    import h5py
    with h5py.File(path, 'w') as f:
        for question_result in question_results:
            data = np.concatenate([question_result.start, question_result.end, question_result.span_logit], -1)
            f.create_dataset(question_result.qas_id, data=data)


def convert_question_features_to_dataloader(query_eval_features, fp16, local_rank, predict_batch_size):
    all_input_ids_ = torch.tensor([f.input_ids for f in query_eval_features], dtype=torch.long)
    all_input_mask_ = torch.tensor([f.input_mask for f in query_eval_features], dtype=torch.long)
    all_example_index_ = torch.arange(all_input_ids_.size(0), dtype=torch.long)
    if fp16:
        all_input_ids_, all_input_mask_ = tuple(t.half() for t in (all_input_ids_, all_input_mask_))

    question_data = TensorDataset(all_input_ids_, all_input_mask_, all_example_index_)

    if local_rank == -1:
        question_sampler = SequentialSampler(question_data)
    else:
        question_sampler = DistributedSampler(question_data)
    question_dataloader = DataLoader(question_data, sampler=question_sampler, batch_size=predict_batch_size)
    return question_dataloader


def get_final_text_(example, feature, start_index, end_index, do_lower_case, verbose_logging):
    tok_tokens = feature.tokens[start_index:(end_index + 1)]
    orig_doc_start = feature.token_to_orig_map[start_index]
    orig_doc_end = feature.token_to_orig_map[end_index]
    orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
    tok_text = " ".join(tok_tokens)

    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    orig_text = " ".join(orig_tokens)
    full_text = " ".join(example.doc_tokens)

    start_pos, end_pos = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
    offset = sum(len(token) + 1 for token in example.doc_tokens[:orig_doc_start])

    return full_text, offset + start_pos, offset + end_pos


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.
    default_out = 0, len(orig_text)

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return default_out
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return default_out

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return default_out

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return default_out

    # output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return orig_start_position, orig_end_position + 1


def float_to_int8(num, offset, factor):
    out = (num - offset) * factor
    out = out.clip(-128, 127)
    out = np.round(out).astype(np.int8)
    return out
