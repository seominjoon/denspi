import collections
import json
import logging
import os
import shutil

import numpy as np
import six
from tqdm import tqdm as tqdm_

import tokenization

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def tqdm(*args, mininterval=5.0, **kwargs):
    return tqdm_(*args, mininterval=mininterval, **kwargs)


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
    vec_count = 0

    predictions = {}
    scores = {}

    for result in tqdm(all_results, total=len(all_features)):
        feature = id2feature[result.unique_id]
        example = id2example[result.unique_id]
        id_ = example.qas_id

        word_count += len(feature.tokens)

        for start_index in range(len(feature.tokens)):
            for end_index in range(start_index, min(len(feature.tokens), start_index + max_answer_length - 1)):
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue

                vec_count += 1

                score = result.all_logits[start_index, end_index]
                if id_ not in scores or score > scores[id_]:
                    orig_text, start_pos, end_pos = get_final_text_(example, feature, start_index, end_index,
                                                                    do_lower_case, verbose_logging)
                    phrase = orig_text[start_pos:end_pos]
                    predictions[id_] = phrase
                    scores[id_] = score

    print('num vecs=%d, num_words=%d, nvpw=%.4f' % (vec_count, word_count, vec_count / word_count))

    with open(output_prediction_file, 'w') as fp:
        json.dump(predictions, fp)


def get_documents(all_examples, all_features, all_results, threshold,
                  max_answer_length, do_lower_case, verbose_logging):
    id2feature = {feature.unique_id: feature for feature in all_features}
    id2example = {id_: all_examples[id2feature[id_].example_index] for id_ in id2feature}

    for result in all_results:
        feature = id2feature[result.unique_id]
        example = id2example[result.unique_id]

        vectors = []
        phrases = []
        spans = []
        for start_index in range(len(feature.tokens)):
            for end_index in range(start_index, min(len(feature.tokens), start_index + max_answer_length - 1)):
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue
                span_logit = result.span_logits[start_index, end_index]
                filter_logit = result.filter_logits[start_index, end_index]
                if filter_logit < threshold:
                    continue

                vector = np.concatenate([np.array([span_logit]), result.start[start_index], result.end[end_index]])
                orig_text, start_pos, end_pos = get_final_text_(example, feature, start_index, end_index, do_lower_case,
                                                                verbose_logging)
                phrase = orig_text[start_pos:end_pos]
                span = [start_pos, end_pos]
                vectors.append(vector)
                phrases.append(phrase)
                spans.append(span)
        metadata = {'title': example.title, 'pid': example.pid, 'sid': feature.doc_span_index,
                    'context': orig_text, 'answer_spans': spans, 'doc_idx': example.doc_idx}
        yield metadata, vectors, phrases


def get_metadata(example, features, results, max_answer_length, do_lower_case, verbose_logging):
    start2char = []
    end2char = []
    start = np.concatenate([result.start[:len(feature.tokens)] for feature, result in zip(features, results)], axis=0)
    end = np.concatenate([result.end[:len(feature.tokens)] for feature, result in zip(features, results)], axis=0)
    span_logits = -1e9 * np.ones([np.shape(start)[0], max_answer_length])
    idx = 0
    for feature, result in zip(features, results):
        for i in range(len(feature.tokens)):
            for j in range(i, min(i + max_answer_length, len(feature.tokens))):
                span_logits[idx, j - i] = result.span_logits[i, j]
            idx += 1

    full_text = ' '.join(example.doc_tokens)
    offset = 0
    for feature in features:
        for idx in range(len(feature.tokens)):
            if idx not in feature.token_to_orig_map or not feature.token_is_max_context.get(idx, False):
                start2char.append(-1)
                end2char.append(-1)
            else:
                _, start_pos, end_pos = get_final_text_(example, feature, idx, idx, do_lower_case,
                                                        verbose_logging)
                start2char.append(offset + start_pos)
                end2char.append(offset + end_pos)
        offset += len(feature.tokens)

    metadata = {'did': example.doc_idx, 'pid': example.pid,
                'context': full_text, 'title': example.title, 'start2char': start2char, 'end2char': end2char,
                'start': start, 'end': end, 'span_logits': span_logits}
    return metadata


def write_hdf5(all_examples, all_features, all_results,
               max_answer_length, do_lower_case, hdf5_path, verbose_logging):
    assert len(all_examples) > 0

    import h5py
    f = h5py.File(hdf5_path, 'w')

    id2feature = {feature.unique_id: feature for feature in all_features}
    id2example = {id_: all_examples[id2feature[id_].example_index] for id_ in id2feature}

    def add(example_, features_, results_):
        metadata = get_metadata(example_, features_, results_, max_answer_length, do_lower_case, verbose_logging)
        did, pid = str(metadata['did']), str(metadata['pid'])
        dg = f[did] if did in f else f.create_group(did)
        pd = dg.create_group(pid)
        pd.create_dataset('start', data=metadata['start'])
        pd.create_dataset('end', data=metadata['end'])
        pd.create_dataset('span_logits', data=metadata['span_logits'])
        pd.create_dataset('start2char', data=metadata['start2char'])
        pd.create_dataset('end2char', data=metadata['end2char'])
        pd.attrs['context'] = metadata['context']

    prev_example = None
    features = []
    results = []
    for result in all_results:
        example = id2example[result.unique_id]
        feature = id2feature[result.unique_id]
        if len(features) > 0 and feature.doc_span_index == 0:
            # consume features
            add(prev_example, features, results)
            features = [feature]
            results = [result]
        else:
            features.append(feature)
            results.append(result)
        prev_example = example
    add(prev_example, features, results)

    f.close()


def write_context(all_examples, all_features, all_results, threshold,
                  max_answer_length, do_lower_case, zip_dir, verbose_logging,
                  one_file_per_doc=False):
    if one_file_per_doc:
        embed_dir = zip_dir
    else:
        embed_dir = 'context_emb/'
        os.makedirs(embed_dir)
    word_count = sum(len(feature.tokens) for feature in all_features)
    vec_count = 0

    # track doc stride index for each question. if not visited, set flag
    doc_tracker = set()

    # Used for storing vecs from previous runs.
    # For docs from the same para or for one_file_per_doc
    vectors_to_store = []
    phrases_to_store = []
    paras_to_store = []
    spans_to_store = []

    for i, (metadata, vectors, phrases) in tqdm(enumerate(get_documents(all_examples, all_features, all_results,
                                                                        threshold, max_answer_length, do_lower_case,
                                                                        verbose_logging)),
                                                total=len(all_features),
                                                desc=zip_dir):
        feature = all_features[i]
        title = metadata['title']
        doc_idx = metadata['doc_idx']
        context = metadata['context']
        pid = metadata['pid']
        spans = metadata['answer_spans']
        if one_file_per_doc:
            spans = [(pid, span) for span in spans]
        name = str(doc_idx).zfill(4) if one_file_per_doc else '%s_%d' % (title, pid)
        name = name.replace('/', '_')  # some titles have slashes
        npz_path = os.path.join(embed_dir, '%s.npz' % name)
        json_path = os.path.join(embed_dir, '%s.json' % name)
        metadata_path = os.path.join(embed_dir, '%s.metadata' % name)

        is_last = i == len(all_features) - 1
        is_last_stride = is_last or all_features[i + 1].example_index != feature.example_index
        is_last_para = is_last or (all_examples[all_features[i + 1].example_index].title !=
                                   all_examples[feature.example_index].title)

        vec_count += len(vectors)

        vectors_to_store.extend(vectors)
        phrases_to_store.extend(phrases)
        paras_to_store.append(context)
        spans_to_store.extend(spans)

        metadata_to_store = {'title': title, 'context': paras_to_store, 'answer_spans': spans_to_store}
        if one_file_per_doc:
            store = is_last_para
        else:
            store = is_last_stride
            metadata_to_store['pid'] = pid

        if store:
            matrix = np.stack(vectors_to_store, 0) if len(vectors_to_store) > 0 else np.array([])
            np.savez_compressed(npz_path, matrix)
            if not one_file_per_doc:
                with open(json_path, 'w') as fp:
                    json.dump(phrases_to_store, fp)
            with open(metadata_path, 'w') as fp:
                json.dump(metadata_to_store, fp)
            vectors_to_store = []
            phrases_to_store = []
            paras_to_store = []
            spans_to_store = []

    print('num vecs=%d, num_words=%d, nvpw=%.4f' % (vec_count, word_count, vec_count / word_count))

    if not one_file_per_doc:
        # archiving
        print('archiving at %s.zip' % zip_dir)
        shutil.make_archive(zip_dir, 'zip', embed_dir)
        shutil.rmtree(embed_dir)


def get_questions(all_examples, all_features, all_results):
    id2feature = {feature.unique_id: feature for feature in all_features}
    id2example = {id_: all_examples[id2feature[id_].example_index] for id_ in id2feature}

    for result in all_results:
        example = id2example[result.unique_id]
        _1 = np.array([[1]])  # change this!
        matrix = np.concatenate([_1, result.start, result.end], axis=1)
        yield example, matrix


def write_question(all_examples, all_features, all_results, zip_dir):
    embed_dir = 'question_emb/'
    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir)

    for example, matrix in tqdm(get_questions(all_examples, all_features, all_results), total=len(all_features)):
        npz_path = os.path.join(embed_dir, '%s.npz' % example.qas_id)
        np.savez_compressed(npz_path, matrix)

    # archiving
    print('archiving at %s.zip' % zip_dir)
    shutil.make_archive(zip_dir, 'zip', embed_dir)
    shutil.rmtree(embed_dir)


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
