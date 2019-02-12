import json
import os
from collections import namedtuple
from time import time

import h5py
import numpy as np
import faiss
from tqdm import tqdm


def int8_to_float(num, offset, factor):
    return num.astype(np.float32) / factor + offset


class DocumentPhraseMIPS(object):
    def __init__(self, document_index, phrase_index, max_answer_length, doc_score_cf, start_index=None, ip2l2=False):
        super(DocumentPhraseMIPS, self).__init__()
        self.document_index = document_index
        self.phrase_index = phrase_index
        self.max_answer_length = max_answer_length
        self.doc_score_cf = doc_score_cf
        self.start_index = start_index
        self.ip2l2 = ip2l2
        if start_index is not None:
            if os.path.exists(start_index):
                self.start_index = faiss.read_index(start_index)
                with open('id2dp.json', 'r') as fp:
                    self.id2dp = json.load(fp)
            else:
                if self.ip2l2:
                    max_norm = 0
                    for key, group in tqdm(self.phrase_index.items(), desc='find max norm'):
                        max_norm = max(max_norm, np.linalg.norm(group['start'][:], axis=1).max())
                    print('max norm:', max_norm)

                    offset = 0
                    self.id2dp = {}
                    starts = []
                    for key, group in tqdm(self.phrase_index.items(), desc='faiss indexing'):
                        ids = np.arange(group['start'].shape[0]) + offset
                        start = int8_to_float(group['start'][:], group.attrs['offset'], group.attrs['scale'])
                        norms = np.linalg.norm(start, axis=1, keepdims=True)
                        consts = np.sqrt(max_norm ** 2 - norms ** 2)
                        start = np.concatenate([consts, start], axis=1)
                        starts.append(start)

                        for i, id_ in enumerate(ids):
                            self.id2dp[int(id_)] = (key, i)
                        offset += ids.shape[0]

                    train_data = np.concatenate(starts, axis=0)
                    # self.start_index = faiss.IndexFlatL2(481)
                    print('index ntotal:', train_data.shape[0])
                    # self.start_index = faiss.index_factory(481, "IVF4096,PQ37")
                    self.start_index = faiss.index_factory(481, "IVF4096_HNSW32,SQ8")
                    print('training')
                    self.start_index.train(train_data)
                    self.start_index.add(train_data)
                    faiss.write_index(self.start_index, start_index)
                    with open('id2dp.json', 'w') as fp:
                        json.dump(self.id2dp, fp)

                else:
                    self.start_index = faiss.IndexFlatIP(480)
                    offset = 0
                    self.id2dp = {}

                    for key, group in tqdm(self.phrase_index.items(), desc='faiss indexing'):
                        ids = np.arange(group['start'].shape[0]) + offset
                        start = int8_to_float(group['start'][:], group.attrs['offset'], group.attrs['scale'])

                        self.start_index.add(start)
                        for i, id_ in enumerate(ids):
                            self.id2dp[id_] = (key, i)
                        offset += ids.shape[0]
                    print('index ntotal:', self.start_index.ntotal)
            self.start_index.nprobe = 30

    def search_document(self, query, top_k=5):
        res = query * self.document_index

        if len(res.data) <= top_k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, top_k)[0:top_k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_indices = res.indices[o_sort]
        doc_scores = res.data[o_sort]
        out = [{'doc_idx': doc_idx, 'doc_score': doc_score}
               for doc_idx, doc_score in zip(doc_indices.tolist(), doc_scores.tolist())]
        return out

    def search_start(self, query_start, top_k=5):
        assert self.start_index is not None
        query_start = query_start.astype(np.float32)
        if self.ip2l2:
            query_start = np.concatenate([np.zeros([query_start.shape[0], 1]).astype(np.float32), query_start], axis=1)
        D, I = self.start_index.search(query_start, top_k)
        results = []
        for d, i in zip(D, I):
            doc_idxs, start_idxs = zip(*[self.id2dp[str(id_)] for id_ in i])
            results.append((doc_idxs, start_idxs, D))
        return results

    def search_phrase(self, doc_idx, query, top_k=5, doc_score=0.0, para_idx=None, start_idx=None):
        t0 = time.time()

        if str(doc_idx) not in self.phrase_index:
            print('phrase index keys:', self.phrase_index.keys())
            return []

        group = self.phrase_index[str(doc_idx)]
        if para_idx is not None:
            group = group[str(para_idx)]

        start, end, span_logits, start2end, word2char_start, word2char_end = [
            group[key][:] for key in
            ['start', 'end', 'span_logits', 'start2end', 'word2char_start', 'word2char_end']]

        # print(start.min(), start.max(), end.min(), end.max())

        context = group.attrs['context']
        title = group.attrs['title']

        if 'offset' in group.attrs:
            start = int8_to_float(start, group.attrs['offset'], group.attrs['scale'])
            end = int8_to_float(end, group.attrs['offset'], group.attrs['scale'])

        query_start, query_end, query_span_logit = query

        PhraseResult = namedtuple('PhraseResult', ('score', 'start_score', 'end_score', 'span_score',
                                                   'start_idx', 'end_idx'))
        if start_idx is None:
            start_scores = np.matmul(start, np.array(query_start).transpose()).squeeze(1)
            if top_k > 0:
                best_start_pairs = sorted(enumerate(start_scores.tolist()), key=lambda item: -item[1])[:top_k]
            else:
                best_start_pairs = enumerate(start_scores.tolist())
        else:
            start_scores = np.matmul(start[start_idx:start_idx + 1], np.array(query_start).transpose()).squeeze(1)
            best_start_pairs = [(start_idx, start_scores[0])]

        query_span_logit = float(query_span_logit[0][0])

        results = []
        for start_idx, start_score in best_start_pairs:
            max_result = None
            max_score = -1e9
            for i, end_idx in enumerate(start2end[start_idx]):
                if i >= self.max_answer_length:
                    break
                if end_idx < 0:
                    continue
                span_logit = span_logits[start_idx, i].item()
                end_score = np.matmul(end[end_idx:end_idx + 1], np.array(query_end).transpose()).squeeze(1).squeeze(0)
                span_score = query_span_logit * span_logit
                score = self.doc_score_cf * doc_score + start_score + end_score + span_score
                if score > max_score:
                    max_result = PhraseResult(score, start_score, end_score, span_score, start_idx, end_idx)
                    max_score = score
            if max_result is not None:
                results.append(max_result)
        results = sorted(results, key=lambda item: -item.score)

        # Non-maximal suppression (might not be needed)
        new_results = []
        for result in results:
            include = True
            for new_result in new_results:
                if overlap(word2char_start, word2char_end,
                           result.start_idx, result.end_idx, new_result.start_idx, new_result.end_idx):
                    include = False
                    break
            if include:
                new_results.append(result)

        results = new_results[:top_k]
        out = [{'context': context,
                'title': title,
                'doc_idx': doc_idx,
                'doc_score': doc_score,
                'start_pos': word2char_start[result.start_idx].item(),
                'end_pos': word2char_end[result.end_idx].item(),
                'answer': context[word2char_start[result.start_idx].item():word2char_end[result.end_idx].item()],
                'start_score': result.start_score,
                'end_score': result.end_score,
                'span_score': result.span_score,
                'phrase_score': result.score - doc_score,
                'score': result.score} for result in results]
        t3 = time.time()
        # print('Finding answer: %dms' % int(1000 * (t3 - t2)))
        out = [adjust(each) for each in out]
        # out = [each for each in out if len(each['context']) > 100 and each['score'] >= 30]

        return out

    def search_phrase_global(self, phrase_query, top_k=5):
        doc_idxs, start_idxs, scores = self.search_start(phrase_query[0], top_k=top_k)
        phrase_rets = sum([self.search_phrase(doc_idx, phrase_query, top_k=top_k)
                           for doc_idx in doc_idxs], [])
        phrase_rets = sorted(phrase_rets, key=lambda ret: -ret['score'])[:top_k]
        return phrase_rets

    def search(self, doc_query, phrase_query, top_k_docs=5, top_k_phrases=5):
        document_rets = self.search_document(doc_query, top_k_docs)
        phrase_rets = sum([self.search_phrase(ret['doc_idx'], phrase_query, top_k=top_k_phrases,
                                              doc_score=ret['doc_score'])
                           for ret in document_rets], [])
        phrase_rets = sorted(phrase_rets, key=lambda ret: -ret['score'])[:top_k_phrases]
        return phrase_rets


def adjust(each):
    last = each['context'].rfind(' [PAR] ', 0, each['start_pos'])
    last = 0 if last == -1 else last + len(' [PAR] ')
    next = each['context'].find(' [PAR] ', each['end_pos'])
    next = len(each['context']) if next == -1 else next
    each['context'] = each['context'][last:next]
    each['start_pos'] -= last
    each['end_pos'] -= last
    return each


def overlap(t1, t2, a1, a2, b1, b2):
    if t1[b1] > t2[a2] or t1[a1] > t2[b2]:
        return False
    return True


def id2idxs(id_):
    return (id_ / 1e6).astype(np.int64), id_ % int(1e6)


def idxs2id(doc_idx, start_idx):
    return doc_idx * int(1e6) + start_idx


class MIPS(object):
    def __init__(self, phrase_index_path, start_index_path, max_answer_length, index_factory='IVF4096,Flat',
                 load_to_memory=False):
        super(MIPS, self).__init__()
        if load_to_memory:
            self.phrase_index = h5py.File(phrase_index_path, 'r', driver="core")
        else:
            self.phrase_index = h5py.File(phrase_index_path, 'r')
        self.max_answer_length = max_answer_length
        if os.path.exists(start_index_path):
            self.start_index = faiss.read_index(start_index_path)
            with open('id2dp.json', 'r') as fp:
                self.id2dp = json.load(fp)
        else:
            max_norm = 0
            for key, group in tqdm(self.phrase_index.items(), desc='find max norm'):
                max_norm = max(max_norm, np.linalg.norm(group['start'][:], axis=1).max())
            print('max norm:', max_norm)

            ids = []
            starts = []
            for key, group in tqdm(self.phrase_index.items(), desc='faiss indexing'):
                cur_ids = idxs2id(int(key), np.arange(group['start'].shape[0]))
                start = int8_to_float(group['start'][:], group.attrs['offset'], group.attrs['scale'])
                norms = np.linalg.norm(start, axis=1, keepdims=True)
                consts = np.sqrt(max_norm ** 2 - norms ** 2)
                start = np.concatenate([consts, start], axis=1)
                starts.append(start)
                ids.extend(cur_ids)

            train_data = np.concatenate(starts, axis=0)
            self.start_index = faiss.index_factory(481, index_factory)
            print('training start index')
            self.start_index.train(train_data)
            self.start_index.add_with_ids(train_data, np.array(ids))
            faiss.write_index(self.start_index, start_index_path)

    def search_start(self, query_start, doc_idxs=None, para_idxs=None, top_k=5, nprobe=16):
        # TODO : use doc_idxs for SQuAD eval
        assert self.start_index is not None
        query_start = query_start.astype(np.float32)
        query_start = np.concatenate([np.zeros([query_start.shape[0], 1]).astype(np.float32), query_start], axis=1)
        self.start_index.nprobe = nprobe
        start_scores, I = self.start_index.search(query_start, top_k)
        doc_idxs, phrase_idxs = id2idxs(I)
        return start_scores, doc_idxs, phrase_idxs

    def search_phrase(self, query, doc_idxs, start_idxs, para_idxs=None, start_scores=None):
        # query = [Q, d]
        # doc_idxs = [Q]
        # start_idxs = [Q]
        # para_idxs = [Q]
        query_start, query_end, query_span_logit = query[:, :480], query[:, 480:960], query[:, 960:961]

        groups = [self.phrase_index[str(doc_idx)] for doc_idx in doc_idxs]
        if para_idxs is not None:
            groups = [group[str(para_idx)] for group, para_idx in zip(groups, para_idxs)]

        def dequant(group, input_):
            if 'offset' in group.attrs:
                return int8_to_float(input_, group.attrs['offset'], group.attrs['scale'])
            return input_

        if start_scores is None:
            start = np.stack([group['start'][start_idx, :]
                              for group, start_idx in zip(groups, start_idxs)], 0)  # [Q, d]
            start = dequant(groups[0], start)
            start_scores = np.sum(query_start * start, 1)  # [Q]

        ends = [group['end'][:] for group in groups]
        spans = [group['span_logits'][:] for group in groups]

        end_idxs = [group['start2end'][start_idx, :] for group, start_idx in zip(groups, start_idxs)]  # [Q, L]
        end_mask = -1e9 * (np.array(end_idxs) < 0)  # [Q, L]
        end = np.stack([[each_end[each_end_idx, :] for each_end_idx in each_end_idxs]
                        for each_end, each_end_idxs in zip(ends, end_idxs)], 0)  # [Q, L, d]
        end = dequant(groups[0], end)
        span = np.stack([[each_span[start_idx, i] for i in range(len(each_end_idxs))]
                         for each_span, start_idx, each_end_idxs in zip(spans, start_idxs, end_idxs)], 0)  # [Q, L]

        end_scores = np.sum(np.expand_dims(query_end, 1) * end, 2)  # [Q, L]
        span_scores = query_span_logit * span  # [Q, L]
        scores = np.expand_dims(start_scores, 1) + end_scores + span_scores + end_mask  # [Q, L]
        pred_end_idxs = np.stack([each[idx] for each, idx in zip(end_idxs, np.argmax(scores, 1))], 0)  # [Q]
        max_scores = np.max(scores, 1)

        out = [{'context': group.attrs['context'],
                'title': group.attrs['title'],
                'doc_idx': doc_idx,
                'start_pos': group['word2char_start'][start_idx].item(),
                'end_pos': group['word2char_end'][end_idx].item(),
                'score': score}
               for doc_idx, group, start_idx, end_idx, score in zip(doc_idxs, groups, start_idxs,
                                                                    pred_end_idxs, max_scores)]

        for each in out:
            each['answer'] = each['context'][each['start_pos']:each['end_pos']]

        out = [adjust(each) for each in out]
        # out = [each for each in out if len(each['context']) > 100 and each['score'] >= 30]

        return out

    def search(self, query, top_k=5, nprobe=64):
        num_queries = query.shape[0]
        query_start = query[:, :480]
        start_scores, doc_idxs, start_idxs = self.search_start(query_start, top_k=top_k, nprobe=nprobe)

        # reshape
        query = np.reshape(np.tile(np.expand_dims(query, 1), [1, top_k, 1]), [-1, query.shape[1]])
        idxs = np.reshape(np.tile(np.expand_dims(np.arange(num_queries), 1), [1, top_k]), [-1])
        start_scores = np.reshape(start_scores, [-1])
        doc_idxs = np.reshape(doc_idxs, [-1])
        start_idxs = np.reshape(start_idxs, [-1])

        out = self.search_phrase(query, doc_idxs, start_idxs)
        new_out = [[] for _ in range(num_queries)]
        for idx, each_out in zip(idxs, out):
            new_out[idx].append(each_out)
        for i in range(len(new_out)):
            new_out[i] = sorted(new_out[i], key=lambda each_out: -each_out['score'])

        return new_out
