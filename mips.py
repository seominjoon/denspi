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


def id2idxs(id_, para=False):
    if para:
        return (id_ / 1e7).astype(np.int64), ((id_ % int(1e7)) / 1e4).astype(np.int64), id_ % int(1e4)
    return (id_ / 1e6).astype(np.int64), id_ % int(1e6)


def idxs2id(doc_idx, first_idx, second_idx=None):
    if second_idx is None:
        assert np.all(first_idx < 1e6)
        return doc_idx * int(1e6) + first_idx
    assert np.all(second_idx < 1e4)
    assert np.all(first_idx < 1e3)
    return doc_idx * int(1e7) + first_idx * int(1e4) + second_idx


class MIPS(object):
    def __init__(self, phrase_index_path, start_index_path, max_answer_length,
                 para=False, index_factory='IVF4096,SQ8', load_to_memory=False):
        if load_to_memory:
            self.phrase_index = h5py.File(phrase_index_path, 'r', driver="core")
        else:
            self.phrase_index = h5py.File(phrase_index_path, 'r')
        self.max_answer_length = max_answer_length
        self.para = para
        if os.path.exists(start_index_path):
            self.start_index = faiss.read_index(start_index_path)
        else:
            max_norm = 0
            if para:
                for key, doc_group in tqdm(self.phrase_index.items(), desc='find max norm'):
                    for _, group in doc_group.items():
                        max_norm = max(max_norm, np.linalg.norm(group['start'][:], axis=1).max())
            else:
                for key, doc_group in tqdm(self.phrase_index.items(), desc='find max norm'):
                    max_norm = max(max_norm, np.linalg.norm(doc_group['start'][:], axis=1).max())
            print('max norm:', max_norm)

            ids = []
            starts = []
            if para:
                for doc_idx, doc_group in tqdm(self.phrase_index.items(), desc='faiss indexing'):
                    for para_idx, group in doc_group.items():
                        cur_ids = idxs2id(int(doc_idx), int(para_idx), np.arange(group['start'].shape[0]))
                        start = int8_to_float(group['start'][:], group.attrs['offset'], group.attrs['scale'])
                        norms = np.linalg.norm(start, axis=1, keepdims=True)
                        consts = np.sqrt(max_norm ** 2 - norms ** 2)
                        start = np.concatenate([consts, start], axis=1)
                        starts.append(start)
                        ids.extend(cur_ids)
            else:
                for doc_idx, doc_group in tqdm(self.phrase_index.items(), desc='faiss indexing'):
                    cur_ids = idxs2id(int(doc_idx), np.arange(doc_group['start'].shape[0]))
                    start = int8_to_float(doc_group['start'][:], doc_group.attrs['offset'], doc_group.attrs['scale'])
                    norms = np.linalg.norm(start, axis=1, keepdims=True)
                    consts = np.sqrt(max_norm ** 2 - norms ** 2)
                    start = np.concatenate([consts, start], axis=1)
                    starts.append(start)
                    ids.extend(cur_ids)
            ids = np.array(ids)

            train_data = np.concatenate(starts, axis=0)
            self.start_index = faiss.index_factory(481, index_factory)
            print('training start index')
            self.start_index.train(train_data)
            self.start_index.add_with_ids(train_data, ids)
            faiss.write_index(self.start_index, start_index_path)

    def search_start(self, query_start, doc_idxs=None, para_idxs=None, top_k=5, nprobe=16):
        # doc_idxs = [Q], para_idxs = [Q]
        assert self.start_index is not None
        query_start = query_start.astype(np.float32)

        if doc_idxs is None:
            query_start = np.concatenate([np.zeros([query_start.shape[0], 1]).astype(np.float32), query_start], axis=1)
            self.start_index.nprobe = nprobe
            start_scores, I = self.start_index.search(query_start, top_k)
            if self.para:
                doc_idxs, para_idxs, start_idxs = id2idxs(I, para=self.para)  # [Q, K]
            else:
                doc_idxs, start_idxs = id2idxs(I, para=self.para)
        else:
            groups = [self.phrase_index[str(doc_idx)][str(para_idx)] for doc_idx, para_idx in zip(doc_idxs, para_idxs)]
            starts = [group['start'][:, :] for group in groups]
            starts = [int8_to_float(start, groups[0].attrs['offset'], groups[0].attrs['scale']) for start in starts]
            all_scores = [np.squeeze(np.matmul(start, query_start[i:i + 1, :].transpose()), -1)
                          for i, start in enumerate(starts)]
            start_idxs = np.array([scores.argsort()[-top_k:][::-1]
                                   for scores in all_scores])
            start_scores = np.array([scores[idxs] for scores, idxs in zip(all_scores, start_idxs)])
            doc_idxs = np.tile(np.expand_dims(doc_idxs, -1), [1, top_k])
            para_idxs = np.tile(np.expand_dims(para_idxs, -1), [1, top_k])
        return start_scores, doc_idxs, para_idxs, start_idxs

    def search_phrase(self, query, doc_idxs, start_idxs, para_idxs=None, start_scores=None):
        # query = [Q, d]
        # doc_idxs = [Q]
        # start_idxs = [Q]
        # para_idxs = [Q]
        query_start, query_end, query_span_logit = query[:, :480], query[:, 480:960], query[:, 960:961]

        groups = [self.phrase_index[str(doc_idx)] for doc_idx in doc_idxs]
        if self.para:
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
               for doc_idx, group, start_idx, end_idx, score in zip(doc_idxs.tolist(), groups, start_idxs.tolist(),
                                                                    pred_end_idxs.tolist(), max_scores.tolist())]

        for each in out:
            each['answer'] = each['context'][each['start_pos']:each['end_pos']]

        out = [adjust(each) for each in out]
        # out = [each for each in out if len(each['context']) > 100 and each['score'] >= 30]

        return out

    def search(self, query, top_k=5, nprobe=64, doc_idxs=None, para_idxs=None):
        num_queries = query.shape[0]
        query_start = query[:, :480]
        start_scores, doc_idxs, para_idxs, start_idxs = self.search_start(query_start, top_k=top_k, nprobe=nprobe,
                                                                          doc_idxs=doc_idxs, para_idxs=para_idxs)

        # reshape
        query = np.reshape(np.tile(np.expand_dims(query, 1), [1, top_k, 1]), [-1, query.shape[1]])
        idxs = np.reshape(np.tile(np.expand_dims(np.arange(num_queries), 1), [1, top_k]), [-1])
        start_scores = np.reshape(start_scores, [-1])
        doc_idxs = np.reshape(doc_idxs, [-1])
        para_idxs = np.reshape(para_idxs, [-1])
        start_idxs = np.reshape(start_idxs, [-1])

        out = self.search_phrase(query, doc_idxs, start_idxs, para_idxs=para_idxs)
        new_out = [[] for _ in range(num_queries)]
        for idx, each_out in zip(idxs, out):
            new_out[idx].append(each_out)
        for i in range(len(new_out)):
            new_out[i] = sorted(new_out[i], key=lambda each_out: -each_out['score'])

        return new_out
