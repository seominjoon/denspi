import argparse
import json
import os
import random
from collections import namedtuple
from time import time

import h5py
import numpy as np
import faiss
import torch
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


class MIPS(object):
    def __init__(self, phrase_dump_dir, start_index_path, idx2id_path, max_answer_length, para=False,
                 num_dummy_zeros=0, cuda=False):
        if os.path.isdir(phrase_dump_dir):
            self.phrase_dump_paths = [os.path.join(phrase_dump_dir, name) for name in os.listdir(phrase_dump_dir)]
            dump_names = [os.path.splitext(os.path.basename(path))[0] for path in self.phrase_dump_paths]
            self.dump_ranges = [list(map(int, name.split('-'))) for name in dump_names]
        else:
            self.phrase_dump_paths = [phrase_dump_dir]
        self.phrase_dumps = [h5py.File(path, 'r') for path in self.phrase_dump_paths]
        self.max_answer_length = max_answer_length
        self.para = para

        print('reading %s' % start_index_path)
        self.start_index = faiss.read_index(start_index_path)
        self.idx_f = h5py.File(idx2id_path, 'r')
        self.has_offset = not 'doc' in self.idx_f
        # with h5py.File(idx2id_path, 'r') as f:
        #     self.idx2doc_id = f['doc'][:]
        #     self.idx2para_id = f['para'][:]
        #     self.idx2word_id = f['word'][:]

        self.num_dummy_zeros = num_dummy_zeros
        self.cuda = cuda

    def get_idxs(self, I):
        if self.has_offset:
            offsets = (I / 1e8).astype(np.int64) * int(1e8)
            idxs = I % int(1e8)
            doc = np.array([[self.idx_f[str(offset)]['doc'][idx] for offset, idx in zip(oo, ii)] for oo, ii in zip(offsets, idxs)])
            word = np.array([[self.idx_f[str(offset)]['word'][idx] for offset, idx in zip(oo, ii)] for oo, ii in zip(offsets, idxs)])
            if self.para:
                para = np.array([[self.idx_f[str(offset)]['para'][idx] for offset, idx in zip(oo, ii)] for oo, ii in zip(offsets, idxs)])
            else:
                para = None
        else:
            doc = np.array([[self.idx_f['doc'][idx] for idx in ii] for ii in I])
            word = np.array([[self.idx_f['word'][idx] for idx in ii] for ii in I])
            if self.para:
                para = np.array([[self.idx_f['para'][idx] for idx in ii] for ii in I])
            else:
                para = None

        return doc, para, word

    def close(self):
        for phrase_dump in self.phrase_dumps:
            phrase_dump.close()

    def get_doc_group(self, doc_idx):
        if len(self.phrase_dumps) == 1:
            return self.phrase_dumps[0][str(doc_idx)]
        for dump_range, dump in zip(self.dump_ranges, self.phrase_dumps):
            if dump_range[0] * 1000 <= int(doc_idx) < dump_range[1] * 1000:
                return dump[str(doc_idx)]
        raise ValueError('%d not found in dump list' % int(doc_idx))

    def search_start(self, query_start, doc_idxs=None, para_idxs=None, top_k=5, nprobe=16):
        # doc_idxs = [Q], para_idxs = [Q]
        assert self.start_index is not None
        query_start = query_start.astype(np.float32)

        if doc_idxs is None:
            query_start = np.concatenate([np.zeros([query_start.shape[0], 1]).astype(np.float32),
                                          query_start], axis=1)
            if self.num_dummy_zeros > 0:
                query_start = np.concatenate([query_start, np.zeros([query_start.shape[0], self.num_dummy_zeros],
                                                                    dtype=query_start.dtype)], axis=1)
            self.start_index.nprobe = nprobe
            start_scores, I = self.start_index.search(query_start, top_k)

            doc_idxs, para_idxs, start_idxs = self.get_idxs(I)
            print(doc_idxs, para_idxs)
        else:
            groups = [self.get_doc_group(doc_idx)[str(para_idx)] for doc_idx, para_idx in zip(doc_idxs, para_idxs)]
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
        bs = int((query.shape[1] - 1) / 2)
        query_start, query_end, query_span_logit = query[:, :bs], query[:, bs:2 * bs], query[:, -1:]

        groups = [self.get_doc_group(doc_idx) for doc_idx in doc_idxs]

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
            if self.cuda:
                cuda0 = torch.device('cuda:0')
                start = torch.FloatTensor(start).to(cuda0)
                query_start = torch.FloatTensor(query_start).to(cuda0)
                start_scores = (query_start * start).sum(1).cpu().numpy()
            else:
                start_scores = np.sum(query_start * start, 1)  # [Q]

        ends = [group['end'][:] for group in groups]
        spans = [group['span_logits'][:] for group in groups]

        default_end = np.zeros(bs).astype(np.float32)
        end_idxs = [group['start2end'][start_idx, :] for group, start_idx in zip(groups, start_idxs)]  # [Q, L]
        end_mask = -1e9 * (np.array(end_idxs) < 0)  # [Q, L]

        end = np.stack([[each_end[each_end_idx, :] if each_end.size > 0 else default_end
                         for each_end_idx in each_end_idxs]
                        for each_end, each_end_idxs in zip(ends, end_idxs)], 0)  # [Q, L, d]
        end = dequant(groups[0], end)
        span = np.stack([[each_span[start_idx, i] for i in range(len(each_end_idxs))]
                         for each_span, start_idx, each_end_idxs in zip(spans, start_idxs, end_idxs)], 0)  # [Q, L]

        if self.cuda:
            cuda0 = torch.device('cuda:0')
            end = torch.FloatTensor(end).to(cuda0)
            query_end = torch.FloatTensor(query_end).to(cuda0)
            end_scores = (query_end.unsqueeze(1) * end).sum(2).cpu().numpy()
        else:
            end_scores = np.sum(np.expand_dims(query_end, 1) * end, 2)  # [Q, L]
        span_scores = query_span_logit * span  # [Q, L]
        scores = np.expand_dims(start_scores, 1) + end_scores + span_scores + end_mask  # [Q, L]
        pred_end_idxs = np.stack([each[idx] for each, idx in zip(end_idxs, np.argmax(scores, 1))], 0)  # [Q]
        max_scores = np.max(scores, 1)

        out = [{'context': group.attrs['context'],
                'title': group.attrs['title'],
                'doc_idx': doc_idx,
                'start_pos': group['word2char_start'][start_idx].item(),
                'end_pos': group['word2char_end'][end_idx].item() if len(group['word2char_end']) > 0
                else group['word2char_start'][start_idx].item() + 1,
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
        bs = int((query.shape[1] - 1) / 2)
        query_start = query[:, :bs]
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
