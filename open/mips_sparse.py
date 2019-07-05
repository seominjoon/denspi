import json
from time import time

import faiss
import numpy as np
import os
import h5py
import re
from drqa.retriever import TfidfDocRanker

from tqdm import tqdm
from scipy.sparse import vstack


def scale_l2_to_ip(l2_scores, max_norm=None, query_norm=None):
    """
    sqrt(m^2 + q^2 - 2qx) -> m^2 + q^2 - 2qx -> qx - 0.5 (q^2 + m^2)
    Note that faiss index returns squared euclidean distance, so no need to square it again.
    """
    if max_norm is None:
        return -0.5 * l2_scores
    assert query_norm is not None
    return -0.5 * (l2_scores - query_norm ** 2 - max_norm ** 2)


def int8_to_float(num, offset, factor):
    return num.astype(np.float32) / factor + offset


def dequant(group, input_):
    if 'offset' in group.attrs:
        return int8_to_float(input_, group.attrs['offset'], group.attrs['scale'])
    return input_


def filter_results(results):
    out = []
    for result in results:
        c = Counter(result['context'])
        if c['?'] > 3:
            continue
        if c['!'] > 5:
            continue
        out.append(result)
    return out


def sparse_slice_ip_from_raw(q_mat, c_data_list, c_indices_list):
    """
    Efficient batch inner product after slicing
    """
    out = []
    for q_i, (c_data, c_indices) in enumerate(zip(c_data_list, c_indices_list)):
        c_map = {c_ii: c_d for c_ii, c_d in zip(c_indices, c_data)}

        if q_mat.shape[0] == len(c_data_list):
            q_data = q_mat.data[q_mat.indptr[q_i]:q_mat.indptr[q_i + 1]]
            q_indices = q_mat.indices[q_mat.indptr[q_i]:q_mat.indptr[q_i + 1]]
        elif q_mat.shape[0] == 1:
            q_data = q_mat.data[q_mat.indptr[0]:q_mat.indptr[1]]
            q_indices = q_mat.indices[q_mat.indptr[0]:q_mat.indptr[1]]
        else:
            raise Exception('Dimension mismatch: %d != %d' % (q_mat.shape[0], len(c_data_list)))

        ip = sum((c_map[q_ii] * q_d if q_ii in c_map else 0.0) for q_ii, q_d in zip(q_indices, q_data))
        out.append(ip)
    return np.array(out)


def adjust(each):
    last = each['context'].rfind(' [PAR] ', 0, each['start_pos'])
    last = 0 if last == -1 else last + len(' [PAR] ')
    next = each['context'].find(' [PAR] ', each['end_pos'])
    next = len(each['context']) if next == -1 else next
    each['context'] = each['context'][last:next]
    each['start_pos'] -= last
    each['end_pos'] -= last
    return each


class MIPSSparse(object):
    def __init__(self, phrase_dump_dir, start_index_path, idx2id_path, ranker_path, max_answer_length, para=False,
                 tfidf_dump_dir=None, sparse_weight=1e-1, sparse_type=None, cuda=False,
                 max_norm_path=None, num_dummy_zeros=0):

        # Save arguments
        self.num_dummy_zeros = num_dummy_zeros
        self.max_answer_length = max_answer_length
        self.num_docs_list = []
        self.sparse_type = sparse_type
        self.sparse_weight = sparse_weight
        self.para = para
        self.cuda = cuda
        if max_norm_path is None:
            self.max_norm = None
        else:
            with open(max_norm_path, 'r') as fp:
                self.max_norm = json.load(fp)

        print('loading phrase dumps from %s' % phrase_dump_dir)
        if os.path.isdir(phrase_dump_dir):
            self.phrase_dump_paths = sorted(
                [os.path.join(phrase_dump_dir, name) for name in os.listdir(phrase_dump_dir) if 'hdf5' in name])
            dump_names = [os.path.splitext(os.path.basename(path))[0] for path in self.phrase_dump_paths]
            self.dump_ranges = [list(map(int, name.split('-'))) for name in dump_names]
        else:
            self.phrase_dump_paths = [phrase_dump_dir]
        self.phrase_dumps = [h5py.File(path, 'r') for path in self.phrase_dump_paths]

        print('loading start index from %s' % start_index_path)
        self.start_index = faiss.read_index(start_index_path, faiss.IO_FLAG_ONDISK_SAME_DIR)

        print('loading idx2id from %s' % idx2id_path)
        self.idx_f = self.load_idx_f(idx2id_path)
        self.has_offset = not 'doc' in self.idx_f

        print('loading tfidf dump from %s' % tfidf_dump_dir)
        assert os.path.isdir(tfidf_dump_dir)
        self.tfidf_dump_paths = sorted(
            [os.path.join(tfidf_dump_dir, name) for name in os.listdir(tfidf_dump_dir) if 'hdf5' in name])
        dump_names = [os.path.splitext(os.path.basename(path))[0] for path in self.tfidf_dump_paths]
        dump_ranges = [list(map(int, name.split('_')[0].split('-'))) for name in dump_names]
        self.tfidf_dumps = [h5py.File(path, 'r') for path in self.tfidf_dump_paths]
        assert dump_ranges == self.dump_ranges

        print('loading DrQA DocRanker from %s' % ranker_path)
        self.ranker = TfidfDocRanker(ranker_path)

    def load_idx_f(self, idx2id_path):
        idx_f = {}
        types = ['doc', 'word']
        if self.para:
            types.append('para')
        with h5py.File(idx2id_path, 'r', driver='core', backing_store=False) as f:
            for key in tqdm(f, desc='loading idx2id'):
                idx_f_cur = {}
                for type_ in types:
                    idx_f_cur[type_] = f[key][type_][:]
                idx_f[key] = idx_f_cur
            return idx_f

    def get_idxs(self, I):
        if self.has_offset:
            offsets = (I / 1e8).astype(np.int64) * int(1e8)
            idxs = I % int(1e8)
            doc = np.array(
                [[self.idx_f[str(offset)]['doc'][idx] for offset, idx in zip(oo, ii)] for oo, ii in zip(offsets, idxs)])
            word = np.array([[self.idx_f[str(offset)]['word'][idx] for offset, idx in zip(oo, ii)] for oo, ii in
                             zip(offsets, idxs)])
            if self.para:
                para = np.array([[self.idx_f[str(offset)]['para'][idx] for offset, idx in zip(oo, ii)] for oo, ii in
                                 zip(offsets, idxs)])
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
                if str(doc_idx) not in dump:
                    raise ValueError('%d not found in dump list' % int(doc_idx))
                return dump[str(doc_idx)]
        raise ValueError('%d not found in dump list' % int(doc_idx))

    def get_tfidf_group(self, doc_idx):
        if len(self.tfidf_dumps) == 1:
            return self.tfidf_dumps[0][str(doc_idx)]
        for dump_range, dump in zip(self.dump_ranges, self.tfidf_dumps):
            if dump_range[0] * 1000 <= int(doc_idx) < dump_range[1] * 1000:
                return dump[str(doc_idx)]
        raise ValueError('%d not found in dump list' % int(doc_idx))

    def get_para_scores(self, q_spvecs, doc_idxs, para_idxs=None, start_idxs=None):
        if para_idxs is None:
            if self.para:
                groups = [self.get_doc_group(doc_idx) for doc_idx in doc_idxs]
                groups = [group[str(para_idx)] for group, para_idx in zip(groups, para_idxs)]
            else:
                groups = [self.get_doc_group(doc_idx) for doc_idx in doc_idxs]
                doc_bounds = [[m.start() for m in re.finditer('\[PAR\]', group.attrs['context'])] for group in
                              groups]
                doc_starts = [group['word2char_start'][start_idx].item() for group, start_idx in
                              zip(groups, start_idxs)]
                para_idxs = [sum([1 if start > bound else 0 for bound in par_bound])
                             for par_bound, start in zip(doc_bounds, doc_starts)]
        tfidf_groups = [self.get_tfidf_group(doc_idx) for doc_idx in doc_idxs]
        tfidf_groups = [group[str(para_idx)] for group, para_idx in zip(tfidf_groups, para_idxs)]
        data_list = [data['vals'][:] for data in tfidf_groups]
        idxs_list = [data['idxs'][:] for data in tfidf_groups]
        par_scores = sparse_slice_ip_from_raw(q_spvecs, data_list, idxs_list)
        return par_scores

    def search_dense(self, query_start, start_top_k, nprobe, all_doc_scores):
        # Search space reduction with Faiss
        query_start = np.concatenate([np.zeros([query_start.shape[0], 1]).astype(np.float32), query_start], axis=1)
        if self.num_dummy_zeros > 0:
            query_start = np.concatenate([query_start, np.zeros([query_start.shape[0], self.num_dummy_zeros],
                                                                dtype=query_start.dtype)], axis=1)
        self.start_index.nprobe = nprobe
        t = time()
        start_scores, I = self.start_index.search(query_start, start_top_k)
        query_norm = np.linalg.norm(np.squeeze(query_start), ord=2)
        start_scores = scale_l2_to_ip(start_scores, max_norm=self.max_norm, query_norm=query_norm)
        ts = time() - t
        # print('on-disk index search:', ts)

        doc_idxs, para_idxs, start_idxs = self.get_idxs(I)
        # print('get idxs:', time() - t)

        num_docs = len(set(doc_idxs.flatten().tolist()))
        self.num_docs_list.append(num_docs)
        # print('unique # docs: %d' % num_docs)
        # print('avg unique # docs: %.2f' % (sum(self.num_docs_list) / len(self.num_docs_list)))

        # Rerank based on sparse + dense (start)
        doc_idxs = np.reshape(doc_idxs, [-1])
        if self.para:
            para_idxs = np.reshape(para_idxs, [-1])
        start_idxs = np.reshape(start_idxs, [-1])
        start_scores = np.reshape(start_scores, [-1])

        start_scores += self.sparse_weight * all_doc_scores[doc_idxs]

        # for now, assume 1D
        doc_idxs = np.squeeze(doc_idxs)
        if para_idxs is not None:
            para_idxs = np.squeeze(para_idxs)
        start_idxs = np.squeeze(start_idxs)
        start_scores = np.squeeze(start_scores)

        return (doc_idxs, para_idxs, start_idxs), start_scores

    def search_sparse(self, query_start, doc_scores, doc_top_k):
        top_doc_idxs = doc_scores.argsort()[-doc_top_k:][::-1]
        top_doc_scores = doc_scores[top_doc_idxs]
        doc_idxs = []
        start_idxs = []
        scores = []
        for doc_idx, doc_score in zip(top_doc_idxs, top_doc_scores):
            try:
                doc_group = self.get_doc_group(doc_idx)
            except ValueError:
                continue
            start = dequant(doc_group, doc_group['start'][:])
            cur_scores = np.sum(query_start * start, 1)
            for i, cur_score in enumerate(cur_scores):
                doc_idxs.append(doc_idx)
                start_idxs.append(i)
                scores.append(cur_score + self.sparse_weight * doc_score)

        doc_idxs, start_idxs, scores = np.array(doc_idxs), np.array(start_idxs), np.array(scores)

        return (doc_idxs, start_idxs), scores

    def search_start(self, query_start, doc_idxs=None, para_idxs=None,
                     start_top_k=100, mid_top_k=20, top_k=5, nprobe=16, q_texts=None,
                     doc_top_k=5, search_strategy='dense_first'):

        # doc_idxs = [Q], para_idxs = [Q]
        assert self.start_index is not None
        query_start = query_start.astype(np.float32)

        # Open-domain setup (doc_idxs, para_idxs are not given)
        if doc_idxs is None:

            # Pre-compute doc_level sparse scores
            q_spvecs = vstack([self.ranker.text2spvec(q) for q in q_texts])
            doc_scores = np.squeeze((q_spvecs * self.ranker.doc_mat).toarray())

            # Branch based on the strategy (dense vs. sparse (doc-level))
            if search_strategy == 'dense_first':
                (doc_idxs, para_idxs, start_idxs), start_scores = self.search_dense(query_start,
                                                                                    start_top_k,
                                                                                    nprobe,
                                                                                    doc_scores)
            elif search_strategy == 'sparse_first':
                (doc_idxs, start_idxs), start_scores = self.search_sparse(query_start, doc_scores, doc_top_k)
            elif search_strategy == 'hybrid':
                (doc_idxs, para_idxs, start_idxs), start_scores = self.search_dense(query_start,
                                                                                    start_top_k,
                                                                                    nprobe,
                                                                                    doc_scores)
                (doc_idxs_, start_idxs_), start_scores_ = self.search_sparse(query_start, doc_scores, doc_top_k)
                doc_idxs = np.concatenate([doc_idxs, doc_idxs_], -1)
                start_idxs = np.concatenate([start_idxs, start_idxs_], -1)
                start_scores = np.concatenate([start_scores, start_scores_], -1)
            else:
                raise ValueError(search_strategy)

            # Rerank and reduce
            rerank_idxs = start_scores.argsort()[-mid_top_k:][::-1]
            doc_idxs = doc_idxs[rerank_idxs]
            start_idxs = start_idxs[rerank_idxs]
            start_scores = start_scores[rerank_idxs]

            # Para and rerank and reduce
            doc_idxs = np.reshape(doc_idxs, [-1])
            if self.para:
                para_idxs = np.reshape(para_idxs, [-1])
            start_idxs = np.reshape(start_idxs, [-1])
            start_scores = np.reshape(start_scores, [-1])
            start_scores += self.sparse_weight * self.get_para_scores(q_spvecs, doc_idxs, start_idxs=start_idxs)

            rerank_scores = np.reshape(start_scores, [-1, mid_top_k])
            rerank_idxs = np.array([scores.argsort()[-top_k:][::-1]
                                    for scores in rerank_scores])

            doc_idxs = doc_idxs[rerank_idxs]
            if para_idxs is not None:
                para_idxs = para_idxs[rerank_idxs]
            start_idxs = start_idxs[rerank_idxs]
            start_scores = start_scores[rerank_idxs]

        # Close-domain setup
        else:
            # Get start vectors and dequantize
            groups = [self.get_doc_group(doc_idx)[str(para_idx)] for doc_idx, para_idx in zip(doc_idxs, para_idxs)]
            starts = [group['start'][:, :] for group in groups]
            starts = [int8_to_float(start, groups[0].attrs['offset'], groups[0].attrs['scale']) for start in starts]

            # Calculate start scores based on dot-product
            all_scores = [np.squeeze(np.matmul(start, query_start[i:i + 1, :].transpose()), -1)
                          for i, start in enumerate(starts)]
            start_idxs = np.array([scores.argsort()[-top_k:][::-1] for scores in all_scores])
            start_scores = np.array([scores[idxs] for scores, idxs in zip(all_scores, start_idxs)])

            # Keep only top_k indices
            doc_idxs = np.tile(np.expand_dims(doc_idxs, -1), [1, top_k])
            para_idxs = np.tile(np.expand_dims(para_idxs, -1), [1, top_k])

        return start_scores, doc_idxs, para_idxs, start_idxs

    def search_phrase(self, query, doc_idxs, start_idxs, para_idxs=None, start_scores=None):
        # query = [Q, d]
        # doc_idxs = [Q]
        # start_idxs = [Q]
        # para_idxs = [Q]
        start_dim = int((query.shape[1] - 1) / 2)
        query_start, query_end, query_span_logit = query[:, :start_dim], query[:, start_dim:2 * start_dim], query[:, -1:]

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

        default_end = np.zeros(start_dim).astype(np.float32)
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

    def search(self, query, top_k=10, nprobe=256, doc_idxs=None, para_idxs=None, start_top_k=1000, mid_top_k=100,
               q_texts=None, filter_=False, search_strategy='dense_first', doc_top_k=5, aggregate=False):

        # Get dimensions
        batch_size = query.shape[0]
        start_dim = int((query.shape[1] - 1) / 2)

        # Search based on the strategy (dense-first, sparse-first, hybrid)
        query_start = query[:, :start_dim]
        start_scores, doc_idxs, para_idxs, start_idxs = self.search_start(query_start,
                                                                          start_top_k=start_top_k,
                                                                          mid_top_k=mid_top_k,
                                                                          top_k=top_k,
                                                                          nprobe=nprobe,
                                                                          doc_idxs=doc_idxs,
                                                                          para_idxs=para_idxs,
                                                                          q_texts=q_texts,
                                                                          search_strategy=search_strategy,
                                                                          doc_top_k=doc_top_k)
        if doc_idxs.shape[1] != top_k:
            print("Warning.. %d only retrieved" % doc_idxs.shape[1])
            top_k = doc_idxs.shape[1]

        # Reshape each output for broadcasting
        query = np.reshape(np.tile(np.expand_dims(query, 1), [1, top_k, 1]), [-1, query.shape[1]])
        idxs = np.reshape(np.tile(np.expand_dims(np.arange(batch_size), 1), [1, top_k]), [-1])
        start_scores = np.reshape(start_scores, [-1])
        doc_idxs = np.reshape(doc_idxs, [-1])
        para_idxs = np.reshape(para_idxs, [-1])
        start_idxs = np.reshape(start_idxs, [-1])

        # Rerank top phrases based on the end/span scores
        out = self.search_phrase(query, doc_idxs, start_idxs, para_idxs=para_idxs, start_scores=start_scores)
        new_out = [[] for _ in range(batch_size)]
        for idx, each_out in zip(idxs, out):
            new_out[idx].append(each_out)
        for i in range(len(new_out)):
            new_out[i] = sorted(new_out[i], key=lambda each_out: -each_out['score'])

        # Filter irrelevant outputs
        if filter_:
            new_out = [filter_results(results) for results in new_out]

        # What is this for?
        if aggregate:
            pass

        return new_out
