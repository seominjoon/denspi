from mips import MIPS, int8_to_float, adjust

import numpy as np


def dequant(group, input_):
    if 'offset' in group.attrs:
        return int8_to_float(input_, group.attrs['offset'], group.attrs['scale'])
    return input_


def linear_mxq(q_idx, q_val, c_idx, c_val):
    q_dict = {}
    for idx, val in zip(q_idx, q_val):
        if val <= 0:
            continue
        if idx not in q_dict:
            q_dict[idx] = [val, 0.0]
        else:
            q_dict[idx][0] += val

    for idx, val in zip(c_idx, c_val):
        if idx in q_dict:
            q_dict[idx][1] += val

    total = sum([a[0]*a[1] for a in q_dict.values()])
    return total


class MIPSSparse(MIPS):
    def search_start(self, query_start, q_sparse, q_input_ids, doc_idxs=None, para_idxs=None, start_top_k=100, out_top_k=5, nprobe=16):
        # doc_idxs = [Q], para_idxs = [Q]
        assert self.start_index is not None
        query_start = query_start.astype(np.float32)

        # Open
        if doc_idxs is None:
            # Search space reduction with Faiss
            query_start = np.concatenate([np.zeros([query_start.shape[0], 1]).astype(np.float32), query_start], axis=1)
            self.start_index.nprobe = nprobe
            start_scores, I = self.start_index.search(query_start, start_top_k)

            doc_idxs = self.idx2doc_id[I]
            start_idxs = self.idx2word_id[I]
            if self.para:
                para_idxs = self.idx2para_id[I]

            # Rerank based on sparse + dense (start)
            query_start = np.reshape(np.tile(np.expand_dims(query_start[:,1:], 1), [1, start_top_k, 1]), [-1, query_start[:,1:].shape[1]])
            doc_idxs = np.reshape(doc_idxs, [-1])
            para_idxs = np.reshape(para_idxs, [-1])
            start_idxs = np.reshape(start_idxs, [-1])
            q_sparse = [sparse for sparse in q_sparse for _ in range(start_top_k)]
            q_input_ids = [input_ids for input_ids in q_input_ids for _ in range(start_top_k)]
            groups = [self.get_doc_group(doc_idx) for doc_idx in doc_idxs]
            if self.para:
                groups = [group[str(para_idx)] for group, para_idx in zip(groups, para_idxs)]
            start = np.stack([group['start'][start_idx, :]
                              for group, start_idx in zip(groups, start_idxs)], 0)  # [Q, d]
            start = dequant(groups[0], start)
            start_scores = np.sum(query_start * start, 1)  # [Q]

            # Sparse start rerank
            sparse = [group['sparse'][start_idx,:]
                      for group, start_idx in zip(groups, start_idxs)]
            input_ids = [group['input_ids'][:] for group in groups]
            common_mask = [(np.expand_dims(ip, 1) == np.expand_dims(q_ip, 0)).astype(np.int) for q_ip, ip in zip(q_input_ids, input_ids)]
            sparse_val = [np.expand_dims(sp, 1) * np.expand_dims(q_sp, 0) * m for q_sp, sp, m in zip(q_sparse, sparse, common_mask)]
            sparse_scores = np.stack([np.sum(sp) for sp in sparse_val])
            # sparse_scores = np.stack([linear_mxq(q_idx, q_val, c_idx, c_val) for q_idx, q_val, c_idx, c_val in zip(q_input_ids, q_sparse, input_ids, sparse)])

            rerank_scores = np.reshape(start_scores + sparse_scores * 3, [-1, start_top_k])
            rerank_idxs = np.array([scores.argsort()[-out_top_k:][::-1]
                                    for scores in rerank_scores])
            new_I = np.array([each_I[idxs] for each_I, idxs in zip(I, rerank_idxs)])

            doc_idxs = self.idx2doc_id[new_I]
            start_idxs = self.idx2word_id[new_I]
            if self.para:
                para_idxs = self.idx2para_id[new_I]

            start_scores = np.array([scores[idxs] for scores, idxs in zip(rerank_scores, rerank_idxs)])[:,:out_top_k]

        # Closed
        else:
            groups = [self.get_doc_group(doc_idx)[str(para_idx)] for doc_idx, para_idx in zip(doc_idxs, para_idxs)]
            starts = [group['start'][:, :] for group in groups]
            starts = [int8_to_float(start, groups[0].attrs['offset'], groups[0].attrs['scale']) for start in starts]
            all_scores = [np.squeeze(np.matmul(start, query_start[i:i + 1, :].transpose()), -1)
                          for i, start in enumerate(starts)]
            start_idxs = np.array([scores.argsort()[-out_top_k:][::-1]
                                   for scores in all_scores])
            start_scores = np.array([scores[idxs] for scores, idxs in zip(all_scores, start_idxs)])
            doc_idxs = np.tile(np.expand_dims(doc_idxs, -1), [1, out_top_k])
            para_idxs = np.tile(np.expand_dims(para_idxs, -1), [1, out_top_k])
        return start_scores, doc_idxs, para_idxs, start_idxs


    # Just added q_sparse / q_input_ids to pass to search_phrase
    def search(self, query, top_k=5, nprobe=64, doc_idxs=None, para_idxs=None, q_sparse=None, q_input_ids=None):
        start_top_k = 100
        num_queries = query.shape[0]
        bs = int((query.shape[1] - 1) / 2)
        query_start = query[:, :bs]
        start_scores, doc_idxs, para_idxs, start_idxs = self.search_start(query_start, q_sparse, q_input_ids, start_top_k=start_top_k, out_top_k=top_k, nprobe=nprobe,
                                                                          doc_idxs=doc_idxs, para_idxs=para_idxs)

        # reshape
        query = np.reshape(np.tile(np.expand_dims(query, 1), [1, top_k, 1]), [-1, query.shape[1]])
        idxs = np.reshape(np.tile(np.expand_dims(np.arange(num_queries), 1), [1, top_k]), [-1])
        start_scores = np.reshape(start_scores, [-1])
        doc_idxs = np.reshape(doc_idxs, [-1])
        para_idxs = np.reshape(para_idxs, [-1])
        start_idxs = np.reshape(start_idxs, [-1])

        out = self.search_phrase(query, doc_idxs, start_idxs, para_idxs=para_idxs, start_scores=start_scores)
        new_out = [[] for _ in range(num_queries)]
        for idx, each_out in zip(idxs, out):
            new_out[idx].append(each_out)
        for i in range(len(new_out)):
            new_out[i] = sorted(new_out[i], key=lambda each_out: -each_out['score'])

        return new_out
