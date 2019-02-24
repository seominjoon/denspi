from mips import MIPS, int8_to_float, adjust, id2idxs

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
    def search_phrase(self, query, doc_idxs, start_idxs, para_idxs=None, start_scores=None, q_sparse=None, q_input_ids=None):
        # query = [Q, d]
        # doc_idxs = [Q]
        # start_idxs = [Q]
        # para_idxs = [Q]
        query_start, query_end, query_span_logit = query[:, :480], query[:, 480:960], query[:, 960:961]

        groups = [self.phrase_index[str(doc_idx)] for doc_idx in doc_idxs]
        if self.para:
            groups = [group[str(para_idx)] for group, para_idx in zip(groups, para_idxs)]


        if start_scores is None:
            start = np.stack([group['start'][start_idx, :]
                              for group, start_idx in zip(groups, start_idxs)], 0)  # [Q, d]
            start = dequant(groups[0], start)
            start_scores = np.sum(query_start * start, 1)  # [Q]

            # Sparse start version
            '''
            sparse = [group['sparse'][start_idx,:]
                      for group, start_idx in zip(groups, start_idxs)]
            input_ids = [group['input_ids'][:] for group in groups]
            common_mask = [(np.expand_dims(ip, 1) == np.expand_dims(q_ip, 0)).astype(np.int) for q_ip, ip in zip(q_input_ids, input_ids)]
            sparse_val = [np.expand_dims(sp, 1) * np.expand_dims(q_sp, 0) * m for q_sp, sp, m in zip(q_sparse, sparse, common_mask)]
            sparse_scores = np.stack([np.sum(sp) for sp in sparse_val])
            # sparse_scores = np.stack([linear_mxq(q_idx, q_val, c_idx, c_val) for q_idx, q_val, c_idx, c_val in zip(q_input_ids, q_sparse, input_ids, sparse)])
            sparse_scores = np.expand_dims(sparse_scores, 1)
            '''

        ends = [group['end'][:] for group in groups]
        spans = [group['span_logits'][:] for group in groups]

        end_idxs = [group['start2end'][start_idx, :] for group, start_idx in zip(groups, start_idxs)]  # [Q, L]
        end_mask = -1e9 * (np.array(end_idxs) < 0)  # [Q, L]
        end = np.stack([[each_end[each_end_idx, :] for each_end_idx in each_end_idxs]
                        for each_end, each_end_idxs in zip(ends, end_idxs)], 0)  # [Q, L, d]
        end = dequant(groups[0], end)
        span = np.stack([[each_span[start_idx, i] for i in range(len(each_end_idxs))]
                         for each_span, start_idx, each_end_idxs in zip(spans, start_idxs, end_idxs)], 0)  # [Q, L]

        # Sparse end version
        '''
        if 'sparse' in groups[0]:
            sparse = [np.stack([group['sparse'][each_end_idx,:] 
                       for each_end_idx in each_end_idxs])
                       for group, each_end_idxs in zip(groups, end_idxs)] # [Q, L, P]
            # sparse = [dequant(groups[0], sp) for sp in sparse]
            # print(np.max(q_sparse), np.min(q_sparse))
            input_ids = [group['input_ids'][:] for group in groups]
            common_mask = [(np.expand_dims(ip, 1) == np.expand_dims(q_ip, 0)).astype(np.int) for q_ip, ip in zip(q_input_ids, input_ids)]
            sparse_val = [np.expand_dims(sp, 2) * np.expand_dims(np.expand_dims(q_sp, 0), 1) * np.expand_dims(m, 0) for q_sp, sp, m in zip(q_sparse, sparse, common_mask)]
            sparse_scores = np.stack([np.sum(sp, (1,2)) for sp in sparse_val])
            # sparse_scores = np.stack([[linear_mxq(q_idx, q_val, c_idx, c_val) for c_val in c_vals] for q_idx, q_val, c_idx, c_vals in zip(q_input_ids, q_sparse, input_ids, sparse)])
        '''

        end_scores = np.sum(np.expand_dims(query_end, 1) * end, 2)  # [Q, L]
        span_scores = query_span_logit * span  # [Q, L]
        scores = np.expand_dims(start_scores, 1) + end_scores + span_scores + end_mask  # [Q, L]
        # scores += sparse_scores * 1
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
            if self.para:
                doc_idxs, para_idxs, start_idxs = id2idxs(I, para=self.para)  # [Q, K]
            else:
                doc_idxs, start_idxs = id2idxs(I, para=self.para)

            # Rerank based on sparse + dense (start)
            query_start = np.reshape(np.tile(np.expand_dims(query_start[:,1:], 1), [1, start_top_k, 1]), [-1, query_start[:,1:].shape[1]])
            doc_idxs = np.reshape(doc_idxs, [-1])
            para_idxs = np.reshape(para_idxs, [-1])
            start_idxs = np.reshape(start_idxs, [-1])
            q_sparse = [sparse for sparse in q_sparse for _ in range(start_top_k)]
            q_input_ids = [input_ids for input_ids in q_input_ids for _ in range(start_top_k)]
            groups = [self.phrase_index[str(doc_idx)] for doc_idx in doc_idxs]
            if self.para:
                groups = [group[str(para_idx)] for group, para_idx in zip(groups, para_idxs)]
            start = np.stack([group['start'][start_idx, :]
                              for group, start_idx in zip(groups, start_idxs)], 0)  # [Q, d]
            start = dequant(groups[0], start)
            start_scores = np.sum(query_start * start, 1)  # [Q]

            # Sparse start rerank
            sparse = [group['sparse'][start_idx,:]
                      for group, start_idx in zip(groups, start_idxs)]
            sparse = [dequant(groups[0], sp) for sp in sparse]
            input_ids = [group['input_ids'][:] for group in groups]
            common_mask = [(np.expand_dims(ip, 1) == np.expand_dims(q_ip, 0)).astype(np.int) for q_ip, ip in zip(q_input_ids, input_ids)]
            sparse_val = [np.expand_dims(sp, 1) * np.expand_dims(q_sp, 0) * m for q_sp, sp, m in zip(q_sparse, sparse, common_mask)]
            sparse_scores = np.stack([np.sum(sp) for sp in sparse_val])
            # sparse_scores = np.stack([linear_mxq(q_idx, q_val, c_idx, c_val) for q_idx, q_val, c_idx, c_val in zip(q_input_ids, q_sparse, input_ids, sparse)])

            rerank_scores = np.reshape(start_scores + sparse_scores * 3, [-1, start_top_k])
            rerank_idxs = np.array([scores.argsort()[-out_top_k:][::-1]
                                    for scores in rerank_scores])
            new_I = np.array([each_I[idxs] for each_I, idxs in zip(I, rerank_idxs)])

            if self.para:
                doc_idxs, para_idxs, start_idxs = id2idxs(new_I, para=self.para)  # [Q, K]
            else:
                doc_idxs, start_idxs = id2idxs(new_I, para=self.para)

        # Closed
        else:
            groups = [self.phrase_index[str(doc_idx)][str(para_idx)] for doc_idx, para_idx in zip(doc_idxs, para_idxs)]
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
        query_start = query[:, :480]
        start_scores, doc_idxs, para_idxs, start_idxs = self.search_start(query_start, q_sparse, q_input_ids, start_top_k=start_top_k, out_top_k=top_k, nprobe=nprobe,
                                                                          doc_idxs=doc_idxs, para_idxs=para_idxs)

        # reshape
        query = np.reshape(np.tile(np.expand_dims(query, 1), [1, top_k, 1]), [-1, query.shape[1]])
        idxs = np.reshape(np.tile(np.expand_dims(np.arange(num_queries), 1), [1, top_k]), [-1])
        start_scores = np.reshape(start_scores, [-1])
        doc_idxs = np.reshape(doc_idxs, [-1])
        para_idxs = np.reshape(para_idxs, [-1])
        start_idxs = np.reshape(start_idxs, [-1])
        # q_sparse = [sparse for sparse in q_sparse for _ in range(top_k)]
        # q_input_ids = [input_ids for input_ids in q_input_ids for _ in range(top_k)]

        out = self.search_phrase(query, doc_idxs, start_idxs, para_idxs=para_idxs)
        new_out = [[] for _ in range(num_queries)]
        for idx, each_out in zip(idxs, out):
            new_out[idx].append(each_out)
        for i in range(len(new_out)):
            new_out[i] = sorted(new_out[i], key=lambda each_out: -each_out['score'])

        return new_out
       
