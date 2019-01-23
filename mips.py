from collections import namedtuple

import h5py
import numpy as np


class DocumentPhraseMIPS(object):
    def __init__(self, document_index, phrase_index, max_answer_length, doc_score_cf):
        super(DocumentPhraseMIPS, self).__init__()
        assert isinstance(phrase_index, h5py.File)
        self.document_index = document_index
        self.phrase_index = phrase_index
        self.max_answer_length = max_answer_length
        self.doc_score_cf = doc_score_cf

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
               for doc_idx, doc_score in zip(doc_indices, doc_scores)]
        return out

    def search_phrase(self, doc_idx, query, top_k=5, doc_score=0.0):
        group = self.phrase_index[str(doc_idx)]
        start, end, span_logits, start2char, end2char = [group[key][:] for key in
                                                         ['start', 'end', 'span_logits', 'start2char', 'end2char']]
        context = group.attrs['context']

        query_start, query_end, query_span_logit = query
        query_span_logit = float(query_span_logit[0][0])
        start_scores = np.matmul(start, np.array(query_start, dtype=np.float16).transpose())
        end_scores = np.matmul(end, np.array(query_end, dtype=np.float16).transpose())
        PhraseResult = namedtuple('PhraseResult', ('score', 'start_idx', 'end_idx'))
        results = []
        for start_idx, start_score in enumerate(start_scores):
            for end_idx in range(start_idx, min(start_idx + self.max_answer_length, len(start_scores))):
                span_logit = span_logits[start_idx, end_idx - start_idx]
                if span_logit < -1e6:
                    continue
                end_score = end_scores[end_idx]
                span_score = query_span_logit * span_logits[start_idx, end_idx - start_idx]
                score = self.doc_score_cf * doc_score + start_score.item() + end_score.item() + span_score.item()
                result = PhraseResult(score, start_idx, end_idx)
                results.append(result)
        results = sorted(results, key=lambda item: -item.score)[:top_k]
        out = [{'context': context,
                'doc_idx': doc_idx,
                'doc_score': doc_score,
                'start_pos': start2char[result.start_idx].item(),
                'end_pos': end2char[result.end_idx].item(),
                'phrase_score': result.score - doc_score,
                'score': result.score} for result in results]
        out = [adjust(each) for each in out]
        return out

    def search(self, doc_query, phrase_query, top_k_docs=5, top_k_phrases=5):
        document_rets = self.search_document(doc_query, top_k_docs)
        phrase_rets = sum([self.search_phrase(ret['doc_idx'], phrase_query, top_k=top_k_phrases,
                                              doc_score=ret['doc_score'])
                           for ret in document_rets], [])
        phrase_rets = sorted(phrase_rets, key=lambda ret: -ret['score'])[:top_k_phrases]
        return phrase_rets


def adjust(each):
    last = each['context'].rfind('[PAR] ', 0, each['start_pos'])
    last = 0 if last == -1 else last + len('[PAR] ')
    next = each['context'].find(' [PAR]', each['end_pos'])
    each['context'] = each['context'][last:next]
    each['start_pos'] -= last
    each['end_pos'] -= last
    return each
