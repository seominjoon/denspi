from collections import namedtuple
import time

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

    def search_phrase(self, doc_idx, query, top_k=5, doc_score=0.0, para_idx=None):
        t0 = time.time()
        group = self.phrase_index[str(doc_idx)]
        start, end, span_logits, start2char, end2char = [group[key][:] for key in
                                                         ['start', 'end', 'span_logits', 'start2char', 'end2char']]
        context = group.attrs['context']
        title = group.attrs['title']
        t1 = time.time()
        # print('Loading index: %dms' % int(1000 * (t1 - t0)))

        if para_idx is not None:
            char2start = {char.item(): start for start, char in enumerate(start2char)}
            char2end = {char.item(): end for end, char in enumerate(end2char)}
            char_start = 0
            for _ in range(para_idx):
                char_start = context.find(' [PAR] ', char_start) + len(' [PAR] ')
            char_end = context.find(' [PAR] ', char_start)
            if char_end == -1:
                char_end = len(context)
            word_start = char2start[char_start]
            if char_end not in char2end:
                count = 0
                while char_end not in char2end:
                    char_end += 1
                    count += 1
                print('char end not found; moved down by %d' % count)

            word_end = char2end[char_end]
            context = context[char_start:char_end]

            start = start[word_start:word_end + 1]
            end = end[word_start:word_end + 1]
            span_logits = span_logits[word_start:word_end + 1]
            start2char = start2char[word_start:word_end + 1] - char_start
            end2char = end2char[word_start:word_end + 1] - char_start

        query_start, query_end, query_span_logit = query
        query_span_logit = float(query_span_logit[0][0])
        start_scores = np.matmul(start, np.array(query_start, dtype=start.dtype).transpose()).squeeze(1)
        end_scores = np.matmul(end, np.array(query_end, dtype=end.dtype).transpose()).squeeze(1)
        t2 = time.time()
        # print('Computing IP: %dms' % int(1000 * (t2 - t1)))

        PhraseResult = namedtuple('PhraseResult', ('score', 'start_idx', 'end_idx'))
        results = []
        if top_k > 0:
            best_start_pairs = sorted(enumerate(start_scores.tolist()), key=lambda item: -item[1])[:top_k]
        else:
            best_start_pairs = enumerate(start_scores.tolist())
        end_scores = end_scores.tolist()
        for start_idx, start_score in best_start_pairs:
            for end_idx in range(start_idx, min(start_idx + self.max_answer_length, len(start_scores))):
                span_logit = span_logits[start_idx, end_idx - start_idx].item()
                if span_logit < -1e6:
                    continue
                end_score = end_scores[end_idx]
                span_score = query_span_logit * span_logits[start_idx, end_idx - start_idx]
                score = self.doc_score_cf * doc_score + start_score + end_score + span_score
                result = PhraseResult(score, start_idx, end_idx)
                results.append(result)
        results = sorted(results, key=lambda item: -item.score)[:top_k]
        out = [{'context': context,
                'title': title,
                'doc_idx': doc_idx,
                'doc_score': doc_score,
                'start_pos': start2char[result.start_idx].item(),
                'end_pos': end2char[result.end_idx].item(),
                'phrase_score': result.score - doc_score,
                'score': result.score} for result in results]
        t3 = time.time()
        # print('Finding answer: %dms' % int(1000 * (t3 - t2)))
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
    last = each['context'].rfind(' [PAR] ', 0, each['start_pos'])
    last = 0 if last == -1 else last + len(' [PAR] ')
    next = each['context'].find(' [PAR] ', each['end_pos'])
    next = len(each['context']) if next == -1 else next
    each['context'] = each['context'][last:next]
    each['start_pos'] -= last
    each['end_pos'] -= last
    return each
