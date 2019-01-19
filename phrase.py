import torch
from torch import nn

from bert import BertModel, BERTLayerNorm


class BertWrapper(nn.Module):
    """
    Abides phrase model's encoder spec:
    1. (input_ids, mask) -> input_vectors
    2. input_ids[0] is a special token
    """
    def __init__(self, bert):
        super(BertWrapper, self).__init__()
        self.bert = bert

    def forward(self, input_ids, mask):
        segment_ids = torch.zeros_like(input_ids)
        layers, _ = self.bert(input_ids, segment_ids, mask)
        return layers[-1]


def encode_phrase(layer, span_vec_size, get_first_only=False):
    boundary_layer = layer[:, :, :-span_vec_size]
    span_layer = layer[:, :, -span_vec_size:]
    start, end = boundary_layer.chunk(2, dim=2)
    span_start, span_end = span_layer.chunk(2, dim=2)
    span_logits = span_start.matmul(span_end.transpose(1, 2))
    if get_first_only:
        start = start[:, :1, :]
        end = end[:, :1, :]
        span_logits = span_logits[:, :1, :1]
    return start, end, span_logits


class PhraseModel(nn.Module):
    def __init__(self, encoder, span_vec_size):
        super(PhraseModel, self).__init__()
        self.encoder = encoder
        self.span_vec_size = span_vec_size
        self.default_value = nn.Parameter(torch.randn(1))

    def forward(self,
                context_ids=None, context_mask=None,
                query_ids=None, query_mask=None,
                start_positions=None, end_positions=None):
        if context_ids is not None:
            context_layer = self.encoder(context_ids, context_mask)
            start, end, span_logits = encode_phrase(context_layer, self.span_vec_size)

            # embed context
            if query_ids is None:
                return start, end, span_logits

        if query_ids is not None:
            question_layer = self.encoder(query_ids, query_mask)
            query_start, query_end, q_span_logits = encode_phrase(question_layer, self.span_vec_size,
                                                                  get_first_only=True)

            # embed question
            if context_ids is None:
                return query_start, query_end, q_span_logits

        # pass this line only if train or eval

        start_logits = start.matmul(query_start.transpose(1, 2)).squeeze(-1)
        end_logits = end.matmul(query_end.transpose(1, 2)).squeeze(-1)
        all_logits = start_logits.unsqueeze(2) + end_logits.unsqueeze(1) + span_logits * q_span_logits

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            span_ignored_index = ignored_index ** 2
            start_positions.clamp_(-1, ignored_index)
            end_positions.clamp_(-1, ignored_index)

            cel_1d = CrossEntropyLossWithDefault(default_value=self.default_value,
                                                 ignore_index=ignored_index)
            cel_2d = CrossEntropyLossWithDefault(default_value=self.default_value,
                                                 ignore_index=span_ignored_index)

            span_target = start_positions * ignored_index + end_positions
            # needed to handle -1
            span_target.clamp_(-1, span_ignored_index)
            valid = (start_positions < ignored_index) & (end_positions < ignored_index)
            span_target = valid.long() * span_target + (1 - valid.long()) * span_ignored_index

            true_loss = cel_2d(all_logits.view(all_logits.size(0), -1), span_target)

            help_loss = 0.5 * (cel_1d(all_logits.mean(2), start_positions) +
                               cel_1d(all_logits.mean(1), end_positions))

            loss = true_loss + help_loss

            return loss
        else:
            return all_logits


class BertPhraseModel(PhraseModel):
    def __init__(self, config, span_vec_size):
        encoder = BertWrapper(BertModel(config))
        super(BertPhraseModel, self).__init__(encoder, span_vec_size)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)


class CrossEntropyLossWithDefault(nn.CrossEntropyLoss):
    def __init__(self, default_value, ignore_index=-100, **kwargs):
        if ignore_index >= 0:
            ignore_index += 1
        super(CrossEntropyLossWithDefault, self).__init__(ignore_index=ignore_index, **kwargs)
        self.default_value = default_value

    def forward(self, input_, target):
        assert len(input_.size()) == 2
        default_value = self.default_value.unsqueeze(0).repeat(input_.size(0), 1)
        new_input = torch.cat([default_value, input_], 1)
        new_target = target + 1
        assert new_target.min().item() >= 0, (new_target.min().item(), target.min().item())
        loss = super(CrossEntropyLossWithDefault, self).forward(new_input, new_target)
        return loss
