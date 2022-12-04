import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel, LongformerModel
torch.manual_seed(4096)

class ArgumentModel(nn.Module):
    def __init__(self, model="bert-base-uncased"):
        super().__init__()
        if model in ['bert-base-uncased', 'bert-base-cased']:
            self.encoder = BertModel.from_pretrained(model)   # BERT for query
            self.max_leng = 512
        elif model in ['roberta-base']:
            self.encoder = RobertaModel.from_pretrained(model)
            self.max_leng = 512
        elif model in ['allenai/longformer-base-4096', 'allenai/longformer-large-4096']:
            self.encoder = LongformerModel.from_pretrained(model)
            self.max_leng = 3000
        else:
            raise NotImplementedError("Not supported pretrained model type.")
        if 'large' in model:
            self.hidden = 1024
        else:
            self.hidden = 768
        # 0: other, 1: in query span, 2: in response span
        # self.span_tagging = nn.LSTM(self.hidden, hidden_size=128, bidirectional=True, num_layers=2, batch_first=True)    
        # self.proj = nn.Linear(128 * 2, 3)
        self.span_tagging = nn.Linear(self.hidden, 3)

    def cal_tagging_loss(self, span_logits, length, S):
        '''
            calculate loss of start and end loss
        '''
        device = span_logits.device
        ground_truth = torch.ones((span_logits.size(0), span_logits.size(1))).long().to(device) * 3     # 3: pad index, do not calculated into gradient
        for i in range(span_logits.size(0)):
            ground_truth[i][:length[i]] = 0
            for [q_s, q_e, r_s, r_e] in S[i]:
                if [q_s, q_e, r_s, r_e] == [-1] * 4:
                    continue
                else:
                    ground_truth[i, q_s: q_e + 1] = 1
                    ground_truth[i, r_s: r_e + 1] = 2

        loss_fct = nn.CrossEntropyLoss(ignore_index=3, label_smoothing=0.0)

        loss = loss_fct(span_logits.view(-1, 3), ground_truth.view(-1))
        
        return loss

    def forward(self, A, S=None):
        # Q_s, Q_e, R_s, R_e = [None for i in range(4)] if S is None else S.split([1, 1, 1, 1], -1)

        a = self.encoder(**A).last_hidden_state
        
        # for LSTM
        # h, _ = self.span_tagging(a)
        # span_logits = self.proj(h).contiguous()
        # for linear only
        span_logits = self.span_tagging(a).squeeze(-1).contiguous()   # the start and end token logits of response
        
        Outputs = dict()
        if S is not None:
            length = torch.sum(A['attention_mask'], -1)
            Outputs['loss'] = self.cal_tagging_loss(span_logits, length, S)
        
        # parse span results
        Outputs['q_span'] = torch.argmax(span_logits, -1) == 1
        Outputs['r_span'] = torch.argmax(span_logits, -1) == 2
        
        return Outputs
