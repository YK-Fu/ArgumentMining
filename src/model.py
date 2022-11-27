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
        elif model in ['allenai/longformer-base-4096']:
            self.encoder = LongformerModel.from_pretrained(model)
            self.max_leng = 4096
        else:
            raise NotImplementedError("Not supported pretrained model type.")

        # Will q_tagging and r_tagging share the same model be better?
        self.span_tagging = nn.Linear(768, 4)          # predict the start and end tokens of query
                                    

    def cal_tagging_loss(self, span_logits, start, end):
        '''
            calculate loss of start and end loss
        '''

        # clip the indexes out of max length
        start = torch.clamp(start, 0, self.max_leng - 1).to(span_logits.device)
        end = torch.clamp(end, 0, self.max_leng - 1).to(span_logits.device)

        start_logits, end_logits = span_logits.split([1, 1], -1)

        loss_fct = nn.CrossEntropyLoss()
        start_loss = loss_fct(start_logits.squeeze(-1), start)

        end_loss = loss_fct(end_logits.squeeze(-1), end)
        
        loss = (start_loss + end_loss) / 2
        
        return loss

    def forward(self, A, S=None):
        Q_s, Q_e, R_s, R_e = [None for i in range(4)] if S is None else S.split([1, 1, 1, 1], -1)

        a = self.encoder(**A).last_hidden_state
        
        span_logits = self.span_tagging(a).squeeze(-1).contiguous()   # the start and end token logits of response
        q_span_logits, r_span_logits = span_logits.split([2, 2], -1)
        
        Outputs = dict()
        if Q_s is not None and Q_e is not None:
            Outputs['q_loss'] = self.cal_tagging_loss(q_span_logits, Q_s.squeeze(-1), Q_e.squeeze(-1))
        if R_s is not None and R_e is not None:
            Outputs['r_loss'] = self.cal_tagging_loss(r_span_logits, R_s.squeeze(-1), R_e.squeeze(-1))
        
        # parse span results
        Outputs['q_start'] = torch.argmax(q_span_logits[:, :, 0], -1).tolist()
        Outputs['q_end'] = torch.argmax(q_span_logits[:, :, 1], -1).tolist()
        Outputs['r_start'] = torch.argmax(r_span_logits[:, :, 0], -1).tolist()
        Outputs['r_end'] = torch.argmax(r_span_logits[:, :, 1], -1).tolist()
        
        return Outputs
