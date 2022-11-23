import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel, LongformerModel
from transformers import BertTokenizer
torch.manual_seed(4096)

class ArgumentModel(nn.Module):
    def __init__(self, model="bert-base-uncased"):
        super().__init__()
        if model in ['bert-base-uncased', 'bert-base-cased']:
            self.query = BertModel.from_pretrained(model)   # BERT for query
            self.res = BertModel.from_pretrained(model)     # BERT for response
            self.max_leng = 512
        elif model in ['roberta-base', 'xlm-roberta-base']:
            self.query = RobertaModel.from_pretrained(model)
            self.res = RobertaModel.from_pretrained(model)
            self.max_leng = 512
        elif model in ['allenai/longformer-base-4096']:
            self.query = LongformerModel.from_pretrained(model)
            self.res = LongformerModel.from_pretrained(model)
            self.max_leng = 4096
        else:
            raise NotImplementedError("Not supported pretrained model type.")

        # Will q_tagging and r_tagging share the same model be better?
        self.q_tagging = nn.Linear(768 * 2, 2)          # predict the start and end tokens of query
        self.r_tagging = nn.Linear(768 * 2, 2)          # predict the start and end tokens of response

        self.argument = nn.Linear(768 * 2, 1)           # argument mining label
                                    

    def cal_tagging_loss(self, span_logits, start, end):
        '''
            calculate loss of start and end loss
        '''

        # clip the indexes out of max length
        start = torch.clamp(start, 0, self.max_leng - 1)
        end = torch.clamp(end, 0, self.max_leng - 1)

        start_logits, end_logits = span_logits.split([1, 1], -1)
        
        loss_fct = nn.CrossEntropyLoss()
        start_loss = loss_fct(start_logits.squeeze(-1), start)
        end_loss = loss_fct(end_logits.squeeze(-1), end)
        
        loss = (start_loss + end_loss) / 2
        
        return loss

    def forward(self, Q, R, y, Q_s=None, Q_e=None, R_s=None, R_e=None):
        q = self.query(**Q).last_hidden_state     # (B, T, H)
        r = self.res(**R).last_hidden_state       # (B, T, H)

        q_cls = q[:, 0, :]  # (B, H) cls embedding of query
        r_cls = r[:, 0, :]  # (B, H) cls embedding of response
        
        q = torch.cat((q, r_cls.detach().unsqueeze(1).repeat(1, q.size(1), 1)), dim=-1) # concat r_cls to all of the embeddings of query
        r = torch.cat((r, q_cls.detach().unsqueeze(1).repeat(1, r.size(1), 1)), dim=-1) # concat q_cls to all of the embeddings of response

        q_logits = self.q_tagging(q).squeeze(-1).contiguous()   # the start and end token logits of response
        r_logits = self.r_tagging(r).squeeze(-1).contiguous()   # the start and end token logits of response

        y_logits = self.argument(torch.cat((q_cls, r_cls), -1)) # whether the query and the response are correspond
        Loss = dict()

        # calculate loss of each parts
        if y is not None:
            criterion = nn.BCEWithLogitsLoss()
            Loss['Argument'] = criterion(y_logits, y)
        if Q_s is not None and Q_e is not None:
            Loss['Query'] = self.cal_tagging_loss(q_logits, Q_s, Q_e)
        if R_s is not None and R_e is not None:
            Loss['Response'] = self.cal_tagging_loss(r_logits, R_s, R_e)
        
        # parse span results
        q_start = torch.argmax(q_logits[:, :, 0], -1).tolist()
        q_end = torch.argmax(q_logits[:, :, 1], -1).tolist()
        r_start = torch.argmax(r_logits[:, :, 0], -1).tolist()
        r_end = torch.argmax(r_logits[:, :, 1], -1).tolist()
        
        return (q_start, q_end), (r_start, r_end), y_logits, Loss
