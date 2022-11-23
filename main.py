import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from src.model import ArgumentModel
from src.utils import longestCommonSubsequence, get_optim
from src.dataset import ArgumentMining
torch.manual_seed(4096)
np.random.seed(4096)

def train(args):
    device = args.device
    model = ArgumentModel(args.model).to(device)
    optimizer, scheduler = get_optim(model, args.optim)
    
    trainset = ArgumentMining(path=args.data_path, split='train_', batch_size=args.batch_size, model=args.model)
    devset = ArgumentMining(path=args.data_path, split='valid_', batch_size=args.batch_size, model=args.model)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=trainset.collate_fn, num_workers=0)
    devloader = DataLoader(devset, batch_size=1, shuffle=False, collate_fn=devset.collate_fn, num_workers=0)

    steps = 0
    update_time = 0
    best_score = 0

    q_loss = 0
    r_loss = 0
    a_loss = 0
    correct = 0
    n = 0

    for e in range(args.epoch):
        for ID, Q, D, Y, Q_s, Q_e, R_s, R_e in tqdm(trainloader):
            
            Q = {k: v.to(args.device) for k, v in Q.items()}
            R = {k: v.to(args.device) for k, v in D.items()}
            Y = Y.to(args.device)
            Q_s = Q_s.to(args.device)
            Q_e = Q_e.to(args.device)
            R_s = R_s.to(args.device)
            R_e = R_e.to(args.device)

            _, _, y_logits, Loss = model(Q, R, Y, Q_s, Q_e, R_s, R_e)
            loss = (Loss['Argument'] + Loss['Query'] + Loss['Response']) / args.grad_steps
            loss.backward()

            # log argument acc and loss
            y = (y_logits.detach().cpu().view(-1).numpy()> 0.5) * 1
            correct += np.sum(y == Y.detach().cpu().view(-1).numpy(), -1)
            q_loss += Loss['Query'].item()
            r_loss += Loss['Response'].item()
            a_loss += Loss['Argument'].item()
            n += y.shape[0]


            steps += 1
            if steps % args.grad_steps == 0:
                update_time += 1

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                
                scheduler.step()

                if update_time % args.log_step == 0:
                    print('[steps {0}] Query loss: {1:.4f}, Res loss: {2:.4f}, Arg loss: {3:.4f}, Arg acc: {4:.4f}'.format(update_time, q_loss / n, r_loss / n, a_loss / n, correct / n))
                    q_loss = 0
                    r_loss = 0
                    a_loss = 0
                    correct = 0
                    n = 0
                if update_time % args.eval_step == 0:
                    print("Validating...", end='\r')
                    valid_score = valid(model, devloader, devset.tokenizer)
                    if valid_score > best_score:
                        torch.save(model.state_dict(), 'best.ckpt')
                        print("Better validation scores, save model to best.ckpt")
                        best_score = valid_score
                    model.train()

def valid(model, devloader, tokenizer):
    # TODO: enumerate all acceptable ground truth, and use the best as acc
    model.eval()
    with torch.no_grad():
        n = 0
        correct = 0
        q_acc = []
        r_acc = []
        for ID, Q, R, Y, Q_s, Q_e, R_s, R_e in devloader:
            Q = {k: v.to(args.device) for k, v in Q.items()}
            R = {k: v.to(args.device) for k, v in R.items()}
            Y = Y.to(args.device)

            Q_s = Q_s.to(args.device)
            Q_e = Q_e.to(args.device)
            R_s = R_s.to(args.device)
            R_e = R_e.to(args.device)

            hyp_q_span, hyp_r_span, y_logits, Loss = model(Q, R, Y, Q_s, Q_e, R_s, R_e)
            
            y = (y_logits.detach().view(-1).cpu().numpy() > 0.5) * 1
            
            q_hyp = []
            r_hyp = []
            q_tgt = []
            r_tgt = []
            
            for i in range(Q['input_ids'].size(0)):
                q_hyp.append(tokenizer.decode(Q['input_ids'][i][hyp_q_span[0][i]: hyp_q_span[1][i] + 1], skip_special_tokens=True))
                r_hyp.append(tokenizer.decode(R['input_ids'][i][hyp_r_span[0][i]: hyp_r_span[1][i] + 1], skip_special_tokens=True))

                q_tgt.append(tokenizer.decode(Q['input_ids'][i][Q_s[i]: Q_e[i] + 1], skip_special_tokens=True))
                r_tgt.append(tokenizer.decode(R['input_ids'][i][R_s[i]: R_e[i] + 1], skip_special_tokens=True))
                
                q_com = longestCommonSubsequence(q_hyp[-1].split(), q_tgt[-1].split())
                r_com = longestCommonSubsequence(r_hyp[-1].split(), r_tgt[-1].split())

                # TODO: the acc of too long ground truth is not calculated
                if len(q_tgt[-1].split()) > 0 and len(r_tgt[-1].split()) > 0:
                    q_acc.append(q_com / (len(q_hyp[-1].split()) + len(q_tgt[-1].split()) - q_com))
                    r_acc.append(r_com / (len(r_hyp[-1].split()) + len(r_tgt[-1].split()) - r_com))

            correct += np.sum(y == Y.detach().view(-1).cpu().numpy(), -1)
            n += y.shape[0]
    
    q_acc = sum(q_acc) / len(q_acc)
    r_acc = sum(r_acc) / len(r_acc)
    mean_acc = (q_acc + r_acc) / 2
    print("[valid] acc: {0:.4f}, q_acc: {1:.4f}, r_acc: {2:.4f}, mean_acc: {3:.4f}".format(correct / n, q_acc, r_acc, mean_acc))
    
    return mean_acc

def inference():
    # TODO
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", '-m', type=str, required=True, help="Train or inference")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Pretrained model name")
    parser.add_argument("--device", '-d', type=str, default="cuda:0", help="Training on which device")
    parser.add_argument("--epoch", '-e', type=int, default=100, help="Numbers of epoch")
    parser.add_argument("--batch_size", '-bs', type=int, default=2, help="Batch size")
    parser.add_argument("--grad_steps", '-gs', type=int, default=16, help="Gradient accumulation steps, 1080 sucks TAT")
    parser.add_argument("--optim", type=str, default='AdamW,0.0001,1500,5000', help="optimizer config: \"type,lr,warmup,allsteps\" ex: AdamW,0.0001,1000,2000")
    parser.add_argument("--log_step", type=int, default=100, help="log steps")
    parser.add_argument("--eval_step", type=int, default=300, help="evaluation steps")
    parser.add_argument("--data_path", type=str, default="./data/", help="Path to a data folder containing [train.csv, valid.csv, test.csv]")
    parser.add_argument("--ckpt", type=str, default="", help="Path to a model ckpt for initialize the model")
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        assert args.ckpt is not None
        inference(args)
    else:
        raise NotImplementedError("Not supported mode, choose from train or inference.")