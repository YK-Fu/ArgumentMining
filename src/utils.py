from torch.optim import Adam, AdamW, SGD
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

def get_optim(parameter, optim_cfg, schedule_cfg):
    optim, lr, momentum = optim_cfg.split(',')
    scheduler, warmup, steps, cycles = schedule_cfg.split(',')

    if optim == 'AdamW':
        optimizer = AdamW(parameter, float(lr))
    elif optim == 'Adam':
        optimizer = Adam(parameter, float(lr))
    elif optim == 'SGD':
        optimizer = SGD(parameter, float(lr), momentum=float(momentum))
    else:
        raise NotImplementedError("Not supported optimizer type.")

    if scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup), num_training_steps=int(steps))
    elif scheduler == 'cosine_warmup':    
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup), num_training_steps=int(steps), num_cycles=int(cycles))
    elif scheduler == 'cosine_warmup_start':
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup), num_training_steps=int(steps), num_cycles=int(cycles))
    else:
        scheduler = None
        print("WARMING: not supported scheduler type, optimize withou scheduler.")
    
    return optimizer, scheduler


def longestCommonSubsequence(text1: list, text2: list) -> int:
    if len(text2) > len(text1):
        text1, text2 = text2, text1
    lcs = [[0] * (len(text2) + 1) for _ in range(2)]
    for i in range(1, len(text1)+1):
        for j in range(1, len(text2)+1):
            if text1[i-1] == text2[j-1]:
                lcs[i % 2][j] = lcs[(i-1) % 2][j-1] + 1
            else:
                lcs[i % 2][j] = max(lcs[(i-1) % 2][j], lcs[i % 2][j-1])
    return lcs[len(text1) % 2][len(text2)]