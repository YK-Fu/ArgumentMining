from torch.optim import Adam, AdamW, SGD
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

def get_optim(model, config, schedule=True):
    # TODO: customized scheduler
    
    optim, lr, warmup, steps = config.split(',')
    lr = float(lr)
    warmup = int(warmup)
    steps = int(steps)

    if optim == 'AdamW':
        optimizer = AdamW(model.parameters(), lr)
    elif optim == 'Adam':
        optimizer = Adam(model.parameters(), lr)
    elif optim == 'SGD':
        optimizer = SGD(model.parameters(), lr)
    else:
        raise NotImplementedError("Not supported optimizer type.")
    
    scheduler = None
    if schedule:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=steps, num_cycles=3)
    
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