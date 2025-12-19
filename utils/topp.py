import torch

def top_p(logits, p):
    logits_s, indices = torch.sort(logits, descending=True)
    logits_n = torch.nn.functional.softmax(logits_s)
    
    logits_sum = torch.cumsum(logits_n)
    
    mask = logits_sum > p
    
    logits_s = logits_s.masked_fill(mask, float('-inf'))
    
    logits_nv2 = torch.nn.functional.softmax(logits_s)
    
    idx = torch.multinomial(logits_nv2, num_samples=1)
    
    idx_f = torch.gather(indices, dim = 1, index=idx)
    
    return idx_f