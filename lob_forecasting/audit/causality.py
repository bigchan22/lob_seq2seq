def max_earlier_logit_difference(model, original, changed, cutoff):
 import torch
 model.eval()
 with torch.no_grad(): return (model(original)[:,:cutoff]-model(changed)[:,:cutoff]).abs().max().item()
