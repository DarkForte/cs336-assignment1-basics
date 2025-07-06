import torch
import math

def cosine_schedule(t, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if t < warmup_iters:
        return max_learning_rate * (t / warmup_iters)
    elif t <= cosine_cycle_iters:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
    else:
        return min_learning_rate

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, weight_decay, eps):
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                m = self.state[p].get('m', torch.zeros_like(p.data))
                v = self.state[p].get('v', torch.zeros_like(p.data))

                m = m * group['beta1'] + (1 - group['beta1']) * grad
                v = v * group['beta2'] + (1 - group['beta2']) * grad * grad
                self.state[p]['m'] = m
                self.state[p]['v'] = v

                t = self.state[p].get('t', 1)

                lr = group['lr'] * math.sqrt(1 - group['beta2'] ** t) / (1 - group['beta1'] ** t)
                p.data = p.data - lr * (m / (v.sqrt() + group["eps"]))
                p.data = p.data - group["lr"] * group['weight_decay'] * p.data
                t += 1
                self.state[p]['t'] = t

