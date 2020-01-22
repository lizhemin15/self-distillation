import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean', smooth_eps=0):
        super().__init__()

        self.reduction = reduction
        self.smooth_eps = smooth_eps

    def forward(self, input, target):
        if self.smooth_eps != 0:
            N = input.size(-1) - 1
            smooth_target = torch.zeros_like(input).fill_(self.smooth_eps / N)
            smooth_target.scatter_(-1, target.unsqueeze(-1),
                                   1 - self.smooth_eps)
            input = F.log_softmax(input, dim=-1)
            output = F.kl_div(input, smooth_target, reduction='none').sum(-1)

            if self.reduction == 'sum':
                output = output.sum()
            elif self.reduction == 'mean':
                output = output.mean()
            return output

        else:
            return F.cross_entropy(input, target, reduction=self.reduction)

class InterpolationLoss1(nn.Module):
    def __init__(self, reduction='mean', lam=0.2):
        super().__init__()

        self.reduction = reduction
        self.lam = lam
        self.steps = 0

    def forward(self, input, target):
        self.steps += 1
        if self.lam != 0:
            N = input.size(-1)
            onehot_target = torch.zeros_like(input)
            onehot_target.scatter_(-1, target.unsqueeze(-1), 1)
            tmp = F.softmax(input, dim=-1).detach()
            interpolation_target = (1-self.lam) * onehot_target + self.lam * tmp
            input = F.log_softmax(input, dim=-1)
            output = F.kl_div(input, interpolation_target, reduction='none').sum(-1)

            if self.reduction == 'sum':
                output = output.sum()
            elif self.reduction == 'mean':
                output = output.mean()
            return output

        else:
            return F.cross_entropy(input, target, reduction=self.reduction)
        
class InterpolationLoss2(nn.Module):
    def __init__(self, reduction='mean', lam=0.1):
        super().__init__()

        self.reduction = reduction
        self.lam = lam
        self.steps = 0

    def forward(self, input, target, predict):
        self.steps += 1
        if self.lam != 0:
            N = input.size(-1)
            onehot_target = torch.zeros_like(input)
            onehot_target.scatter_(-1, target.unsqueeze(-1), 1)
            tmp = F.softmax(predict, dim=-1).detach()
            interpolation_target = (1-self.lam) * onehot_target + self.lam * tmp
            input = F.log_softmax(input, dim=-1)
            output = F.kl_div(input, interpolation_target, reduction='none').sum(-1)
           
            if self.reduction == 'sum':
                output = output.sum()
            elif self.reduction == 'mean':
                output = output.mean()
            return output

        else:
            return F.cross_entropy(input, target, reduction=self.reduction)
        
class InterpolationLoss3(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()

        self.reduction = reduction
        self.steps = 0

    def forward(self, input, target):
        self.steps += 1
        if 1 != 0:
            N = target.size(-1)
            onehot_target = torch.zeros_like(input)
            onehot_target.scatter_(-1, target.unsqueeze(-1), 1)
            tmp = F.softmax(input, dim=-1).detach()
            
            _, predicted = input.max(1)
            lam =1*(predicted.eq(target).sum().item())/N
  
            _, tmp = input.max(1)
            pre_target = torch.zeros_like(input)
            pre_target.scatter_(-1, tmp.unsqueeze(-1), 1)
            interpolation_target = ((1-lam) * onehot_target + lam * pre_target).detach()

            input = F.log_softmax(input, dim=-1)
            output = F.kl_div(input, interpolation_target, reduction='none').sum(-1)
           
            if self.reduction == 'sum':
                output = output.sum()
            elif self.reduction == 'mean':
                output = output.mean()
            return output

        else:
            return F.cross_entropy(input, target, reduction=self.reduction)

class InterpolationLoss4(nn.Module):
    def __init__(self, reduction='mean', lam=0.1):
        super().__init__()

        self.reduction = reduction
        self.lam = lam
        self.steps = 0

    def forward(self, input, target):
        self.steps += 1
        if self.lam != 0:
            N = input.size(-1)
            onehot_target = torch.zeros_like(input)
            onehot_target.scatter_(-1, target.unsqueeze(-1), 1)
            _, tmp = input.max(1)
            pre_target = torch.zeros_like(input)
            pre_target.scatter_(-1, tmp.unsqueeze(-1), 1)
            interpolation_target = ((1-self.lam) * onehot_target + self.lam * pre_target).detach()
            input = F.log_softmax(input, dim=-1)
            output = F.kl_div(input, interpolation_target, reduction='none').sum(-1)

            if self.reduction == 'sum':
                output = output.sum()
            elif self.reduction == 'mean':
                output = output.mean()
            return output

        else:
            return F.cross_entropy(input, target, reduction=self.reduction)

class InterpolationLoss5(nn.Module):
    def __init__(self, reduction='mean', t=0):
        super().__init__()

        self.reduction = reduction
        self.steps = 0
        self.t=t

    def forward(self, input, target):
        self.steps += 1
        if 1 != 0:
            N = target.size(-1)
            onehot_target = torch.zeros_like(input)
            onehot_target.scatter_(-1, target.unsqueeze(-1), 1)
            tmp = F.softmax(input, dim=-1).detach()

            _, predicted = input.max(1)
            lam = 0.9*(predicted.eq(target).sum().item())/N

            if self.t ==0:
               _, tmp = input.max(1)
               pre_target = torch.zeros_like(input)
               pre_target.scatter_(-1, tmp.unsqueeze(-1), 1)
               interpolation_target = ((1-lam) * onehot_target + lam * pre_target).detach()
            else:
               pre_target = F.softmax(input / self.t, dim=-1).detach()

            interpolation_target = ((1-lam) * onehot_target + lam * pre_target).detach()
#            interpolation_target = ((1-lam) * onehot_target + lam * tmp).detach()
            input = F.log_softmax(input, dim=-1)
            output = F.kl_div(input, interpolation_target, reduction='none').sum(-1)

            if self.reduction == 'sum':
                output = output.sum()
            elif self.reduction == 'mean':
                output = output.mean()
            return output

        else:
            return F.cross_entropy(input, target, reduction=self.reduction)

class TestCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean', smooth_eps=0):
        super().__init__()

        self.reduction = reduction
        self.smooth_eps = smooth_eps

    def forward(self, input, target):
        N = input.size(-1) - 1
        smooth_target = torch.zeros_like(input).fill_(self.smooth_eps / N)
        smooth_target.scatter_(-1, target.unsqueeze(-1), 1 - self.smooth_eps)
        input = F.log_softmax(input, dim=-1)
        output = F.kl_div(input, smooth_target, reduction='none').sum(-1)

        if self.reduction == 'sum':
            output = output.sum()
        elif self.reduction == 'mean':
            output = output.mean()
        return output


if __name__ == "__main__":
    c = TestCrossEntropyLoss(reduction='none', smooth_eps=0)
    # c = TestCrossEntropyLoss(smooth_eps=0.1)
    logits = torch.ones(2, 2)
    labels = torch.Tensor([0, 1]).to(torch.long)
    print(c(logits, labels))
