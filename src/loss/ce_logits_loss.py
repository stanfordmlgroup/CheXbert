import torch
import torch.nn as nn
import torch.nn.functional as F

class CELogitsLoss(nn.Module):
        """A module to implement cross entropy with logits on soft labels"""
        def __init__(self, temp=1):
                """ Initialize the focal loss module
                @param temp (float): temperature parameter before applying softmax
                """
                super(CELogitsLoss, self).__init__()
                self.temp = temp

        def forward(self, logits, targets):
                """Compute the ce loss in forward pass
                @param logits (torch.Tensor): outputs from network (logits) with shape
                                              (batch_size, num_classes)
                @param targets (torch.LongTensor): target logits of shape (batch_size, num_classes)

                @returns (torch.Tensor): scalar Tensor representing the ce loss
                """
                #shape of log_prob is (batch_size, num_classes)
                log_probs = F.log_softmax(logits, dim=-1)
                target_probs = F.softmax(targets/self.temp, dim=-1)
                loss = torch.sum(-log_probs * target_probs)
                return loss
                
