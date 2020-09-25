import torch
import torch.nn as nn

class FocalLoss(nn.Module):
        """A module to implement the focal loss"""
        def __init__(self, alpha=1.0, gamma=5.0):
                """ Initialize the focal loss module
                @param alpha (float): alpha parameter in focal loss
                @param gamma (float): gamma parameter in focal loss
                """
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')

        def forward(self, inputs, targets):
                """Compute the focal loss in forward pass
                @param inputs (torch.Tensor): outputs from network (logits) with shape
                                              (batch_size, num_classes)
                @param targets (torch.LongTensor): target classes of shape (batch_size)

                @returns (torch.Tensor): scalar Tensor representing the focal loss
                """
                neg_log_p = self.CrossEntropyLoss(inputs, targets)
                p = torch.exp(-neg_log_p)
                focal_loss = self.alpha * (1 - p)**(self.gamma) * neg_log_p
                return focal_loss.sum()


#TESTS
#alpha = 1
#gamma = 2
#inputs = torch.Tensor([[5.3, 4.6, 6.1], [1.1, .9, .4]])
#targets = torch.LongTensor([1, 0])
#loss = FocalLoss(alpha=alpha, gamma=gamma)
#output = loss(inputs, targets)
#print(output)  
