import torch
import torch.nn as nn
from transformers import AutoModel

class albert_labeler(nn.Module):
    def __init__(self):
        """ Init the labeler module """
        super(albert_labeler, self).__init__()

        self.albert = AutoModel.from_pretrained('albert-base-v2')
        #size of the output of transformer's last layer
        hidden_size = 768
        #classes: present, absent, unknown, blank for 12 conditions + support devices
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        #classes: yes, no for the 'no finding' observation
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    def forward(self, source_padded, attention_mask):
        """ Forward pass of the labeler
        @param source_padded (torch.LongTensor): Tensor of word indices with padding, shape (batch_size, max_len)
        @param attention_mask (torch.Tensor): Mask to avoid attention on padding tokens, shape (batch_size, max_len)
        @returns out (List[torch.Tensor])): A list of size 14 containing tensors. The first 13 have shape 
                                            (batch_size, 4) and the last has shape (batch_size, 2)  
        """
        #shape (batch_size, max_len, hidden_size)
        final_hidden = self.albert(source_padded, attention_mask=attention_mask)[0]
        #shape (batch_size, hidden_size)
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        out = []
        for i in range(14):
            out.append(self.linear_heads[i](cls_hidden))
        return out
