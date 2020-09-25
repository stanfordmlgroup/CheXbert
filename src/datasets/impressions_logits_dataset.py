import torch
import pandas as pd
import numpy as np
from bert_tokenizer import load_list
from torch.utils.data import Dataset, DataLoader

class ImpressionsLogitsDataset(Dataset):
        """The dataset to contain report impressions and their labels."""
        
        def __init__(self, logits_path, list_path):
                """ Initialize the dataset object
                @param logits_path (string): path to the list of logits
                @param list_path (string): path to the list of encoded impressions
                """
                raw_logits = load_list(path=logits_path)
                self.logits_13 = torch.Tensor(raw_logits[:13])
                self.logits_13 = self.logits_13.permute(1, 0, 2)
                self.logits_no_finding = torch.Tensor(raw_logits[13])
                self.encoded_imp = load_list(path=list_path)

        def __len__(self):
                """Compute the length of the dataset

                @return (int): size of the dataset
                """
                return len(self.encoded_imp)

        def __getitem__(self, idx):
                """ Functionality to index into the dataset
                @param idx (int): Integer index into the dataset

                @return (dictionary): Has keys 'imp', 'logits_13', logits_no_finding, and 'len'. 
                                      The value of 'imp' is a LongTensor of an encoded impression. 
                                      The values of 'logits_13' and 'logits_no_finding' are tensors 
                                      containing the logits for 13 conditions and no finding respectively,
                                      and 'the value of 'len' is an integer representing the length of 
                                      imp's value
                """
                if torch.is_tensor(idx):
                        idx = idx.tolist()
                logits_13 = self.logits_13[idx]
                logits_no_finding = self.logits_no_finding[idx]
                imp = self.encoded_imp[idx]
                imp = torch.LongTensor(imp)
                return {"imp": imp, 
                        "logits_13": logits_13, 
                        "logits_no_finding": logits_no_finding, 
                        "len": imp.shape[0]}
