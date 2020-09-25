import pandas as pd
import transformers
from transformers import BertTokenizer, AutoTokenizer
import json
from tqdm import tqdm

def get_impressions_from_csv(path):	
        df = pd.read_csv(path)
        imp = df['Report Impression']
        imp = imp.str.strip()
        imp = imp.replace('\n',' ', regex=True)
        imp = imp.replace('\s+', ' ', regex=True)
        imp = imp.str.strip()
        return imp

def tokenize(impressions, tokenizer):
        new_impressions = []
        print("\nTokenizing report impressions. All reports are cut off at 512 tokens.")
        for i in tqdm(range(impressions.shape[0])):
                tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
                if tokenized_imp: #not an empty report
                        res = tokenizer.encode_plus(tokenized_imp)['input_ids']
                        if len(res) > 512: #length exceeds maximum size
                                #print("report length bigger than 512")
                                res = res[:511] + [tokenizer.sep_token_id]
                        new_impressions.append(res)
                else: #an empty report
                        new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id]) 
        return new_impressions

def load_list(path):
        with open(path, 'r') as filehandle:
                impressions = json.load(filehandle)
                return impressions

if __name__ == "__main__":
        tokenizer = BertTokenizer.from_pretrained('/data3/aihc-winter20-chexbert/bluebert/pretrain_repo')
        #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        #tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
        
        #impressions = get_impressions_from_csv('/data3/aihc-winter20-chexbert/chexpert_data/master_train.csv')
        #new_impressions = tokenize(impressions, tokenizer)
        #with open('/data3/aihc-winter20-chexbert/bluebert/impressions_lists/train_impressions_list', 'w') as filehandle:
        #        json.dump(new_impressions, filehandle)

        #impressions = get_impressions_from_csv('/data3/aihc-winter20-chexbert/chexpert_data/master_dev.csv')
        #new_impressions = tokenize(impressions, tokenizer)
        #with open('/data3/aihc-winter20-chexbert/bluebert/impressions_lists/dev_impressions_list', 'w') as filehandle:
        #        json.dump(new_impressions, filehandle)

        impressions = get_impressions_from_csv('/data3/aihc-winter20-chexbert/MIMIC-CXR/mimic_686/train_beamx1.csv')
        new_impressions = tokenize(impressions, tokenizer)
        with open('/data3/aihc-winter20-chexbert/bluebert/impressions_lists/mimic_686/train_beamx1', 'w') as filehandle:
                json.dump(new_impressions, filehandle)
