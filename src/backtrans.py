import copy
import torch
import pandas as pd
import numpy as np

BEAM = 2  #how many translations are produced upon each generate call
TOPK = 20  #use top-k sampling to generate diverse outputs

def get_impressions(path):
    """Return the report impressions in the csv 
    @param path (str): path to the csv file

    @returns (List[str]): list of report impressions 
    """
    imp = pd.read_csv(path)['Report Impression']
    return imp.to_list()

def generate(model, text, beam, sampling_topk):
    """Generate translations for input string 
    @param model (torch.nn.Module): the fairseq nmt model
    @param text (str): input string
    @param beam (int): number of translations generated 
    @param sampling_topk (int): top-k sampling for more diverse output 

    @param res (List[str]): list of translations in target language
    """
    toks = model.tokenize(text)
    bpe = model.apply_bpe(toks)
    source_bin = model.binarize(bpe)
    target_bin = model.generate(source_bin, beam=beam, sampling=False)

    res = []
    for i in range(beam):
        target_sample = target_bin[i]['tokens']
        target_bpe = model.string(target_sample)
        target_toks = model.remove_bpe(target_bpe)
        target = model.detokenize(target_toks)
        res.append(target)

    return res
        
def back_trans(imps, beam=BEAM, sampling_topk=TOPK):
    """Perform back-translation 
    @param imps (List[str]): list of report impressions
    @param beam (int): number of translations generated
    @param sampling_topk (int): top-k sampling for more diverse output

    @param res (List[List[str]]): list of translations for each report in
                                  the imps parameter
    """
    
    #German models
    en2de = torch.hub.load('pytorch/fairseq',
                           'transformer.wmt19.en-de.single_model',
                           tokenizer='moses',
                           bpe='fastbpe')
    de2en = torch.hub.load('pytorch/fairseq',
                           'transformer.wmt19.de-en.single_model',
                           tokenizer='moses',
                           bpe='fastbpe')

    print("\nBegin translation")
    res = [[] for _ in range(len(imps))]
    for j in range(len(imps)):
        print("Working on report %d" % j)
        imp = imps[j]
        #translate to German
        translations = generate(en2de, imp, beam=beam, sampling_topk=sampling_topk)

        #back translate each German report to English
        back_trans = []
        for trans in translations:
            back_trans += generate(de2en, trans, beam=beam, sampling_topk=sampling_topk)
        back_trans = list(set(back_trans))
        res[j] += back_trans
            
    return res

def write_output(translations, csv_path, save_path):
    """Write the translations with labels to a csv file.
    @param translations (List[List[str]]): has a list of translations for each report
    @param csv_path (str): path to the file with labels
    @param save_path (str): path for saving output
    """
    assert(save_path != csv_path)
    df = pd.read_csv(csv_path)
    imp_idx = df.columns.to_list().index('Report Impression')

    output_df = []
    for i in range(len(translations)):
        orig_row = df.iloc[i].to_list()
        output_df.append(orig_row)

        for j in range(len(translations[i])):
            new_row = copy.deepcopy(orig_row)
            new_row[imp_idx] = translations[i][j]
            output_df.append(new_row)

    output_df = pd.DataFrame.from_records(output_df, columns=df.columns)
    output_df = output_df.sample(frac=1)
    output_df.to_csv(save_path, index=False)
    
        
if __name__ == '__main__':
    imp_250 = get_impressions('250.csv')
    imp_750 = get_impressions('750.csv')

    res = back_trans(imp_250)
    write_output(res, '250.csv', '250_btx2.csv')

    res = back_trans(imp_750)
    write_output(res, '750.csv', '750_btx2.csv')
