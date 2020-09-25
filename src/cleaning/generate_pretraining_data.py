import pandas as pd
from nltk.tokenize import sent_tokenize

def generate_pretraining_file(in_files, out_file):
    """ Generates pretraining txt file in proper form for BERT.
    File contains sentence per line separated by newline for new report.

    @param in_files (List): list of csv files to process
    @param out_file (string): filename to write to
    """
    imps = []
    for in_file in in_files:
        df = pd.read_csv(in_file)
        imp = df['Report Impression']
        imp = imp.str.strip()
        imp = imp.replace('\n',' ', regex=True)
        imp = imp.replace('[0-9]\.', '', regex=True)
        imp = imp.replace('\s+', ' ', regex=True)
        imp = imp.str.strip()
        imp = imp.to_list()
        imps = imps + imp

    # Write sentences to file
    f = open(out_file, "w")
    for impression in imps:
        sent_list = sent_tokenize(impression)
        for sent in sent_list:
            f.write(sent + '\n')
        f.write('\n')
    f.close()

if __name__ == "__main__":
    input_files = ["/data3/aihc-winter20-chexbert/master_train.csv", 
                   "/data3/aihc-winter20-chexbert/master_dev.csv",
                   "/data3/aihc-winter20-chexbert/MIMIC-CXR/mimic_master_notest.csv"]
    generate_pretraining_file(input_files, "/data3/aihc-winter20-chexbert/bert/BERT_pretraining_corpus.txt")
