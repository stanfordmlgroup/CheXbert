import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def mimic_train_dev():
    """Create 85-15% train-dev split on MIMIC-CXR reports labeled by CheXpert labeler.
    """
    df = pd.read_csv('/data3/aihc-winter20-chexbert/MIMIC-CXR/chexpert_labeled/mimic_notest.csv')
    train, dev = train_test_split(df, test_size=.15, random_state=42)
    print("train: ", train.shape)
    print("dev: ", dev.shape)
    dev.to_csv("/data3/aihc-winter20-chexbert/MIMIC-CXR/chexpert_labeled/mimic_dev.csv", index=False)
    train.to_csv("/data3/aihc-winter20-chexbert/MIMIC-CXR/chexpert_labeled/mimic_train.csv", index=False)
    
def clean_chexpert_master(k=None):
    """ Function removes 1000 test set reports from master csv file, conducts train / dev split,
    and saves train / dev data as master_train.csv and master_dev.csv respectively. If k is not
    None, then it creates k train-dev splits that are non-overlapping and saves them to disk.
    """
    df_master = pd.read_csv('/data3/CXR-CHEST/DATA/CheXpert/Master/master.csv')
    df_test = pd.read_csv('/data3/aihc-winter20-chexbert/chexpert_data/original_test_set.csv')
    master_id = df_master['SimpleTestReportID']
    report_id = df_test['SimpleTestReportID'].to_numpy()
    mask = [False if master_id.iloc[i] in report_id else True for i in range(len(df_master))]
    df_new_master = df_master[mask]
    #df_new_master.to_csv('/data3/aihc-winter20-chexbert/new_master.csv', index=False)
    mask = ~df_new_master['Report #'].duplicated()
    df_new_master = df_new_master[mask]

    if k is None:
        master_train, master_dev = train_test_split(df_new_master, test_size=.15, random_state=42)
        master_train.to_csv('/data3/aihc-winter20-chexbert/master_train.csv', index=False)
        master_dev.to_csv('/data3/aihc-winter20-chexbert/master_dev.csv', index=False)
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        kf.get_n_splits(df_new_master)
        c = 0
        for train_idx, dev_idx in kf.split(df_new_master):
            train_df, dev_df = df_new_master.iloc[train_idx], df_new_master.iloc[dev_idx]
            train_df.to_csv('/data3/aihc-winter20-chexbert/chexpert_data/folds/train_%d.csv' % c, index=False)
            dev_df.to_csv('/data3/aihc-winter20-chexbert/chexpert_data/folds/dev_%d.csv' % c, index=False)
            c += 1

if __name__ == "__main__":
    mimic_train_dev()
