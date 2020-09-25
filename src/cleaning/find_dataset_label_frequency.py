import pandas as pd

def getLabelFrequencies(df, conditions):
    """ Gets frequency of labels in a dataframe.
    
    @param df (pandas dataframe): dataframe containing labels for conditions
                            in each report
    @param conditions: list of conditions to report statistics for
    """
    for condition in conditions:
        positive = master[condition].value_counts()[1]
        blank = len(master) - master[condition].value_counts().sum()
        if condition == "No Finding":
            print(condition, positive, blank)
            print(condition, positive / len(master), blank / len(master))
        else:
            negative = master[condition].value_counts()[0]
            uncertain = master[condition].value_counts()[-1]
            print(condition, positive, negative, uncertain, blank)
            print(condition, positive / len(master), negative / len(master), uncertain / len(master), blank / len(master))

if __name__ == '__main__':
    master_train = pd.read_csv("/data3/aihc-winter20-chexbert/master_train.csv")
    master_dev = pd.read_csv("/data3/aihc-winter20-chexbert/master_dev.csv")
    master = pd.concat([master_train, master_dev], ignore_index=True)
    conditions = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
    getLabelFrequencies(master, conditions)
    
