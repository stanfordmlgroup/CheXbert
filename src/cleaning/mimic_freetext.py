import os
import pandas as pd

def extract(directory):
    """ Extract full MIMIC reports from directory
    
    @parameter directory (string): directory name containing reports
    @returns (pandas dataframe): pandas dataframe with two columns:
                                 report and filename. 
    """
    imps = {"filename": [], "report": []}
    i = 0
    for filename in os.listdir(directory):
        f = open(directory + "/" + filename, "r")
        s = f.read()
        imps["report"].append(s)
        imps["filename"].append(filename)
        i += 1
        if i % 10000 == 0:
            print(i)
        
    df = pd.DataFrame(data=imps)
    df.to_csv("/data3/aihc-winter20-chexbert/MIMIC-CXR/mimic_freetext.csv", index=False)
    return df

if __name__ == "__main__":
    extract("/data3/aihc-winter20-chexbert/MIMIC-CXR/mimic-combined")
