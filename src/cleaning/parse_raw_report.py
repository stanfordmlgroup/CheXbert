import os
import pandas as pd

def getIndexOfLast(l, element):
    """ Get index of last occurence of element

    @param l (list): list of elements
    @param element (string): element to search for
    @returns (int): index of last occurrence of element
    """
    i = max(loc for loc, val in enumerate(l) if val == element)
    return i

def parse_ct(path):
    """ Parse CT scan reports in csv file for impression

    @parameter path (string): path to CT scan csv file
    @returns (pandas dataframe): pandas dataframe with two columns:
                                 index and impression
    """
    orig_df = pd.read_csv(path)
    reports = orig_df['REPORT']
    imps = {"impression": []}
    for i in range(len(reports)):
        s = reports.iloc[i]
        s_split = s.split()
        if "IMPRESSION:" in s_split:
            begin = s_split.index("IMPRESSION:") + 1
            s_split = s_split[begin:]
            end = None
            end_cand1 = None
            end_cand2 = None
            # remove recommendation(s) and notification
            if "RECOMMENDATION(S):" in s_split:
                end_cand1 = s_split.index("RECOMMENDATION(S):")
            elif "RECOMMENDATION:" in s_split:
                end_cand1 = s_split.index("RECOMMENDATION:")
            elif "RECOMMENDATIONS:" in s_split:
                end_cand1 = s_split.index("RECOMMENDATIONS:")

            if "NOTIFICATION:" in s_split:
                end_cand2 = s_split.index("NOTIFICATION:")
            elif "NOTIFICATIONS:" in s_split:
                end_cand2 = s_split.index("NOTIFICATIONS:")

            if end_cand1 and end_cand2:
                end = min(end_cand1, end_cand2)
            elif end_cand1:
                end = end_cand1
            elif end_cand2:
                end = end_cand2

            if end == None:
                imp = " ".join(s_split)
            else:
                imp = " ".join(s_split[:end])

            if 'I have personally reviewed the images for this examination' in imp:
                ind = imp.index('I have personally reviewed the images for this examination')
                imp = imp[:ind]

            if 'Physician to Physician Radiology Consult Line' in imp:
                ind = imp.index('Physician to Physician Radiology Consult Line')
                imp = imp[:ind]

            if 'Reference:' in imp:
                ind = imp.index('Reference:')
                imp = imp[:ind]
                
            imps["impression"].append(imp)
            
        else:
            print("No impression for %d" % i)

    df = pd.DataFrame(data=imps)
    df.rename(columns = {'impression': 'Report Impression'}, inplace=True)
    df.to_csv("/data3/aihc-winter20-chexbert/CT/ct_master.csv", index=False)
    return df

def parse_mimic(directory):
    """ Parse radiology reports in directory for impression.
    
    @parameter directory (string): directory name containing reports
    @returns (pandas dataframe): pandas dataframe with two columns:
                                 impression and filename. 
    """
    imps = {"filename": [], "impression": []}
    for filename in os.listdir(directory):
        f = open(directory + "/" + filename, "r")
        s = f.read()
        s_split = s.split()
        print(filename)
        if "IMPRESSION:" in s_split:
            begin = getIndexOfLast(s_split, "IMPRESSION:") + 1
            end = None
            end_cand1 = None
            end_cand2 = None
            # remove recommendation(s) and notification
            if "RECOMMENDATION(S):" in s_split:
                end_cand1 = s_split.index("RECOMMENDATION(S):")
            elif "RECOMMENDATION:" in s_split:
                end_cand1 = s_split.index("RECOMMENDATION:")
            elif "RECOMMENDATIONS:" in s_split:
                end_cand1 = s_split.index("RECOMMENDATIONS:")

            if "NOTIFICATION:" in s_split:
                end_cand2 = s_split.index("NOTIFICATION:")
            elif "NOTIFICATIONS:" in s_split:
                end_cand2 = s_split.index("NOTIFICATIONS:")

            if end_cand1 and end_cand2:
                end = min(end_cand1, end_cand2)
            elif end_cand1:
                end = end_cand1
            elif end_cand2:
                end = end_cand2            

            if end == None:
                imp = " ".join(s_split[begin:])
            else:
                imp = " ".join(s_split[begin:end])
            imps["impression"].append(imp)
            imps["filename"].append(filename)
        else:
            print("No impression")
    df = pd.DataFrame(data=imps)
    df.to_csv("/data3/aihc-winter20-chexbert/MIMIC-CXR/mimic_master.csv", index=False)
    return df

if __name__ == "__main__":
    parse_ct("/data3/aihc-winter20-chexbert/CT/chest_CT.csv")
