import os
import copy
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import utils
from models.bert_labeler import bert_labeler
from run_bert import load_test_data
from bert_tokenizer import load_list
from scipy import stats
from statsmodels.stats.inter_rater import cohens_kappa
from sklearn.metrics import confusion_matrix

def metrics_from_lists(y_true_path, y_pred_path, true_csv):
    """Compute kappa and weighted-F1 from predictions lists
    @param y_true_path (str): path to list of ground truth
    @param y_pred_path (str): path to list of predictions
    @param true_csv (str): path to the ground truth csv file
    """
    f1_weights = utils.get_weighted_f1_weights(true_csv)
    y_true = np.array(load_list(y_true_path))
    y_pred = np.array(load_list(y_pred_path))
    
    mention_f1 = utils.compute_mention_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred))
    negation_f1 = utils.compute_negation_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred))
    uncertain_f1 = utils.compute_uncertain_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred))
    positive_f1 = utils.compute_positive_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred))
    blank_f1 = utils.compute_blank_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred))

    weighted = []
    kappas = []
    for j in range(len(y_pred)):
        cond = utils.CONDITIONS[j]
        avg = utils.weighted_avg([negation_f1[j], uncertain_f1[j], positive_f1[j]], f1_weights[cond])
        weighted.append(avg)

        mat = confusion_matrix(y_true[j], y_pred[j])
        kappas.append(cohens_kappa(mat, return_results=False))

    print("\nWeighted-F1:")
    for i in range(len(utils.CONDITIONS)):
        cond = utils.CONDITIONS[i]
        print("%s: %.3f" % (cond, weighted[i]))
    print("avg: %.3f" % np.mean(weighted))
        
    #print("\nKappa:")
    #for i in range(len(utils.CONDITIONS)):
    #    cond = utils.CONDITIONS[i]
    #    print("%s: %.3f" % (cond, kappas[i]))
    #print("avg: %.3f" % np.mean(kappas))
    
def save_model_preds(model, checkpoint_path, test_ld, save_path, f1_weights):
    """Load up a model and save its predictions as a list
    @param model (nn.Module): labeler module 
    @param checkpoint_path (string): location of saved model checkpoint 
    @param test_ld (dataloader): dataloader for test set
    @param save_path (string): path to save the predictions list
    @param f1_weights (dictionary): maps conditions to f1 task weights
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) #to utilize multiple GPU's 
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, y_pred, y_true = utils.evaluate(model, test_ld, device, return_pred=True, f1_weights=f1_weights)
    y_pred = [t.tolist() for t in y_pred]
    y_true = [t.tolist() for t in y_true]
    
    with open(os.path.join(save_path, 'y_true'), 'w') as filehandle:
        json.dump(y_true, filehandle)
    with open(os.path.join(save_path, 'student_radio_beamx1_y_pred'), 'w') as filehandle:
        json.dump(y_pred, filehandle)

def compute_paired_diff(true_list_path, pred_list_paths, true_csv_path,
                         bootstrap_size, num_bootstraps, use_kappa=False):
    """ Function to compute paired differences of kappa/F1 
    @param true_list_path (str): path to list containing ground truth
    @param pred_list_paths (List[str]): list of two paths to predictions by
                                        two models. The second model's metrics
                                        are subtracted from the first model's
                                        metrics
    @param true_csv_path (str): path to ground truth csv
    @param bootstrap_size (int): number of reports in each bootstrap
    @param num_bootstrap_size (int): number of bootstrap samples to create
    @param use_kappa (bool): whether to report kappa CI's or F1 CI's
    """
    metric_list_a, avg_list_a = compute_ci(true_list_path, pred_list_paths[0], true_csv_path,
                                           bootstrap_size, num_bootstraps, use_kappa=use_kappa)
    metric_list_b, avg_list_b = compute_ci(true_list_path, pred_list_paths[1], true_csv_path,
                                           bootstrap_size, num_bootstraps, use_kappa=use_kappa)

    diff_list = np.array(metric_list_a) - np.array(metric_list_b)
    diff_list.sort()
    diff_avg_list = np.array(avg_list_a) - np.array(avg_list_b)
    diff_avg_list.sort()

    if use_kappa:
        metric = 'kappa'
    else:
        metric = 'weighted f1'
    
    print("Printing paired differences for %s\n" % metric)
    for i in range(len(utils.CONDITIONS)):
        cond = utils.CONDITIONS[i]
        hi = diff_list[i][974]
        lo = diff_list[i][24]
        print("%s: %.3f, %.3f" % (cond, lo, hi))
    print("avg: %.3f, %.3f" % (diff_avg_list[24], diff_avg_list[974]))

    print("\nPrinting p-value:")
    for i in range(len(utils.CONDITIONS)):
        cond = utils.CONDITIONS[i]
        if np.any(diff_list[i] > 0):
            p = np.argmax(diff_list[i] > 0) / len(diff_list[i])
        else:
            p = 1.0
        print("%s: %.3f" % (cond, p))
    if np.any(diff_avg_list > 0):
        p = np.argmax(diff_avg_list > 0) / len(diff_avg_list)
    else:
        p = 1.0
    print("avg: %.3f" % p)
    
def compute_ci(true_list_path, pred_list_path, true_csv_path,
               bootstrap_size, num_bootstraps, use_kappa=False):
    """ Function to compute confidence intervals
    @param true_list_path (str): path to list containing ground truth
    @param pred_list_path (str): path to list containing predictions
    @param true_csv_path (str): path to ground truth csv
    @param bootstrap_size (int): number of reports in each bootstrap
    @param num_bootstrap_size (int): number of bootstrap samples to create 
    @param use_kappa (bool): whether to report kappa CI's or F1 CI's

    @param returns (tuple): returns two lists of size 14. The first list
                            contains kappa/F1 scores for each of the 14
                            conditions, for each replicate. The second 
                            list contains average kappa/F1 scores for 
                            each replicate
    """
    df = pd.read_csv(true_csv_path)
    y_true = np.array(load_list(true_list_path))
    y_pred = np.array(load_list(pred_list_path))
    np.random.seed(42) #Crucial for deterministic behavior!

    if use_kappa:
        kappa_list = []
        kappa_avg_list = []
        for _ in range(num_bootstraps):
            choices = np.random.choice(np.arange(0, len(df)), size=bootstrap_size, replace=True)
            y_true_new = [t[choices] for t in y_true]
            y_pred_new = [t[choices] for t in y_pred]
            df_new = df.iloc[choices]

            kappas = []
            for i in range(len(utils.CONDITIONS)):
                mat = confusion_matrix(y_true_new[i], y_pred_new[i])
                res = cohens_kappa(mat, return_results=False)
                kappas.append(res)
                
            kappa_list.append(kappas)
            kappa_avg_list.append(np.mean(kappas))

        kappa_list = np.array(kappa_list).T

        copy_list = np.copy(kappa_list)
        copy_list.sort()
        copy_avg_list = np.copy(kappa_avg_list)
        copy_avg_list.sort()
        
        print("\nPrinting unpaired kappa CI's")
        for i in range(len(utils.CONDITIONS)):
            cond = utils.CONDITIONS[i]
            hi = copy_list[i][974]
            lo = copy_list[i][24]
            print("%s: %.3f, %.3f" % (cond, lo, hi))
        print("avg: %.3f, %.3f" % (copy_avg_list[24], copy_avg_list[974]))
        
        return kappa_list, kappa_avg_list
    
    weighted_f1_list = []
    weighted_avg_list = []
    for _ in range(num_bootstraps):
        choices = np.random.choice(np.arange(0, len(df)), size=bootstrap_size, replace=True)
        y_true_new = [t[choices] for t in y_true]
        y_pred_new = [t[choices] for t in y_pred]
        df_new = df.iloc[choices]
        f1_weights = utils.get_weighted_f1_weights(df_new)

        for cond in utils.CONDITIONS:
            if np.sum(f1_weights[cond]) == 0:
                print("bootstrap %d condition %s had zero weights" % (_, cond))
        
        negation_f1 = utils.compute_negation_f1(copy.deepcopy(y_true_new), copy.deepcopy(y_pred_new))
        uncertain_f1 = utils.compute_uncertain_f1(copy.deepcopy(y_true_new), copy.deepcopy(y_pred_new))
        positive_f1 = utils.compute_positive_f1(copy.deepcopy(y_true_new), copy.deepcopy(y_pred_new))

        weighted = []
        for j in range(len(y_pred_new)):
            cond = utils.CONDITIONS[j]
            avg = utils.weighted_avg([negation_f1[j], uncertain_f1[j], positive_f1[j]], f1_weights[cond])
            weighted.append(avg)
            
        weighted_f1_list.append(weighted)
        weighted_avg_list.append(np.mean(weighted))
        
    weighted_f1_list = np.array(weighted_f1_list).T

    copy_list = np.copy(weighted_f1_list)
    copy_list.sort()
    copy_avg_list =np.copy(weighted_avg_list)
    copy_avg_list.sort()

    print("\nPrinting unpaired weighted-F1 CI's")
    for i in range(len(utils.CONDITIONS)):
        cond = utils.CONDITIONS[i]
        hi = copy_list[i][974]
        lo = copy_list[i][24]
        print("%s: %.3f, %.3f" % (cond, lo, hi))
    print("avg: %.3f, %.3f" % (copy_avg_list[24], copy_avg_list[974]))
    
    return weighted_f1_list, weighted_avg_list
    
if __name__ == '__main__':
    #model = bert_labeler()
    #checkpoint_path = '/data3/aihc-winter20-chexbert/bluebert/tblue-rad-bt-labels/labeler_ckpt/student_radio_beamx1/model_epoch1_iter84'
    #save_path = '/data3/aihc-winter20-chexbert/bootstraps/bluebert/tblue-rad-bt-labels/mimic_test_ann1/'
    #test_loader = load_test_data('/data3/aihc-winter20-chexbert/MIMIC-CXR/mimic_test_ann1.csv',
    #                             '/data3/aihc-winter20-chexbert/bluebert/impressions_lists/mimic_test_ann1')
    #f1_weights = utils.get_weighted_f1_weights('/data3/aihc-winter20-chexbert/MIMIC-CXR/mimic_test_ann1.csv')
    #save_model_preds(model, checkpoint_path, test_loader, save_path, f1_weights)


    #metrics_from_lists('/data3/aihc-winter20-chexbert/bootstraps/bluebert/tblue-rad-bt-labels/mimic_test_ann1/y_true',
    #                   '/data3/aihc-winter20-chexbert/bootstraps/bluebert/tblue-rad-bt-labels/mimic_test_ann1/student_radio_beamx1_y_pred',
    #                   '/data3/aihc-winter20-chexbert/MIMIC-CXR/mimic_test_ann1.csv')

    pred_list_paths = ['/data3/aihc-winter20-chexbert/bootstraps/bluebert/mimic_test_ann1/auto_radio_beamx1_y_pred',
                       '/data3/aihc-winter20-chexbert/bootstraps/bluebert/mimic_test_ann1/chexpert_y_pred']
    compute_paired_diff(true_list_path='/data3/aihc-winter20-chexbert/bootstraps/bluebert/mimic_test_ann1/y_true',
                        pred_list_paths=pred_list_paths,
                        true_csv_path='/data3/aihc-winter20-chexbert/MIMIC-CXR/mimic_test_ann1.csv',
                        bootstrap_size=687,
                        num_bootstraps=1000,
                        use_kappa=False)

    
