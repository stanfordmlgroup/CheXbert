import copy
import pandas as pd
import numpy as np
import utils
from sklearn.metrics import confusion_matrix
from statsmodels.stats.inter_rater import cohens_kappa

def do_eval(label_path, truth_path):
    """Print evaluation of chexpert labeled reports.
    @param label_path (string): path to chexpert labeled reports
    @param truth_path (string): path to ground truth
    """
    label = pd.read_csv(label_path)
    truth = pd.read_csv(truth_path)
    f1_weights = utils.get_weighted_f1_weights(truth_path)
    
    label.replace(0, 2, inplace=True)
    label.replace(-1, 3, inplace=True)
    label.fillna(0, inplace=True)
    
    truth.replace(0, 2, inplace=True)
    truth.replace(-1, 3, inplace=True)
    truth.fillna(0, inplace=True)

    y_true = np.array([truth[cond].to_list() for cond in utils.CONDITIONS])
    y_pred = np.array([label[cond].to_list() for cond in utils.CONDITIONS])

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

    for j in range(len(utils.CONDITIONS)):
        print('%s kappa: %.3f' % (utils.CONDITIONS[j], kappas[j]))
    print('average: %.3f' % np.mean(kappas))

    print()
    for j in range(len(utils.CONDITIONS)):
        print('%s weighted_f1: %.3f' % (utils.CONDITIONS[j], weighted[j]))
    print('Average of weighted_f1: %.3f' % (np.mean(weighted)))
    
    print()
    for j in range(len(utils.CONDITIONS)):
         print('%s blank_f1:  %.3f, negation_f1: %.3f, uncertain_f1: %.3f, positive_f1: %.3f' % (utils.CONDITIONS[j],
                                                                                                 blank_f1[j],
                                                                                                 negation_f1[j],
                                                                                                 uncertain_f1[j],
                                                                                                 positive_f1[j]))
    men_macro_avg = np.mean(mention_f1)
    neg_macro_avg = np.mean(negation_f1[:-1]) #No Finding has no negations
    unc_macro_avg = np.mean(uncertain_f1[:-2]) #No Finding, Support Devices have no uncertain labels in test set
    pos_macro_avg = np.mean(positive_f1)
    blank_macro_avg = np.mean(blank_f1)
    
    print("blank macro avg: %.3f, negation macro avg: %.3f, uncertain macro avg: %.3f, positive macro avg: %.3f" % (blank_macro_avg,
                                                                                                                    neg_macro_avg,
                                                                                                                    unc_macro_avg,
                                                                                                                    pos_macro_avg))
    print()
    for j in range(len(utils.CONDITIONS)):
        print('%s mention_f1: %.3f' % (utils.CONDITIONS[j], mention_f1[j]))
    print('mention macro avg: %.3f' % men_macro_avg)
    
        
if __name__ == '__main__':
    truth_path = '/data3/aihc-winter20-chexbert/MIMIC-CXR/mimic_test_ann1.csv'
    label_path = '/data3/aihc-winter20-chexbert/chexpert-labeler/mimic_ann1_labeled.csv'
    do_eval(label_path, truth_path)
