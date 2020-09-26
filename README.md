
# CheXbert: Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT

CheXbert is an accurate, automated deep-learning based chest radiology report labeler that can label for the following 14 common medical conditions: Pneumonia, Fracture, Consolidation, Enlarged Cardiomediastinum, No Finding, Pleural Other, Cardiomegaly, Pneumothorax, Atelectasis, Support Devices, Edema, Pleural Effusion, Lung Lesion, Lung Opacity

Preprint: https://arxiv.org/abs/2004.09167

## Abstract

The extraction of labels from radiology text reports enables large-scale training of medical imaging models. Existing approaches to report labeling typically rely either on sophisticated feature engineering based on medical domain knowledge or manual annotations by experts. In this work, we introduce a BERT-based approach to medical image report labeling that exploits both the scale of available rule-based systems and the quality of expert annotations. We demonstrate superior performance of a biomedically pretrained BERT model first trained on annotations of a rulebased labeler and then finetuned on a small set of expert annotations augmented with automated backtranslation. We find that our final model, CheXbert, is able to outperform the previous best rules-based labeler with statistical significance, setting a new SOTA for report labeling on one of the largest datasets of chest x-rays.

## Prerequisites 

Create conda environment

- conda env create -f environment.yml

Activate environment

- conda activate chexbert

By default, all available GPU's will be used for labeling in parallel. If there is no GPU, the CPU is used. You can control which GPU's are used by appropriately setting CUDA_VISIBLE_DEVICES. The batch size by default is 18, but can be changed inside constants.py

## Usage

Put all reports in a csv file under the column name "Report Impression". Let the absolute path to this csv be {path to reports}. Download the pytorch checkpoint and let the absolute path to it be {path to checkpoint}. Let the absolute path to your desired output folder by {path to output dir}. 

- python label.py -d={path to reports} -o={path to output dir} -c={path to checkpoint} 

The output file with labeled reports is {path to output dir}/labeled_reports.csv

Run the following for descriptions of all command line arguments:

- python label.py -h

** Ignore any error messages about the size of the report exceeding 512 tokens. All reports are automatically cut off at 512 tokens. **

# Label Convention

The labeler outputs the following numbers corresponding to classes:

- Blank: 0
- Positive: 1
- Negative: 2
- Uncertain: 3
