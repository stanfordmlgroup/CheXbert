
# CheXbert: Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT

CheXbert is an accurate, automated deep-learning based chest radiology report labeler that can label for the following 14 common medical conditions: Pneumonia, Fracture, Consolidation, Enlarged Cardiomediastinum, No Finding, Pleural Other, Cardiomegaly, Pneumothorax, Atelectasis, Support Devices, Edema, Pleural Effusion, Lung Lesion, Lung Opacity

Preprint: https://arxiv.org/abs/2004.09167

## Abstract

The extraction of labels from radiology text reports enables large-scale training of medical imaging models. Existing approaches to report labeling typically rely either on sophisticated feature engineering based on medical domain knowledge or manual annotations by experts. In this work, we introduce a BERT-based approach to medical image report labeling that exploits both the scale of available rule-based systems and the quality of expert annotations. We demonstrate superior performance of a biomedically pretrained BERT model first trained on annotations of a rulebased labeler and then finetuned on a small set of expert annotations augmented with automated backtranslation. We find that our final model, CheXbert, is able to outperform the previous best rules-based labeler with statistical significance, setting a new SOTA for report labeling on one of the largest datasets of chest x-rays.

![The CheXbert approach](figures/chexbert.png)

## Prerequisites 

Create conda environment

```
conda env create -f environment.yml
```

Activate environment

```
conda activate chexbert
```

By default, all available GPU's will be used for labeling in parallel. If there is no GPU, the CPU is used. You can control which GPU's are used by appropriately setting CUDA_VISIBLE_DEVICES. The batch size by default is 18, but can be changed inside constants.py

## Usage

### Label reports with CheXbert

Put all reports in a csv file under the column name "Report Impression". Let the path to this csv be {path to reports}. Download the pytorch checkpoint and let the path to it be {path to checkpoint}. Let the path to your desired output folder by {path to output dir}. 

```
python label.py -d={path to reports} -o={path to output dir} -c={path to checkpoint} 
```

The output file with labeled reports is {path to output dir}/labeled_reports.csv

Run the following for descriptions of all command line arguments:

```
python label.py -h
```

** Ignore any error messages about the size of the report exceeding 512 tokens. All reports are automatically cut off at 512 tokens. **

### Train a model on labeled reports

Put all train/dev set reports in csv files under the column name "Report Impression". The labels for each of the 14 conditions should be in columns with the corresponding names, and the class labels should follow the convention described in this README.

Training is a two-step process. First, you must tokenize and save all the report impressions in the train and dev sets:

```
python bert_tokenizer.py 
```

# Label Convention

The labeler outputs the following numbers corresponding to classes. This convention is the same as that of the CheXpert labeler.

- Blank: NaN
- Positive: 1
- Negative: 0
- Uncertain: -1

# Citation

If you use the CheXbert labeler in your work, please cite our paper:

```
@misc{smit2020chexbert,
	title={CheXbert: Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT},
	author={Akshay Smit and Saahil Jain and Pranav Rajpurkar and Anuj Pareek and Andrew Y. Ng and Matthew P. Lungren},
	year={2020},
	eprint={2004.09167},
	archivePrefix={arXiv},
	primaryClass={cs.CL}
}
```
