
# CheXbert: Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT

Preprint: https://arxiv.org/abs/2004.09167

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
