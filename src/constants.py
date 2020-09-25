NUM_EPOCHS = 80
BATCH_SIZE = 18    #Refer to load_data docstring before changing this!
NUM_WORKERS = 4    #A value of 0 means the main process loads the data
LEARNING_RATE = 2e-5
PAD_IDX = 0        #padding index as required by the tokenizer
LOG_EVERY = 200    #iterations after which to log status
VALID_NITER = 2000 #iterations after which to evaluate model and possibly save
TEMP = 1.0         #softmax temperature for self training

#CONDITIONS is a list of all 14 medical observations 
CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
              'Support Devices', 'No Finding']
CLASS_MAPPING = {0: "Blank", 1: "Positive", 2: "Negative", 3: "Uncertain"}
