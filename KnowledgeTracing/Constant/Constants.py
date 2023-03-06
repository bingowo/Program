import os
import torch
Dpath =  os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'EOJDataset')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print("device:",device)
datasets = {
    'contest513' : 'contest513',
    'static2011' : 'static2011',
}

# question number of each dataset
numbers = {
    'contest513' : 141,  # use 1 - 136
    'static2011' : 1224, 
}

student_n = 195
exer_n = 141 
knowledge_n = 25
use_knowledge = False
use_code = True
code_length = 768

DATASET = datasets['contest513']
NUM_OF_QUESTIONS = numbers['contest513']
# the max step of RNN model
MAX_STEP = 50
BATCH_SIZE = 32
WORKERS = 0
LR = 0.00005
EPOCH = 10000
#input dimension
INPUT = NUM_OF_QUESTIONS * 2 + (knowledge_n if use_knowledge else 0)
# embedding dimension
EMBED = NUM_OF_QUESTIONS
# hidden layer dimension
HIDDEN = 200
# nums of hidden layers
LAYERS = 1
# output dimension
OUTPUT = NUM_OF_QUESTIONS
