import torch
import torch.nn as nn
import pandas as pd
import pycaret
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from sklearn.decomposition import PCA
#import tensorflow as tf
import tensorflow_hub as hub
from pycaret.classification import * 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix
import sklearn.preprocessing as preprocessing

class BERT_Arch(nn.Module):

    def __init__(self, bert):

        super(BERT_Arch, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768,512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,2)

        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

        #pass the inputs to the model
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x

# PREDICT
path = "awesome-model.pt"

bert = AutoModel.from_pretrained('bert-base-uncased')
model = BERT_Arch(bert)

model.load_state_dict(torch.load(path))
model.eval()

fake = ["Querdenker holt sich Booster-Telegram-Gruppe, um Immunität gegen Fakten aufzufrischen"]
true = ["Die letzten Stunden des NSU – und was bis heute ungeklärt ist"]
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tokens_test = tokenizer.batch_encode_plus(
    true,
    max_length = 11,
    pad_to_max_length=True,
    truncation=True
)

print(tokens_test)
quit()

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
#test_y = torch.tensor(['title', 'label'])

# deactivate autograd
with torch.no_grad():
    # model predictions
    
    # sent_id = tokenizer.batch_encode_plus(text, padding=True)

    device = torch.device("cpu")
    preds = model(test_seq.to(device), test_mask.to(device))
    print(preds)
    a = preds.detach().cpu().numpy()
    print(a)
    x = np.argmax(preds, axis = 1)
    x = x.detach().cpu().numpy()
    print(x[0])
    y = nn.functional.softmax(preds, dim=-1)
    print(y)
    z = y.detach().cpu().numpy()
    print("Probability True: " + str(z[0][0]))
    print("Probability False: " + str(z[0][1]))
    #print(classification_report(test_y, preds))

