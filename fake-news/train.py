import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from sklearn.decomposition import PCA
import tensorflow_hub as hub
from pycaret.classification import * 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix
import sklearn.preprocessing as preprocessing
import pandas as pd
import pycaret
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from bert_arch import BERT_Arch

########## READ/PREPARE DATA ##########

# Read the real data csv into a pandas dataframe, limit to 500 rows for basic testing
true_data = pd.read_csv('True.csv', nrows=1000)
# Remove the columns of the real data that we do not need. We base our predictions on the title of the article.
true_data = true_data.drop(columns=['text', 'tags', 'source', 'author', 'published'])
# Add a new column "label" that always has the value "0" (which stands for real news) in this dataframe
true_data['label']=[0]*len(true_data)

# Read the fake data csv into a pandas dataframe, limit to 500 rows for basic testing
fake_data = pd.read_csv('Fake.csv', nrows=1000)
# Remove the columns of the fake data that we do not need. We base our predictions on the title of the article.
fake_data = fake_data.drop(columns=['id', 'url', 'Body', 'Kategorie', 'Datum', 'Quelle', 'Fake', 'Art'])
# Rename the column "Titel" to "title" to match with the real data
fake_data = fake_data.rename(columns={"Titel": "title"})
# Add a new column "label" that always has the value "1" (which stands for fake news) in this dataframe
fake_data['label']=[1]*len(fake_data)

# Combine the two dataframes (fake news / real news) into one new dataframe
data=true_data.append(fake_data).sample(frac=1)

# Optional code to see the dataframes content
#print("TRUE: ")
#print(true_data.head())
#print("FAKE: ")
#print(fake_data.head())
#print("FULL: ")
#print(len(data))
#print(data.head())

# Optional code to see the share of fake/real news
# cat_tar=pd.get_dummies(data.label)[1]
# label_size = [cat_tar.sum(),len(cat_tar)-cat_tar.sum()]
# plt.pie(label_size,explode=[0.1,0.1],colors=['firebrick','navy'],startangle=90,shadow=True,labels=[0,1],autopct='%1.1f%%')
# plt.show()


########## PREPARE TRAINING DATA ##########

# Split the data into test and training data
# The column "title" contains our "input" the column "label" our "output" 
# We use 70% of data for training and 30% for testing
# Input:
# - Column title of the dataset
# - Column label of the dataset
# - random_state: Ensures that shuffling in split is reproducible
# - test_size: Optional (Default 25%); Proportion (30%) of the dataset that is used for testing; in our case 70% for training
# - stratify: Ensures that the real/fake news proportion in the training set is the same as in the testing set. That's why we use the label column as it defines real/fake news.
# Output: 
# - train_text: 70% of the column title data
# - temp_text: 30% of the column title data
# - train_labels: 70% of the column label data
# - temp_labels: 30% of the column label data
train_text, temp_text, train_labels, temp_labels = train_test_split(data['title'], data['label'], 
                                                                    random_state=2018, 
                                                                    test_size=0.3, 
                                                                    stratify=data['label'])

# Same behaviour as above but with a 50% split
# Output:
# - val_text: 15% of the column title data
# - test_text: 15% of the column title data
# - val_labels: 15% of the column label data
# - test_labels: 15% of the column label data
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                random_state=2018, 
                                                                test_size=0.5, 
                                                                stratify=temp_labels)


########## ANALYZE DISTRIBUTION AND SET MAX LENGTH ##########

# Optional: Visualize the word count distribution among the training titles
# seq_len = [len(i.split()) for i in train_text]
# pd.Series(seq_len).hist(bins = 40,color='firebrick')
# plt.xlabel('Number of Words')
# plt.ylabel('Number of texts')
# plt.show()

# Set the max length of the titles to cut extremely long records 
MAX_LENGHT = 11


########## TOKENIZE ##########

# Use a pre-trained BERT model
# BERT was developed by Google and is able to detect language patterns.
# Orginally developed to detected masked words in sentences and make next sentence predictions
# https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi3mf-Kx_7zAhWMOuwKHcBrDy8QFnoECAwQAQ&url=https%3A%2F%2Ftowardsdatascience.com%2Fbert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270&usg=AOvVaw3i0maaLfw6E-LA5MR__GV3
# Can be finetuned fast for specific language related use cases 
bert = AutoModel.from_pretrained('bert-base-uncased')
# Tokenizer is in charge of preparing the inputs for a model
# It will convert a word-string to a numerical value
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Uses the tokenizer to convert all training texts to numerical value arrays
# Will cut the title to MAX_LENGTH
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = MAX_LENGHT,
    pad_to_max_length=True,
    truncation=True
)

# Uses the tokenizer to convert all val? texts to numerical value arrays
# Will cut the title to MAX_LENGTH
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = MAX_LENGHT,
    pad_to_max_length=True,
    truncation=True
)

# Uses the tokenizer to convert all testing texts to numerical value arrays
# Will cut the title to MAX_LENGTH
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = MAX_LENGHT,
    pad_to_max_length=True,
    truncation=True
)

# Output
# {'input_ids': [[1, 2, 3], [4, 5, 6]], 'token_type_ids': [[0, 0,0 ], [0, 0, 0]], 'attention_mask': [[1, 1, 1], [1, 1, 1]]}


########## TENSORS ##########
# A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
# convert lists to tensors

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

# A TensorDataset is a dataset of tensors.
# Stores a single tensor internally, which is then indexed inside get().
train_data = TensorDataset(train_seq, train_mask, train_y)
val_data = TensorDataset(val_seq, val_mask, val_y)

########## TRAINING ##########

# Batch size for training
batch_size = 32

# Samplers for Datasets
# RandomSampler: Samples elements randomly.
# SequentialSampler: Samples elements sequentially, always in the same order.
train_sampler = RandomSampler(train_data)
val_sampler = SequentialSampler(val_data)

# Dataloader for training set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Dataloader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)
for param in bert.parameters():
    param.requires_grad = False

# Instance of our model
model = BERT_Arch(bert)

# Optimizer
# Implements AdamW algorithm.
optimizer = AdamW(model.parameters(), lr = 1e-5) # learning rate

# Compute the class weights
# Proportion of fake and real news data
class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
print("Class Weights: ",class_weights)

# Tensor for class weights
weights = torch.tensor(class_weights, dtype=torch.float)

# Define the loss function
# A loss function maps decisions to their associated costs to 
# minimize the error for each training example during the learning process.
cross_entropy  = nn.NLLLoss(weight=weights) 

# Number of training epochs
epochs = 10


########## TRAINING / EVALUATE FUNCTIONS ##########

def train():

    model.train()

    total_loss = 0

    # empty list to save model predictions
    total_preds=[]

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 10 batches.
        if step % 10 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r for r in batch]
        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss, total_preds

def evaluate():

    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss = 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step,batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:

            # Calculate elapsed time in minutes.
            #elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds,labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
for epoch in range(epochs):

    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    #train model
    train_loss, _ = train()

    #evaluate model
    valid_loss, _ = evaluate()

    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'awesome-model.pt')

    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

#load weights of best model
path = 'awesome-model.pt'
model.load_state_dict(torch.load(path))

with torch.no_grad():
    preds = model(test_seq, test_mask)
    preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis = 1)
print(classification_report(test_y, preds))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(preds,test_labels))
