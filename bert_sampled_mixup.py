from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
import os
import random
import argparse
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import logging
import json
from sklearn.metrics import precision_score, recall_score, f1_score
import string
from nltk import word_tokenize
import pickle

logger = logging.getLogger()
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger('numexpr').setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)

logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

parser_1 = argparse.ArgumentParser()

parser_1.add_argument('--bert_pretrained', default='bert-base-uncased', type=str, help='the model version we are using')
parser_1.add_argument('--num_classes', default=2, type=int, help='2, 3, or 6 way classification')
parser_1.add_argument('--learning_rate', default=5e-5, type=float, help="lower -> slower training, initial learning rate")
parser_1.add_argument("--train_file", default='/data/scratch/projects/punim0478/ailis/Fakeddit/multimodal_train.tsv', type=str, help="the train file") 
parser_1.add_argument("--dev_file", default='/data/scratch/projects/punim0478/ailis/Fakeddit/multimodal_validate.tsv', type=str, help="the dev file")     
parser_1.add_argument("--test_file", default='/data/scratch/projects/punim0478/ailis/Fakeddit/multimodal_test_public.tsv', type=str, help="the test file")
parser_1.add_argument('--max_length', default=60, type=int, help='the maximum length for processed text')
parser_1.add_argument('--batch_size', default=64, type=int, help='batch training size')
parser_1.add_argument('--nb_epochs', default=15, type=int, help="number of training epochs")
parser_1.add_argument('--successive_decrease', default=5, type=int, help="number of successive decrease performance in validation dataset")
parser_1.add_argument('--saved_model_name', default='/data/scratch/projects/punim0478/ailis/Fakeddit/text_best_over_dev.pt', type=str, help='the file storing the best model over dev set')
parser_1.add_argument('--test_prediction', default='/data/scratch/projects/punim0478/ailis/Fakeddit/bert_prediction_test.txt', type=str, help='the file storing the predicted result over the test set')
parser_1.add_argument('--device_id', default='0',type=str, help='the GPU device id if multiple GPUs are available')
parser_1.add_argument('--mixup_lambda', default=2,type=int, help='determine whether adopting mixup strategy, 0: no; 2, yes')

parser = parser_1.parse_args()


class TextDataset(Dataset):

  def __init__(self, page_ids, text, target, tokenizer):
    self.page_id = page_ids
    self.text = text
    self.target = target
    self.tokenizer = tokenizer
    
  
  def __len__(self):
    return len(self.page_id)
  
  def __getitem__(self, item):
    text = self.text[item]
    target = int(self.target[item])

    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=parser.max_length,
      return_token_type_ids=False,
      truncation=True,
      #pad_to_max_length=True,
      padding='max_length',
      return_attention_mask=True,
      return_tensors='pt',
    )

    return encoding['input_ids'].flatten(), encoding['attention_mask'].flatten(), target, self.page_id[item]
   

class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        #self.bert = BertModel.from_pretrained('/data/scratch/projects/punim0478/ailis/fijitsu/code/bert-base-cased')
        self.bert = BertModel.from_pretrained(parser.bert_pretrained, output_hidden_states = True)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, parser.num_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)

def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred>threshold, dtype=float)
    precision = precision_score(y_true=target, y_pred=pred, average='micro')
    recall = recall_score(y_true=target, y_pred=pred, average='micro')
    f1 = f1_score(y_true=target, y_pred=pred, average='micro')
    return precision, recall, f1


def get_lambda(alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    return lam

def train_epoch(model, epoch, data_loader, criterion, optimizer, device):
    model = model.train()

    sum_loss = 0
    total = 0
    global_preds = []
    global_labels = []
    counter = 0
    for input_ids, attention_mask, target, _ in data_loader:
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = target.to(device)

        #zero the parameter gradients
        optimizer.zero_grad()
        torch.set_grad_enabled(True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, targets)
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = torch.max(outputs, dim=1)[1]

        acc = (preds==targets).float().sum()/float(input_ids.size(0))
        #logger.info(acc.cpu().numpy())
        targets = targets.cpu().numpy()
        preds = preds.cpu().numpy()

        #precision, recall, f1 = calculate_metrics(preds.cpu().numpy(), targets)

        global_preds.extend(preds)
        global_labels.extend(targets)

        sum_loss+= loss.item()*input_ids.size(0)
        total+=input_ids.size(0)

        if counter%100 == 0:
            logger.info("Epoch " + str(epoch+1) + ", Minibatch Loss= " + \
                                "{:.6f}".format(loss) + ", Training Acc = " + \
                                "{:.5f}".format(acc))
            #logger.info('Accuracy'+str(acc.cpu().numpy()))

        
        counter+=1

    #precision, recall, f1 = calculate_metrics(np.array(global_preds), np.array(global_labels))
    '''logger.info("Epoch " + str(epoch+1) + ", Loss= " + \
                                "{:.6f}".format(sum_loss/float(total)) + ", Training Precision= " + \
                                "{:.5f}".format(precision)+ ", Training Recall= " + \
                                "{:.5f}".format(recall)+ ", Training F1 score = " + \
                                "{:.5f}".format(f1))'''


    accuracy = (np.array(global_preds)==np.array(global_labels)).sum()/float(len(global_preds))     
    logger.info("Training Acc= "+"{:.5f}".format(accuracy))

def eval_model(model, dev_test, epoch, data_loader, criterion, device):
    model = model.eval()

    sum_loss = 0
    total = 0
    global_preds = []
    global_labels = []
    page_ids = []

    with torch.no_grad():
        for input_ids, attention_mask, target, ids in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = target.to(device)
            total+=input_ids.size(0)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets)

            preds = torch.max(outputs, dim=1)[1]
            global_preds.extend(preds.detach().cpu().numpy())
            global_labels.extend(targets.cpu().numpy())
            page_ids.extend(ids)    

            sum_loss+=loss.item()*input_ids.size(0)
            total+=input_ids.size(0)

    #precision, recall, f1 = calculate_metrics(np.array(global_preds), np.array(global_labels))
    accuracy = (np.array(global_preds)==np.array(global_labels)).sum()/float(len(global_preds))
    loss = sum_loss/float(total)
    logger.info("Epoch " + str(epoch+1) +' '+ dev_test+": Loss= " + \
                                "{:.6f}".format(loss) + ", Accuracy = " + \
                                "{:.5f}".format(accuracy))

    return loss, accuracy, global_labels, global_preds, page_ids

def make_predictions(model, data_loader, device):
    model = model.eval()

    global_preds = []
    page_ids = []

    with torch.no_grad():
        for input_ids, attention_mask, _, ids in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            sig_outputs = torch.sigmoid(outputs)

            global_preds.extend(sig_outputs.detach().cpu().numpy())
            page_ids.extend(ids)    

    return  page_ids, global_preds

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_everything(42)

    device = torch.device("cuda:"+parser.device_id if torch.cuda.is_available() else "cpu")

    with open(parser.train_file, 'r') as f:
        train_ids_labels = f.readlines()
        
    train_text = []
    train_ids = []
    train_labels = []

    for i in range(0, len(train_ids_labels)):
        tem_list = train_ids_labels[i].strip().split('\t')
        tem_id = tem_list[0].split('/')[-1].replace('.jpg', '')
        train_ids.append(tem_id) 
        train_labels.append(int(tem_list[1]))
        with open('/data/scratch/projects/punim0478/ailis/Fakeddit/fakenews_full/text/'+tem_id+'.txt') as f:
            data = f.readline().strip()
        train_text.append(data)

    with open(parser.dev_file, 'r') as f:
        dev_ids_labels = f.readlines()
        
    dev_text = []
    dev_ids = []
    dev_labels = []

    for i in range(0, len(dev_ids_labels)):
        tem_list = dev_ids_labels[i].strip().split('\t')
        tem_id = tem_list[0].split('/')[-1].replace('.jpg', '')
        dev_ids.append(tem_id) 
        dev_labels.append(int(tem_list[1]))
        with open('/data/scratch/projects/punim0478/ailis/Fakeddit/fakenews_full/text/'+tem_id+'.txt') as f:
            data = f.readline().strip()
        dev_text.append(data)
    
    with open(parser.test_file, 'r') as f:
        test_ids_labels = f.readlines()
        
    test_text = []
    test_ids = []
    test_labels = []

    for i in range(0, len(test_ids_labels)):
        tem_list = test_ids_labels[i].strip().split('\t')
        tem_id = tem_list[0].split('/')[-1].replace('.jpg', '')
        test_ids.append(tem_id) 
        test_labels.append(int(tem_list[1]))
        with open('/data/scratch/projects/punim0478/ailis/Fakeddit/fakenews_full/text/'+tem_id+'.txt') as f:
            data = f.readline().strip()
        test_text.append(data)
    
    
    logger.info('train: '+str(len(train_ids))+' dev: '+str(len(dev_ids))+' test: '+str(len(test_ids)))

    tokenizer = BertTokenizer.from_pretrained(parser.bert_pretrained)

    train_dataset = TextDataset(train_ids, train_text, train_labels, tokenizer)
    dev_dataset = TextDataset(dev_ids, dev_text, dev_labels, tokenizer)
    test_dataset = TextDataset(test_ids, test_text, test_labels, tokenizer)


    dataloaders_train = torch.utils.data.DataLoader(train_dataset, batch_size=parser.batch_size, shuffle=True, num_workers=4)
    dataloaders_dev = torch.utils.data.DataLoader(dev_dataset, batch_size=parser.batch_size, shuffle=False, num_workers=4)
    dataloaders_test = torch.utils.data.DataLoader(test_dataset, batch_size=parser.batch_size, shuffle=False, num_workers=4)

    model = TextClassifier()
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=parser.learning_rate)

    #criterion = nn.BCEWithLogitsLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    counter = 0
    optimal_dev_accuracy = 0
    optimal_dev_predictions = []
    optimal_dev_gold = []
    optimal_dev_page_ids = []

    for epoch in range(0, parser.nb_epochs):

        logger.info('Epoch: %d', epoch)
        logger.info('----------')

        train_epoch(model, epoch, dataloaders_train, criterion, optimizer, device)

        loss, accuracy, overall_labels, predictions, page_ids = eval_model(model, 'Dev', epoch, dataloaders_dev, criterion, device)

        if accuracy < optimal_dev_accuracy:
            counter+=1
        else:
            counter = 0
            optimal_dev_accuracy = accuracy
            optimal_dev_predictions = predictions
            optimal_dev_gold = overall_labels
            optimal_dev_page_ids = page_ids

            torch.save(model.state_dict(), parser.saved_model_name)
            #optimal_test_accuracy = test_accuracy
            #optimal_test_predictions = test_predictions
            #optimal_test_confusion = metrics.confusion_matrix(gold_labels, test_predictions)
        
        if counter == parser.successive_decrease:
            break

    logger.info('Optimization finished!')
    logger.info('optimal dev accuracy is %f', optimal_dev_accuracy)


    model = TextClassifier()
    model = model.to(device)

    model.load_state_dict(torch.load(parser.saved_model_name))
    logger.info('model loaded')

    loss, accuracy, overall_labels, predictions, page_ids = eval_model(model, 'Test', 0, dataloaders_test, criterion, device)

    with open(parser.test_prediction, 'w') as f:
        for j in range(0, len(predictions)):
            f.write(str(page_ids[j])+'\t'+str(predictions[j])+'\t'+str(overall_labels[j])+'\n')
