import argparse
import json
import re
import math
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertTokenizerFast
import torch.nn.functional as F

import emoji

def remove_emoji(text):
    return re.sub('\s+', ' ', re.sub(':\S+?:', ' ', emoji.demojize(text)))

def remove_long_words(text):
    words = text.split()
    filtered_words = [word for word in words if len(word) <= 40]
    return " ".join(filtered_words)

def preprocess_text(text):
    # reemplazar las urls
    text = re.sub(r'http\S+', 'http', text)
    # drop letters that start at @&
    text = re.sub(r'[&][^\s]+', '', text)
    text = re.sub(r'[@][^\s]+', '@user', text)
    #drop all new lines
    text = re.sub(r"[\r\n]+", " ", text)
    #drop emojis
    text = remove_emoji(text)
    # Convertir todo el texto a minúsculas
    text = text.lower()
    #Sustituir los Retweets
    text = re.sub(r'\b(RT|rt)\b\s+', ' ', text)
    # reemplazar múltiples espacios con un solo espacio
    text = re.sub(r'\s+', ' ', text)
    #eliminar secuencia de palabras repetidas
    text = re.sub(r'([@]*[a-zA-Z]{1,}\s+)\1+', r'\1', text)
    text = remove_long_words(text)
    # reemplazar múltiples espacios con un solo espacio
    text = re.sub(r'\s+', ' ', text)
    #remove white space at begin and end of line
    text = text.strip()

    return text

def compute_prob(prototypes, embeddings):
    # Calcula la distancia euclidiana entre cada prototipo y las incrustaciones de las muestras de consulta
    distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)
    
    # Calcula la probabilidad mediante la aplicación de Softmax sobre la negación de las distancias
    neg_distances = -distances
    probability = F.softmax(neg_distances, dim=-1)
    
    # Encuentra la clase predicha para cada muestra de consulta
    y_hat, index = torch.max(probability, dim=-1)

    return y_hat, index

class PrototypicalNetwork(nn.Module):
    def __init__(self, bert_model):
        super(PrototypicalNetwork, self).__init__()
        self.bert_model = bert_model
    def forward(self, ids, ids_mask):
        """Prototype Networks forward.
        Args:
            ids: torch.Tensor, [-1, N, K, max_length]
            ids_mask: torch.Tensor, [-1, N, K, max_length]
            
        Returns:
            ids: torch.Tensor, [B, N*K, D]"""
        B, N, K, max_length = ids.size()
        ids = ids.view(-1, max_length)  # [B * N * K, max_length]
        ids_mask = ids_mask.view(-1, max_length)
        ids = self.bert_model(input_ids=ids, attention_mask=ids_mask)[1]  # [B * N * K, D]
        ids = ids.view(B, -1, ids.size()[-1])  # [B, N*K, D] D=self.bert_model.config.hidden_size
        return ids
    
class InputExample(object):
    def __init__(self, twitter_id, text, label='unk'):
        self.twitter_id = twitter_id
        self.text = text
        self.label = label
      
def load_prototypes(proto_dict):
    proto_stack = torch.stack(list(proto_dict.values()), dim=1)
    labels = list(proto_dict.keys())
    return proto_stack, labels

def load_file(input_directory: Path, max_length=80):
    examples = []
    labels = []
    with open(input_directory, 'r') as f:
        for line in f:
            obj = json.loads(line)
            text = [text['text'] for text in obj['texts']]
            text = ' '.join(text)
            text = preprocess_text(text)
            text_split = text.split()
            text_length = len(text_split)
            new_text = []
            for i in range(math.ceil(text_length/max_length)):
                if (i+1)*max_length>= text_length:
                    new_text.append(' '.join(text_split[-max_length:]))
                else:
                    new_text.append(' '.join(text_split[i*max_length:(i+1)*max_length]))
            label = 'unk'
            if 'class' in obj:
                label = obj['class']
            labels.append(label)
            if len(new_text)==0:
                new_text.append("")
            examples.append(InputExample(twitter_id=obj['twitter user id'], text=new_text, label=label))
    return examples, labels

def run_inference(df_to_predict, args):
    user_id=[]
    user_probs=[]
    user_label=[]  
    tokenizer = BertTokenizerFast.from_pretrained('/tokenizer')
    checkpoint = torch.load('/weights.pth', map_location=torch.device(args.device))
    model = checkpoint['model_state_dic']
    model.to(args.device)
    model.eval()

    with torch.no_grad():
        prototypes, labels = load_prototypes(checkpoint['prototypes'])
        for example in df_to_predict:
            encode = tokenizer.batch_encode_plus(
                        example.text,
                        max_length = args.tokenizer_length,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        add_special_tokens = True,
                        return_tensors = 'pt',
                        )
            input_ids = encode['input_ids'].to(args.device)
            input_ids_shape = input_ids.size()
            attention_mask = encode['attention_mask'].to(args.device)
            input_ids = input_ids.view(1,1,*input_ids_shape)  
            attention_mask = attention_mask.view(1,1,*input_ids_shape)
            input_embeding = model(input_ids, attention_mask)
            input_embeding = torch.mean(input_embeding, 1, keepdim=True)  
            y_hat, y_hat_id = compute_prob(prototypes, input_embeding)
            user_label.append(labels[y_hat_id.item()])
            user_probs.append(round(y_hat.item(),3))
            user_id.append(example.twitter_id)
    df_output = pd.DataFrame(list(zip(user_id, user_label, user_probs)), columns =['twitter user id', 'class', 'probability'])
    df_output.to_json(args.output, orient='records', lines=True)
    

def main():
    parser = argparse.ArgumentParser(description='This is a baseline for subtask2 influencer intent identification.')
    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)
    args = parser.parse_args()
    args.device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')
    args.tokenizer_length = 95
    args.text_seq_length = 80
    if args.output is None:
        args.output = 'subtask2.json'
    df_to_predict,_ = load_file(Path(args.input), args.text_seq_length)   
    run_inference(df_to_predict, args)


if __name__ == '__main__':
    main()