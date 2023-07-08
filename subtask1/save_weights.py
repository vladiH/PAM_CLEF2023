import json
import re
import random
import numpy as np
import pandas as pd
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, BertTokenizerFast

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

device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')
bert = AutoModel.from_pretrained('microsoft/MiniLM-L12-H384-uncased')
tokenizer = BertTokenizerFast.from_pretrained('microsoft/MiniLM-L12-H384-uncased',do_lower_case=True)
tokenizer.save_pretrained("./tokenizer")
model = PrototypicalNetwork(bert)
checkpoint = torch.load('./weights_0.pth', map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dic'])
torch.save({
      'model_state_dic': model,
      'prototypes': checkpoint['prototypes']
    }, 'weights.pth')