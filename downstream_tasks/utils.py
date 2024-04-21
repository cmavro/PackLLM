import torch
#import openai
import json
from tqdm import tqdm
#from sentence_transformers import SentenceTransformer
from transformers import GPTJForCausalLM
from collections import OrderedDict
#import sqlparse
#from nltk import tokenize
import torch.nn.functional as F
#import seaborn as sns
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


TOKENIZER_MAPS = {
    "AdaptLLM/medicine-LLM": "AdaptLLM/medicine-LLM",
    "BioMistral/BioMistral-7B": "BioMistral/BioMistral-7B",
    "chaoyi-wu/PMC_LLAMA_7B": "chaoyi-wu/PMC_LLAMA_7B",
    "llSourcell/medllama2_7b": "meta-llama/Llama-2-7b-hf",
    "microsoft/phi-2":  "microsoft/phi-2",
    "google/gemma-7b": "google/gemma-7b", 
    "Deci/DeciLM-7B": "Deci/DeciLM-7B",
    "mistralai/Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",  
    'meta-llama/Llama-2-7b-hf': 'meta-llama/Llama-2-7b-hf',
    "openlm-research/open_llama_7b" : "openlm-research/open_llama_7b",
    "openlm-research/open_llama_7b_v2": "openlm-research/open_llama_7b_v2",
    "mosaicml/mpt-7b": "mosaicml/mpt-7b",
    "Wanfq/FuseLLM-7B": 'meta-llama/Llama-2-7b-hf',
    "EleutherAI/llemma_7b": "EleutherAI/llemma_7b",
    "rombodawg/LosslessMegaCoder-llama2-7b-mini": "rombodawg/LosslessMegaCoder-llama2-7b-mini",
    "medalpaca/medalpaca-7b": "medalpaca/medalpaca-7b",
    "lmsys/vicuna-7b-v1.5-16k": "lmsys/vicuna-7b-v1.5-16k", 
    "garage-bAInd/Platypus2-7B": "garage-bAInd/Platypus2-7B",
    "GOAT-AI/GOAT-7B-Community": "GOAT-AI/GOAT-7B-Community",
    "Aspik101/trurl-2-7b-pl-instruct_unload": "Aspik101/trurl-2-7b-pl-instruct_unload",
    "Charlie911/vicuna-7b-v1.5-lora-mctaco": "Charlie911/vicuna-7b-v1.5-lora-mctaco",
    "ashercn97/manatee-7b": "ashercn97/manatee-7b", 
    "julianweng/Llama-2-7b-chat-orcah": "julianweng/Llama-2-7b-chat-orcah", 
    "meta-llama/Llama-2-13b-hf" : "meta-llama/Llama-2-7b-hf",
    "lmsys/vicuna-13b-v1.3": "lmsys/vicuna-7b-v1.5-16k", 
    "upstage/SOLAR-10.7B-v1.0": "upstage/SOLAR-10.7B-v1.0",
    "AdaptLLM/law-LLM": 'meta-llama/Llama-2-7b-hf',
    "AdaptLLM/law-LLM-13B": 'meta-llama/Llama-2-7b-hf',
    'AdaptLLM/medicine-LLM-13B': 'meta-llama/Llama-2-7b-hf',
    'AdaptLLM/medicine-LLM': 'meta-llama/Llama-2-7b-hf',
    "luodian/llama-7b-hf": 'meta-llama/Llama-2-7b-hf',
    "google/gemma-2b": "google/gemma-2b",
    "stabilityai/stablelm-3b-4e1t": "stabilityai/stablelm-3b-4e1t",
    "Qwen/Qwen1.5-1.8B": "Qwen/Qwen1.5-1.8B",
    "microsoft/phi-1_5":  "microsoft/phi-2",
    


}



#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def calculate_sentence_transformer_embedding(text_to_encode,args,accelerator=None):
    num = len(text_to_encode)
    if accelerator is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = accelerator.device #f"cuda:{accelerator.process_index}"

    emb_model = AutoModel.from_pretrained(args.embedding_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    embeddings = []
    bar = tqdm(range(0,num,20),desc='calculate embeddings')
    for i in range(0,num,20):
        # Compute token embeddings
        encoded_input = tokenizer(text_to_encode[i:i+20], padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = emb_model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        embeddings += sentence_embeddings.tolist()
        bar.update(1)
    embeddings = torch.tensor(embeddings)
    #embeddings = F.normalize(embeddings, p=2, dim=-1) #embeddings / (embeddings.norm(dim=1)[:, None] + 1e-6)

    # mean_embeddings = torch.mean(embeddings, 0, True)
    # embeddings = embeddings #- mean_embeddings
    emb_model = emb_model.to("cpu")
    return embeddings

def calculate_sentence_transformer_embedding_m(accelerator, text_to_encode,args):
    num = len(text_to_encode)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_model = accelerator.prepare(AutoModel.from_pretrained(args.embedding_model)) .to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    embeddings = []
    bar = tqdm(range(0,num,20),desc='calculate embeddings')
    for i in range(0,num,20):
        # Compute token embeddings
        encoded_input = accelerator.prepare(tokenizer(text_to_encode[i:i+20], padding=True, truncation=True, return_tensors='pt')) #.to(device)
        with torch.no_grad():
            model_output = emb_model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        embeddings += sentence_embeddings.tolist()
        bar.update(1)
    embeddings = torch.tensor(embeddings)
    #embeddings = F.normalize(embeddings, p=2, dim=-1) #embeddings / (embeddings.norm(dim=1)[:, None] + 1e-6)

    # mean_embeddings = torch.mean(embeddings, 0, True)
    # embeddings = embeddings #- mean_embeddings
    return embeddings


def get_sub_answers(answers, begin=0, end=None):
    return [" ".join(x.split(" ")[begin:end]) for x in answers if len(x.split(" ")) > 1]

PUNCTUATION_SET_TO_EXCLUDE = set(''.join(['‘', '’', '´', '`', '.', ',', '-', '"']))
def expand_to_aliases(given_answers, make_sub_answers=False):
    if make_sub_answers:
        given_answers = given_answers + get_sub_answers(given_answers, begin=1) + get_sub_answers(given_answers, end=-1)
    answers = []
    for answer in given_answers:
        alias = answer.replace('_', ' ').lower()
        alias = ''.join(c if c not in PUNCTUATION_SET_TO_EXCLUDE else ' ' for c in alias)
        answers.append(' '.join(alias.split()).strip())
    return set(answers)



def compute_acc(gold, pred, n_slot=30):

    if type(gold) == dict:
        gold = [f"{k}-{v}" for k, v in gold.items()]
    if type(pred) == dict:
        pred = [f"{k}-{v}" for k, v in pred.items()]

    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = n_slot
    ACC = n_slot - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC


def compute_prf(gold, pred):

    if type(gold) == dict:
        gold = [f"{k}-{v}" for k, v in gold.items()]
    if type(pred) == dict:
        pred = [f"{k}-{v}" for k, v in pred.items()]

    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP) != 0 else 0
        recall = TP / float(TP+FN) if (TP+FN) != 0 else 0
        F1 = 2 * precision * recall / \
            float(precision + recall) if (precision+recall) != 0 else 0
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count


def evaluate(preds: dict, golds: dict):

    gold_slots = list(golds.keys())
    for k in gold_slots:
        if '|' in golds[k]:
            gold_values = golds[k].split('|')
            if k in preds and preds[k] in gold_values:
                golds[k] = preds[k]

    jga, acc, f1 = 0, 0, 0

    if preds == golds:
        jga = 1
    acc = compute_acc(golds, preds)
    f1 = compute_prf(golds, preds)[0]

    return jga, acc, f1

