import numpy as np
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast, BatchEncoding
from sklearn.metrics.pairwise import cosine_similarity
import os, sys
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from typing import Dict, List
import torch
import json
from tqdm import tqdm, trange
from torch import Tensor
import torch.nn.functional as F
import pandas as pd
from joblib import Parallel, delayed


def input_transform_func(
    tokenizer: PreTrainedTokenizerFast,
    examples: Dict[str, List],
    always_add_eos: bool,
    max_length: int,
    instruction: str,
) -> BatchEncoding:
    if always_add_eos:
        examples['input_texts'] = [instruction + input_example + tokenizer.eos_token for input_example in examples['input_texts']]
    batch_dict = tokenizer(
        examples['input_texts'],
        max_length=max_length,
        padding=True,
        return_token_type_ids=False,
        return_tensors="pt",
        truncation=True)
    return batch_dict

    
model_path = ""
print(model_path)

max_length = 512
batch_size = 32
device = f"cuda:{sys.argv[1]}" if len(sys.argv) > 1 else "cuda:0"

model = AutoModel.from_pretrained(model_path,trust_remote_code=True)
model.eval()
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

if 'MedCPT' in model_path:
    model_q = AutoModel.from_pretrained(model_path.replace("MedCPT-d", "MedCPT-q"))
    model_q.to(device)
    tokenizer_q = AutoTokenizer.from_pretrained(model_path.replace("MedCPT-d", "MedCPT-q"))
    
    def embed_query(text):
        tokenized = tokenizer_q(text, padding = True, truncation = True, max_length = max_length, return_tensors = "pt")
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        with torch.no_grad():
            output = model_q(input_ids = input_ids, attention_mask = attention_mask)
            output = output.last_hidden_state[:, 0, :]
        output = output.detach().cpu().numpy()
        
        return output
    
if 'Qwen' in model_path:
    # Qwen & SFR
    def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


    def get_detailed_instruct(query: str) -> str:
        return f'Instruct: Given the medical entity, retrieve relevant paragraphs of patients\' medical records\nQuery: {query}'


    def embed(input_texts):
        # Tokenize the input texts
        batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy()
elif "BMRetriever" in model_path:
    # BMRetriever 
    # Don't forget passage prefix
    def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            embedding = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            embedding = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
        return embedding


    def get_detailed_instruct(query: str) -> str:
        return f'Given the medical entity, retrieve relevant paragraphs of patients\' medical records\nQuery: {query}'


    def embed(input_texts):
        batch_dict = tokenizer(input_texts, max_length=max_length-1, padding=True, truncation=True)

        # Important! Adding EOS token at the end
        batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
        batch_dict['attention_mask'] = [attention_mask + [1] for attention_mask in batch_dict['attention_mask']]
        batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embeddings.detach().cpu().numpy()
elif "NV-Embed" in model_path or "DR.EHR-large" in model_path:
    def embed(texts, query = True):
        if query:
            prefix = "Instruct: Given the medical entity, retrieve relevant paragraphs of patients\' medical records\nQuery: "
        else:
            prefix = ''
        embeddings = None
        if len(texts) >= batch_size:
            for i in trange(len(texts) // batch_size):
                batch_dict = input_transform_func(tokenizer,
                                                {"input_texts": [prompt for prompt in texts[i * batch_size : (i+1) * batch_size]]},
                                                always_add_eos=True,
                                                max_length=max_length,
                                                instruction=prefix)
                batch_dict = {k : v.to(device) for k, v in batch_dict.items()}
                attention_mask = batch_dict['attention_mask'].clone()
                if prefix:
                    instruction_lens = len(tokenizer.tokenize(prefix))
                    attention_mask[:, :instruction_lens] = 0
                features = {
                    'input_ids': batch_dict['input_ids'].long(),
                    'attention_mask': batch_dict['attention_mask'],
                    'pool_mask': attention_mask,
                }

                with torch.no_grad():
                    output = model(**features)["sentence_embeddings"].squeeze(1)
                
                # output = model.encode(texts[i * batch_size : (i+1) * batch_size], instruction = prefix, max_length = max_length)
                output = output.detach().cpu().numpy()
                if embeddings is not None:
                    embeddings = np.concatenate((embeddings, output), axis=0)
                else:
                    embeddings = output
        else:
            i = -1
        if len(texts) % batch_size != 0:
            output = model.encode(texts[(i+1) * batch_size:], instruction = prefix, max_length = max_length)
            output = output.detach().cpu().numpy()
            if embeddings is not None:
                embeddings = np.concatenate((embeddings, output), axis=0)
            else:
                embeddings = output
        
        return embeddings
else:
    def embed(texts):
        embeddings = None
        if len(texts) >= batch_size:
            for i in trange(len(texts) // batch_size):
                tokenized = tokenizer(texts[i * batch_size : (i+1) * batch_size], padding = True, truncation = True, max_length = max_length, return_tensors = "pt")
                input_ids = tokenized['input_ids'].to(device)
                attention_mask = tokenized['attention_mask'].to(device)
                with torch.no_grad():
                    output = model(input_ids = input_ids, attention_mask = attention_mask)
                output = output.last_hidden_state[:, 0, :]
                output = output.detach().cpu().numpy()
                if embeddings is not None:
                    embeddings = np.concatenate((embeddings, output), axis=0)
                else:
                    embeddings = output
        else:
            i = -1
        if len(texts) % batch_size != 0:
            tokenized = tokenizer(texts[(i+1) * batch_size:], padding = True, truncation = True, max_length = max_length, return_tensors = "pt")
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            with torch.no_grad():
                output = model(input_ids = input_ids, attention_mask = attention_mask)
            output = output.last_hidden_state[:, 0, :]
            output = output.detach().cpu().numpy()
            if embeddings is not None:
                embeddings = np.concatenate((embeddings, output), axis=0)
            else:
                embeddings = output
        
        return embeddings

def calculate_mrr(predicted_ranks, correct_ranks):
    for idx, pr in enumerate(predicted_ranks):
        if pr in correct_ranks:
            return 1.0 / (idx + 1)
    return 0


def calculate_ndcg_at_k(predicted_ranks, correct_ranks, k):
    def dcg_at_k(r, k):
        r = np.asarray(r)[:k]
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    
    def ideal_dcg_at_k(r, k):
        ir = sorted(r, reverse=True)
        return dcg_at_k(ir, k)
    
    relevances = [1 if rank in correct_ranks else 0 for rank in predicted_ranks[:k]]
    if 1 in relevances:
        return dcg_at_k(relevances, k) / ideal_dcg_at_k(relevances, k)
    else:
        return 0


def recall_at_k(predicted_ranks, correct_ranks, k):
    # 计算前k个预测中有多少是正确的
    correct_predictions_in_top_k = sum(1 for rank in predicted_ranks[:k] if rank in correct_ranks)

    # 如果没有正确的答案，返回0
    if len(correct_ranks) == 0:
        return 0
    else:
        # 计算所有正确答案中被预测到的比例
        return correct_predictions_in_top_k / len(correct_ranks)
    
    
def average_precision(predicted_ranks, correct_ranks):  
    if not correct_ranks:  
        return 0.0  

    score = 0.0  
    num_hits = 0.0  

    for i, p in enumerate(predicted_ranks):  
        if p in correct_ranks:  
            num_hits += 1.0  
            score += num_hits / (i + 1.0)  

    return score / len(correct_ranks)
    
    
with open("benchmark/corpus.jsonl") as f:
    corpus = [json.loads(l) for l in f.readlines()]
corpus_ids = [c['_id'] for c in corpus]
corpus = [c['text'] for c in corpus]
if "Qwen" in model_path:
    corpus_embeds = np.vstack([embed([text]) for text in tqdm(corpus)])
elif "BMRetriever" in model_path:
    corpus_embeds = np.vstack([embed(["Represent this passage\npassage: " + text]) for text in tqdm(corpus)])
elif "NV-Embed" in model_path:
    corpus_embeds = embed(corpus, query = False)
else:
    corpus_embeds = embed(corpus)
print(len(corpus_ids))
print(corpus_embeds.shape)

inner_metrics = {"string": [[], [], []], 
                "synonym": [[], [], []], 
                "abbreviation": [[], [], []], 
                "hyponym": [[], [], []], 
                "implication": [[], [], []], 
                "overall": [[], [], []]}
inter_metrics = [[], [], []]
for split in ['disease', 'procedure', 'drug']:
    with open(f"benchmark/queries_{split}.jsonl") as f:
        queries = [json.loads(l) for l in f.readlines()]
    queries = {q['_id']: q['text'] for q in queries}
    qrels_df = pd.read_csv(f"benchmark/qrels_{split}.tsv", sep = '\t', header = None, dtype = str)
    qrels = {}
    for _, row in qrels_df.iterrows():
        q = queries[row[0]]
        if q not in qrels:
            qrels[q] = {"string": [], "synonym": [], "abbreviation": [], "hyponym": [], "implication": [], "overall": []}
        qrels[q][row[2]].append(row[1])
        qrels[q]['overall'].append(row[1])
    query_list = list(queries.values())
    if "Qwen" in model_path or "BMRetriever" in model_path:
        query_embs = np.vstack([embed([get_detailed_instruct(query, split)])[0] for query in tqdm(query_list)])
    elif "MedCPT" in model_path:
        query_embs = embed_query(query_list)
    else:
        query_embs = embed(query_list)

    def process_query(q_emb):
        similarities = cosine_similarity([q_emb], corpus_embeds)[0]
        predicted_ranks = [corpus_ids[rank] for rank in np.argsort(-similarities)]
        return predicted_ranks

    results = Parallel(n_jobs = -1)(delayed(process_query)(query_embs[i]) for i in trange(len(queries)))
    tmp_inner_metrics = [[], [], []]
    tmp_inter_metrics = [[], [], []]
    for i in trange(len(queries)):
        predicted_ranks = results[i]
        q = query_list[i]
        correct_ranks = qrels[q]['overall']
        tmp_inter_metrics[0].append(calculate_mrr(predicted_ranks, correct_ranks))
        tmp_inter_metrics[1].append(calculate_ndcg_at_k(predicted_ranks, correct_ranks, 10))
        tmp_inter_metrics[2].append(recall_at_k(predicted_ranks, correct_ranks, 100))
        
        inner_corpus = set(doc_id.split('_')[0] for doc_id in qrels[q]['overall'])
        for doc_id in inner_corpus:
            tmp_pred = [d for d in predicted_ranks if d.split('_')[0] == doc_id]
            tmp_corr = [d for d in correct_ranks if d.split('_')[0] == doc_id]
            if len(tmp_pred) < 5:
                continue
            if len(tmp_pred) == len(tmp_corr):
                continue
            tmp_inner_metrics[0].append(calculate_mrr(tmp_pred, tmp_corr))
            tmp_inner_metrics[1].append(calculate_ndcg_at_k(tmp_pred, tmp_corr, len(tmp_pred)))
            tmp_inner_metrics[2].append(average_precision(tmp_pred, tmp_corr))
            
            for typ in ["string", "synonym", "abbreviation", "hyponym", "implication"]:
                typ_corr = [d for d in qrels[q][typ] if d in tmp_corr]
                if not typ_corr:
                    continue
                remove = [x for x in tmp_corr if x not in typ_corr]
                typ_pred = [x for x in tmp_pred if x not in remove]
                if len(typ_pred) == len(typ_corr):
                    continue
                inner_metrics[typ][0].append(calculate_mrr(typ_pred, typ_corr))
                inner_metrics[typ][1].append(calculate_ndcg_at_k(typ_pred, typ_corr, len(typ_pred)))
                inner_metrics[typ][2].append(average_precision(typ_pred, typ_corr))
                
    inner_metrics['overall'][0] += tmp_inner_metrics[0]
    inner_metrics['overall'][1] += tmp_inner_metrics[1]
    inner_metrics['overall'][2] += tmp_inner_metrics[2]
    
    inter_metrics[0] += tmp_inter_metrics[0]
    inter_metrics[1] += tmp_inter_metrics[1]
    inter_metrics[2] += tmp_inter_metrics[2]
    
    print(f"==========={split}==========")
    print(" & {:.2f} & {:.2f} & {:.2f}".format(np.mean(tmp_inner_metrics[0])*100, np.mean(tmp_inner_metrics[1])*100, np.mean(tmp_inner_metrics[2])*100))
    print(" & {:.2f} & {:.2f} & {:.2f}".format(np.mean(tmp_inter_metrics[0])*100, np.mean(tmp_inter_metrics[1])*100, np.mean(tmp_inter_metrics[2])*100))
        
for typ in ["string", "synonym", "abbreviation", "hyponym", "implication"]:
    print(f"==========={typ}==========")
    print(" & {:.2f} & {:.2f} & {:.2f}".format(np.mean(inner_metrics[typ][0])*100, np.mean(inner_metrics[typ][1])*100, np.mean(inner_metrics[typ][2])*100))
    
print("===========overall==========")
print(" & {:.2f} & {:.2f} & {:.2f}".format(np.mean(inner_metrics['overall'][0])*100, np.mean(inner_metrics['overall'][1])*100, np.mean(inner_metrics['overall'][2])*100))
print(" & {:.2f} & {:.2f} & {:.2f}".format(np.mean(inter_metrics[0])*100, np.mean(inter_metrics[1])*100, np.mean(inter_metrics[2])*100))
    