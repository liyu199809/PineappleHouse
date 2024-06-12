# from reranker import BgeM3Reranker
# from bge_retriever import BgeRetriever
import dill
from tqdm import tqdm
from .preprocess import AQAdata
import faiss
import os.path as osp
import sys
import torch
import numpy as np
import json
import random
from sentence_transformers import SentenceTransformer
from utils import gte_retriever

path_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
device = "cuda"

def main_v3(data):
    # adjust the faiss index path and model path here
    
    retriever = gte_retriever(data, 
                              device= "cuda", 
                              path=osp.join(path_root, 'data/process_data/gte_retriever_finetune/680/database.index'), 
                              model_path=osp.join(path_root, 'result/mine_hard3/try/model/1'))
    emb_batch_size = 2
    searching_batch_size = 100
    embeddings = []
    result = []
    
    queries = data.question["train"]
    passage_ids = list(data.doc.keys())
    
    length = len(queries)
    for k in tqdm(range(length//emb_batch_size + 1)):
        start = k*emb_batch_size
        end = k*emb_batch_size+emb_batch_size
        if end>length:
            end=length
        if start >= end:
            break
        sentences = [item["query"] for item in queries[start:end]]
        embs = retriever.encode(sentences)
        embeddings.append(embs.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    
    for k in tqdm(range(length//searching_batch_size + 1)):
        start = k*searching_batch_size
        end = k*searching_batch_size+searching_batch_size
        if end > length:
            end=length
        if start >= end:
            break
        query_embeddings = embeddings[start:end, :]
        
        # embeddings.append(self.embedding(sentence))
        # embs = self.embedding(sentence, batch_size=end-start)
        res = retriever.search(query_embeddings, k=150)
        result += res
    
    with open(osp.join(path_root, 'data/process_data/rerank_dataset.json'),'w') as f:
        for res, query in zip(result, queries):
            pos_pid = query["pids"]
            query_question = query["query"]
            pos_set = set(pos_pid)
            random_select = random.sample(passage_ids, k=100)
            neg_0 = random.sample(list(set(res[30:90])-pos_set), k=20)
            neg_1 = random.sample(list(set(res[90:150])-pos_set), k=20)
            neg_2 = random.sample(list(set(random_select)-pos_set-set(res)), k=20)
            
            waiting_lists = [neg_0, neg_1, neg_2, pos_pid]
            number_list = ['0', '1', '2', '3']
            
            for wait_list, number in zip(waiting_lists, number_list):
                for pid in wait_list:
                    f.write(json.dumps({"query":query_question, "doc":data.doc[pid], "type":number}, ensure_ascii=False)+'\n')
            
            

if __name__ == '__main__':
    
    with open(osp.join(path_root, 'data/process_data', f'AQAdata.dill'), 'rb') as f:
        data:AQAdata = dill.load(f)
    
    with torch.no_grad():
        main_v3(data)
    