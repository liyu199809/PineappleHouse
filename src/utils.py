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

path_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
device = "cuda"

class gte_embedding:
    def __init__(self, device="cuda", model_path=osp.join(path_root, "result/mine_hard/try/model/3")):
        model_name = model_path
        self.device = torch.device(device)
        self.embedding = SentenceTransformer(model_name, trust_remote_code=True).to(self.device)
    
    def __call__(self, sentence):
        embeddings = self.embedding.encode(sentence, normalize_embeddings=True, convert_to_tensor=True)
        return embeddings

class gte_retriever:
    def __init__(self, data, device= "cuda", path=osp.join(path_root, 'data/process_data/test/gte340.index'), model_path=osp.join(path_root, "result/mine_hard/try/model/3")):
        self.dict = data.doc
        self.batch_size = 3
        try:
            self.index = faiss.read_index(path)
            self.ids = list(data.doc.keys())
            self.index = faiss.index_cpu_to_all_gpus(self.index)
            self.embedding = gte_embedding(device, model_path)
        except:
            print("error!")
            sys.exit()
    
    def encode(self, query):
        return self.embedding(query)
    
    def search(self, query, k):
        _, ids = self.index.search(query, k)
        result = []
        for line in ids:
            result.append([self.ids[id] for id in line])
        return result