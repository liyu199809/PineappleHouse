import torch
import faiss

import os.path as osp
import numpy as np
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup
from typing import List
from sentence_transformers import SentenceTransformer

class baseRetriever:
    """
    A class for retrieving related documents based on a query.

    Attributes:
        db_path (str): The path to the database file.
        embedding_model_path (str): The path to the sentence embedding model.
        faiss_index_path (str): The path to the FAISS index file.

    Methods:
        __init__(self, db_path:str, embedding_model_path:str, faiss_index_path:str):
            Initializes the baseRetriever object.
        get_related_doc(self, query:List[str], top_k:int):
            Retrieves the top k related documents based on a query.
        get_pid(self, metadata_batch):
            Extracts the document IDs from the metadata batch.
        __call__(self, query:str, top_k:int=20):
            Retrieves the top k related documents based on a query.
    """
    def __init__(self, db_path:str, embedding_model_path:str, faiss_index_path:str):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.sentences, emb, self.metadatas = torch.load(db_path)
        
        if not osp.exists(faiss_index_path):
            print('build faiss index')
            
            self.vectorstore = faiss.IndexFlatIP(1024) 
            self.vectorstore = faiss.IndexIDMap(self.vectorstore)
    
            self.vectorstore.add_with_ids(emb, np.arange(len(self.sentences)))
            print('add faiss successfully')
            faiss.write_index(self.vectorstore, faiss_index_path)
        else:
            self.vectorstore = faiss.read_index(faiss_index_path)
        
        # if self.device == 'cuda':
        #     self.vectorstore = faiss.index_cpu_to_all_gpus(self.vectorstore)

        self.query_embedder = SentenceTransformer(embedding_model_path, device=self.device, trust_remote_code=True)
        # self.sentences = np.array(self.sentences) # TODO: OOM bug
        self.metadatas = np.array(self.metadatas)
        print('baseRetriever init successfully')

    def get_related_doc(self, query:List[str], top_k:int):
        """
        Retrieves the top k related documents based on a query.

        Args:
            query (List[str]): The query to use for retrieval.
            top_k (int): The number of top related documents to retrieve.

        Returns:
            numpy.ndarray: The top k related documents.
        """
        query = self.query_embedder.encode(query, normalize_embeddings=True, device=self.device)
        score, ids = self.vectorstore.search(query, k=top_k)
        return self.metadatas[ids]

    def get_pid(self, metadata_batch):
        """
        Extracts the document IDs from the metadata batch.

        Args:
            metadata_batch (numpy.ndarray): The metadata batch to extract IDs from.

        Returns:
            list: The list of document IDs.
        """
        metadata_batch = metadata_batch.tolist()
        doc_id_list = []
        for metadata in metadata_batch:
            temp_list = []
            for meta in metadata:
                temp_list.append(meta['pid'])
            doc_id_list.append(temp_list)
        return doc_id_list

    def __call__(self, query:str, top_k:int=20):
        # TODO: batch infer query
        metadata_batch = self.get_related_doc(query, top_k)
        doc_id_list = self.get_pid(metadata_batch)
        return doc_id_list


if __name__ == '__main__':
    work_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    mnt_root = '/mnt/bn/liyullm2/kdd_output/' # 云盘路径
    with open(osp.join(work_root, 'AQA', 'qa_test_wo_ans_new.txt'), 'r') as f:
        val_query_list = f.readlines()

    ckpt_id = 4
    embedding_model_path = f'/mnt/bn/liyullm2/kdd_output/train_data_woh_wb/runs/checkpoints/checkpoint_{ckpt_id}'
    vectorstore_path = osp.join(mnt_root, f'bsz16_woh_wb_4_emb.bin')
    faiss_index_path = osp.join(mnt_root, 'train_data_woh_wb', f'train_data_woh_wb_{ckpt_id}')

    retriever = baseRetriever(vectorstore_path, embedding_model_path, faiss_index_path)
    
    file_write_obj = open(osp.join(work_root, 'result', f"woh_wb_{ckpt_id}.txt"), 'a')

    for query in tqdm(val_query_list):
        query = eval(query)
        
        question = query['question']
        soup = BeautifulSoup(query.get('body', ''), 'html.parser')
        body = soup.get_text()
        retriever_input = "# question:" + question + "\n\nDescription:" + body

        doc_id_list = retriever([retriever_input])
        file_write_obj.write(','.join(doc_id_list[0]))
        file_write_obj.write('\n')
    file_write_obj.close()