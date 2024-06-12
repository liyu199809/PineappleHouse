import json
import torch

import os.path as osp

from glob import glob
from tqdm import tqdm
from langchain_core.documents import Document
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.vectorstores.utils import DistanceStrategy
from sentence_transformers import SentenceTransformer


class vectorstoreBuilder:
    """
    This class is used to build a vector store for document retrieval.

    Attributes:
        type (str): The type of the vector store.
        work_root (str): The root directory of the work.
        mnt_root (str): The root directory of the mount.

    Methods:
        load_documents(): Loads documents from the processed data directory.
        build_vectorstore_bge(embedding_model_path): Builds a vector store using the BGE embeddings.
        build_vectorstore_mutilGPU(embedding_model_path): Builds a vector store using multiple GPUs.
    """
    def __init__(self, db_type:str):
        self.type = db_type 
        self.work_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
        self.mnt_root = '/mnt/bn/liyullm2/kdd_output/'
    
    def load_documents(self):
        """
        Loads documents from the processed data directory.

        Returns:
            list: A list of documents.
        """
        doc_path = osp.join(self.work_root, 'processed_data', 'documents.pt')
        if osp.exists(doc_path):
            doc_list = torch.load(osp.join(self.work_root, 'processed_data', 'documents.pt'))
        else:
            raw_doc1 = json.load(open(osp.join(self.work_root, 'AQA', 'pid_to_title_abs_new.json'), 'r'))
            raw_doc2 = json.load(open(osp.join(self.work_root, 'AQA', 'pid_to_title_abs_update_filter.json'), 'r'))
            raw_doc = {**raw_doc1, **raw_doc2}

            doc_list = []
            for doc_id in raw_doc.keys():
                if raw_doc[doc_id]['title'] is None:
                    title = ' '
                else:
                    title = raw_doc[doc_id]['title']
                doc = Document(page_content= '# ' + title + '\n\n' + raw_doc[doc_id]['abstract'],
                                          metadata={'pid':doc_id})
                doc_list.append(doc)
            torch.save(doc_list, doc_path)

        return doc_list

    def build_vectorstore_bge(self, embedding_model_path):
        """
        Builds a vector store using the BGE embeddings.

        Args:
            embedding_model_path (str): The path to the embedding model.

        Returns:
            FAISS: The built vector store.
        """
        doc_list = self.load_documents()
        
        model_kwargs = {'device': 'cuda', 'trust_remote_code':True}
        encode_kwargs = {'normalize_embeddings': True, 'batch_size':16, 'show_progress_bar': True} # set True to compute cosine similarity
        embedding_hf = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            max_seq_length=1024,
        )

        vectorstore = FAISS.from_documents(doc_list, embedding_hf, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        vectorstore.save_local(osp.join(self.mnt_root, 'vector_db', f'{self.type}'))
        return vectorstore

    def build_vectorstore_mutilGPU(self, embedding_model_path):
        """
        Builds a vector store using multiple GPUs.

        Args:
            embedding_model_path (str): The path to the embedding model.

        Returns:
            None: The function does not return any value.
        """
        doc_list = self.load_documents()
        
        sentences, metadatas = [], []
        for doc in doc_list:
            sentences.append(doc.page_content)
            metadatas.append(doc.metadata)

        model = SentenceTransformer(embedding_model_path, trust_remote_code=True)
        model.max_seq_length = 1024

        pool = model.start_multi_process_pool()
        emb = model.encode_multi_process(sentences, pool, batch_size=384, normalize_embeddings=True)
        model.stop_multi_process_pool(pool)

        torch.save([sentences, emb, metadatas], osp.join(self.mnt_root, f'{self.type}_emb.bin'))


if __name__ == '__main__':
    check_point_root = '/mnt/bn/liyullm2/kdd_output/train_data_woh_wb/runs/checkpoints/checkpoint_4'
    # check_point_list = glob(osp.join(check_point_root, 'checkpoint_*'))
    # print(check_point_list)

    # for ckpt in tqdm(check_point_list):
    builder = vectorstoreBuilder('bsz16_woh_wb_4')
    vectorstore = builder.build_vectorstore_mutilGPU(check_point_root)

