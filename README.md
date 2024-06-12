# OAG-AQA

## Prerequisites
Linux
Python 3.9.19
PyTorch 2.0.1

Please refer to requirements.txt for detail environments

## How to reproduce our experimental results？

### Overview of the project structure
- AQA: dataset from offical
- model_finetune: finetune embedding models and rerank models
    - config: accelerate trainning config
    - rag-retrieval: trainging code
- processed data: save processed data
- results: save results txt file
- src: main pipline of retriver

### base model
- embedding model: NV-Embed-v1,Linq-Embed-Mistral,GritLM-7B, gte-large-en-v1.5, SFR-Embedding-Mistral
- rerank model: bge-reranker-v2-m3

### Finetune embedding model
```
cd model_finetune/rag-retrieval/embedding
bash train_embedding.sh
```
Note that you need to set up your relevant parameters (e.g., model name, number of negative examples) in .sh file before running it.

### Finetune rerank model
```
cd model_finetune/rag-retrieval/reranker
bash train_rerank.sh
```
Note that you need to set up your relevant parameters (e.g., model name, number of negative examples) in .sh file before running it.

### get documents embedding
```
python3 src/build_retriever.py
```
Note that you need to set up your relevant parameters (e.g., embedding model path, save_path) in `build_retriever.py` file before running it.

### retrival
```
python3 src/get_related_doc.py
```
See the code comments for details on how to use.

### rerank
```
python3 src/rerank.py
```
See the code comments for details on how to use.

### RRF
@ctl
### hard nagative sampling
@ctl

### other function
- hyde.py: Generating hypothetical answers for retrieval via LLM
- doc_classifier.py: Classify documents into different categories
- query_rewrite: Rewrite query