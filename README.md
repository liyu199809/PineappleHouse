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
- In addition we provide the checkpoint of finetuned gte-large-en-v1.5, you can download it from the Baidu Netdisk (link：https://pan.baidu.com/s/1OOhfzIhedVWpgBYnBcTwKQ?pwd=8zoc, password：8zoc). You can convert the checkpoint to sentence-transformer checkpint form following the guidence of sentence-transformer official site [SentenceTransformers Documentation — Sentence Transformers documentation (sbert.net)](https://sbert.net/)

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

See the code comments for details on how to use. You need to specify the model name of model path in the code to produce your results for each model.

### rerank

```
python3 src/rerank.py
```

See the code comments for details on how to use.

### hard nagative sampling

```
python src/preprocess.py
python src/hard_negative_mining.py
```

See the code comments for details on how to use.  We use this code to mine the hard negative examples, according to the similarity between docs and queries.

### RRF

```
python3 src/RRF.PY
```

See the code comments for details on how to use. This code can merge the retrieval result produced  by different models, using reciprocal rank fusion(RRF).

### other function

- hyde.py: Generating hypothetical answers for retrieval via LLM
- doc_classifier.py: Classify documents into different categories
- query_rewrite: Rewrite query

## How to reproduce our best result？

We have explored a lot of ways to improve the effectiveness of our model, but some proves to be only useful in the valid dataset (reranker and hard negative), so we only apply some of the functions mentioned above in our best-result-model.

### finetune embedding model

```
cd model_finetune/rag-retrieval/embedding
bash train_embedding.sh
```

We finetune gte-large-en-v1.5 using contrastive learning.

### build retriever and retrieve docs

```
python3 src/build_retriever.py
python3 src/get_related_doc.py
```

Here we construct retrievers of five models: gte-large-en-v1.5(finetuned), GritLm-7B, SFR-Embedding-Mistral, NV-Embed-v1, Linq-Embed-Mistral. We retrieve 100 docs for each query by those five retrievers, and get five result files respectively in the result folder.

### combine multiple result sets

```
python3 src/RRF.PY
```

We finally combine the result of the mentioned fine models using RRF, and after voting, top-20 results of each query are selected as the final results.
