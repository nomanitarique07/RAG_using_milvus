# RAG_using_milvus

This repository demonstrates a Retrieval-Augmented Generation (RAG) pipeline using the [Milvus vector database](https://milvus.io/). The pipeline performs the following steps:

1. **Data Scraping**: The pipeline scrapes data from the [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/).
2. **Embedding Creation**: The scraped text is converted into embedding vectors using the open-source embedding model [Alibaba-NLP/gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct).
3. **Embedding Storage**: The embedding vectors are stored in the Milvus vector database.
4. **Semantic Similarity Search**: Milvus performs a semantic similarity search on the knowledge base embeddings for a given user query.
5. **Result Re-ranking**: The retrieved results are re-ranked using the [CrossEncoder](https://huggingface.co/cross-encoder) re-rank function.
6. **Response Augmentation**: The final re-ranked retrieved result is sent to the open-source LLM [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) for response augmentation.


