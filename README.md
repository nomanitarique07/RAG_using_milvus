# RAG_using_milvus

Above Notebook have the RAG pipeline that uses the URL: https://docs.nvidia.com/cuda/ to scrape the data and create the embeddings vector of the scraped text using an open source embedding model - "Alibaba-NLP/gte-Qwen2-1.5B-instruct" 

Milvus vector database is being used to store the embeddings and through its search method which performs semantic similarity search on the knowledge base embeddings for a user query.

CrossEncoder Rerank function is being used to rerank the retrieved result and final reranked retrieved result being sent to open source LLM - "HuggingFaceH4/zephyr-7b-beta" for response augmentation.


# RAG_using_milvus

This repository demonstrates a Retrieval-Augmented Generation (RAG) pipeline using the [Milvus vector database](https://milvus.io/). The pipeline performs the following steps:

1. **Data Scraping**: The pipeline scrapes data from the [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/).
2. **Embedding Creation**: The scraped text is converted into embedding vectors using the open-source embedding model [Alibaba-NLP/gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct).
3. **Embedding Storage**: The embedding vectors are stored in the Milvus vector database.
4. **Semantic Similarity Search**: Milvus performs a semantic similarity search on the knowledge base embeddings for a given user query.
5. **Result Re-ranking**: The retrieved results are re-ranked using the [CrossEncoder](https://huggingface.co/cross-encoder) re-rank function.
6. **Response Augmentation**: The final re-ranked retrieved result is sent to the open-source LLM [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) for response augmentation.

## Setup and Usage

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/RAG_using_milvus.git
    cd RAG_using_milvus
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Notebook**:
    Open and run the `RAG_pipeline.ipynb` notebook to execute the pipeline.

## Details

- **Data Scraping**: Utilizes web scraping techniques to extract text data from the [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/).
- **Embedding Creation**: Uses the [Alibaba-NLP/gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct) model to create embeddings from the scraped text.
- **Milvus Vector Database**: Stores the embeddings in Milvus for efficient retrieval and similarity search.
- **Semantic Similarity Search**: Performs a search on the stored embeddings to find the most relevant results for a user query.
- **Re-ranking**: Re-ranks the retrieved results using the [CrossEncoder](https://huggingface.co/cross-encoder) re-rank function to improve relevance.
- **Response Augmentation**: Uses the [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) model to generate a comprehensive response based on the re-ranked results.

For more details, refer to the `RAG_pipeline.ipynb` notebook.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
