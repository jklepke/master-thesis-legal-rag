# master-thesis-legal-rag

This is a repository for my master's thesis on Retrieval-Augemented Generation in the legal domain specifically the law and ordinance landscape of the German energy supply system. 

This repository contains the code, experimental data, and evaluation results developed as part of my master's thesis at University of Cologne. The work investigates how retrieval-augmented generation (RAG) can improve the reliability of answering legal questions in the context of the German energy supply system. 

The final chatbot application (baseline, vector-based, and hybrid RAG models) is publicly deployed on Hugging Face Spaces. This repository, however, focuses exclusively on the experimental pipelines, data, and results.

---

## Research Context

Large Language Models (LLMs) show impressive performance in natural language generation, but they are prone to hallucinations and limited factual accuracy. These issues are especially critical in the legal domain. 

This thesis explored targeted optimizations of the three core phases of RAG:  
- **Indexing** (e.g., paragraph-based chunking)  
- **Retrieval** (e.g., hybrid retrieval, filtering, reranking)  
- **Generation** (e.g., self-consistency)  

A systematic experimental analysis demonstrated that optimized RAG pipelines, particularly vector-based retrieval and hybrid vector–graph approaches, significantly outperform both a non-optimized baseline RAG and a standard LLM in terms of semantic similarity, factual accuracy, and legal correctness. 

---

## Repository Structure

- **data/** – Input data and supporting datasets  
  - **evaluation/** – Evaluation datasets
    - **manual/** – Manual dataset (n=200)
    - **synthetic/** – Synthetic data generation (n=50)
  - **external_knowledge_source/** – External domain knowledge sources  
- **ipynb_notebooks/** – Jupyter notebooks for analysis
  - **standard_llm/** – Standard LLM baseline (no retrieval)  
  - **baseline/** – Baseline RAG implementation  
  - **retrieval_inputs/** – Retrieval inputs and indexing data  
  - **single_stage_enhancements/** – Individual enhancement modules
  - **multi_stage_enhancements/** – Combined RAG enhancements into a fully optimized 
  - **evaluation_datasets/** –  Creation of datasets for evaluation, implementation of metrics and llm-as-a-judge approach
- **visualization_and_figures/** – Scripts and figures for result visualization

---

## Installation and Usage

### Requirements
- Python version: **>= 3.10.11**
- Virtual environment recommended (e.g., `venv` or `conda`)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/jklepke/master-thesis-legal-rag.git
   cd master-thesis-legal-rag.git
   
2. **Create and activate a virtual environment**

    ```bash
    python -m venv venv
    source venv/bin/activate        # Linux/MacOS
    venv\Scripts\activate           # Windows

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt

4. **Configure API keys**
Create a .env file in the root directory and add the following keys:
    
    ```bash
    OPENAI_API_KEY=your_key
    COHERE_API_KEY=your_key
    GEMINI_API_KEY=your_key
    HUGGING_FACE_API_KEY=your_key
    TOGETHER_API_KEY=your_key

5. **Knowledge Graph (Neo4j)**

    Either run Neo4j Community Edition with Docker

    ```bash
    version: '3.8'
    
    services:
      neo4j:
        container_name: neo4j
        image: neo4j:latest
        ports:
          - "7474:7474"
          - "7687:7687"
        env_file:
          - .env.neo4j
        environment:
          - NEO4J_apoc_export_file_enabled=true
          - NEO4J_apoc_import_file_enabled=true
          - NEO4J_apoc_import_file_use__neo4j__config=true
          - NEO4JLABS_PLUGINS=["apoc", "graph-data-science"]
        volumes:
          - ./neo4j_db/data:/data
          - ./neo4j_db/logs:/logs
          - ./neo4j_db/import:/var/lib/neo4j/import
          - ./neo4j_db/plugins:/plugins

Or use [Neo4j Aura](https://neo4j.com/product/auradb/) (cloud-hosted version)

## Running Experiments

All experiments are provided as Jupyter notebooks in the ipynb_notebooks/ folder.
After installation, you can start Jupyter Lab or Jupyter Notebook and run the experiments step by step

---

## Single Stage Experiments

| **RAG Phase**  | **Experiment**                      | **Description** |
|----------------|-------------------------------------|-----------------|
| **Indexing**   | Optimal Chunking                    | Analysis of different chunk sizes and overlaps to improve context quality |
|                | Vector vs. Graph Store              | Comparison of two types of storage: vector representation vs. graph representation |
| **Retrieval**  | Retriever Selection                 | Selection of suitable retrieval strategies and hybrid combinations |
|                | Retrieval Process                   | Investigation of iterative, recursive, and adaptive strategies in the retrieval process |
| **Generation** | Reranking, Filtering & Summarization| Effect analysis of context selection, sorting, and compression in the language model input |
|                | Self vs. Cross-Consistency          | Consistency checks through multiple generations of responses and selection of the most suitable |

---

## Multi Stage Experiments (i.e. Final RAG Pipelines)

| **Optimization**                  | **Baseline RAG** | **VecRAG** (Vector) | **HyRAG** (Hybrid: Vector + Graph) | **Status** |
|-----------------------------------|------------------|---------------------|------------------------------------|------------|
| 1. Optimal Chunking               | (1024, 128)    | ✓  (paragraph-wise)                 | ✓    (paragraph-wise)                                  | Successful |
| 2. Vector vs. Graph Store         | Vector only      | Vector only         | Vector + Graph                     | Partial    |
| 3. Retriever Selection            | Dense                | ✓ Hybrid (MMR+BM25)                  | ✓ Hybrid (MMR+BM25)                                  | Successful |
| 4. Retrieval Process              | Single                | ✗                   | ✗                                  | Not included |
| 5. Reranking, Filtering, Summarization | ✗          | ✓ Filtering (0.25) + RankGPT                   | ✓ Filtering (0.25) + RankGPT                                  | Successful |
| 6. Self- & Cross-Consistency      | ✗                | ✓ Self-Consistency                  | ✓ Self-Consistency                                  | Successful |

**Legend:**  
✓ = integrated, ✗ = not included  

---

## Deployment

The final chatbot is accessible on Hugging Face Spaces:  
[Legal RAG Chatbot](https://huggingface.co/spaces/jonas61099/legal-rag-chatbot)

This repository contains only the experimental code, data, and evaluation results.

## License

This project is released under the MIT License (see LICENSE file).

## Used AI Tools

ChatGPT was used for error handling and code creation

