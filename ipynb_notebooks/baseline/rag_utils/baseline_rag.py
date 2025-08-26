# === Standard Library ===
import json
import os
import re
import shutil
import sys
import uuid
from urllib.request import urlopen

# === Third-Party Libraries ===
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from langdetect import detect, DetectorFactory
import pycountry
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# === LangChain Core ===
from langchain.docstore.document import Document
from langchain.schema import Document  # (Optional: doppelt zu obigem)
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import TokenTextSplitter

# === LangChain Community Integrationen ===
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

# === OpenAI / LangChain OpenAI ===
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

# === Lokale Projektmodule ===
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from ipynb_notebooks.evaluation_datasets.generation_eval.generation_metrics import run_generation_evaluation
from ipynb_notebooks.evaluation_datasets.retrieval_eval.eval_vector_dataset_generator import generate_evalset
from ipynb_notebooks.evaluation_datasets.retrieval_eval.retrieval_metrics import run_retrieval_evaluation



# Configurations

# Load environment variables. Assumes that the project directory contains a .env file with API keys
load_dotenv()

# Set the OpenAI API key from the environment variables
# Make sure to update "OPENAI_API_KEY" to match the variable name in your .env file
openai.api_key = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=openai.api_key)

# Define constants for paths
DATA_PATH = "../../data/laws_and_ordinances.json"  # Directory containing the url to the law and ordinance documents
DATA_PATH_SHORT_VERSION = "../../data/laws_and_ordinances_short_version.json" # Directory containing a subset of all urls for testing purposes
CHROMA_PATH = "chroma_dbs/chroma"  # Directory to save the Chroma vector store


# Helper Functions

def clean_text(content: str) -> str:
    soup = BeautifulSoup(content, 'html.parser')

    # Define replacements: Marker → Paragraph Break
    replacements = {
        "Nichtamtliches Inhaltsverzeichnis": "\n\n",
        "zum Seitenanfang": "",
        "zurück": "",
        "weiter": "",
        "Impressum": "",
        "Datenschutz": "",
        "Barrierefreiheitserklärung": "",
        "Feedback-Formular": ""
    }

    # Replace values
    for old_text, new_text in replacements.items():
        for element in soup.find_all(string=re.compile(re.escape(old_text), re.IGNORECASE)):
            element.replace_with(element.replace(old_text, new_text))

    # Cleaning of additional linebreaks and whitespaces
    cleaned_content = soup.get_text(separator='\n', strip=True)
    cleaned_content = re.sub(r'\n\s*\n', '\n\n', cleaned_content)

    return cleaned_content


def save_cleaning_diff(raw_text, cleaned_text, title):
    
    save_dir="../../data/extracted_contents"
    
    os.makedirs(save_dir, exist_ok=True)

    raw_path = os.path.join(save_dir, "raw_contents", f"{title}_raw.txt")
    clean_path = os.path.join(save_dir, "cleaned_contents", f"{title}_cleaned.txt")

    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)


def load_documents(datapath: str, baseline: bool = False):
    with open(datapath, "r", encoding="utf-8") as file:
        data = json.load(file)

    documents = []

    entries = []
    for category in ["laws", "ordinances"]:
        for entry in data.get(category, []):
            entry["category"] = category
            entries.append(entry)

    if baseline:
        for entry in tqdm(entries, desc="Loading documents"):
            title = entry.get("title", "Unknown Title")
            base_url = entry.get("base_url", "")
            category = entry["category"]

            if base_url:
                loader = WebBaseLoader(base_url)
                docs = loader.load()
                for doc in docs:
                    raw_content = doc.page_content
                    cleaned_content = clean_text(raw_content)
                    doc.page_content = cleaned_content

                    save_cleaning_diff(raw_content, cleaned_content, title)
                    doc.metadata.update({"title": title, "category": category})
                    documents.append(doc)
            else:
                print(f"Missing base URL for: {title}")

    else:
        cleaned_dir = os.path.join("..", "..", "data", "extracted_contents", "cleaned_contents")
        expected_files = {}
        for entry in entries:
            title = entry["title"]
            expected_filename = f"{title}_cleaned.txt"
            expected_files[expected_filename] = {
                "source": entry.get("base_url", ""),
                "title": entry.get("title", "Unknown Title"),
                "language": entry.get("language", "unknown"),
                "category": entry.get("category", "unknown")
            }

        for filename in tqdm(os.listdir(cleaned_dir), desc="Loading cleaned files"):
            if filename in expected_files:
                filepath = os.path.join(cleaned_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                metadata = expected_files[filename]
                metadata["source_file"] = filename

                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)

    if not documents:
        raise ValueError("No documents loaded.")

    print(f"Loaded {len(documents)} documents.")
    return documents

def documents_to_jsonable(documents: list[Document]) -> list[dict]:
    return [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in documents
    ]
    
def save_documents_for_sparse_retrieval(documents: list[Document], chunk_size: int, chunk_overlap: int, optimization: str, baseline: bool = False):
    filename = f"{len(documents)}_documents_for_sparse_retrieval_{chunk_size}_{chunk_overlap}_{optimization}{'_baseline' if baseline else ''}.json"
    filepath = f"../retrieval_inputs/stored_chunks_for_sparse_retrieval/{filename}"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    jsonable_docs = documents_to_jsonable(documents)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(jsonable_docs, f, ensure_ascii=False, indent=2)
        
def load_documents_for_sparse_retrieval(json_path: str) -> list[Document]:
    with open(f"../retrieval_inputs/stored_chunks_for_sparse_retrieval/{json_path}", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    return [
        Document(page_content=entry["page_content"], metadata=entry["metadata"])
        for entry in raw_data
    ]

# ### 2. Creation of Vector Database

def generate_data_store(datapath, chunk_size=256, chunk_overlap=32, baseline: bool = False, optimization: str = "default"):
    documents = load_documents(datapath=datapath, baseline=baseline)
    chunks = split_text(documents, chunk_size, chunk_overlap)
    save_documents_for_sparse_retrieval(chunks, chunk_size, chunk_overlap, optimization, baseline)
    chroma_path = save_to_chroma(chunks, chunk_size, chunk_overlap, baseline, optimization)
    return chroma_path

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

def token_length(text):
    return len(encoding.encode(text))

def split_text(documents: list[Document], chunk_size, chunk_overlap):
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name="gpt-4o-mini"
    )
    chunks = text_splitter.split_documents(documents)
    
    chunk_index = 1
    
    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunk.metadata["chunk_index"] = chunk_index
        chunk_index+= 1

    if len(chunks) > 10:
        document = chunks[10]
    
    return chunks

def save_to_chroma(chunks: list[Document], chunk_size, chunk_overlap, baseline, optimization, batch_size=10):
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing directory: {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    
    if baseline: 
        chroma_path = f"../chroma_dbs/chroma_chunksize{chunk_size}_overlap{chunk_overlap}_{str(uuid.uuid4())[:8]}_baseline"
    else: 
        chroma_path = f"../chroma_dbs/chroma_chunksize{chunk_size}_overlap{chunk_overlap}_{str(uuid.uuid4())[:8]}_{optimization}"

    
    # preprare embeddings 
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # initialize Chroma
    db = Chroma(embedding_function=embeddings, persist_directory=chroma_path)
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="🔢 Store Chunks with Embeddings"):
        batch = chunks[i:i+batch_size]
        db.add_documents(batch)
        
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    return chroma_path



PROMPT_TEMPLATE = """
Du bist ein hilfreicher, juristischer KI-Assistent für Gesetzestexte im deutschen Energie- und Versorgungsbereich. 
Generiere eine kurze, präzise, konsistente und vollständige Gesamtantwort von max. 200 Tokens basierend auf folgenden Kontext: 

Frage:
{question}
---
Kontext:
{context}
---
Sprache in der geantwortet werden soll: 
{language}
"""



def translate_query_to_german_if_needed(query: str) -> str:
    try:
        detected_lang = detect(query)
    except LangDetectException:
        detected_lang = "unknown"

    if detected_lang == "de":
        return query
    else: 
        translation_prompt = f"Translate the following question accurately and correctly into German:\n\n{query}"
        
        try:
            response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are an AI Translator specialized in translating texts to German."},
                    {"role": "user", "content": translation_prompt}],
            temperature=0.3
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error while translating: {e}")
            return query 
        

# Reproduzierbare Ergebnisse
DetectorFactory.seed = 42

def detect_language_name(text):
    try:
        lang_code = detect(text)
        # Ländercode in Klartext-Sprache umwandeln
        language = pycountry.languages.get(alpha_2=lang_code)
        if language is not None:
            return language.name  # z.B. 'German'
        else:
            return lang_code  # fallback
    except LangDetectException:
        return "unbekannt"



def load_vector_database(chroma_path):
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
    return db


def retrieve_documents(query_text, db, k=6):
    if len(db) == 0:
        return [], "No documents available in the database."

    query_de = translate_query_to_german_if_needed(query_text)
    results = db.similarity_search_with_relevance_scores(query_de, k=k)
    
    return results



def generate_answer(results, query_text, model_name, temperature: float = 0.7, custom_prompt: str = None):
    
    # Use passed-in prompt directly if provided
    if custom_prompt is not None:
        final_prompt = custom_prompt
    else:
        # Build context from results if no prompt is given
        if isinstance(results, str):
            context_text = results
        elif isinstance(results, list):
            context_text = "\n\n---\n\n".join(
                doc.page_content if isinstance(doc, Document) else str(doc)
                for doc in results
            )
        else:
            raise ValueError("Unsupported format for 'results': expected str or list of Document")

        detected_language = detect_language_name(query_text)

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        final_prompt = prompt_template.format(
            context=context_text,
            question=query_text,
            language=detected_language
        )


    model = ChatOpenAI(model_name=model_name, temperature=temperature)
    response_text = model.predict(final_prompt)

    return response_text


def rag_pipeline(query, database, model_name="gpt-4o-mini"):
    
    results = retrieve_documents(query, db=database)

    response = generate_answer(results, query, model_name)
    
    if results and isinstance(results[0], tuple):
        results = [doc for doc, _ in results]

    sources = [doc.metadata.get("source") for doc in results]
    retrieved_chunk_contexts = [doc.page_content for doc in results]
    retrieved_chunk_ids = [doc.metadata.get("chunk_id") for doc in results]
    retrieved_chunk_indices = [doc.metadata.get("chunk_index") for doc in results]

    return response, sources, retrieved_chunk_contexts, retrieved_chunk_ids, retrieved_chunk_indices


def enrich_eval_dataset_with_rag_responses(eval_dataset, chroma_path, model_name="gpt-4o-mini"):
    
    db = load_vector_database(chroma_path)

    with open(eval_dataset, "r", encoding="utf-8") as f:
        eval_dataset_json = json.load(f)

    enriched_dataset = []
    
    for entry in tqdm(eval_dataset_json, desc="Processing RAG responses"):
        query = entry["query"]

        # Run RAG pipeline
        response, _, retrieved_chunk_contexts, retrieved_chunk_ids, retrieved_chunk_indices = rag_pipeline(query, db, model_name=model_name)

        # Add new fields to file
        entry["generated_response"] = response
        entry["retrieved_chunk_contexts"] = retrieved_chunk_contexts
        entry["retrieved_chunk_ids"] = retrieved_chunk_ids
        entry["retrieved_chunk_indices"] = retrieved_chunk_indices

        enriched_dataset.append(entry)

    output_path = f"{eval_dataset.replace('.json', '')}_rag_enriched.json"
    # Store results as new json file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_dataset, f, indent=2, ensure_ascii=False)
        
    return output_path
