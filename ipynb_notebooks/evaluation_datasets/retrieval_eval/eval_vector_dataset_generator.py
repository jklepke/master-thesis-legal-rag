from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from typing import List
import webbrowser

from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer, MultiHopSpecificQuerySynthesizer
from ragas.testset.persona import Persona
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain_core.documents import Document
import openai

# .env-Datei laden
load_dotenv()

# Zugriff auf den Token
ragas_token = os.getenv("RAGAS_APP_TOKEN")
os.environ["RAGAS_APP_TOKEN"] = ragas_token

openai.api_key = os.environ['OPENAI_API_KEY']

import json
import random
import logging
from pathlib import Path
from typing import List
from uuid import uuid4
import pandas as pd

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from openai import OpenAI

from tqdm import tqdm

# == SETUP ==
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
# Set client for chat completion 
client = OpenAI(api_key=openai.api_key)
embedding_model = OpenAIEmbeddings()

OUTPUT_FOLDER = "datasets/"

# == CONFIG ==
SINGLE_HOP_RATIO = 0.8  # 80% Single Hop
TOTAL_QUERIES = 100
MAX_CONTEXT_CHUNKS = 3


persona = Persona(
    name="energy_law_practitioner",
    role_description=(
        "Ein juristischer Fachanwender mit Spezialisierung im deutschen Energier- und Versorgungsrecht. "
        "Hat tiefes Verständnis für gesetzliche Regelungen wie das EnWG, StromNEV oder WPG und sucht gezielt nach "
        "Antworten auf konkrete rechtliche Fragestellungen im Kontext von Energieversorgung, Netzregulierung und Planungspflichten."
    )
)

# == Load documents from Chroma ==
def load_documents_from_chroma(chroma_db: str) -> List[Document]:
    db = Chroma(
        persist_directory=chroma_db,
        embedding_function=embedding_model
    )

    data = db.get(include=["embeddings", "documents", "metadatas"])
    embeddings = data["embeddings"]
    documents = data["documents"]
    metadatas = data["metadatas"]

    # Reconstruct as LangChain Documents
    docs = [Document(embedding=embedding, page_content=doc, metadata=meta) for embedding, doc, meta in zip(embeddings, documents, metadatas)]

    return docs


# == Group chunks by title ==
def group_chunks_by_title(documents: List[Document]):
    grouped = {}
    
    for doc in documents:
        paragraph_wise_chunking = bool(doc.metadata.get("law_title"))
        if paragraph_wise_chunking:
            title = doc.metadata.get("law_title", "unknown")
            grouped.setdefault(title, []).append(doc)
        else:
            title = doc.metadata.get("title", "unknown")
            grouped.setdefault(title, []).append(doc)
    return grouped


# == Find adjacent chunks ==
def get_adjacent_chunks(chunk_list: List[Document], num_chunks: int = 3) -> List[Document]:
    if num_chunks < 1:
        raise ValueError("num_chunks must be at least 1")

    sorted_chunks = sorted(chunk_list, key=lambda x: x.metadata.get("chunk_index", 0))
    total_chunks = len(sorted_chunks)
    
    if total_chunks <= num_chunks:
        return sorted_chunks

    half = num_chunks // 2
    valid_range_start = half
    valid_range_end = total_chunks - (num_chunks - half - 1)

    if valid_range_end <= valid_range_start:
        return sorted_chunks[:num_chunks]

    center_idx = random.randint(valid_range_start, valid_range_end - 1)
    start_idx = center_idx - half
    end_idx = start_idx + num_chunks

    return sorted_chunks[start_idx:end_idx]

def get_random_intra_document_chunks(chunk_list: List[Document], max_chunks: int = 5) -> List[Document]:
    
    num_chunks = random.randint(2, max_chunks)
    
    if len(chunk_list) <= num_chunks:
        return chunk_list 
    
    return random.sample(chunk_list, num_chunks)


def extract_question_answer(response):
    # Extract the response (raw text)
    response_content = response.choices[0].message.content
    
    # Extracting the question and answer using custom delimiters
    try:        
        question_start = response_content.index("Frage:") + len("Frage:")
        
        question_end = response_content.index("Antwort:")        
        
        answer_start = response_content.index("Antwort:") + len("Antwort:")
        
        # Extract question and answer
        question = response_content[question_start:question_end].strip()
        answer = response_content[answer_start:].strip()
        
        # Clean up extra whitespace or newline characters
        question = question.strip().replace("\n", " ")
        answer = answer.strip().replace("\n", " ")
        
        return question, answer

    except ValueError:
        # Handle case where "Summary Question" or "Answer" not found
        print("Error: Unable to extract question and answer.")
        return None

# == Generate single-hop query ==
def generate_single_query(doc: Document) -> dict:
    
    title = {doc.metadata.get("title")} 
    
    prompt = f"""Erstelle eine präzise Frage und eine genaue Antwort, die auf dem folgenden Ausschnitt eines Gesetzes/einer Verordnung basieren:

                Ausschnitt aus {title}:
                \"\"\"{doc.page_content}\"\"\"
                
                Bitte gib die Ausgabe genau im folgenden Format zurück und verzichte bitte auf Formatierungen wie z.B. Fett, Kursiv, etc.:

                Frage: [deine Frage hier]
                Antwort: [deine Antwort hier]
             """
    system_prompt = f"""Du bist ein KI-System, das als folgende Person agiert:

                        Name: {persona.name}
                        Rolle: {persona.role_description}

                        Deine Aufgabe ist es, rechtliche Fragen und Antworten basierend auf Gesetzestexten zu generieren.
                        Achte darauf, dass die Fragen deinem Wissensstand und deiner Perspektive entsprechen.
                     """
                     
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": prompt}]
    )
    
    question, answer = extract_question_answer(response)
    
    return {
        "query": question or "",
        "ground_truth": answer or "",
        "ground_truth_chunk_contexts": [doc.page_content],
        "ground_truth_chunk_ids": [doc.metadata.get("chunk_id", str(uuid4()))],
        "ground_truth_chunk_indices": [doc.metadata.get("chunk_index", "chunk_index not found")],
        "synthesizer": "single-hop",
    }
    
# == Generate multi-hop query ==
def generate_multi_query(docs: List[Document], query_synthesizer_type: str = "") -> dict:
    
    context_text = "\n\n".join([d.page_content for d in docs])
    titles = list(dict.fromkeys(doc.metadata.get("title", "No title") for doc in docs))
    
    prompt = f"""Erstelle eine komplexere Frage, deren Beantwortung mehrere zusammenhängende Textabschnitte eines Gesetzestexts erfordert:

                Ausschnitte aus {titles}:
                \"\"\"{context_text}\"\"\"

                Bitte gib die Ausgabe genau im folgenden Format zurück und verzichte bitte auf Formatierungen wie z.B. Fett, Kursiv, etc.:

                Frage: [deine Frage hier]
                Antwort: [deine Antwort hier]
             """
                 
    system_prompt = f"""Du bist ein KI-System, das als folgende Person agiert:

                        Name: {persona.name}
                        Rolle: {persona.role_description}

                        Deine Aufgabe ist es, rechtliche Fragen und Antworten basierend auf Gesetzestexten zu generieren.
                        Achte darauf, dass die Fragen deinem Wissensstand und deiner Perspektive entsprechen.
                     """
                     
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": prompt}]
    )
    
    question, answer = extract_question_answer(response)
    
    return {
        "query": question or "",
        "ground_truth": answer or "",
        "ground_truth_chunk_contexts": [d.page_content for d in docs],  
        "ground_truth_chunk_ids": [d.metadata.get("chunk_id", str(uuid4())) for d in docs],
        "ground_truth_chunk_indices": [d.metadata.get("chunk_index", "chunk_index not found") for d in docs],
        "synthesizer": f"multi-hop-{query_synthesizer_type}",
    }
    
def generate_evalset(chroma_db: str, test_set_size: int, optimization: str = "", query_distribution: dict = None):
    
    documents = load_documents_from_chroma(chroma_db)
    grouped = group_chunks_by_title(documents)

    testset = []

    if query_distribution is None:
        query_distribution = {
            "single": 0.6, # single chunks 
            "multi_specific": 0.2, # adjacent chunks
            "multi_intra_document": 0.1, # chunks in the same document
            # "multi_cross_context": 0.1 # chunks in different documents
        }
    
    single_ratio = query_distribution.get("single", 0.0) 
    multi_specific_ratio = query_distribution.get("multi_specific", 0.0)
    multi_intra_document_ratio = query_distribution.get("multi_intra_document", 0.0)
    # multi_cross_context_ratio = query_distribution.get("multi_cross_context", 0.0)
    
    num_single = int(single_ratio * test_set_size)
    num_multi_specific = int(multi_specific_ratio * test_set_size)
    num_multi_intra_document = int(multi_intra_document_ratio * test_set_size)
    # num_multi_cross_context = int(multi_cross_context_ratio * test_set_size)
    
    remaining = test_set_size - (num_single + num_multi_specific + num_multi_intra_document)
    
    if(remaining>0 or remaining<0):
        num_single += remaining

    query_id = 1
    
    print(f"🔹 Generating {num_single} Single-Hop Queries")
    single_candidates = random.sample(documents, k=min(num_single, len(documents)))
    for doc in tqdm(single_candidates):
        try:
            entry = generate_single_query(doc)
            entry = {
                    "query_id": query_id,
                    **entry
                    }
            testset.append(entry)
            query_id += 1
            # testset.append(generate_single_query(doc))
        except Exception as e:
            logging.warning(f"Single-Hop Fehler: {e}")

    print(f"🔸 Generating {num_multi_specific} Multi-Hop-Specific Queries (i.e. Adjacent Chunks)")
    for _ in tqdm(range(num_multi_specific)):
        try:
            title = random.choice(list(grouped.keys()))
            chunk_group = grouped[title]
            adjacent_chunks = get_adjacent_chunks(chunk_group, num_chunks=5)
            entry = generate_multi_query(adjacent_chunks, query_synthesizer_type="adjacent")
            entry = {
                    "query_id": query_id,
                    **entry
                    }
            testset.append(entry)
            query_id += 1
            # testset.append(generate_multi_query(adjacent_chunks, query_synthesizer_type="adjacent"))
        except Exception as e:
            logging.warning(f"Multi-Hop-Specific Fehler: {e}")
            
    print(f"🔹 Generating {num_multi_intra_document} Multi-Hop-Intra-Document Queries")
    for _ in tqdm(range(num_multi_intra_document)):
        try:
            title = random.choice(list(grouped.keys()))
            chunk_group = grouped[title]
            intra_document_chunks = get_random_intra_document_chunks(chunk_group, max_chunks=5)
            entry = generate_multi_query(intra_document_chunks, query_synthesizer_type="intra-doc")
            entry = {
                    "query_id": query_id,
                    **entry
                    }
            testset.append(entry)
            query_id += 1
            # testset.append(generate_multi_query(intra_document_chunks, query_synthesizer_type="intra-doc"))
        except Exception as e:
            logging.warning(f"Multi-Hop-Intra-Document Fehler: {e}")
    
    chroma_db_name = chroma_db.split("/")[-1]
            
    output_json_path = f"eval_datasets/{optimization}artificial_evaluation_dataset_for_{chroma_db_name}.json"

    print(f"Storing testset with {len(testset)} entries in {output_json_path}...")
        
    with open(output_json_path, "w", encoding="utf-8") as outfile:
        json.dump(testset, outfile, indent=4, ensure_ascii=False)
    
    return output_json_path