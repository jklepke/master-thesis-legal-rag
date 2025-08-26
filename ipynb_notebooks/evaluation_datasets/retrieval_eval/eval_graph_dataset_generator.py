# === Standard Library Imports ===
import os
import json
import logging
import random
from typing import List, Dict
from uuid import uuid4
from collections import defaultdict

# === Third-Party Library Imports ===
import openai
from dotenv import load_dotenv
from tqdm import tqdm

# === LangChain Imports ===

# Experimental and Other Integrations

# === RAGAS (Custom) Imports ===
from ragas.testset.persona import Persona

# === OpenAI Imports ===
from openai import OpenAI

# === Neo4j Imports ===
from neo4j import GraphDatabase

# .env-Datei laden
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

# == SETUP - OpenAI API ==
client = OpenAI(api_key=openai.api_key)

# == SETUP - Neo4j Connection ==
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "../.."))

# Construct the path to .env.neo4j in the base directory
env_path = os.path.join(BASE_DIR, ".env.neo4j")

# Load environment variables from .env and .env.neo4j files
load_dotenv()
load_dotenv(env_path, override=True)

NEO4J_AUTH = os.getenv("NEO4J_AUTH")
NEO4J_URI = os.getenv("NEO4J_URI")

# Split NEO4J_AUTH into user name and password
NEO4J_USERNAME, NEO4J_PASSWORD = NEO4J_AUTH.split("/")

persona = Persona(
    name="energy_law_practitioner",
    role_description=(
        "Ein juristischer Fachanwender mit Spezialisierung im deutschen Energierecht. "
        "Hat tiefes Verständnis für gesetzliche Regelungen wie das EnWG, StromNEV oder WPG und sucht gezielt nach "
        "Antworten auf konkrete rechtliche Fragestellungen im Kontext von Energieversorgung, Netzregulierung und Planungspflichten."
    )
)



def load_graph_relations_from_neo4j(uri: str, user: str, password: str) -> list[dict]:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    relations = []

    query = """
    MATCH (c:Chunk)-[:HAS_ENTITY]->(h:Entity)-[r]->(t:Entity)
    RETURN {
        chunk_id: c.chunk_id,
        chunk_index: c.chunk_index,
        title: c.title,
        relation_type: type(r),
        head: h.id,
        tail: t.id,
        context: h.id + ' - ' + type(r) + ' -> ' + t.id
    } AS relation
    """

    with driver.session() as session:
        result = session.run(query)
        print("Running query...")
        for record in tqdm(result, desc="Extracting Relations from Neo4J..."):
            data = record.data()
            if "relation" in data:
                relations.append(data["relation"])
            else:
                print("No 'relation'-Key in Record:", data)
            
    driver.close()
    
    return relations


def group_relations_by_chunk(relations: list[dict]) -> list[dict]:
    
    grouped = defaultdict(lambda: {
        "chunk_id": None,
        "chunk_index": None,
        "contexts": [],
        "relation_count": 0
    })

    for rel in relations:
        chunk_id = rel["chunk_id"]
        index = rel["chunk_index"]
        grouped[chunk_id]["chunk_id"] = chunk_id
        grouped[chunk_id]["chunk_index"] = index
        grouped[chunk_id]["contexts"].append(rel["context"])
        grouped[chunk_id]["relation_count"] += 1

    return list(grouped.values())

from collections import defaultdict
import random

def find_multi_hop_candidates_by_entity_bridge(uri: str, user: str, password: str, limit: int = 20) -> list[dict]:
    """
    Identify multi-hop candidates by selecting Entity nodes connected to multiple Chunk nodes.
    Limit chunk selection per entity to a maximum of 5.
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    candidates = []

    query = """
    MATCH (e:Entity)<-[:HAS_ENTITY]-(c:Chunk)
    WITH e, collect(DISTINCT c.chunk_id)[..3] AS chunk_ids
    WHERE size(chunk_ids) > 1
    RETURN e.id AS entity, chunk_ids
    LIMIT 20
    """

    with driver.session() as session:
        results = list(session.run(query, {"limit": limit}))
        for record in results:
            chunk_ids = record["chunk_ids"]

            subgraph_query = """
            MATCH (c:Chunk)-[:HAS_ENTITY]->(h:Entity)-[r]->(t:Entity)
            WHERE c.chunk_id IN $chunk_ids
            RETURN DISTINCT
                c.chunk_id AS chunk_id,
                c.chunk_index AS chunk_index,
                c.title AS law_title,
                h.id AS head,
                type(r) AS relation_type,
                t.id AS tail,
                h.id + ' - ' + type(r) + ' -> ' + t.id AS context
            """
            subgraph_result = session.run(subgraph_query, {"chunk_ids": chunk_ids})

            grouped = defaultdict(list)
            chunk_index_map = {}
            law_titles = set()

            for row in subgraph_result:
                index = row["chunk_index"]
                grouped[index].append(row["context"])
                chunk_index_map[index] = row["chunk_id"]
                law_titles.add(row["law_title"])

            contexts = [", ".join(v) for v in grouped.values()]
            chunk_indices = sorted(grouped.keys())
            chunk_id_list = [chunk_index_map[i] for i in chunk_indices]
            relation_counts = [len(grouped[i]) for i in chunk_indices]

            candidates.append({
                "chunk_ids": chunk_id_list,
                "chunk_indices": chunk_indices,
                "contexts": contexts,
                "relation_counts": relation_counts,
                "law_titles": list(law_titles)
            })


    driver.close()
    return candidates


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


def generate_single_query_graph_based(selected_relation: Dict) -> Dict:

    # Join the subgraph edges as string context
    contexts = selected_relation["contexts"]
    chunk_id = selected_relation["chunk_id"]
    chunk_index = selected_relation["chunk_index"]
    title = selected_relation.get("title", "Unbekanntes Gesetz")

    # Prompt to LLM
    prompt = f"""
    Du siehst eine Sammlung juristischer Relationen, die aus einem Abschnitt des Gesetzes "{title}" extrahiert wurden.

    Kontext:
    \"\"\"{contexts}\"\"\"

    Erstelle daraus eine möglichst konkrete juristische Frage und eine passende Antwort.
    Gib die Ausgabe exakt in folgendem Format zurück:

    Frage: [deine Frage hier]
    Antwort: [deine Antwort hier]
    """

    # Persona-Systemprompt
    system_prompt = f"""Du bist ein juristisches KI-System, das als folgende Person agiert:

        Name: {persona.name}
        Rolle: {persona.role_description}

        Deine Aufgabe ist es, rechtliche Fragen und Antworten aus einem relationalen Wissensgraphen, der aus Gesetzestexten stammt, zu generieren.
        Achte darauf, dass die Fragen deinem Wissensstand und deiner Perspektive entsprechen.
    """

    # LLM call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    question, answer = extract_question_answer(response)

    return {
        "query": question or "",
        "ground_truth": answer or "",
        "ground_truth_chunk_contexts": [contexts],
        "ground_truth_chunk_ids": [chunk_id],
        "ground_truth_chunk_indices": [chunk_index],
        "synthesizer": "single-hop"
    }

    
# == Generate multi-hop query ==
def generate_multi_query_graph_based(adjacent_relations: dict, query_synthesizer_type: str = "") -> dict:

    contexts = adjacent_relations["contexts"]
    chunk_ids = adjacent_relations["chunk_ids"]
    chunk_indices = adjacent_relations["chunk_indices"]
    titles = adjacent_relations.get("law_titles", "Unbekanntes Gesetz")


    prompt = f"""
    Du siehst mehrere juristische Relationen, die aus unterschiedlichen, aber zusammenhängenden Abschnitten der Gesetze/Verordnungen {titles} extrahiert wurden.

    Kontext:
    \"\"\"{contexts}\"\"\"

    Erstelle auf Basis dieses vernetzten Wissens eine juristische Frage, deren Beantwortung Informationen aus mehreren Stellen voraussetzt.
    Gib die Ausgabe exakt in folgendem Format zurück:

    Frage: [deine Frage hier]
    Antwort: [deine Antwort hier]
    """

    system_prompt = f"""Du bist ein juristisches KI-System, das als folgende Person agiert:

        Name: {persona.name}
        Rolle: {persona.role_description}

        Deine Aufgabe ist es, rechtliche Fragen und Antworten aus einem relationalen Wissensgraphen zu generieren, die mehrere gesetzliche Abschnitte kombinieren.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    question, answer = extract_question_answer(response)

    return {
        "query": question or "",
        "ground_truth": answer or "",
        "ground_truth_chunk_contexts": contexts,
        "ground_truth_chunk_ids": chunk_ids,
        "ground_truth_chunk_indices": chunk_indices,
        "synthesizer": f"multi-hop-{query_synthesizer_type}"
    }

    
def generate_graph_evalset(test_set_size: int, query_distribution: dict = None):
    
    if query_distribution is None:
        query_distribution = {
            "single": 0.7, # single chunks 
            "multi_specific": 0.3, # adjacent chunks
            }
    
    single_ratio = query_distribution.get("single", 0.0) 
    multi_specific_ratio = query_distribution.get("multi_specific", 0.0)
    
    num_single = int(single_ratio * test_set_size)
    num_multi_specific = int(multi_specific_ratio * test_set_size)
    
    remaining = test_set_size - (num_single + num_multi_specific)
    
    if(remaining>0 or remaining<0):
        num_single += remaining
    
    testset = []

    relations = load_graph_relations_from_neo4j(uri=NEO4J_URI, user=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    grouped_relations = group_relations_by_chunk(relations)
    selected_chunks = random.sample(grouped_relations, k=min(num_single, len(grouped_relations)))

    query_id = 1

    print(f"🔹 Generating {num_single} Single-Hop Queries")
    for doc in tqdm(selected_chunks):

        try:
            entry = generate_single_query_graph_based(doc)
            
            entry = {
                "query_id": query_id,
                **entry
            }
            testset.append(entry)
            query_id += 1

        except Exception as e:
            logging.warning(f"Single-Hop Fehler: {e}")
    
    print(f"🔸 Generating {num_multi_specific} Multi-Hop-Specific Queries (i.e. Adjacent Relations)")
    for i in tqdm(range(num_multi_specific)):

        try:
            candidates = find_multi_hop_candidates_by_entity_bridge(uri=NEO4J_URI, user=NEO4J_USERNAME, password=NEO4J_PASSWORD, limit=10)
            candidate = candidates[i]
            entry = generate_multi_query_graph_based(candidate, query_synthesizer_type="adjacent")
            entry = {
                "query_id": query_id,
                **entry
            }
            testset.append(entry)
            query_id += 1
        except Exception as e:
            logging.warning(f"Multi-Hop-Specific Fehler: {e}")

    output_json_path = f"eval_datasets/2_graph_database/artificial_evaluation_dataset_for_graph_rag.json"

    print(f"Storing testset with {len(testset)} entries in {output_json_path}...")
        
    with open(output_json_path, "w", encoding="utf-8") as outfile:
        json.dump(testset, outfile, indent=4, ensure_ascii=False)
        
    return output_json_path