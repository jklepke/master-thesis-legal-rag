# Code was taken from Sun et al. (2024) under the Apache 2.0 License. 
# Title: Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents
# Authors: Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, DaweiYin, Zhaochun Ren
# Year: 2024
# Paper: https://arxiv.org/pdf/2304.09542, 
# Code: https://github.com/sunnweiwei/RankGPT/tree/main 


from copy import deepcopy
import copy
from tqdm import tqdm
import time
import json
import os
from openai import OpenAI
import openai
from dotenv import load_dotenv
import re

# Load environment variables. Assumes that the project directory contains a .env file with API keys
load_dotenv()

# Set the OpenAI API key from the environment variables
# Make sure to update "OPENAI_API_KEY" to match the variable name in your .env file
openai.api_key = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=openai.api_key)

def get_prefix_prompt(query, num):
    return [{'role': 'system',
             'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]


def get_post_prompt(query, num):
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."


def create_permutation_instruction(item=None, rank_start=0, rank_end=100, model_name='gpt-4o-mini'):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])

    max_length = 300

    messages = get_prefix_prompt(query, num)
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        # For Japanese should cut by character: content = content[:int(max_length)]
        content = ' '.join(content.split()[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_prompt(query, num)})

    return messages


def run_llm(messages, model_name="gpt-4o-mini"):

    response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
        )
        
    response_only = response.choices[0].message.content.strip()
    
    return response_only


def clean_response(response: str) -> list[int]:
    """
    Extrahiert die in eckigen Klammern angegebenen Indizes aus dem LLM-Response.
    Wandelt sie in 0-basierte Integer-Indizes um.
    """
    numbers = re.findall(r"\[(\d+)\]", response)
    return [int(n) - 1 for n in numbers]


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):

    response = clean_response(permutation)
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item


def permutation_pipeline(item=None, rank_start=0, rank_end=100, model_name='gpt-4o-mini'):

    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end,
                                              model_name=model_name)
    
    permutation = run_llm(messages, model_name=model_name)

    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
    return item


def sliding_windows(item=None, rank_start=0, rank_end=100, window_size=2, step=2, model_name='gpt-4o-mini'):
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        item = permutation_pipeline(item, start_pos, end_pos, model_name=model_name)
        end_pos = end_pos - step
        start_pos = start_pos - step
    return item

def rankgpt_rerank(query, documents, model_name="gpt-4o-mini", window_size=2, step=2):
    
    hits = []
    for doc in documents:
        hit = {
            "content": doc.page_content,
            "chunk_index": doc.metadata.get("chunk_index", None)  # Fallback falls nicht gesetzt
        }
        hits.append(hit)
        
    item =  {
            "query": query,
            "hits": hits
            }           

    item = sliding_windows(item=item, rank_start=0, rank_end=len(hits), window_size=window_size, step=step, model_name=model_name)
    
    reordered_docs = []
    for hit in item["hits"]:
        chunk_idx = hit.get("chunk_index")
        matched_doc = next((doc for doc in documents if doc.metadata.get("chunk_index") == chunk_idx), None)
        if matched_doc:
            reordered_docs.append(matched_doc)
    return reordered_docs
