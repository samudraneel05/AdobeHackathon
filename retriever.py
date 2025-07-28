# retriever.py (Final version, works with the simple content-aware parser)

import os
import json
from pathlib import Path
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util, CrossEncoder

def create_chunks_from_semantic_json(parsed_json_data):
    """Processes the JSON from our custom content-aware parser."""
    document_chunks, chunk_metadata = [], []
    doc_title = parsed_json_data.get('file_name', 'Unknown Document')
    
    for block in parsed_json_data.get('content_tree', []):
        heading = block.get('heading', 'General Content')
        content = block.get('content', '')
        page_num = block.get('page', 0)
        
        if content:
            chunk_text = f"Document: {doc_title}\nSection: {heading}\n\n{content}"
            document_chunks.append(chunk_text)
            chunk_metadata.append({"doc_name": doc_title, "page_number": page_num, "section_path": heading})
    return document_chunks, chunk_metadata

def load_and_chunk_all_documents(parsed_json_directory):
    json_files = list(Path(parsed_json_directory).glob("*.json"))
    all_chunks, all_metadata = [], []
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        new_chunks, new_metadata = create_chunks_from_semantic_json(data)
        all_chunks.extend(new_chunks); all_metadata.extend(new_metadata)
    return all_chunks, all_metadata

def run_retrieval(user_intent_model: dict, parsed_json_directory: str):
    """The main retrieval pipeline (non-hardcoded version)."""
    print("\n--- Running Full Retrieval Pipeline ---")
    
    job_text_query = user_intent_model["job_to_be_done_analysis"]["text"]
    document_chunks, chunk_metadata = load_and_chunk_all_documents(parsed_json_directory)
    if not document_chunks: return []
    print(f"✅ Created {len(document_chunks)} chunks from source documents.")

    print("⏳ Starting Hybrid Search + Reranking...")
    tokenized_corpus = [doc.split(" ") for doc in document_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(job_text_query.split(" "))

    semantic_model = SentenceTransformer('./models/mpnet_qa')
    query_embedding = semantic_model.encode("Question: " + job_text_query, convert_to_tensor=True)
    corpus_embeddings = semantic_model.encode(document_chunks, convert_to_tensor=True, show_progress_bar=True)
    search_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=25, score_function=util.dot_score)[0]
    
    bm25_indices = set(sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:25])
    vector_indices = {hit['corpus_id'] for hit in search_hits}
    fused_indices = list(bm25_indices.union(vector_indices))
    
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    rerank_pairs = [[job_text_query, document_chunks[i]] for i in fused_indices]
    rerank_scores = cross_encoder.predict(rerank_pairs, show_progress_bar=True)
    
    results = sorted([{"score": score, "index": index} for score, index in zip(rerank_scores, fused_indices)], key=lambda x: x['score'], reverse=True)
    
    print("✅ Retrieval and ranking complete.")
    final_ranked_results = []
    for result in results[:5]:
        final_ranked_results.append({
            "score": result['score'],
            "chunk_text": document_chunks[result['index']],
            "metadata": chunk_metadata[result['index']]
        })
    return final_ranked_results