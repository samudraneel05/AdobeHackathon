# main.py (Version 3.0 - The Final, Dynamic Orchestrator)

import json
import os
import sys
import glob
import subprocess
from datetime import datetime
from transformers import T5ForConditionalGeneration, T5Tokenizer

# --- Import your refactored controller functions ---
# Make sure these files (intent_analyzer.py, retriever.py) are in the same directory as main.py
from intent_analyzer import create_intent_model_from_text
#from retriever import run_retrieval
from retriever import run_retrieval


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def run_pdf_parsing(pdf_folder, output_folder):
    """
    Calls the external 'process_pdfs.py' script to perform semantic parsing.
    """
    print(f"--- STAGE 0: Parsing PDFs from '{pdf_folder}' ---")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print("  -> Cleaning old parsed files...")
    for old_file in glob.glob(os.path.join(output_folder, '*.json')):
        os.remove(old_file)

    script_to_run = 'process_pdfs.py'
    command = [sys.executable, script_to_run, pdf_folder, output_folder]
    
    print(f"DEBUG: Running command -> {' '.join(command)}")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"!!!!!! ERROR: '{script_to_run}' FAILED. !!!!!!!")
        print(f"\n--- Stderr: ---\n{e.stderr}")
        raise
        
    print("✅ PDF parsing complete.")
    return [os.path.basename(p) for p in glob.glob(os.path.join(pdf_folder, '*.pdf'))]

def setup_generative_model():
    """Initializes and returns the Flan-T5 model and tokenizer."""
    print("--- Setting up Generative Model (Flan-T5) ---")
    model_name = "google/flan-t5-base"
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        print("✅ Generative model loaded.")
        return tokenizer, model
    except Exception as e:
        print(f"ERROR: Could not load generative model. Check internet for first run. Error: {e}")
        return None, None

def generate_refined_text(tokenizer, model, user_job, chunk_text):
    """Generates a concise summary for a given chunk."""
    prompt = f"""Based on the user's task, provide a one-sentence summary of the most relevant information in the document section below.
    USER TASK: "{user_job}"
    DOCUMENT SECTION: "{chunk_text}"
    CONCISE SUMMARY:"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return "Could not generate analysis for this section."

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    
    # --- DYNAMIC SETUP from Command-Line Argument ---
    if len(sys.argv) < 2:
        print("ERROR: Please provide the path to a collection directory.")
        print("Usage: python main.py \"path/to/Collection 1\"")
        sys.exit(1)
        
    collection_dir = sys.argv[1]
    if not os.path.isdir(collection_dir):
        print(f"ERROR: Directory not found at '{collection_dir}'")
        sys.exit(1)

    # Define all paths dynamically based on the input directory
    PDF_DIR = os.path.join(collection_dir, "PDFs")
    PARSED_DIR = os.path.join(collection_dir, "parsed_output") # A dedicated folder for parsed JSONs
    INPUT_FILE = os.path.join(collection_dir, "challenge1b_input.json")
    FINAL_OUTPUT_FILE = os.path.join(collection_dir, "challenge1b_generated_output.json")
    
    # --- STAGE 0: Parse all PDFs for the given collection ---
    processed_pdf_names = run_pdf_parsing(PDF_DIR, PARSED_DIR)

    # --- STAGE 1: Load Challenge Input ---
    print(f"\n--- STAGE 1: Loading challenge input from '{INPUT_FILE}' ---")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        challenge_data = json.load(f)
    persona = challenge_data["persona"]["role"]
    job_to_be_done = challenge_data["job_to_be_done"]["task"]
    print("✅ Challenge input loaded.")

    # --- STAGE 2: Run Intent Analysis ---
    user_intent_model = create_intent_model_from_text(persona, job_to_be_done)

    # --- STAGE 3: Run Retrieval Pipeline ---
    # We must pass the path to the parsed files to the retriever!
    top_ranked_chunks = run_retrieval(user_intent_model, PARSED_DIR)
    if not top_ranked_chunks:
        print("--- No relevant chunks found. Cannot generate final output. ---")
        sys.exit(1)

    # --- STAGE 4: Run Generation and Final Assembly ---
    gen_tokenizer, gen_model = setup_generative_model()
    if not gen_model: sys.exit(1)
        
    final_output_data = {
        "metadata": { # Lowercase to match ideal output
            "input_documents": processed_pdf_names,
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [], # Lowercase to match ideal output
        "subsection_analysis": []  # Lowercase to match ideal output
    }

    print("\n--- STAGE 4: Starting Final Analysis and Generation ---")
    for rank, result in enumerate(top_ranked_chunks, 1): # Start rank from 1
        rank_score = result['score']
        print(f"  -> Generating analysis for Rank #{rank} (Score: {rank_score:.4f})")
        refined_text = generate_refined_text(gen_tokenizer, gen_model, job_to_be_done, result['chunk_text'])
        
        final_output_data["extracted_sections"].append({
            "document": result['metadata']['doc_name'],
            "page_number": result['metadata']['page_number'],
            "section_title": result['metadata']['section_path'],
            "importance_rank": rank # Use integer rank
        })
        
        final_output_data["subsection_analysis"].append({
            "document": result['metadata']['doc_name'],
            "page_number": result['metadata']['page_number'], # Corrected key
            "refined_text": refined_text
        })

    # --- STAGE 5: Save the Final Output ---
    print(f"\n--- STAGE 5: Saving final output to '{FINAL_OUTPUT_FILE}' ---")
    with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_output_data, f, indent=4)
        
    print(f"✨✨✨ Pipeline Complete for '{collection_dir}'. Results are in '{FINAL_OUTPUT_FILE}' ✨✨✨")