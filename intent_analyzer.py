import json
from gliner import GLiNER
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

def analyze_user_intent(persona_text: str, job_to_be_done_text: str) -> dict:
    """
    Analyzes user persona and job-to-be-done to create a structured intent model.

    This function performs three main tasks:
    1. Named Entity Recognition (NER) to extract key entities.
    2. Key Phrase Extraction to identify the most important concepts.
    3. Semantic Analysis to create a vector embedding of the user's goal.

    Args:
        persona_text: The string describing the user's persona.
        job_to_be_done_text: The string describing the task to be accomplished.

    Returns:
        A dictionary representing the structured user intent model.
    """
    
    # --- 1. MODEL INITIALIZATION ---
    # In a real application, you might load these models once and reuse them.
    # For this script, we initialize them inside the function for self-containment.
    print("Initializing models...")
    ner_model = GLiNER.from_pretrained('./models/gliner_small')
    keyphrase_model = KeyBERT(model='./models/mini_lm')
    semantic_model = SentenceTransformer('./models/mpnet_qa')
    print("Models initialized.")

    # --- 2. NAMED ENTITY RECOGNITION (NER) ---
    print("Performing NER...")
    persona_labels = ["user role", "field of expertise", "specific focus"]
    job_labels = ["financial metric", "business strategy", "document type", "core task"]
    
    persona_entities = ner_model.predict_entities(persona_text, persona_labels, threshold=0.5)
    job_entities = ner_model.predict_entities(job_to_be_done_text, job_labels, threshold=0.5)

    # --- 3. KEY PHRASE EXTRACTION ---
    print("Extracting key phrases...")
    # We focus on the job-to-be-done for the most relevant action phrases.
    key_phrases = keyphrase_model.extract_keywords(job_to_be_done_text, 
                                                   keyphrase_ngram_range=(1, 3), 
                                                   stop_words='english',
                                                   top_n=5)
    
    # Convert to a more JSON-friendly format
    formatted_key_phrases = [{"phrase": phrase, "score": round(score, 4)} for phrase, score in key_phrases]


    # --- 4. SEMANTIC ANALYSIS (EMBEDDING) ---
    print("Performing semantic analysis...")
    # We create an embedding for the core task, as it's the semantic anchor.
    # The .tolist() is crucial for making the numpy array JSON serializable.
    #semantic_embedding = semantic_model.encode(job_to_be_done_text).tolist()
    question_prefix = "Question: "
    transformed_job_text = question_prefix + job_to_be_done_text
    print(f"DEBUG: Transformed query for embedding: '{transformed_job_text}'")

    # Use the same semantic model object to encode the NEW transformed text
    semantic_embedding = semantic_model.encode(transformed_job_text).tolist()

    # --- 5. ASSEMBLE THE FINAL JSON OBJECT ---
    print("Assembling final intent model...")
    user_intent_model = {
      "persona_analysis": {
        "text": persona_text,
        "entities": persona_entities
      },
      "job_to_be_done_analysis": {
        "text": job_to_be_done_text,
        "entities": job_entities,
        "key_phrases": formatted_key_phrases,
        "semantic_embedding": semantic_embedding
      }
    }
    
    return user_intent_model

def create_intent_model_from_text(persona, job_to_be_done):
    """
    Orchestrates the creation of the user intent model and saves it to a file.
    
    Args:
        persona (str): The user persona text.
        job_to_be_done (str): The user job-to-be-done text.
        
    Returns:
        The dictionary containing the full intent model.
    """
    final_intent_model = analyze_user_intent(persona, job_to_be_done)
    
    output_filename = 'user_intent_model.json'
    print(f"--- 💾 Saving intent model to '{output_filename}' ---")
    with open(output_filename, 'w') as f:
        json.dump(final_intent_model, f, indent=2)
    print("✅ Intent model file saved successfully.")
    
    return final_intent_model


# --- MAIN EXECUTION BLOCK (for standalone testing) ---
if __name__ == "__main__":
    print("--- Running intent_analyzer.py as a standalone script for testing ---")
    test_persona = "The user is an Investment Analyst."
    test_job = "Analyze revenue trends from the annual reports."
    
    # Run the controller function
    create_intent_model_from_text(test_persona, test_job)

'''# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # --- Define our sample inputs ---
    input_persona = "A project manager for a government library initiative."
input_job_to_be_done = "Find the project timeline and key report delivery dates for the Ontario Digital Library business plan."
    # --- Run the analysis pipeline ---
    final_intent_model = analyze_user_intent(input_persona, input_job_to_be_done)

    # --- Print the final, structured output (This is good for checking) ---
    print("\n--- 🚀 FINAL USER INTENT MODEL (Displaying on screen) ---")
    print(json.dumps(final_intent_model, indent=2))
    
    # ==========================================================
    # >> ADD THIS SECTION TO SAVE THE FILE <<
    # ==========================================================
    output_filename = 'user_intent_model.json'
    print(f"\n--- 💾 Saving intent model to '{output_filename}' ---")
    
    with open(output_filename, 'w') as f:
        # Use json.dump() to write the dictionary to the file handle 'f'
        json.dump(final_intent_model, f, indent=2)
        
    print("✅ File saved successfully.")'''