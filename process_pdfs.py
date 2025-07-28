# process_pdfs.py (The Simple, Reliable, Content-Aware Parser)

import os
import sys
import json
import fitz  # PyMuPDF
from pathlib import Path

def extract_content_tree(pdf_path: str) -> dict:
    """
    Extracts a list of content blocks, each with a detected heading and
    its associated paragraph content.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"ERROR opening {pdf_path}: {e}")
        return {} # Return empty dict on failure

    # --- Heuristics to find the average font size for body text ---
    text_spans = [span for page in doc for block in page.get_text("dict", flags=0)['blocks'] if block['type'] == 0 for line in block['lines'] for span in line['spans']]
    avg_font_size = sum(s['size'] for s in text_spans) / len(text_spans) if text_spans else 10.0
    HEADING_THRESHOLD = avg_font_size * 1.15 # A heading is 15% bigger

    content_tree = []
    current_heading = "Introduction"
    content_buffer = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] != 0 or 'lines' not in block: continue
            
            block_text = " ".join(span['text'] for line in block['lines'] for span in line['spans']).strip()
            if not block_text: continue
            
            # Check if the block is likely a heading
            first_span = block['lines'][0]['spans'][0]
            is_heading = (first_span['size'] > HEADING_THRESHOLD or "bold" in first_span['font'].lower()) and len(block_text) < 150

            if is_heading:
                # Save the previously buffered content under the last heading
                if content_buffer:
                    content_tree.append({"heading": current_heading, "content": "\n".join(content_buffer), "page": page_num + 1})
                content_buffer = [] # Reset the buffer
                current_heading = block_text # This block is the new heading
            else:
                # This block is content, add it to the buffer
                content_buffer.append(block_text)
    
    # After the last page, save any remaining content in the buffer
    if content_buffer:
        content_tree.append({"heading": current_heading, "content": "\n".join(content_buffer), "page": page_num + 1})

    return {
        "file_name": os.path.basename(pdf_path),
        "title": doc.metadata.get('title', os.path.basename(pdf_path)),
        "content_tree": content_tree
    }

def main(input_dir, output_dir):
    """Processes all PDFs in input_dir and saves results to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to process in '{input_dir}'.")
    
    for pdf_file in pdf_files:
        print(f"  -> Parsing: {pdf_file.name}")
        result = extract_content_tree(str(pdf_file))
        if not result: continue
        
        output_file = Path(output_dir) / f"{pdf_file.stem}.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_pdfs.py <input_pdf_directory> <output_json_directory>")
        sys.exit(1)
    # This now correctly uses the arguments from the command line
    main(sys.argv[1], sys.argv[2])