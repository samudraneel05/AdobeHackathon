#!/usr/bin/env python3
"""
Adobe Hackathon Challenge 1A: PDF Outline Extraction
High-performance PDF processing with heading detection and structured output.
"""

import os
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

import fitz  # PyMuPDF
import jsonschema

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFOutlineExtractor:
    """Extract structured outlines from PDF documents with heading detection."""
    
    def __init__(self, schema_path: Optional[str] = None):
        """Initialize the extractor with optional schema validation."""
        self.schema = None
        if schema_path and os.path.exists(schema_path):
            try:
                with open(schema_path, 'r') as f:
                    self.schema = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load schema: {e}")
    
    def extract_title(self, doc: fitz.Document, pdf_path: Optional[str] = None) -> str:
        """Extract document title from metadata, page content, or filename."""
        import os
        def is_generic_title(title, pdf_path=None):
            if not title:
                return True
            t = title.strip().lower()
            if t in {"untitled", "document", "untitled document", "pdf document"}:
                return True
            if pdf_path:
                fname = os.path.splitext(os.path.basename(pdf_path))[0].lower()
                if fname in t or t in fname:
                    return True
            if t.startswith("microsoft word -"):
                return True
            return False

        metadata = doc.metadata
        meta_title = metadata.get('title', '').strip() if metadata else ''
        if meta_title and not is_generic_title(meta_title, pdf_path):
            # Remove file extension if present
            meta_title_clean = re.sub(r'\.[a-zA-Z0-9]{1,5}$', '', meta_title).strip()
            return meta_title_clean

        # Try largest, highest text block on first page
        if len(doc) > 0:
            page = doc[0]
            blocks = page.get_text("dict").get("blocks", [])
            # Filter text blocks only
            text_blocks = [b for b in blocks if b.get("type", 1) == 0 and b.get("lines")]
            if text_blocks:
                # Sort by (-font size, y0)
                def block_score(b):
                    max_size = max((span["size"] for line in b["lines"] for span in line["spans"]), default=0)
                    y0 = b["bbox"][1]
                    return (-max_size, y0)
                text_blocks.sort(key=block_score)
                # Get the best candidate
                for b in text_blocks:
                    candidate = " ".join(span["text"] for line in b["lines"] for span in line["spans"]).strip()
                    if 10 < len(candidate) < 200 and not re.match(r'^\d+$|^page \d+|^\d+/\d+', candidate.lower()):
                        return candidate

            # Fallback: original line-based heuristic
            text = page.get_text()
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            for line in lines[:10]:
                if 10 < len(line) < 200:
                    if not re.match(r'^\d+$|^page \d+|^\d+/\d+', line.lower()):
                        return line

        # Fallback: use filename if provided
        if pdf_path:
            return os.path.splitext(os.path.basename(pdf_path))[0]
        return "Untitled Document"
    
    def detect_heading_level(self, text: str, font_size: float, font_flags: int, 
                            avg_font_size: float, page_position: Optional[float] = None) -> Optional[str]:
        """Detect heading level based on text properties, content, and position."""
        # Clean text for analysis
        clean_text = text.strip()
        if not clean_text or len(clean_text) < 2:  # Allow shorter headings (like "1")
            return None
            
        # Skip very long texts (likely paragraphs)
        if len(clean_text) > 200:
            return None
        
        # Skip decorative lines with repetitive characters
        if re.match(r'^[\-=_*#]{4,}$', clean_text) or re.match(r'^[\|\+]{4,}$', clean_text):
            return None
            
        # Skip form fields that are not headings
        if re.match(r'^\d+\.\s*$', clean_text):  # Just a number with a period like "5."
            return None
        
        # Skip standalone dates or simple form labels
        if re.match(r'^Date\s*$', clean_text) or re.match(r'^Signature\s*$', clean_text):
            return None
            
        # Font size based detection
        font_ratio = font_size / avg_font_size if avg_font_size > 0 else 1.0
        is_bold = bool(font_flags & 2**4)  # Bold flag
        
        # Check for document title (usually at the top of first page)
        is_likely_title = False
        if page_position is not None and page_position < 0.15 and len(clean_text) > 10 and len(clean_text) < 100:
            is_likely_title = True
        
        # Form document patterns
        form_title_pattern = r'^(Application|Form|Request)\s+\w+'
        if re.match(form_title_pattern, clean_text) and (font_ratio >= 1.2 or is_bold):
            return "H1"
        
        # Heading patterns
        h1_patterns = [
            r'^Chapter\s+\d+[:\.]?$',  # "Chapter 1:" or "Chapter 1"
            r'^[A-Z][A-Z\s]{5,}$',  # ALL CAPS longer titles
            r'^[IVX]{1,5}\.\s+[A-Z]',  # Roman numerals with text
            r'^PART\s+[IVX]+',  # "PART IV"
            r'^[A-Z][A-Z\s]{2,}:\s',  # ALL CAPS followed by colon and space
            r'^\d+\.\s+[A-Z][a-zA-Z\s]{5,}',  # "1. Major Heading" - conditionally H1 (see below)
        ]
        
        h2_patterns = [
            r'^\d+\.\d+\.?\s+[A-Z]',  # "1.1 Subheading"
            r'^[A-Z][a-z]?\. [A-Z]',  # "A. Subheading"
            r'^\([a-z]\)\s+[A-Z][a-zA-Z\s]{3,}',  # "(a) Subheading" with at least a few words
            r'^\d+\.\s+[A-Z][a-zA-Z\s]{5,}',  # "1. Major Heading"
        ]
        
        h3_patterns = [
            r'^\d+\.\d+\.\d+\.?\s',  # "1.1.1 Sub-subheading"
            r'^[a-z]\. [A-Z]',  # "a. Sub-subheading"
            r'^\d+\)\s+[A-Z]',  # "1) Sub-subheading"
            r'^S\.No',  # "S.No" (Serial Number in a table)
            r'^Name|Age|Relationship', # Common form field labels
        ]
        
        has_h1_pattern = any(re.match(pattern, clean_text) for pattern in h1_patterns)
        has_h2_pattern = any(re.match(pattern, clean_text) for pattern in h2_patterns)
        has_h3_pattern = any(re.match(pattern, clean_text) for pattern in h3_patterns)
        
        # Document title detection - give preference to H1 for the document title
        if is_likely_title and (font_ratio >= 1.3 or is_bold):
            return "H1"
            
        # Determine heading level based on patterns, font characteristics, and position
        
        # H1 Detection
        if (font_ratio >= 1.6 or 
            (font_ratio >= 1.4 and is_bold) or 
            has_h1_pattern or 
            clean_text.startswith("CHAPTER ")):
            return "H1"
            
        # H2 Detection
        elif (font_ratio >= 1.25 or 
             (font_ratio >= 1.1 and is_bold) or 
             has_h2_pattern):
            return "H2"
            
        # H3 Detection - more restrictive to avoid catching regular text
        elif ((font_ratio >= 1.1 and is_bold) or 
             has_h3_pattern):
            return "H3"
                
        return None
    
    def calculate_average_font_size(self, doc: fitz.Document) -> float:
        """Calculate average font size across the document for baseline comparison."""
        font_sizes = []
        sizes_by_frequency = {}
        
        # Sample first few pages to get representative font sizes
        max_pages = min(5, len(doc))
        for page_num in range(max_pages):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["size"] > 0:
                                size = round(span["size"], 1)  # Round to avoid minor differences
                                font_sizes.append(size)
                                # Track frequency of each font size
                                if size in sizes_by_frequency:
                                    sizes_by_frequency[size] += len(span.get("text", ""))
                                else:
                                    sizes_by_frequency[size] = len(span.get("text", ""))
        
        # If we have enough samples, use the most common font size as base
        if sizes_by_frequency:
            # Get the font size that appears most frequently by text length
            most_common_size = max(sizes_by_frequency.items(), key=lambda x: x[1])[0]
            # Make sure we have a reasonable number of samples
            if sizes_by_frequency[most_common_size] > 100:
                return most_common_size
        
        # Fallback: use mean font size
        return sum(font_sizes) / len(font_sizes) if font_sizes else 12.0
    
    def merge_spans_by_line(self, block: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Merge spans that belong to the same line to handle fragmented text."""
        merged_lines = []
        
        if "lines" not in block:
            return merged_lines
            
        for line in block["lines"]:
            if not line.get("spans"):
                continue
                
            spans = line["spans"]
            if not spans:
                continue
                
            # Calculate weighted average of font sizes and flags
            total_length = sum(len(span.get("text", "")) for span in spans)
            if total_length == 0:
                continue
                
            # Use the most common font size and flags
            font_sizes = [span["size"] for span in spans]
            font_flags = [span["flags"] for span in spans]
            
            # Use the largest font as representative
            max_size_idx = font_sizes.index(max(font_sizes)) if font_sizes else 0
            
            # Combine text
            merged_text = "".join(span.get("text", "") for span in spans).strip()
            if not merged_text:
                continue
                
            # Create merged span
            merged_line = {
                "text": merged_text,
                "size": font_sizes[max_size_idx] if font_sizes else 0,
                "flags": font_flags[max_size_idx] if font_flags else 0,
                "bbox": line.get("bbox", [0, 0, 0, 0]),
                "origin_spans": len(spans)
            }
            
            merged_lines.append(merged_line)
            
        return merged_lines

    def extract_outline(self, pdf_path: str) -> Dict[str, Any]:
        """Extract title and hierarchical outline from PDF."""
        start_time = time.time()
        
        try:
            doc = fitz.open(pdf_path)
            
            # Extract title
            title = self.extract_title(doc, pdf_path)
            
            # Calculate baseline font size
            avg_font_size = self.calculate_average_font_size(doc)
            
            outline = []
            seen_headings = set()  # Avoid duplicates
            
            # First pass: collect all headings and their font sizes per page
            page_heading_data = {}
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_height = page.rect.height
                blocks = page.get_text("dict")["blocks"]
                
                page_texts = []
                page_heading_sizes = {"H1": [], "H2": [], "H3": []}
                
                for block in blocks:
                    # First merge spans by line to handle fragmented text
                    merged_lines = self.merge_spans_by_line(block)
                    
                    for merged_line in merged_lines:
                        text = merged_line["text"]
                        font_size = merged_line["size"]
                        font_flags = merged_line["flags"]
                        
                        # Calculate relative vertical position on page
                        bbox = merged_line.get("bbox", [0, 0, 0, 0])
                        y_position = bbox[1] if bbox else 0
                        relative_position = y_position / page_height if page_height else 0
                        
                        # Initial heading level detection
                        level = self.detect_heading_level(
                            text, font_size, font_flags, avg_font_size, relative_position
                        )
                        
                        if level and text:
                            # Store candidate heading with its characteristics for post-processing
                            heading_item = {
                                "text": text,
                                "level": level,
                                "font_size": font_size,
                                "position": relative_position,
                                "page": page_num + 1,
                                "is_numbered": bool(re.match(r'^\d+\.\s+[A-Z]', text))
                            }
                            page_texts.append(heading_item)
                            
                            # Collect font sizes by heading level
                            page_heading_sizes[level].append(font_size)
                
                # Store the heading data for this page
                page_heading_data[page_num] = {
                    "texts": page_texts,
                    "sizes": page_heading_sizes
                }
            
            # Second pass: adjust heading levels based on relative sizes
            for page_num, data in page_heading_data.items():
                page_texts = data["texts"]
                sizes = data["sizes"]
                
                # Calculate average sizes for each heading level on this page
                avg_h1_size = sum(sizes["H1"]) / len(sizes["H1"]) if sizes["H1"] else 0
                avg_h2_size = sum(sizes["H2"]) / len(sizes["H2"]) if sizes["H2"] else 0
                
                # Adjust numbered headings (possible "1. Major Heading" pattern)
                for item in page_texts:
                    if item["is_numbered"] and item["level"] == "H1":
                        # If H2s exist and this heading is smaller than average H2, demote to H2
                        if avg_h2_size > 0 and item["font_size"] < avg_h2_size * 0.9:
                            item["level"] = "H2"
                
                # Post-process headings on this page to avoid fragmented headings
                self.process_page_headings(page_texts, outline, seen_headings)
            
            doc.close()
            
            # Clean up the outline by removing any potential duplicates and sorting by page
            cleaned_outline = self.clean_outline(outline)
            
            result = {
                "title": title,
                "outline": cleaned_outline
            }
            
            # Validate against schema if available
            if self.schema:
                try:
                    jsonschema.validate(result, self.schema)
                    logger.debug(f"Schema validation passed for {pdf_path}")
                except jsonschema.ValidationError as e:
                    logger.error(f"Schema validation failed for {pdf_path}: {e}")
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {pdf_path} in {processing_time:.2f}s, found {len(cleaned_outline)} headings")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {
                "title": f"Error processing {Path(pdf_path).name}",
                "outline": []
            }
            
    def process_page_headings(self, page_texts: List[Dict[str, Any]], outline: List[Dict[str, Any]], seen_headings: set[str]):
        """Process page headings to avoid fragmentation and improve accuracy."""
        if not page_texts:
            return
            
        # Sort texts by vertical position
        page_texts.sort(key=lambda x: x.get("position", 0))
        
        # Group texts that are likely part of the same heading
        i = 0
        while i < len(page_texts):
            current = page_texts[i]
            combined_text = current["text"]
            combined_level = current["level"]
            page = current["page"]
            
            # Look ahead for fragments that should be combined
            j = i + 1
            while j < len(page_texts):
                next_item = page_texts[j]
                
                # Check if this is likely a continuation of the current heading
                # Criteria: same level, close position, sensible text combination
                position_diff = next_item.get("position", 0) - current.get("position", 0)
                
                # Don't combine numbered items in forms (like "1." and "2.")
                if (re.match(r'^\d+\.\s*$', current["text"]) and re.match(r'^\d+\.\s*$', next_item["text"])):
                    break
                    
                # Don't combine items that look like table headers
                if any(x in current["text"].lower() for x in ["name", "age", "relationship", "s.no"]):
                    break
                
                if (next_item["level"] == current["level"] and 
                    position_diff < 0.015 and  # Very close vertically - more strict
                    len(next_item["text"]) < 40):  # Not too long - more strict
                    
                    # Check if combining makes linguistic sense
                    combined_candidate = combined_text + " " + next_item["text"]
                    
                                # If adding the next piece creates a sensible heading, combine them
                    if len(combined_candidate) < 150:  # More reasonable heading length
                        # Check if combined text looks sensible (not mixing different contexts)
                        if "form" in combined_text.lower() and "chapter" in next_item["text"].lower():
                            break  # Don't mix form and chapter headings
                            
                        combined_text = combined_candidate
                        j += 1
                    else:
                        break
                else:
                    break
            
            # Skip items that are likely form field numbers without content
            if re.match(r'^\d+\.\s*$', combined_text):
                i = j
                continue
                
            # Create unique identifier to avoid duplicates
            heading_id = f"{combined_level}:{combined_text}:{page}"
            if heading_id not in seen_headings and combined_text.strip():
                outline.append({
                    "level": combined_level,
                    "text": combined_text.strip(),
                    "page": page
                })
                seen_headings.add(heading_id)
            
            # Skip over the items we've combined
            i = j
    
    def clean_outline(self, outline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean up the outline by removing duplicates and sorting by page."""
        # Sort by page number first
        sorted_outline = sorted(outline, key=lambda x: x.get("page", 0))
        
        # Remove duplicates with similar text
        cleaned = []
        seen_similar = set()
        
        for item in sorted_outline:
            # Simplify text for comparison (remove extra spaces, lowercase)
            simple_text = re.sub(r'\s+', ' ', item["text"].lower()).strip()
            
            # Skip very short items that might be fragments
            if len(simple_text) < 3:
                continue
                
            # Skip standalone form field numbers ("5.", "6.", etc.)
            if re.match(r'^\d+\.\s*$', item["text"]):
                continue
                
            # Skip items that look like they belong to a table structure
            if simple_text.lower() in ["s.no", "sl.no", "name", "age", "relationship"]:
                continue
                
            # Create a fingerprint of this heading
            text_fingerprint = f"{item['level']}:{simple_text[:30]}"
            
            # Only add if we haven't seen something very similar
            if text_fingerprint not in seen_similar:
                cleaned.append(item)
                seen_similar.add(text_fingerprint)
                
        # If this looks like a form, ensure the document title is H1
        if cleaned and any("application" in item["text"].lower() or "form" in item["text"].lower() for item in cleaned[:3]):
            # Find the title and make it H1
            title_found = False
            for item in cleaned[:3]:  # Check first few items
                if len(item["text"]) > 10 and "application" in item["text"].lower() or "form" in item["text"].lower():  
                    item["level"] = "H1"
                    title_found = True
                    break
                    
            # If no specific form title found, use the first substantial item
            if not title_found and cleaned and len(cleaned[0]["text"]) > 10:
                cleaned[0]["level"] = "H1"
        
        return cleaned


def process_pdfs():
    """Main processing function for Challenge 1A."""
    logger.info("Starting PDF processing for Challenge 1A")
    
    # Get input and output directories
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load schema for validation
    schema_path = "/app/schema/output_schema.json"
    if not os.path.exists(schema_path):
        schema_path = None
        logger.warning("Schema file not found, skipping validation")
    
    # Initialize extractor
    extractor = PDFOutlineExtractor(schema_path)
    
    # Get all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    total_start_time = time.time()
    
    for pdf_file in pdf_files:
        try:
            # Extract outline
            result = extractor.extract_outline(str(pdf_file))
            
            # Create output JSON file
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processed {pdf_file.name} -> {output_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}")
    
    total_time = time.time() - total_start_time
    logger.info(f"Completed processing {len(pdf_files)} files in {total_time:.2f}s")


if __name__ == "__main__":
    process_pdfs()