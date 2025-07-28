#!/usr/bin/env python3
"""
Test Challenge 1A with your own PDF files
"""
import sys
import os
import json
sys.path.append('.')
from process_pdfs import PDFOutlineExtractor

def test_your_pdf(pdf_path):
    """Test with your own PDF file."""
    from pathlib import Path
    # Always resolve pdf_path relative to repo root if not absolute
    pdf_path = Path(pdf_path)
    if not pdf_path.is_absolute():
        # Try resolving relative to current working directory (repo root)
        candidate = (Path.cwd() / pdf_path).resolve()
        if candidate.exists():
            pdf_path = candidate
        else:
            # fallback: resolve relative to script
            pdf_path = (Path(__file__).parent / pdf_path).resolve()
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return
    print(f"🔍 Testing Challenge 1A with: {pdf_path}")
    # Initialize extractor
    schema_path = (Path(__file__).parent / "sample_dataset" / "schema" / "output_schema.json").resolve()
    extractor = PDFOutlineExtractor(str(schema_path))
    # Extract outline
    result = extractor.extract_outline(str(pdf_path))
    
    # Show results
    print(f"\n📄 Title: {result['title']}")
    print(f"📊 Total headings found: {len(result['outline'])}")
    
    print(f"\n🏗️ Document Structure:")
    for heading in result['outline']:
        level_indent = "  " * (int(heading['level'][1]) - 1)
        print(f"{level_indent}{heading['level']}: \"{heading['text']}\" (Page {heading['page']})")
    
    # Save output
    output_name = os.path.basename(pdf_path).replace('.pdf', '_output.json')
    with open(output_name, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Output saved to: {output_name}")

if __name__ == "__main__":
    # Example usage:
    # python test_your_pdf.py path/to/your/file.pdf
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        test_your_pdf(pdf_path)
    else:
        print("Usage: python test_your_pdf.py <path_to_pdf>")
        print("Example: python test_your_pdf.py sample_dataset/pdfs/file02.pdf")
