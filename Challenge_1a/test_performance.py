#!/usr/bin/env python3
"""
Performance testing harness for Challenge 1A
Tests processing time and validates output against schema.
"""

import time
import json
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add current directory to path to import process_pdfs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Challenge_1a.process_pdfs import PDFOutlineExtractor

import jsonschema


def create_test_pdf(num_pages: int = 50) -> str:
    """Create a test PDF with specified number of pages for performance testing."""
    try:
        import fitz
        
        # Create a temporary PDF with multiple pages and headings
        doc = fitz.open()
        
        for page_num in range(num_pages):
            page = doc.new_page()
            
            # Add some headings at different levels
            if page_num == 0:
                # Title page
                page.insert_text((100, 100), "Test Document Title", fontsize=20)
            
            if page_num % 10 == 0:
                # H1 every 10 pages
                page.insert_text((50, 150), f"Chapter {page_num // 10 + 1}", fontsize=16)
            
            if page_num % 5 == 0:
                # H2 every 5 pages
                page.insert_text((70, 200), f"Section {page_num // 5 + 1}", fontsize=14)
            
            if page_num % 2 == 0:
                # H3 every 2 pages
                page.insert_text((90, 250), f"Subsection {page_num // 2 + 1}", fontsize=12)
            
            # Add some body text
            body_text = f"This is page {page_num + 1} content. " * 20
            page.insert_text((50, 300), body_text, fontsize=10)
        
        # Save to temporary file
        temp_path = tempfile.mktemp(suffix='.pdf')
        doc.save(temp_path)
        doc.close()
        
        return temp_path
        
    except ImportError:
        raise RuntimeError("PyMuPDF is required for creating test PDFs")


def test_performance(pdf_path: str, max_time: float = 10.0) -> dict:
    """Test processing performance on a PDF file."""
    # Always resolve pdf_path to absolute, relative to script if not already absolute
    pdf_path = Path(pdf_path)
    if not pdf_path.is_absolute():
        pdf_path = Path(__file__).parent / pdf_path
    pdf_path = pdf_path.resolve()

    schema_path = Path(__file__).parent / "sample_dataset" / "schema" / "output_schema.json"
    schema_path = schema_path.resolve()

    # Load schema
    schema = None
    if schema_path.exists():
        with open(schema_path, 'r') as f:
            schema = json.load(f)
    
    # Initialize extractor
    extractor = PDFOutlineExtractor(str(schema_path) if schema_path.exists() else None)
    
    # Time the extraction
    start_time = time.time()
    result = extractor.extract_outline(pdf_path)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    # Validate schema compliance
    schema_valid = False
    schema_error = None
    if schema:
        try:
            jsonschema.validate(result, schema)
            schema_valid = True
        except jsonschema.ValidationError as e:
            schema_error = str(e)
    
    # Performance check
    performance_pass = processing_time <= max_time
    
    return {
        "processing_time": processing_time,
        "max_time": max_time,
        "performance_pass": performance_pass,
        "schema_valid": schema_valid,
        "schema_error": schema_error,
        "title": result.get("title", ""),
        "outline_count": len(result.get("outline", [])),
        "result": result
    }


def run_performance_tests():
    """Run comprehensive performance tests."""
    print("=" * 60)
    print("Adobe Hackathon Challenge 1A - Performance Test Suite")
    print("=" * 60)
    
    # Test 1: Performance with generated 50-page PDF
    print("\n1. Testing with generated 50-page PDF...")
    try:
        test_pdf_path = create_test_pdf(50)
        
        result = test_performance(test_pdf_path, max_time=10.0)
        
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Performance requirement (≤10s): {'PASS' if result['performance_pass'] else 'FAIL'}")
        print(f"   Schema validation: {'PASS' if result['schema_valid'] else 'FAIL'}")
        if result['schema_error']:
            print(f"   Schema error: {result['schema_error']}")
        print(f"   Document title: {result['title']}")
        print(f"   Headings extracted: {result['outline_count']}")
        
        # Cleanup
        os.unlink(test_pdf_path)
        
        if not result['performance_pass']:
            print(f"\n❌ PERFORMANCE TEST FAILED: Processing took {result['processing_time']:.2f}s (max: 10s)")
            return False
        
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test 2: Sample dataset files
    print("\n2. Testing with sample dataset...")
    sample_dir = Path(__file__).parent / "sample_dataset" / "pdfs"
    if sample_dir.exists():
        pdf_files = list(sample_dir.glob("*.pdf"))
        if pdf_files:
            for pdf_file in pdf_files[:3]:  # Test first 3 files
                result = test_performance(str(pdf_file), max_time=5.0)
                print(f"   {pdf_file.name}: {result['processing_time']:.2f}s, "
                      f"schema: {'✓' if result['schema_valid'] else '✗'}, "
                      f"headings: {result['outline_count']}")
        else:
            print("   No sample PDF files found")
    else:
        print("   Sample dataset directory not found")
    
    print("\n✅ All performance tests completed successfully!")
    return True


if __name__ == "__main__":
    success = run_performance_tests()
    sys.exit(0 if success else 1)
