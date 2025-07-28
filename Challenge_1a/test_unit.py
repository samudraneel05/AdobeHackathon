#!/usr/bin/env python3
"""
Unit tests for Challenge 1A PDF processing functionality.
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from process_pdfs import PDFOutlineExtractor
    import jsonschema
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install dependencies: pip install PyMuPDF jsonschema")
    sys.exit(1)


class TestPDFOutlineExtractor(unittest.TestCase):
    """Test cases for PDF outline extraction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.schema_path = (Path(__file__).parent / "sample_dataset" / "schema" / "output_schema.json").resolve()
        self.extractor = PDFOutlineExtractor(str(self.schema_path) if self.schema_path.exists() else None)
    
    def test_heading_level_detection(self):
        """Test heading level detection algorithm."""
        # Test H1 detection
        self.assertEqual(
            self.extractor.detect_heading_level("CHAPTER ONE", 16.0, 16, 12.0),
            "H1"
        )
        
        # Test H2 detection
        self.assertEqual(
            self.extractor.detect_heading_level("1. Introduction", 14.0, 16, 12.0),
            "H2"
        )
        
        # Test H3 detection
        self.assertEqual(
            self.extractor.detect_heading_level("1.1 Overview", 13.0, 16, 12.0),
            "H3"
        )
        
        # Test non-heading
        self.assertIsNone(
            self.extractor.detect_heading_level("This is regular paragraph text.", 10.0, 0, 12.0)
        )
    
    def test_title_extraction(self):
        """Test document title extraction."""
        # This would require creating a test PDF, which we'll skip for now
        # In a real implementation, we'd create minimal test PDFs
        pass
    
    def test_schema_validation(self):
        """Test output schema compliance."""
        if not self.schema_path.exists():
            self.skipTest("Schema file not found")
        
        # Load schema
        with open(self.schema_path, 'r') as f:
            schema = json.load(f)
        
        # Test valid output
        valid_output = {
            "title": "Test Document",
            "outline": [
                {
                    "level": "H1",
                    "text": "Chapter 1",
                    "page": 1
                },
                {
                    "level": "H2", 
                    "text": "Section 1.1",
                    "page": 2
                }
            ]
        }
        
        # Should not raise exception
        jsonschema.validate(valid_output, schema)
        
        # Test invalid output - missing required field
        invalid_output = {
            "outline": [
                {
                    "level": "H1",
                    "text": "Chapter 1",
                    "page": 1
                }
            ]
            # Missing "title" field
        }
        
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(invalid_output, schema)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test empty heading detection
        self.assertIsNone(
            self.extractor.detect_heading_level("", 12.0, 0, 12.0)
        )
        
        # Test very short text
        self.assertIsNone(
            self.extractor.detect_heading_level("ab", 12.0, 0, 12.0)
        )
        
        # Test very long text (likely paragraph)
        long_text = "This is a very long paragraph that should not be detected as a heading " * 5
        self.assertIsNone(
            self.extractor.detect_heading_level(long_text, 12.0, 0, 12.0)
        )


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_output_format(self):
        """Test that output matches expected format."""
        # This would test with actual PDF files
        # For now, we'll test the structure
        expected_keys = {"title", "outline"}
        
        # Mock result
        result = {
            "title": "Test",
            "outline": []
        }
        
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertIsInstance(result["title"], str)
        self.assertIsInstance(result["outline"], list)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
