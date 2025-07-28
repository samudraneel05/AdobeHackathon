### Methodology: A Hybrid, Reranking Pipeline for Intelligent Document Analysis

Our solution addresses the challenge of building an intelligent document analyst by implementing a state-of-the-art, multi-stage retrieval architecture. The core design philosophy is to combine classical information retrieval techniques with modern deep learning models to achieve high accuracy and relevance, while adhering strictly to the CPU-only, offline, and size constraints. The pipeline is fully automated and containerized, moving from raw PDFs to a final, analyzed JSON output.

**Phase 1: Semantic Document Parsing**

The foundation of any analysis is high-quality data. Our pipeline begins with a custom semantic parser built using PyMuPDF. This parser goes beyond simple text extraction; it analyzes the document's structure using heuristics based on font size, style (e.g., bold), and layout. This allows it to reliably differentiate between headings and their associated paragraph content. The output is a semantically structured JSON tree for each PDF, which preserves the crucial context of the document's hierarchy. This structured data is vastly superior to raw text blocks, enabling more precise retrieval in later stages.

**Phase 2: Deep User Intent Modeling**

To "Connect What Matters," we must first deeply understand what the user is asking for. The input persona and job-to-be-done are processed by an advanced NLP module. We employ GLiNER for fast and accurate Named Entity Recognition to extract key concepts. Most importantly, we transform the user's task into a question-like format (e.g., prepending "Question:") and use a powerful, query-focused Sentence Transformer (`multi-qa-mpnet-base-dot-v1`) to generate a high-fidelity semantic vector embedding. This embedding acts as a numerical representation of the user's core intent.

**Phase 3: Hybrid Retrieval and Precision Reranking**

This two-stage process is the heart of our system's intelligence.

*   **Stage 1 - Hybrid Search:** To ensure we find all potentially relevant information (high recall), we use a hybrid search strategy. It combines the strengths of BM25, a classical keyword-based algorithm that excels at finding exact term matches, with our semantic vector search, which uses the user's intent embedding to find conceptually similar content, even if the wording is different. The results from both searches are fused into a robust list of candidate sections.

*   **Stage 2 - Cross-Encoder Reranking:** This is our precision-focused stage. The candidate list is passed to a Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`). Unlike the previous step, a cross-encoder performs a more computationally intensive but far more accurate full comparison between the user's query and each candidate document. This allows it to precisely re-order the candidates, pushing the most relevant sections to the absolute top of the list with high confidence.

**Phase 4: Generative Analysis**

Finally, the top-ranked sections from the reranker are passed to an instruction-tuned generative model (Google's Flan-T5) to produce the final `Refined Text`. This adds a final layer of AI-powered analysis, delivering a concise, relevant summary for each key finding. This robust, multi-layered approach ensures the system doesn't just find information, but prioritizes and analyzes it with a deep understanding of the user's specific goals.
