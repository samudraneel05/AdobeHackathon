# Dockerfile for the Adobe Document Analyst Project

# --- Stage 1: The Builder ---
# This stage installs dependencies and downloads models to be copied later.
FROM python:3.10-slim as builder

WORKDIR /app

# Copy only the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install all Python dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# --- This is the key step for handling large models ---
# It runs a small Python script INSIDE the build to download and save all models.
# This ensures they are part of the image, but not part of your Git repository.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; from gliner import GLiNER; from transformers import T5ForConditionalGeneration, T5Tokenizer; \
    print('Downloading MPNet-QA model...'); \
    SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1').save('./models/mpnet_qa'); \
    print('Downloading GLiNER model...'); \
    GLiNER.from_pretrained('urchade/gliner_small-v2.1').save_pretrained('./models/gliner_small'); \
    print('Downloading Cross-Encoder model...'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
    print('Downloading Flan-T5 model...'); \
    T5ForConditionalGeneration.from_pretrained('google/flan-t5-base'); \
    T5Tokenizer.from_pretrained('google/flan-t5-base'); \
    print('All models downloaded.')"

# --- Stage 2: The Final, Lean Application Image ---
FROM python:3.10-slim

WORKDIR /app

# Copy the installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the downloaded models from the builder stage's specific save locations and cache
COPY --from=builder /app/models /app/models
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy all your application source code into the final image
# This includes main.py, process_pdfs.py, retriever.py, etc.
COPY . .

# This is the command that will run when the container starts.
# It executes your main orchestrator script.
ENTRYPOINT ["python", "main.py"]

# This provides a default argument. If someone runs the container without specifying
# a collection path, it will show your script's usage instructions.
CMD [""]