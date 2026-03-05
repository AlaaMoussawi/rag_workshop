# rag_workshop

This repository is meant to accompany a 15 minute tutorial on creating your own rag model, for a workshop held on March 6th, 2026.
It utilizes minimal technology, to get a basic RAG model running as quickly as possible.

To run the RAG model:

Install Ollama, and run: "ollama serve"
Download a small model with: "ollama pull qwen3.5:2b"

Creating a virtual environment is recommended.

Install requirements from requirements.txt: "pip3 install requirements.txt"

Download "punkt" from nltk by uncommenting the necessary line and running: "python3 rag_demo.py". This can be commented after running once.
This will also run the model!