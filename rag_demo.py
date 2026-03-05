import requests
from bs4 import BeautifulSoup
import nltk
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

print("\n--- RAG WORKSHOP DEMO ---\n")

############################################
# 1. DOWNLOAD DATA
############################################

print("Downloading dataset...")

url = "https://www.nasa.gov/the-apollo-program/"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

paragraphs = [p.get_text() for p in soup.find_all("p")]
text = "\n".join(paragraphs)

print("Downloaded", len(text), "characters")

############################################
# 2. CHUNK THE TEXT
############################################

print("\nTokenizing into sentences...")

# nltk.download("punkt") #ONLY DOWNLOAD ONCE, then make sure the line is commented out!

from nltk.tokenize import sent_tokenize

sentences = sent_tokenize(text)

chunk_size = 1
chunks = []

for i in range(0, len(sentences), chunk_size):
    chunk = " ".join(sentences[i:i+chunk_size])
    chunks.append(chunk)

print("Created", len(chunks), "chunks")

############################################
# 3. CREATE EMBEDDINGS
############################################

print("\nCreating embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("Embeddings created")

############################################
# 4. CREATE VECTOR DATABASE
############################################

print("\nCreating vector database...")

client = chromadb.Client()

collection = client.create_collection("apollo_docs")

for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        embeddings=[embeddings[i]],
        ids=[str(i)]
    )

print("Stored", len(chunks), "documents in ChromaDB")

############################################
# 5. ASK A QUESTION
############################################

while True:

    print("\nAsk a question (or type 'exit'):")

    query = input("> ")

    if query.lower() == "exit":
        break

    ############################################
    # Retrieve similar documents
    ############################################

    query_embedding = model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    print("\nRetrieved context:\n")
    print(context[:500], "...\n")

    ############################################
    # Send to LLM
    ############################################

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY information from the context below, but providing a response customized to the question asked. If the answer can't be found in the context, respond with "I have no information pertaining to your question."

Context:
{context}

Question:
{query}
"""

    response = ollama.chat(
        model="Qwen3.5:2b",
        messages=[{"role": "user", "content": prompt}],
        think=False
    )

    print("\nAnswer:\n")
    print(response["message"]["content"])
