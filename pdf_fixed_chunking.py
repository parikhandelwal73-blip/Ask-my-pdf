#Import Libraries
import chromadb
from pypdf import PdfReader
from ollama import chat
from sentence_transformers import SentenceTransformer
#Read the PDF
reader = PdfReader("/Users/parikhandelwal/Desktop/folder/Rag/PDF_Chatbot_RAG_Learning_Roadmap.pdf")
text = ""
#Extract text from every page
for page in reader.pages:
    page_text = page.extract_text()
#ignore empty pages
    if page_text:
        text += page_text
print("First 1000 characters:\n")
print(text[:1000])
#Chunk the PDF
chunk_size = 200
chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size)
]
print("\nNumber of chunks:", len(chunks))
print("\nFirst 5 chunks:\n")
for chunk in chunks[:5]:
    print(chunk)
    print("#" * 50)
#Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2",device="cpu")
#Generate Embeddings
embeddings = model.encode(chunks)
print("\nEmbedding shape:")
print(embeddings.shape)
#Create ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="pdf_collection")
#Store Chunks
collection.add(ids=[str(i) for i in range(len(chunks))],documents=chunks,embeddings=embeddings.tolist())
print("\nAll chunks stored successfully")
#Accept User Question
query = input("\nEnter your question: ")
query_embedding = model.encode([query])
#Giving top 3 ans
results = collection.query(query_embeddings=query_embedding.tolist(),n_results=3)
print("\nTop 3 Retrieved Chunks:\n")
for i, doc in enumerate(results["documents"][0], start=1):
    print(f"Result {i}")
    print(doc)
    print("-" * 80)
#Create Prompt
retrieved_text = "\n".join(results["documents"][0])
prompt = f"""
You are an expert AI assistant.

First, use the provided context to answer the question.

If the context contains the answer, answer from it.

If the context is incomplete or does not contain the answer, use your own knowledge to provide a helpful response.

If you use your own knowledge, clearly mention:

"Note: The following information is based on my general knowledge and not on the provided PDF."

Context:
{retrieved_text}

Question:
{query}

Answer:
"""
#Generate Answer using Ollama
response = chat(
    model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)
answer = response["message"]["content"]
print("\nFinal Answer:\n")
print(answer)


