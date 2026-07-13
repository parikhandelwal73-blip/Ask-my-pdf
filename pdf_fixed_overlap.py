# Import Libraries
import chromadb
from pypdf import PdfReader
from ollama import chat
from sentence_transformers import SentenceTransformer
# Read the PDF
reader = PdfReader("/Users/parikhandelwal/Desktop/folder/Rag/PDF_Chatbot_RAG_Learning_Roadmap.pdf")
text = ""
# Extract text from every page
for page in reader.pages:
    page_text = page.extract_text()
# Ignore empty pages
    if page_text:
        text += page_text
print("First 1000 characters:\n")
print(text[:1000])
#Chunk and overlap the PDF
def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks
chunks = chunk_text(text, chunk_size=200, overlap=50)
print("\nNumber of chunks:", len(chunks))
print("\nFirst 5 chunks:\n")
for i, chunk in enumerate(chunks[:5], start=1):
    print(f"Chunk {i}")
    print(chunk)
    print("#" * 60)
#Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2",device="cpu")
#Generate Embeddings
embeddings = model.encode(chunks)
print("\nEmbedding shape:")
print(embeddings.shape)
#Create ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="pdf_collection"
)
#Store Chunks
collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist()
)
print("\nAll chunks stored successfully")
#Accept User Question
query = input("\nEnter your question: ")
query_embedding = model.encode([query])
#Giving top 4 ans
results = collection.query(query_embeddings=query_embedding.tolist(),n_results=4)
print("\nTop 4 Retrieved Chunks:\n")
for i, doc in enumerate(results["documents"][0], start=1):
    print(f"Result {i}")
    print(doc)
    print("-" * 80)
retrieved_text = "\n".join(results["documents"][0])
#Create Prompt
prompt = f"""Answer only using the context below.Context:{retrieved_text}Question:{query}"""
#Generate Answer using Ollama
response = chat(model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",messages=[{"role": "user","content": prompt }])
answer = response["message"]["content"]
print("\nFinal Answer:\n")
print(answer)