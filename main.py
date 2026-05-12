from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from dotenv import load_dotenv

import requests
import os

#load environment variables
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "mistral")

#fast api
app = FastAPI()

#files:

UPLOAD_DIR = "files"

os.makedirs(UPLOAD_DIR, exist_ok=True)

#embbeding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# qdrant
qdrant = QdrantClient(":memory:")

COLLECTION_NAME = "rag_collection"

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

#vector store

vector_store = QdrantVectorStore(
    client=qdrant,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model
)

#home
@app.get("/")
def home():
    return {"message": "RAG WORKING"}

# user interface (ui)

@app.get("/ui", include_in_schema=False)
def ui():

    return FileResponse("templates/index.html")

# upload files :

@app.post("/upload")
async def upload(file: UploadFile):

    path = f"{UPLOAD_DIR}/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    return {
        "message": "uploaded successfully",
        "file": file.filename
    }

#process file and store chunks
@app.post("/process/{file_name}")
async def process(file_name: str):

    global vector_store
    path = f"{UPLOAD_DIR}/{file_name}"
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    #clear old collection
    qdrant.delete_collection(COLLECTION_NAME)

    #recreate collection
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    #recreate vector store
    vector_store = QdrantVectorStore(
        client=qdrant,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model
    )

    #add documents
    vector_store.add_documents(chunks)

    return {
        "chunks_count": len(chunks),
        "message": "stored in qdrant"
    }


#search for relevant chunks
@app.post("/search")
def search(query: str):

    results = vector_store.similarity_search_with_score(
        query=query,
        k=3
    )

    output = []

    for doc, score in results:

        output.append({
            "score": float(score),
            "text": doc.page_content
        })

    return output

#rag answer
@app.post("/answer")
def answer(query: str):

    results = vector_store.similarity_search_with_score(
        query=query,
        k=5
    )

    #filter weak matches
    filtered_results = []

    for doc, score in results:

        if score < 0.5:
            filtered_results.append((doc, score))

    context = "\n\n".join([
        doc.page_content for doc, score in filtered_results
    ])

    prompt = f"""
You are a strict AI assistant.

Answer ONLY from the provided context.

Rules:
- Do NOT guess.
- Do NOT invent information.
- If the information is not explicitly written in the context, say:
"I could not find the answer in the documents."
- Mention only names explicitly connected to the requested skill or experience.

Context:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        "http://host.docker.internal:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    return {
        "query": query,
        "answer": data["response"]
    }