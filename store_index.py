from src.helper import load_pdf_files, text_split, load_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


docs = load_pdf_files(data="./data")
chunks = text_split(docs)
embedding = load_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)
pc.create_index(
    name="med",
    dimension=384, 
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

doc_search = PineconeVectorStore.from_documents(
    documents=chunks,
    index_name="med",
    embedding=embedding
)