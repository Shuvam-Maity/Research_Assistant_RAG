from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def get_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )


def create_vector_store(chunks):

    embeddings = get_embeddings()

    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    return db