from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()

def main():
    print("Loading...")
    loader = TextLoader("C:\\langchain\\the-gist-of-rag\\mediumblog1.txt",encoding="utf-8")
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chuncks") 

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",api_key=GOOGLE_API_KEY)
    print("Ingesting...")
    PineconeVectorStore.from_documents(texts,embeddings,index=os.environ['INDEX_NAME'])
    print("Finish")
if __name__ == "__main__":
    main()
