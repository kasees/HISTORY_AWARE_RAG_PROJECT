import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv() 

def load_documents(docs_path="docs"):
    """ Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}")

    # checks if the docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} doesn't exist. Please create it and add your company files.")

    # load all the txt files from th docs directory
    loader=DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents=loader.load()

    if len(documents)==0:
        raise FileNotFoundError(f"No .txt files are found in {docs_path}. Please add your company documents.")

    for i,doc in enumerate(documents[:2]): # show the first 2 documents
        print(f"\n Document {i+1}:")
        print(f" Source:{doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}....")
        print(f"metadata: {doc.metadata}")

    return documents

#chunking

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """ Split documents into small chunks with overlap"""
    print("Splitting documents into chunks...")

    text_splitter=CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks=text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n---- Chunk{i+1}")
            print(f"Source:{chunk.metadata['source']}")
            print(f"length:{len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-"*50)

        if len(chunks)>5:
            print(f"\n ... and {len(chunks)-5} more chunks")
    return chunks

# embedding

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """ Create an persists ChromaDb vector store"""
    print("Creating ebeddings and Storing in Chromadb")

    embedding_model=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2") #initalizing embedding model

    #create chromad vector store

    print("----Creating vector store")
    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print("--Finished creating vector store---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore





def main():
    print("Main function")


    # Step1: loading all the files
    documents=load_documents(docs_path="docs")
    #Step2: Chunking all the files
    chunks=split_documents(documents)
    #Step3: Embedding and Storing it in vector DB
    vectorstore=create_vector_store(chunks)

if __name__=="__main__":
    main()