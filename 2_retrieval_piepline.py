from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory="db/chroma_db"

#loading embeddings and vector store
embedding_model=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2") #initalizing embedding model

db=Chroma( #recreating the vector store
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

#search for relevant documents
query="How much did Microsoft pay to acquire GitHub?"

# retriever=d.as_retriever(search_kwargs={"k":3}) or # it will retrieve top 3 chunks which has highest similarity score with query embeddings
retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k":5,
        "score_threshold":0.2 #only return chunks with cosine similarity >=0.3
    }
)
relevant_docs=retriever.invoke(query)

print(f"User query:{query}")
#display results

print("---Context----")
for i,doc in enumerate(relevant_docs,1):
    print(f"Document{i}:\n{doc.page_content}\n")


#combine the query and the relevant document contents
combined_input=f""" Based on the following documents, please answer this question:{query}

Documents:
{chr(10).join([f"-- {doc.page_content}" for doc in relevant_docs])}
Please provide a clear,helpful answer using only thr information from the documents. 
If you cannot find answer in the documents, say "I don have enough information to answer that question based on the provided documents."""

 
# create a openchatAi model
model = ChatGroq(model="openai/gpt-oss-20b")


#define the  messages for the model

messages=[
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combined_input),

]

#invoke the model with the combined input
result=model.invoke(messages)

#display the full result and content only
print("\n---Generated Response---")

print("Content only:")
print(result.content)
