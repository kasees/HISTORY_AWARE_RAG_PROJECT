from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()
embedding_model=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2") #initalizing embedding model

persistent_directory="db/chroma_db"

#loading embeddings and vector store


db=Chroma( #recreating the vector store
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
)

model = ChatGroq(model="openai/gpt-oss-20b")
#storing our conversation as messages
chat_history=[]

def ask_question(user_question):
    print(f"\n--You asked:{user_question}")

    #Step 1: Make the question clear usinh conversation history
    if chat_history:
        #ask ai to make the question standalone
        messages=[
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question")
        
        ]+chat_history+[
            HumanMessage(content=f"New question:{user_question}")
        ]

        result=model.invoke(messages)
        search_question=result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question=user_question

    #step2: Find relevant documents
    retriever=db.as_retriever(search_kwargs={"k":3})
    docs=retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i,doc in enumerate(docs,1):
        #show first 2 lines of each document
        lines=doc.page_content.split('\n')[:2]
        preview='\n'.join(lines)
        print(f" Doc {i}: {preview}...")

    #step 3: create the final prompt
    combined_input=f""" Based on the following documents, please answer this question:{user_question}

    Documents:
    {chr(10).join([f"-- {doc.page_content}" for doc in docs])}
    Please provide a clear,helpful answer using only thr information from the documents. 
    If you cannot find answer in the documents, say "I don have enough information to answer that question based on the provided documents."""

    #step 4: get the answer
    
    messages=[
    SystemMessage(content="You are a helpful assistant that answers the questions based on the provived documents and conversation history"),
    ]+chat_history+[
    HumanMessage(content=combined_input)

]
    result=model.invoke(messages)
    answer=result.content

    #remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer:{answer}")
    return answer






#simple chat loop
def start_chat():
    print("Ask me questions ! type 'quit' to exit.")

    while True:
        question=input("\n Your Question:")

        if question.lower()=="quit":
            print("Goodbye")
            break
        ask_question(question)

if __name__=="__main__":
    start_chat()

    
