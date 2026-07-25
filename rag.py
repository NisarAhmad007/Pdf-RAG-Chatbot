import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

def main():
    # --- Configuration ---
    pdf_path = "Eat That Frog! book.pdf"
    llm_model = "qwen2.5:1.5b"           
    embedding_model = "nomic-embed-text" 
    persist_directory = "./chroma_db"   
    
    if not os.path.exists(pdf_path):
        print(f"Error: Please place your PDF at '{pdf_path}'")
        return

    # Initialize Embeddings
    print(f"[{time.strftime('%H:%M:%S')}] Initializing local embedding engine...")
    embeddings = OllamaEmbeddings(model=embedding_model)

    # Check if the database already exists on your hard drive
    if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
        print(f"\n[{time.strftime('%H:%M:%S')}] ---> FOUND SAVED EMBEDDINGS!")
        print(f"[{time.strftime('%H:%M:%S')}] Loading database from '{persist_directory}' (this will be instant)...")
        vector_store = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embeddings
        )
    else:
        print(f"\n[{time.strftime('%H:%M:%S')}] ---> NO SAVED EMBEDDINGS FOUND.")
        print(f"[{time.strftime('%H:%M:%S')}] Processing '{pdf_path}' for the first time...")
        
        # 1. Load PDF
        print(f"[{time.strftime('%H:%M:%S')}] Reading pages from PDF...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # 2. Split into chunks
        print(f"[{time.strftime('%H:%M:%S')}] Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        # 3. Generate embeddings and write to disk
        print(f"[{time.strftime('%H:%M:%S')}] Generating embeddings for {len(chunks)} chunks on GPU...")
        print("*(Please wait, this first-time setup takes a moment...)*")
        
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=persist_directory
        )
        print(f"[{time.strftime('%H:%M:%S')}] Success! Database saved locally to '{persist_directory}'")

    # 4. Create Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 4}) 

    # 5. Initialize LLM & Chains
    print(f"[{time.strftime('%H:%M:%S')}] Loading local LLM ({llm_model})...")
    llm = ChatOllama(model="qwen2.5:1.5b" , temperature=0.0)
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print(f"[{time.strftime('%H:%M:%S')}] RAG System Ready!")

    # --- Interactive Loop ---
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.strip().lower() == 'exit':
            break
        if not query.strip():
            continue
        
        print("\nThinking...")
        start_time = time.time()
        response = rag_chain.invoke({"input": query})
        elapsed = time.time() - start_time
        
        print(f"\n--- Answer (Generated on GPU in {elapsed:.2f}s) ---")
        print(response['answer'])
        print("-" * 50)

if __name__ == "__main__":
    main()