
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Step 1: Load resume PDF
loader = PyPDFLoader(
    r"D:\practiceai\RAGpracticeproject\Documents\DEVOPS Anil Kumar Resume.pdf"
)

documents = loader.load()

print("PDF loaded successfully!")


# Step 2: Split resume into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# Step 3: Load local Ollama embedding model
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

print("Ollama embedding model loaded!")


# Step 4: Generate embeddings and store chunks
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="resume_collection",
    persist_directory="./chroma_db"
)

print("Resume chunks stored in ChromaDB!")


# Step 5: Load local Ollama LLM
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


# Step 6: Create prompt template
prompt_template = ChatPromptTemplate.from_template("""
You are a professional Resume Analyzer and AI Recruitment Assistant.

Answer the user's questions based ONLY on the provided context.

Instructions:
1. Analyze skills, experience, education, certifications and projects.
2. Provide clear and accurate answers.
3. Use bullet points wherever appropriate.
4. Do not assume or invent information.
5. If information is unavailable, say:
   "The information is not available in the document."
6. Keep answers concise and easy to understand.

Context:
{context}

Question:
{question}

Answer:
""")


# Step 7: Create RAG chain
rag_chain = prompt_template | llm | StrOutputParser()


# Step 8: Ask questions
while True:

    question = input("\nEnter your question (type exit to stop): ")

    if question.lower() == "exit":
        print("Chatbot stopped.")
        break

    # Step 9: Retrieve top 3 similar chunks
    retrieved_docs = vector_store.similarity_search(
        query=question,
        k=3
    )

    # Step 10: Extract retrieved context
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Step 11: Generate answer using Ollama
    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print("\nAI Response:")
    print(response)
