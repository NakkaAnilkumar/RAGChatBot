
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Step 1: Load text file
loader = TextLoader(
    r"D:\practiceai\RAGpracticeproject\Documents\tasks.txt",
    encoding="utf-8"
)

documents = loader.load()

print("Document loaded successfully!")


# Step 2: Split document into chunks
textsplitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = textsplitter.split_documents(documents)

print("Total chunks:", len(chunks))


# Step 3: Load Ollama embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)


# Step 4: Store chunks in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="resume_collection",
    persist_directory="./chroma.db"
)

print("Documents stored successfully!")


# Step 5: Load Ollama LLM
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


# Step 6: Create prompt template
prompt_template = PromptTemplate.from_template("""
Act like a Resume Analyzer.

Answer the question using only the given context.

Context:
{context}

Question:
{question}

Answer:
""")


# Step 7: Create RAG chain
ragchain = prompt_template | llm | StrOutputParser()


# Step 8: Ask questions
while True:

    question = input("\nAsk a question or type exit: ")

    if question.lower() == "exit":
        print("Chatbot stopped!")
        break

    # Step 9: Retrieve relevant chunks
    retrieved_docs = vectorstore.similarity_search(
        query=question,
        k=3
    )

    # Step 10: Extract context
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Step 11: Generate answer
    response = ragchain.invoke({
        "context": context,
        "question": question
    })

    print("\nAI Response:")
    print(response)
