
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_path="https://en.wikipedia.org/wiki/Cricket"
)
documents = loader.load()

print("PDF loaded successfully!")


# Step 2: Split resume into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)

# Step 3: Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))



# Step 4: Generate embeddings and store chunks
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="resume_collection",
    persist_directory="./chroma_db"
)

print("Resume chunks stored in ChromaDB!")

# Step 5: Load local Ollama model
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

# Step 6: Create prompt template
prompt_template = ChatPromptTemplate.from_template("""
You are a professional Cricket Information Assistant.

Your task is to answer the user's questions about cricket
using ONLY the provided website context.

Instructions:
1. Analyze the provided cricket information carefully.
2. Answer questions about players, teams, matches,
   tournaments, scores, records, and statistics.
3. Provide clear, accurate, and professional answers.
4. Use bullet points wherever appropriate.
5. Do not assume or invent any information.
6. If the requested information is not available in the
   provided context, say:
   "The information is not available on the website."
7. Keep answers concise and easy to understand.


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
