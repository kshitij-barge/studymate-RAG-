from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector database
vector_store = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

# Retriever setup
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# Load Mistral LLM
llm = ChatMistralAI(model="mistral-small")


# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use only the provided context to answer the user's question. If the answer is not present in the context, say you don't know."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        ),
    ]
)

print("RAG system ready. Ask your question!")
print("Type 0 to exit.")

while True:
    query = input("\nYour question: ")

    # Exit condition
    if query.lower() == "0":
        print("Exiting...")
        break

    try:
        # Retrieve relevant documents
        docs = retriever.invoke(query)

        # Combine retrieved content
        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        # Debug preview
        print("\nRetrieved Context Preview:")
        print(context[:500])

        # Build final prompt
        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": query
            }
        )

        # Generate response
        response = llm.invoke(final_prompt)

        # Print answer
        print("\nAI:", response.content)

    except Exception:
        import traceback 
        print("\nFULL ERROR:")
        traceback.print_exc()