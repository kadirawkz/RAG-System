from langchain_chroma import Chroma
from langchain_nomic import NomicEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
import re


load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store (Nomic)
embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# Create a Google Gemini chat model
model_name = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")

# Accept both "gemini-..." and "models/gemini-..." values.
if model_name.startswith("models/"):
    model_name = model_name.split("/", 1)[1]

fallback_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]
model_candidates = [model_name] + [m for m in fallback_models if m != model_name]

# Require API key
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError(
        "The GOOGLE_API_KEY environment variable is not set.\n"
        "Set it in your environment or in the .env file (GOOGLE_API_KEY=your_key)."
    )

model = None

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with fallback attempts for model availability differences
last_error = None
result = None
for candidate in model_candidates:
    try:
        model = ChatGoogleGenerativeAI(model=candidate)
        result = model.invoke(messages)
        print(f"Using Google model: {candidate}")
        break
    except Exception as exc:
        last_error = exc

def retrieval_only_fallback_answer(user_query, docs):
    """Generate a minimal answer directly from retrieved documents when LLM is unavailable."""
    combined_text = "\n".join(doc.page_content for doc in docs)
    amount_match = re.search(r"\$\s?\d+(?:\.\d+)?\s?(?:billion|million|trillion)", combined_text, re.IGNORECASE)

    if "github" in user_query.lower() and amount_match:
        return f"Based on the retrieved documents, Microsoft paid {amount_match.group(0)} to acquire GitHub."

    return (
        "I couldn't reach Gemini due to API quota limits, and I don't have enough reliable "
        "information to produce a high-confidence generated answer."
    )


if result is None:
    error_text = str(last_error) if last_error else ""
    if "RESOURCE_EXHAUSTED" in error_text or "quota" in error_text.lower():
        print("Gemini quota exceeded; using retrieval-only fallback response.")
        result_content = retrieval_only_fallback_answer(query, relevant_docs)
    else:
        raise RuntimeError(
            "Unable to generate an answer with the configured Gemini model. "
            "Set GOOGLE_MODEL to a model available in your account, e.g. gemini-1.5-flash."
        ) from last_error
else:
    result_content = result.content

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result_content)
