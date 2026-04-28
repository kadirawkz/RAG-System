import os
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs={"k": 5})

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
    }
)

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

# Create a local, free text-to-text model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
generator_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Invoke the model with the combined input
inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    output_tokens = generator_model.generate(**inputs, max_new_tokens=256)
result = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# Display the full result and content only
print("\n--- Generated Response ---")
print("Content only:")
print(result)
