from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load environment variables
load_dotenv()

# Connect to your document database
persistent_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Set up local, free text-to-text model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
generator_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Store our conversation history as text
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    search_question = user_question
    if chat_history:
        # Use local model to rewrite question as standalone
        history_text = "\n".join(chat_history[-4:])  # Keep last 2 exchanges
        prompt = f"""Given this chat history:
{history_text}

Rewrite this new question to be standalone and searchable: {user_question}

Just return the rewritten question:"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output_tokens = generator_model.generate(**inputs, max_new_tokens=128)
        search_question = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()
        print(f"Searching for: {search_question}")
    
    # Step 2: Find relevant documents
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.3
        }
    )
    docs = retriever.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")
    
    # Step 3: Create final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
    
    # Step 4: Get the answer from local model
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output_tokens = generator_model.generate(**inputs, max_new_tokens=256)
    answer = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    # Step 5: Remember this conversation
    chat_history.append(f"User: {user_question}")
    chat_history.append(f"Assistant: {answer}")
    
    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()