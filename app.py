from flask import Flask, render_template, request, jsonify
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings("ignore")

# --- Load RAG Components (Run this once) ---
print("Loading RAG components...")

# Load the AfriMed-QA dataset
dataset = load_dataset("intronhealth/afrimedqa_v2")


def is_valid(example):
    return example["answer_rationale"] is not None and example["question_type"] != "consumer_queries"


filtered_dataset = dataset.filter(is_valid)

# IMPORTANT: Switching to a more robust, biomedical-specific embedding model
# This model is fine-tuned on a vast corpus of medical literature, which should
# drastically improve the retrieval of relevant documents.
print("Using a more robust biomedical-specific sentence transformer for better retrieval...")
embedding_model = SentenceTransformer('NeuML/pubmedbert-base-embeddings')

# IMPORTANT: Correcting the column name from 'answer' to 'correct_answer'
docs = [f"Question: {q} Answer: {a} Rationale: {r}" for q, a, r in
        zip(filtered_dataset['train']['question'], filtered_dataset['train']['correct_answer'],
            filtered_dataset['train']['answer_rationale'])]
doc_embeddings = embedding_model.encode(docs, convert_to_tensor=True)

dimension = doc_embeddings.shape[1]
index = IndexFlatL2(dimension)
index.add(doc_embeddings.cpu().numpy())

# IMPORTANT: Using a slightly larger generative model for better responses
rag_model_name = "EleutherAI/gpt-neo-125m"
rag_tokenizer = AutoTokenizer.from_pretrained(rag_model_name)
rag_model = AutoModelForCausalLM.from_pretrained(rag_model_name)
rag_tokenizer.pad_token = rag_tokenizer.eos_token
rag_model.config.pad_token_id = rag_model.config.eos_token_id

if torch.cuda.is_available():
    rag_model.to('cuda')


# --- RAG Functions ---
def retrieve(query, k=3):
    """Retrieves the top k most relevant documents from the vector database."""
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return [docs[i] for i in indices.flatten()]


def generate(question, retrieved_answers):
    """Generates an answer based on the retrieved context using a language model."""
    context = "\n".join(retrieved_answers)

    # IMPORTANT: New, stricter prompt to prevent hallucinations
    prompt = f"Answer the following question based ONLY on the provided context. If the answer is not in the context, say 'I am sorry, but I cannot find an answer in my knowledge base.'\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    inputs = rag_tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    outputs = rag_model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=rag_tokenizer.eos_token_id
    )
    generated_text = rag_tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.split("Answer:")[-1].strip()
    return answer


# --- Flask App Routes ---
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({'error': 'No question provided'}), 400

    retrieved_answers = retrieve(user_question)
    final_answer = generate(user_question, retrieved_answers)

    return jsonify({'answer': final_answer})


if __name__ == '__main__':
    # Make sure to install Flask: pip install flask
    app.run(debug=True, host='0.0.0.0')
