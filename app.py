# app.py

import os
import math
from flask import Flask, request, jsonify, Response, render_template, stream_with_context

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- 1. API Key and Model Initialization ---
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")

try:
    llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("✅ Models initialized successfully.")
except Exception as e:
    print(f"🔥 Error initializing models: {e}")
    llm, embeddings = None, None

# --- 2. Prompt Engineering ---
ANALYSIS_PROMPT_TEMPLATE = """
You are an expert AI Legal Assistant. Your task is to analyze a legal document for a non-lawyer.
Your analysis must be clear, simple, and focus on practical implications. Avoid legal jargon.
Analyze the following document text and provide a response in structured Markdown format.
**DO NOT** invent any information. If a section is not applicable, state that clearly.
---
DOCUMENT TEXT:
{document_text}
---
REQUIRED ANALYSIS:
### 📝 Plain-English Summary
Provide a concise summary of the document's main purpose and key terms.
### 🎯 Key Clauses & Obligations
Identify and explain important clauses: Term & Termination, Payments, Responsibilities, Auto-Renewal, Penalties.
### 🚩 Potential Red Flags & Risks
Highlight unusual, one-sided, or risky clauses and explain why.
### 🧑‍⚖️ Recommendation
Provide a recommendation (e.g., "appears straightforward" or "contains complex clauses, consider consulting a professional").
"""
RAG_PROMPT_TEMPLATE = """
You are an AI Legal Assistant. Answer the user's question based ONLY on the following context.
If the context does not contain the answer, state "The document does not seem to contain specific information on this topic."
Quote the relevant part of the document if it helps, but always explain it in simple terms.
Context:
{context}
Question:
{question}
Answer:
"""

# --- 3. Lightweight AI Logic ---
rag_chain = None

def perform_initial_analysis(docs):
    full_document_text = "\n\n".join([doc.page_content for doc in docs])
    if len(full_document_text) > 25000:
        return "The document is too long for a full automatic analysis. Please use the chat to ask specific questions."
    analysis_prompt = PromptTemplate(template=ANALYSIS_PROMPT_TEMPLATE, input_variables=["document_text"])
    analysis_chain = analysis_prompt | llm | StrOutputParser()
    return analysis_chain.invoke({"document_text": full_document_text})

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    return 0.0 if norm_a == 0 or norm_b == 0 else dot_product / (norm_a * norm_b)

def create_rag_chain(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)
    chunk_texts = [doc.page_content for doc in split_docs]
    chunk_embeddings = embeddings.embed_documents(chunk_texts)

    def retrieve_chunks(query_text: str, k: int = 5):
        query_embedding = embeddings.embed_query(query_text)
        similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]
        indexed_chunks = sorted(list(zip(similarities, split_docs)), key=lambda x: x[0], reverse=True)
        return [doc for sim, doc in indexed_chunks[:k]]

    prompt = PromptTemplate(template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"])
    retriever_runnable = RunnableLambda(lambda question: retrieve_chunks(question))
    
    return {
        "context": retriever_runnable,
        "question": RunnablePassthrough()
    } | prompt | llm | StrOutputParser()


# --- 4. Flask App Definition ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global rag_chain
    if 'pdf_file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['pdf_file']
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Please select a valid PDF file'}), 400
    try:
        temp_dir = "/tmp"
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)
        filepath = os.path.join(temp_dir, file.filename)
        file.save(filepath)

        loader = PyPDFLoader(filepath)
        docs = loader.load()
        initial_analysis_markdown = perform_initial_analysis(docs)
        rag_chain = create_rag_chain(docs)
        os.remove(filepath)

        return jsonify({'message': f'Analyzed "{file.filename}"', 'analysis': initial_analysis_markdown})
    except Exception as e:
        print(f"🔥 Error during PDF processing: {e}")
        return jsonify({'error': f'Failed to process PDF: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global rag_chain
    if not rag_chain: return Response("Error: Document not processed yet.", status=400)
    data = request.json
    question = data.get('question')
    if not question: return Response("Error: No question provided.", status=400)
    def generate():
        for chunk in rag_chain.stream(question): yield chunk
    return Response(stream_with_context(generate()), mimetype='text/plain')
