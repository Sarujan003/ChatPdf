# app.py

import os
from flask import Flask, request, jsonify, Response, render_template, stream_with_context
from werkzeug.utils import secure_filename

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq

app = Flask(__name__)

# --- Model and RAG Initialization ---
# Ensure you set GROQ_API_KEY in your deployment environment
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Global variable for the RAG chain
rag_chain = None

def create_rag_chain(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(documents=split_docs, embedding=embeddings)
    retriever = vector_store.as_retriever()

    prompt_template = """
    You are a helpful AI assistant... (rest of your prompt)
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global rag_chain
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.pdf'):
        # Use a secure temporary directory provided by the OS
        upload_folder = '/tmp'
        os.makedirs(upload_folder, exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        
        try:
            rag_chain = create_rag_chain(filepath)
            return jsonify({'message': f'PDF "{filename}" processed!'})
        except Exception as e:
            return jsonify({'error': f'Failed to process PDF: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    global rag_chain
    if not rag_chain:
        return Response("Error: PDF not processed.", status=400)
        
    data = request.json
    question = data.get('question')
    if not question:
        return Response("Error: No question provided.", status=400)

    def generate():
        for chunk in rag_chain.stream(question):
            yield chunk

    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == '__main__':
    # This is for local testing only. gunicorn will run the app in production.
    app.run(debug=True, port=5001)