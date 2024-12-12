# Save the below requirements as a requirements.txt file for deployment
# requirements.txt content:
# transformers
# PyPDF2
# python-docx
# flask
# boto3

from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from docx import Document
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from concurrent.futures import ThreadPoolExecutor
import torch

app = Flask(__name__)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ''.join(page.extract_text() for page in reader.pages if page.extract_text())
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

# Function to extract text from a DOCX file
def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        return f"Error reading DOCX: {e}"

# Load T5 and BERT models concurrently
def load_models():
    with ThreadPoolExecutor() as executor:
        t5_future = executor.submit(load_t5_model)
        qa_future = executor.submit(load_qa_model)
        t5_model, t5_tokenizer = t5_future.result()
        qa_model = qa_future.result()
    return t5_model, t5_tokenizer, qa_model

# Load T5 model for summarization and key point extraction
def load_t5_model():
    model_name = "t5-large"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

# Load BERT model for QA
def load_qa_model():
    return pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Process text using T5 model for summarization or key points
def process_text_with_t5(model, tokenizer, context, task, num_points=5):
    task_prompt = {"summarize": f"summarize: {context}",
                   "keypoints": f"extract {num_points} key points: {context}"}
    prompt = task_prompt.get(task)
    if not prompt:
        return "Invalid task specified."

    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=5, length_penalty=2.0, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Answer general queries with the BERT QA model
def answer_query(qa_model, context, query):
    result = qa_model(question=query, context=context)
    return result.get('answer', 'Sorry, I could not find an answer.')

# Load models globally
t5_model, t5_tokenizer, qa_model = load_models()

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = f"/tmp/{file.filename}"
    file.save(file_path)

    # Extract text based on file type
    if file.filename.endswith(".pdf"):
        context = extract_text_from_pdf(file_path)
    elif file.filename.endswith(".docx"):
        context = extract_text_from_docx(file_path)
    else:
        return jsonify({"error": "Unsupported file format. Please upload a PDF or DOCX file."}), 400

    # Determine task
    task = request.form.get("task")
    query = request.form.get("query")

    if task == "keypoints":
        keypoints = process_text_with_t5(t5_model, t5_tokenizer, context, task="keypoints", num_points=5)
        return jsonify({"keypoints": keypoints.split("\n")})
    elif task == "summarize":
        summary = process_text_with_t5(t5_model, t5_tokenizer, context, task="summarize")
        return jsonify({"summary": summary})
    elif task == "query":
        if not query:
            return jsonify({"error": "Query is required for question-answering."}), 400
        answer = answer_query(qa_model, context, query)
        return jsonify({"answer": answer})
    else:
        return jsonify({"error": "Invalid task specified."}), 400

print(torch.__version__)

if __name__ == "__main__":
    app.run() # Run the Flask app
