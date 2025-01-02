from flask import Flask, render_template, redirect, request, url_for, jsonify, session, flash
from PyPDF2 import PdfReader
import os
from werkzeug.utils import secure_filename
import re
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import io
import logging
from werkzeug.exceptions import RequestEntityTooLarge
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
import secrets

app = Flask(__name__)

secret_key = secrets.token_hex(32)

app.config['SECRET_KEY'] = secret_key
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

uploads_dir = os.path.join(os.getcwd(), "uploads")
os.makedirs(uploads_dir, exist_ok=True)
app.config["UPLOADS_FOLDER"] = uploads_dir

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    return 'File is too large', 413

@app.route("/")
def index():
    return render_template("index.html")

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.lower()  
    return text

def extract_keywords_tfidf(text, top_n=10):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
    tfidf_features = vectorizer.get_feature_names_out()
    keywords = sorted(zip(tfidf_features, tfidf_scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [word for word, score in keywords]

def extract_keywords_bert(text):
    ner_results = nlp(text)
    keywords = [result["word"] for result in ner_results if result["entity"].startswith("I-")]
    return list(set(keywords))

def search_vacancies(keywords, top_n=10):
    query = " ".join(keywords)
    try:
        response = requests.get("https://api.hh.ru/vacancies", params={"text": query, "per_page": top_n})
        response.raise_for_status()  
        vacancies = response.json().get("items", [])
        return [
            {
                "title": vacancy.get("name", "No title"),
                "url": vacancy.get("alternate_url", ""),
                "snippet": vacancy.get("snippet", {}).get("requirement", "No description"),
                "salary": vacancy.get("salary", {}),
                "employment_type": vacancy.get("employment", {}).get("name", "Не указано")
            }
            for vacancy in vacancies
        ]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching vacancies: {str(e)}")
        return f"Error fetching vacancies: {str(e)}", 500

def compute_similarity_scores(keywords, vacancies):
    """
    Считает схожесть между ключевыми словами и текстами вакансий.
    """
    # Преобразование входных данных в текст
    documents = [str(keywords)]  # ключевые слова добавляем первым элементом, приводим к строке
    
    # Добавляем вакансии, приводя их snippet к строкам
    for vacancy in vacancies:
        snippet = vacancy.get('snippet', '')
        documents.append(str(snippet))  # приводим snippet к строке

    # Векторизация документов
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Расчёт схожести
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    similarity_scores = {vacancies[i]['title']: similarity_matrix[0, i] for i in range(len(vacancies))}

    return similarity_scores

@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == "POST":
        if "resume" not in request.files:
            logger.error("No file part in the request")
            return "No file part", 400

        file = request.files["resume"]

        if file.filename == '':
            logger.error("No selected file")
            return "No selected file", 400

        if not file.filename.lower().endswith(('.pdf', '.doc', 'docx')):
            logger.error(f"Invalid file type: {file.filename}")
            return "File is not PDF, DOC, DOCX", 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOADS_FOLDER"], filename)
        file.save(file_path)

        try:
            if file.filename.endswith('.pdf'):
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                text = preprocess_text(text)
            elif file.filename.endswith(('.doc', '.docx')):
                doc = Document(file_path)
                text = []
                for para in doc.paragraphs:
                    text.append(para.text)
                text = preprocess_text(text)
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500

        text1 = extract_keywords_tfidf(text)
        text2 = extract_keywords_bert(text)
        keywords = list(set(text1 + text2))

        vacancies = search_vacancies(keywords)
        similarity_scores = compute_similarity_scores(keywords, vacancies)

        return jsonify({"keywords": keywords, "vacancies": vacancies, "similarity_scores": similarity_scores})
    
    return render_template("upload.html")



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Читаем порт из переменной окружения PORT или используем 10000 по умолчанию
    app.run(host="0.0.0.0", port=port)
