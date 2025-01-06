from flask import Flask, render_template, redirect, request, url_for, jsonify, session, flash
from werkzeug.utils import secure_filename
import os
from utils.text_processing import preprocess_text, extract_keywords_tfidf
from utils.vacancy_search import search_vacancies, compute_similarity_scores
from utils.file_processing import process_file

app = Flask(__name__)

secret_key = os.environ.get('SECRET_KEY', 'default-secret-key')
app.config['SECRET_KEY'] = secret_key
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

uploads_dir = os.path.join(os.getcwd(), "uploads")
os.makedirs(uploads_dir, exist_ok=True)
app.config["UPLOADS_FOLDER"] = uploads_dir

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == "POST":
        if "resume" not in request.files:
            return "No file part", 400

        file = request.files["resume"]
        if file.filename == '':
            return "No selected file", 400

        if not file.filename.lower().endswith(('.pdf', '.doc', 'docx')):
            return "File is not PDF, DOC, DOCX", 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOADS_FOLDER"], filename)
        file.save(file_path)

        try:
            text = process_file(file_path)
            text = preprocess_text(text)
            keywords = extract_keywords_tfidf(text)
            vacancies = search_vacancies(keywords)
            similarity_scores = compute_similarity_scores(keywords, vacancies)

            return jsonify({
                "keywords": keywords,
                "vacancies": vacancies,
                "similarity_scores": similarity_scores
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template("upload.html")

if __name__ == "__main__":
    app.run()

# utils/text_processing.py
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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

# utils/vacancy_search.py
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def search_vacancies(keywords, top_n=10):
    query = " ".join(keywords)
    try:
        response = requests.get("https://api.hh.ru/vacancies", 
                              params={"text": query, "per_page": top_n})
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
        return f"Error fetching vacancies: {str(e)}", 500

def compute_similarity_scores(keywords, vacancies):
    documents = [str(keywords)]
    for vacancy in vacancies:
        snippet = vacancy.get('snippet', '')
        documents.append(str(snippet))

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    similarity_scores = {vacancies[i]['title']: similarity_matrix[0, i] 
                        for i in range(len(vacancies))}
    return similarity_scores

# utils/file_processing.py
from PyPDF2 import PdfReader
from docx import Document

def process_file(file_path):
    if file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    elif file_path.endswith(('.doc', '.docx')):
        doc = Document(file_path)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        text = " ".join(text)
    return text
    
