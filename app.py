from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
import os
from werkzeug.utils import secure_filename
import re
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
import secrets
import logging
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)  # Уменьшили размер ключа
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Уменьшили максимальный размер файла до 5MB

# Настройка директории для загрузок
uploads_dir = os.path.join(os.getcwd(), "uploads")
os.makedirs(uploads_dir, exist_ok=True)
app.config["UPLOADS_FOLDER"] = uploads_dir

# Базовая настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    return 'Файл слишком большой (максимум 5MB)', 413

@app.route("/")
def index():
    return render_template("index.html")

def preprocess_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def extract_keywords_tfidf(text, top_n=10):
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")  # Уменьшили max_features
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
    return [word for word, _ in sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)[:top_n]]

def search_vacancies(keywords, top_n=5):  # Уменьшили количество вакансий
    try:
        response = requests.get(
            "https://api.hh.ru/vacancies",
            params={"text": " ".join(keywords), "per_page": top_n},
            timeout=5  # Добавили timeout
        )
        response.raise_for_status()
        return [{
            "title": v.get("name", "Без названия"),
            "url": v.get("alternate_url", ""),
            "snippet": v.get("snippet", {}).get("requirement", "Нет описания"),
            "salary": v.get("salary", {}),
            "employment_type": v.get("employment", {}).get("name", "Не указано")
        } for v in response.json().get("items", [])]
    except Exception as e:
        logger.error(f"Ошибка при поиске вакансий: {str(e)}")
        return []

def compute_similarity_scores(keywords, vacancies):
    if not vacancies:
        return {}
    
    documents = [" ".join(keywords)] + [v.get('snippet', '') for v in vacancies]
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        return {v['title']: float(similarities[0, i]) for i, v in enumerate(vacancies)}
    except Exception as e:
        logger.error(f"Ошибка при расчете схожести: {str(e)}")
        return {v['title']: 0.0 for v in vacancies}

@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method != "POST":
        return render_template("upload.html")

    if "resume" not in request.files:
        return jsonify({"error": "Файл не найден"}), 400

    file = request.files["resume"]
    if not file or file.filename == '':
        return jsonify({"error": "Файл не выбран"}), 400

    if not file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
        return jsonify({"error": "Неподдерживаемый формат файла"}), 400

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOADS_FOLDER"], filename)
        file.save(file_path)

        text = ""
        if file.filename.endswith('.pdf'):
            reader = PdfReader(file_path)
            text = " ".join(page.extract_text() for page in reader.pages)
        else:
            doc = Document(file_path)
            text = " ".join(para.text for para in doc.paragraphs)

        os.remove(file_path)  # Удаляем файл после обработки
        
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
        logger.error(f"Ошибка обработки файла: {str(e)}")
        return jsonify({"error": f"Ошибка обработки файла: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
