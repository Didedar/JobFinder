from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
import os
from werkzeug.utils import secure_filename
import re
import requests
from collections import Counter
from docx import Document
import secrets
import logging
from werkzeug.exceptions import RequestEntityTooLarge
from math import sqrt

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

uploads_dir = os.path.join(os.getcwd(), "uploads")
os.makedirs(uploads_dir, exist_ok=True)
app.config["UPLOADS_FOLDER"] = uploads_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STOP_WORDS = {'and', 'the', 'or', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

def handle_file_too_large(error):
    return 'Файл слишком большой (максимум 5MB)', 413

@app.route("/")
def index():
    return render_template("index.html")

def preprocess_text(text):
    # Простая предобработка текста
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]

def extract_keywords(text, top_n=10):
    # Простое извлечение ключевых слов на основе частоты
    words = preprocess_text(text)
    word_freq = Counter(words)
    return [word for word, _ in word_freq.most_common(top_n)]

def cosine_similarity_simple(text1, text2):
    # Простая реализация косинусного сходства
    words1 = set(preprocess_text(text1))
    words2 = set(preprocess_text(text2))
    
    intersection = words1.intersection(words2)
    
    if not words1 or not words2:
        return 0.0
        
    similarity = len(intersection) / (sqrt(len(words1)) * sqrt(len(words2)))
    return similarity

def search_vacancies(keywords, top_n=5):
    try:
        response = requests.get(
            "https://api.hh.ru/vacancies",
            params={"text": " ".join(keywords), "per_page": top_n},
            timeout=5
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
    
    keyword_text = " ".join(keywords)
    return {
        v['title']: cosine_similarity_simple(keyword_text, v.get('snippet', ''))
        for v in vacancies
    }

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
        
        keywords = extract_keywords(text)
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
