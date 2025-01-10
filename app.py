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
import psutil
from typing import Optional, Tuple
import magic
import chardet

class PDFHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mime = magic.Magic(mime=True)
        
    def validate_file(self, file) -> Optional[str]:
        if not file:
            return "No file provided"
        file.seek(0)
        file_bytes = file.read(2048)
        mime_type = self.mime.from_buffer(file_bytes)
        file.seek(0)
        allowed_mimes = {
            'application/pdf': '.pdf',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx'
        }
        if mime_type not in allowed_mimes:
            return f"Invalid file type. Detected: {mime_type}"
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > 16 * 1024 * 1024:
            return "File size exceeds 16MB limit"
        return None

    def extract_text_from_pdf(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                if reader.is_encrypted:
                    return None, "PDF file is encrypted. Please provide an unencrypted PDF."
                extracted_text = []
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            encoding = chardet.detect(page_text.encode())['encoding']
                            if encoding and encoding.lower() != 'utf-8':
                                page_text = page_text.encode(encoding).decode('utf-8', errors='ignore')
                            extracted_text.append(page_text)
                    except Exception as e:
                        self.logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
                        continue
                text = "\n".join(extracted_text)
                if not text.strip():
                    return None, "No readable text found in PDF. The file might be scanned or contain only images."
                text = " ".join(line.strip() for line in text.split("\n") if line.strip())
                if len(text) < 50:
                    return None, "Extracted text is too short to be a valid resume."
                return text, None
        except Exception as e:
            error_msg = str(e)
            if "file has not been decrypted" in error_msg.lower():
                return None, "PDF file is encrypted. Please provide an unencrypted PDF."
            elif "pdf header not found" in error_msg.lower():
                return None, "Invalid or corrupted PDF file."
            else:
                self.logger.error(f"Error processing PDF: {error_msg}")
                return None, f"Error processing PDF: {error_msg}"

with open('requirements.txt', 'r') as file:
    for line in file:
        print(line.strip())

process = psutil.Process(os.getpid())
print(f"Memory used: {process.memory_info().rss / 1024 ** 2:.2f} MB")

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
    return jsonify({"error": "File size exceeds the limit (16MB)"}), 413

@app.route("/")
def index():
    return render_template("index.html")
    
def preprocess_text(text):
    if not text:
        return ""
    text = "".join(char for char in text if char.isprintable())
    text = " ".join(text.split())
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
    try:
        if not keywords or not vacancies:
            logger.error("Empty keywords or vacancies list")
            return {}

        documents = [' '.join(str(kw) for kw in keywords)] 
        
        vacancy_texts = []
        for vacancy in vacancies:
            title = vacancy.get('title', '').strip()
            snippet = vacancy.get('snippet', '').strip()
            combined_text = f"{title} {snippet}".strip()
            
            if combined_text: 
                vacancy_texts.append(combined_text)

        if not vacancy_texts:
            logger.error("No valid vacancy texts found")
            return {}

        documents.extend(vacancy_texts)

        if len(documents) < 2:  
            logger.error("Insufficient documents for comparison")
            return {}

        vectorizer = TfidfVectorizer(min_df=1)  
        try:
            tfidf_matrix = vectorizer.fit_transform(documents)
        except Exception as e:
            logger.error(f"Error in TF-IDF vectorization: {str(e)}")
            return {}
        try:
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            
            similarity_scores = {}
            for i, vacancy in enumerate(vacancies[:len(vacancy_texts)]):
                title = vacancy.get('title', f'Vacancy {i+1}')
                score = similarity_matrix[0, i] if i < similarity_matrix.shape[1] else 0.0
                similarity_scores[title] = float(score)  
                
            return similarity_scores
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return {}
            
    except Exception as e:
        logger.error(f"Unexpected error in compute_similarity_scores: {str(e)}")
        return {}

@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == "POST":
        try:
            if "resume" not in request.files:
                logger.error("No file part in the request")
                return jsonify({"error": "No file part"}), 400

            file = request.files["resume"]
            if file.filename == '':
                logger.error("No selected file")
                return jsonify({"error": "No selected file"}), 400

            pdf_handler = PDFHandler()
            error = pdf_handler.validate_file(file)
            if error:
                return jsonify({"error": error}), 400

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOADS_FOLDER"], filename)

            try:
                file.save(file_path)

                if file.filename.endswith('.pdf'):
                    text, error = pdf_handler.extract_text_from_pdf(file_path)
                    if error:
                        return jsonify({"error": error}), 400

                elif file.filename.endswith(('.doc', '.docx')):
                    try:
                        doc = Document(file_path)
                        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

                        if not text.strip():
                            return jsonify({"error": "No readable text found in document"}), 400

                    except Exception as e:
                        logger.error(f"Error processing DOC/DOCX file: {str(e)}")
                        return jsonify({"error": f"Error processing document: {str(e)}"}), 400

                processed_text = preprocess_text(text)
                if not processed_text:
                    return jsonify({"error": "No valid text content found after processing"}), 400

                text1 = extract_keywords_tfidf(processed_text)
                text2 = extract_keywords_bert(processed_text)
                keywords = list(set(text1 + text2))

                if not keywords:
                    return jsonify({"error": "Could not extract keywords from document"}), 400

                vacancies = search_vacancies(keywords)
                if isinstance(vacancies, tuple) and vacancies[1] == 500:
                    return jsonify({"error": "Error fetching vacancies"}), 500

                if not vacancies:
                    return jsonify({"error": "No matching vacancies found"}), 404

                similarity_scores = compute_similarity_scores(keywords, vacancies)

                if not similarity_scores:
                    logger.warning("Could not compute similarity scores")
                    similarity_scores = {v['title']: 0.0 for v in vacancies}

                return jsonify({
                    "keywords": keywords, 
                    "vacancies": vacancies, 
                    "similarity_scores": similarity_scores
                })

            except Exception as e:
                logger.error(f"Unexpected error processing file: {str(e)}")
                return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

        except Exception as e:
            logger.error(f"Unexpected error in upload route: {str(e)}")
            return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)

