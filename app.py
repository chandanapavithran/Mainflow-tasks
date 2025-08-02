from flask import Flask, request, render_template, redirect
import os
import spacy
import re
from reportlab.pdfgen import canvas
from flask import send_file
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient

# Connect to MongoDB (local server or cloud)
client = MongoClient("mongodb://localhost:27017/")
db = client["resumeDB"]
collection = db["resumes"]


nlp = spacy.load("en_core_web_sm")

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        return "❌ No file part"

    file = request.files['resume']

    if file.filename == '':
        return "❌ No file selected"

    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    text = ""

    # Extract text based on file type
    if filename.lower().endswith('.pdf'):
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
            text = "\n".join(pages)

    elif filename.lower().endswith('.docx'):
        from docx import Document
        doc = Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs])

    else:
        return "❌ Unsupported file type"
    # ✅ spaCy + regex analysis
    doc = nlp(text)

    # Extract Name
    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    # Extract Email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = email_match.group(0) if email_match else "Not found"

    # Extract Phone
    phone_match = re.search(r'\+?\d[\d\s\-]{8,}', text)
    phone = phone_match.group(0) if phone_match else "Not found"
    
    # Define common skills to look for
    SKILLS = ['Python', 'Java', 'C++', 'SQL', 'HTML', 'CSS', 'JavaScript', 'React', 'Flask', 'Django', 'Machine Learning', 'Data Science']

    # Normalize text
    lower_text = text.lower()

    # Extract matching skills
    matched_skills = [skill for skill in SKILLS if skill.lower() in lower_text]

        # --- Job Matching Code ---
    jobs = [
        "Software Developer with experience in Python, Django, and SQL",
        "Frontend Developer with React and JavaScript",
        "Data Scientist with knowledge of Python, Machine Learning, and Data Analysis",
        "Web Developer with HTML, CSS, and JavaScript",
        "Java Developer with experience in Spring Boot"
    ]

    # Add resume text to the list
    job_texts = jobs + [text]

    # Vectorize
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(job_texts)

    # Cosine similarity: compare resume (last) vs jobs (all before)
    cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1])

    best_match_index = cosine_similarities.argmax()
    best_match_score = cosine_similarities[0][best_match_index]
    best_job = jobs[best_match_index]


    if not text.strip():
        return "<h3>✅ File uploaded but no text found.</h3>"
    
        # 📦 Store resume data in MongoDB
    resume_data = {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": matched_skills,
        "best_job": best_job,
        "match_score": round(float(best_match_score), 2),
        "resume_text": text
    }

    collection.insert_one(resume_data)
     # Generate PDF
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica", 12)

    y = 800
    p.drawString(100, y, "Resume Analysis Report")
    y -= 30
    p.drawString(100, y, f"Name: {name}")
    y -= 20
    p.drawString(100, y, f"Email: {email}")
    y -= 20
    p.drawString(100, y, f"Phone: {phone}")
    y -= 20
    p.drawString(100, y, f"Skills: {', '.join(matched_skills) if matched_skills else 'None found'}")
    y -= 20
    p.drawString(100, y, f"Best Job Match: {best_job}")
    y -= 20
    p.drawString(100, y, f"Match Score: {best_match_score:.2f}%")

    p.showPage()
    p.save()
    buffer.seek(0)

    from flask import Response
    ...
    pdf_data = buffer.getvalue()
    buffer.close()

    response = Response(pdf_data, mimetype='application/pdf')
    response.headers['Content-Disposition'] = 'inline; filename=Resume_Report.pdf'
    return response


if __name__ == "__main__":
    app.run(debug=True)



