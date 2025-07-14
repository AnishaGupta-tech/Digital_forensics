# 🕵️‍♀️ FORGESCAN - AI-Powered Forensic Document Analysis

![Forged vs Authentic]

---

## 🔍 Overview

**FORGESCAN** is a cutting-edge forensic document analysis system designed to detect document forgery with high precision. Leveraging advanced AI models and 17 forensic markers, FORGESCAN simulates real-world investigation techniques such as ink spectral analysis, signature stroke validation, and metadata tampering detection. The system generates a digital forensic report in real-time — compliant with ISO 27001 standards and visually rich with interpretive data.

> "The Future of Forensics is AI-driven — and FORGESCAN is leading that charge."

---

## 🎯 Key Features

| Feature | Description |
|--------|-------------|
| 🖊️ **Signature Analysis** | Validates stroke order, speed, and pressure of handwritten signatures |
| 🌈 **Ink Composition Check** | Differentiates between original and altered ink using spectral properties |
| 🧾 **Paper Quality Inspection** | Assesses texture noise, age simulation, and degradation patterns |
| 💻 **Digital Metadata Scan** | Detects manipulation through hidden metadata layers and inconsistencies |
| 🧠 **AI Verdict Generator** | Compiles all findings and gives a confidence-based forensic verdict |
| 📊 **Interactive Dashboard** | Visual confidence bars, tool metrics, and categorized analysis |

---

## 📦 Folder Structure

```bash
CASEFILE/
│
├── ai_models/                          # 🔍 Trained AI models
│   ├── signature_model.pkl             # Signature verification model
│   ├── ink_model.pkl                   # Ink pattern analysis model
│   ├── paper_model.pkl                 # Paper authenticity classifier
│   ├── metadata_model.pkl              # Metadata anomaly detector
│   ├── ela_model.pkl                   # Error Level Analysis model
│   └── README.md                       # Model training docs & metadata
│
├── case/                               # 🧪 Individual analysis modules
│   ├── ela/                            # Error Level Analysis logic
│   ├── ink/                            # Ink composition analysis logic
│   ├── metadata/                       # Metadata extraction logic
│   ├── paper/                          # Paper feature analysis logic
│   └── signature/                      # Signature analysis logic
│
├── uploads/                            # 📤 User uploaded documents/images
│   └── [uploaded_files].jpg
│
├── reports/                            # 📑 Generated forensic reports
│   └── [report_id].json
│
├── venv/                               # 🐍 Python virtual environment
│
├── static/                             # 🎨 Static assets (optional enhancement)
│   ├── images/
│   │   ├── ela.jpg
│   │   ├── org.png
│   │   ├── for.png
│   │   └── cyber-attack.png
│   └── styles/
│       └── styles.css
│
├── templates/                          # 📄 HTML UI Templates
│   ├── index.html
│   ├── analysis.html
│   ├── case.html
│   ├── register.html
│   └── report.html
│
├── database/
│   └── users.db                        # 🔐 SQLite user credentials & logs
│
├── api.py                              # 🚀 Main Flask API endpoints
├── analysis.py                         # 📊 Main logic for calling models
├── requirements.txt                    # 📦 Python dependencies
├── .gitignore                          # 🧹 Git ignore rules
├── temp.jpg                            # 🧪 Temporary file for testing
├── README.md                           # 🧾 Project overview and usage guide
└── report_generator.py (optional)     # 📄 PDF/HTML report builder script

```

---

## 🛠️ Technologies & Tools

### 🧠 Machine Learning
- **TensorFlow & Keras** — Deep learning models for signature & forgery detection
- **NumPy, Scikit-learn, Pandas** — Data wrangling and metrics processing
- **OpenCV & PIL** — Image preprocessing and enhancement

### 💻 Backend
- **Flask** — Lightweight API server
- **Werkzeug & Logging** — Secure and traceable request handling

### 🎨 Frontend
- **HTML5/CSS3** — Fully responsive layout
- **Vanilla JavaScript** — Dynamic interactions, preview rendering
- **Font Awesome** — Modern UI icons

---

## 🚀 Getting Started

### 🔧 Prerequisites
- Python 3.12.3
- pip

### 📥 Installation Steps
```bash
# Clone Repository
https://github.com/AnishaGupta-tech/FORGESCAN
cd casefile

# Install Python dependencies
pip install -r requirements.txt

# Run the Flask server
python api.py
```

### 🌐 Accessing the Frontend
Open `casefile/index.html` directly in your browser. Ensure the backend is running at `http://127.0.0.1:5000`.

---

## 📸 Complete Demo
CLICK HERE -> https://drive.google.com/drive/u/0/folders/1Gm8gIBJPa7MXcI-DMy8dkHkBuJTMWGbN


## 📊 Output & Verdict

Each document is analyzed across:
- **Signature Authenticity Score** ✅
- **Ink Pattern Consistency** 🖋️
- **Paper Noise & Degradation** 📄
- **Digital Traceability** 🔧

**Final Output:** A structured forensic verdict with explanation, confidence %, and tool-based indicators.

```yaml
Verdict: FORGED_DOCUMENT
Confidence: 92%
Indicators:
  - Ink mismatch in two signature points
  - Stroke delay inconsistent with training pattern
  - PDF metadata reveals date tampering
```

---

## 🎯 Use Cases

- 🔍 Academic simulations in **Cyber Forensics** courses
- 🏛️ Court-admissible documentation (ISO 27001 alignment)
- 🧾 Business/Legal document verification
- 📑 Training forensic analysts on document fraud detection

---

## 🚧 Future Scope

- ✅ React + Vite migration for scalable frontend
- 🔗 Blockchain-based hash stamping for audit trails
- 🔍 OCR integration for content-text vs layout detection
- 🌐 API wrapper for external enterprise integration
- 🧠 Model improvement with synthetic + multilingual datasets

---

## 👩‍💻 Author

**Anisha Gupta**  
B.Tech (CSE - AI), IGDTUW (2028) 
📧 Email: `anishagupta1100@example.com`  
🔗 [LinkedIn](https://linkedin.com/in/anisha-gupta-33582b311)  |  [GitHub](https://github.com/AnishaGupta-tech)

---

## 📃 License

This project is licensed under the **MIT License**. Use it ethically and give credit where due.

---

## 🙌 Acknowledgements

- IGDTUW Faculty for guidance
- OpenAI, TensorFlow & Keras
- Dataset curators for public signature and document forgery repositories
- Cyber Forensics community

---

> "Every pixel leaves a trace. FORGESCAN follows them all."
