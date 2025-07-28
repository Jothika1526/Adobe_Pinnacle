#  Adobe Pinnacle AI Hackathon - Round 1a Solution 

## 📄 Project Description
This solution accurately identifies and hierarchically classifies headings within diverse PDF documents, generating a structured JSON output. It's a robust, offline-capable system packaged in a Docker container, adhering to Adobe Pinnacle AI Hackathon Round 1a requirements.

## 📁 Folder Structure
All core solution files for Round 1a are meticulously organized within the `solution_round1a/` directory.

```
.
├── .gitignore
├── README.md
└── solution_round1a/
    ├── Dockerfile
    ├── requirements.txt
    ├── predict.py
    ├── heading_level_classifier_model.pkl
    ├── heading_level_label_encoder.pkl
    ├── heading_level_scaler.pkl
    ├── is_heading_classifier_model.pkl
    ├── is_heading_label_encoder.pkl
    ├── is_heading_scaler.pkl
    ├── input/  # Contains sample input PDFs (e.g., round1.pdf)
    └── output/ # Contains generated JSON output for sample inputs (e.g., round1.json)
```

## 💡 Solution Approach

Our solution employs a comprehensive, multi-stage machine learning pipeline for high-accuracy heading detection and classification:

1.  **📊 Data Collection:** Curated over 2000 diverse PDF documents via a custom search API to build a robust training dataset, ensuring broad generalization across document layouts.

2.  **⚙️ Feature Extraction (`feature_extractor.py`):**
    * Uses `PyMuPDF` for granular text and layout extraction (bounding boxes, font sizes, bold status).
    * Computes rich numerical features: positional, typographical, contextual, Part-of-Speech (POS) tagging with `spaCy` for linguistic insights, and pattern-based features.

3.  **🎯 Ground Truth & Labeling:** Leveraged **Adobe's Extract PDF API** for initial structural data, followed by meticulous manual labeling to create precise `is_heading` and `level` (Title, H1, H2, H3) annotations.

4.  **⚖️ Data Preprocessing & Balancing (`lbm.ipynb`):** Prepares data by imputing missing values, converting booleans, and addressing class imbalance. Employs **undersampling** for over-represented classes and **SMOTE** for oversampling under-represented heading classes (H3, Title). Features are scaled using `StandardScaler`.

5.  **🧠 Two-Stage LightGBM Training (`lbm.ipynb`):**
    * **Model 1 (`is_heading_classifier_model.pkl`):** Binary classifier for heading detection.
    * **Model 2 (`heading_level_classifier_model.pkl`):** Multi-class classifier for hierarchical level prediction (Title, H1, H2, H3).
    * Both models use `early_stopping` for optimal performance.

6.  **🚀 Inference (`predict.py`):** Orchestrates the pipeline for new PDFs: performs feature extraction, applies both trained LightGBM models sequentially, and compiles the final structured JSON output.

## 🛠️ Models and Libraries Used
This solution leverages powerful Python libraries and pre-trained machine learning models:

* **Core Libraries:**
    * `PyMuPDF==1.26.3`: PDF parsing and text/layout extraction.
    * `spaCy==3.8.7`: Natural Language Processing (POS tagging).
    * `numpy`, `scipy`, `pandas`: Core libraries for data handling and numerical operations.
    * `scikit-learn`: Machine learning utilities (preprocessing, evaluation).
    * `joblib`: Model serialization/deserialization.

* **Machine Learning Models:**
    * `lightgbm==4.6.0`: Primary ML framework for efficient tabular data classification.
    * **Auxiliary Assets:** `label_encoder.pkl` files (for categorical data transformation) and `scaler.pkl` files (for numerical data normalization).

## 💻 Requirements
* **Docker Desktop:** Recommended version 4.x or later (compatible with Linux containers, using Docker Engine 24.x or later).
* **Python:** Version 3.10 (as specified in the Dockerfile).
* **Required Python Packages:** These are automatically installed via `requirements.txt` during the Docker build.

## 🚀 Execution Instructions
These instructions are designed for execution in a Unix-like shell environment (Linux, macOS, or Git Bash on Windows), which is typical for Docker workflows.

### 1. Clone the Repository
Open your terminal  and clone the project:

```bash
git clone https://github.com/Jothika1526/Adobe_Pinnacle.git
```
### 2. Navigate to the Solution Directory
Change your current directory to the `solution_round1a` folder within the cloned repository:

```bash
cd Adobe_Pinnacle/solution_round1a
```

### 3. Build the Docker Image

```bash
docker build --platform linux/amd64 -t my_heading_extractor:latest .
```

### 4. Prepare Input Files
Ensure your input PDF files (e.g., `round1.pdf`) are placed inside the `input/` directory within the `solution_round1a` folder.

### 5. Run the Container

```bash
docker run --rm -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" --network none my_heading_extractor:latest 
```

### 📋 Expected Container Behavior:
* The container will automatically process all PDF files found in the `/app/input` directory.
* For each `filename.pdf` in `/app/input`, a corresponding `filename.json` will be generated and saved into the `/app/output` directory.
* The container will run without any network access.
* Upon completion, the container will automatically remove itself (`--rm`).

## 📤 Expected Output Format
For each `filename.pdf` in the input, a `filename.json` file will be created in the `output/` directory. The structure of the JSON will contain extracted headings and their predicted levels.
