# Adobe Pinnacle AI Hackathon - Round 1a Solution

## Project Description
This solution is designed to process PDF documents to identify headings and their hierarchical levels, generating a structured JSON output for each processed PDF. It adheres to the specific requirements of Adobe Hackathon Round 1a, including offline execution within a Docker container.

## Folder Structure
All core solution files for Round 1a are located within the `solution_round1a/` directory.

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
1.  **Extensive Data Collection:** A robust training dataset of over 2000 diverse PDF documents was curated using a custom search API. This broad collection ensures the models are trained on a wide variety of real-world document layouts and styles, enhancing generalization.

2.  **Advanced Feature Extraction (`feature_extractor.py`):**
    * Utilizes `PyMuPDF` for granular text and layout extraction, capturing text content, bounding boxes, font sizes, and bold statuses for each span.
    * Computes a rich set of numerical features for each span, including:
        * **Positional & Dimensional:** Normalized margins, width, height, aspect ratio.
        * **Typographical:** Font size, bold status, font threshold flags, text case.
        * **Contextual Layout:** Space above/below, font/bold status changes relative to adjacent lines, line emptiness, and indentation.
        * **Part-of-Speech (POS) Tagging:** Employs `spaCy` to assign grammatical categories (e.g., noun, verb) to words, deriving features like counts of specific POS tags. This provides crucial linguistic insights.
        * **Other Linguistic & Pattern-Based:** Character/word counts, character density, capitalization ratios, and common heading patterns (e.g., starts/ends with specific characters/numerals).

3.  **Ground Truth Generation and Labeling:** Initial structural information was obtained by leveraging **Adobe's Extract PDF API**. This high-quality output was then meticulously refined through manual labeling to create precise `is_heading` (boolean) and `level` (Title, H1, H2, H3) annotations for the training dataset.

4.  **Data Preprocessing & Balancing (`lbm.ipynb`):**
    * Prepares the dataset by imputing missing numerical values and converting boolean features.
    * Addresses class imbalance, which is critical for robust model performance. This involves **undersampling** over-represented classes (e.g., non-headings, common heading levels) and intelligently **oversampling** under-represented heading classes (e.g., H3, Title) using techniques like **SMOTE (Synthetic Minority Over-sampling Technique)**.
    * All numerical features are scaled using `StandardScaler` to ensure uniform contribution during model training.

5.  **Two-Stage LightGBM Model Training (`lbm.ipynb`):** Two separate, highly efficient LightGBM (Light Gradient Boosting Machine) models are trained:
    * **Model 1 (`is_heading_classifier_model.pkl`):** A binary classifier that determines whether a given text span is a heading or not.
    * **Model 2 (`heading_level_classifier_model.pkl`):** A multi-class classifier that predicts the precise hierarchical level (Title, H1, H2, H3) for text spans identified as headings by Model 1.
    * Both models incorporate `early_stopping` during training to prevent overfitting and optimize generalization performance.

6.  **Inference (`predict.py`):** The `predict.py` script orchestrates the entire prediction pipeline. It takes a new input PDF, performs the identical feature extraction as used during training, and then passes these features sequentially through the trained `is_heading_classifier_model` and `heading_level_classifier_model`. Finally, it compiles the extracted and classified headings into the specified structured JSON output.

This sophisticated pipeline, from diverse data acquisition and rich feature engineering to a carefully balanced, two-stage machine learning classification, ensures a highly accurate, efficient, and adaptable solution for structured PDF outline extraction.



## Requirements
* **Docker Desktop:** Recommended version 4.x or later (compatible with Linux containers, using Docker Engine 24.x or later).
* **Python:** Version 3.10 (as specified in the Dockerfile).
* **Required Python Packages:** These are automatically installed via `solution_round1a/requirement.txt` during the Docker build.

## Execution Instructions (for Adobe Judges)

### 1. Build the Docker Image
Navigate to the root directory of this repository (where `README.md` and `solution_round1a` are located) and execute the following command:

```bash
docker build --platform linux/amd64 -t my_heading_extractor:latest .
```

### 2. Run the Container
After successfully building the image, ensure your input PDF files are placed in the `input/` directory . Then, run the solution using the following command:

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none my_heading_extractor:latest
```

### Expected Container Behavior:
* The container will automatically process all PDF files found in the `/app/input` directory.
* For each `filename.pdf` in `/app/input`, a corresponding `filename.json` will be generated and saved into the `/app/output` directory.
* The container will run without any network access.

## Expected Output Format
For each `filename.pdf` in the input, a `filename.json` file will be created in the `output/` directory. The structure of the JSON will contain extracted headings and their predicted levels.
