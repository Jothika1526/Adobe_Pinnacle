# predict.py
import os
import json
import pandas as pd
import numpy as np
import joblib
import fitz # PyMuPDF
import spacy
from collections import Counter # Import Counter for font frequency

# --- Global SpaCy NLP Model Loading ---
# Load the pre-downloaded spaCy model for POS tagging.
# This ensures it's loaded only once when the script starts.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. "
          "Please ensure it's downloaded and available within the Docker image. "
          "During local testing, run 'python -m spacy download en_core_web_sm'.")
    # For Docker, this needs to be part of the build process.
    exit(1) # Exit if model is not found, as linguistic features won't work.

# --- Feature Extraction Functions (Adapted from your feature_extractor.py) ---

def extract_text_and_layout(pdf_path):
    """
    Extracts text, page number, bounding box, font size, and bold status
    for each **visual line** in a PDF document, treating each line as a single span.
    This approach aims to provide more robust, logical text units by merging
    all sub-spans within a single visual line.
    """
    try:
        document = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return []

    extracted_spans = []

    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        
        # Capture page dimensions here for normalized features later
        page_width = page.rect.width
        page_height = page.rect.height

        # Sort blocks for consistent processing order
        sorted_blocks = sorted(blocks, key=lambda b: (b['bbox'][1], b['bbox'][0]))

        for block in sorted_blocks: # Use sorted_blocks here
            if block['type'] == 0:  # Text block
                # Sort lines within a block
                sorted_lines = sorted(block["lines"], key=lambda l: l['bbox'][1])
                for line in sorted_lines:
                    line_text = ""
                    max_font_size_in_line = 0.0
                    is_line_bold = False

                    # Sort spans within a line by their horizontal position
                    sorted_line_spans = sorted(line["spans"], key=lambda s: s['bbox'][0])

                    # Iterate through all raw spans within the current line to consolidate them
                    for i, s in enumerate(sorted_line_spans):
                        span_text = s['text']
                        
                        # Add space between spans if necessary
                        if i > 0 and line_text and not line_text.endswith((' ', '\n')) and not span_text.startswith(' '):
                            line_text += " "
                        line_text += span_text 
                        
                        # Find the maximum font size in the line
                        if s['size'] > max_font_size_in_line:
                            max_font_size_in_line = s['size']
                        
                        # Check if ANY span in the line is bold using the font name logic
                        if not is_line_bold: # Only check if not already determined as bold
                            font_name = s['font']
                            if ("bold" in font_name.lower() or 
                                "black" in font_name.lower() or
                                "heavy" in font_name.lower() or
                                "demi" in font_name.lower() or
                                "extrabold" in font_name.lower()):
                                is_line_bold = True

                    full_line_text = line_text.strip()
                    
                    if not full_line_text: # Skip if the consolidated line text is empty
                        continue

                    # The bbox for the consolidated line span is the bbox of the line itself
                    line_bbox = line["bbox"]

                    extracted_spans.append({
                        'text': full_line_text,
                        'page_number': page_num + 1, # PyMuPDF pages are 0-indexed
                        'font_size': max_font_size_in_line, # Use the largest font size found in the line
                        'is_bold': is_line_bold, # True if any part of the line was bold
                        'bbox': {
                            'x0': float(line_bbox[0]), 'y0': float(line_bbox[1]),
                            'x1': float(line_bbox[2]), 'y1': float(line_bbox[3])
                        },
                        'page_width': page_width, # Added for normalized features
                        'page_height': page_height, # Added for normalized features
                        'features': {} # Placeholder for computed features
                    })
    document.close()
    return extracted_spans

def compute_features(span_data_list):
    """
    Computes all specified features for each text span.
    Requires the full list of spans from a document to compute document-level features like Font Threshold Flag.
    Uses the globally loaded nlp object for spaCy processing.
    """
    if not nlp:
        print("SpaCy model not loaded. Cannot compute linguistic features.")
        return span_data_list

    if not span_data_list:
        return []

    # --- Document-level calculation for Font Threshold Flag ---
    font_sizes = [span['font_size'] for span in span_data_list if span['font_size'] is not None]
    most_frequent_font_size = None
    if font_sizes:
        font_size_counts = Counter(font_sizes)
        most_frequent_font_size = font_size_counts.most_common(1)[0][0]

    # Pre-calculate adjacent span features (Space Above/Below, Font/Bold Change)
    # This loop populates the 'features' dictionary for these specific features
    for i, span in enumerate(span_data_list):
        if 'features' not in span:
            span['features'] = {}

        # Initialize with default values for safety
        span['features']['Space Above'] = 0.0
        span['features']['Font Size Change From Previous'] = 0.0
        span['features']['Bold Status Change From Previous'] = False
        span['features']['Is Previous Line Empty'] = False
        span['features']['Space Below'] = 0.0
        span['features']['Font Size Change To Next'] = 0.0
        span['features']['Bold Status Change To Next'] = False
        span['features']['Is Next Line Empty'] = False

        # Previous span features
        if i > 0:
            prev_span = span_data_list[i-1]
            span['features']['Space Above'] = span['bbox']['y0'] - prev_span['bbox']['y1']
            
            if prev_span['font_size'] is not None and span['font_size'] is not None:
                span['features']['Font Size Change From Previous'] = span['font_size'] - prev_span['font_size']
            else:
                span['features']['Font Size Change From Previous'] = 0.0

            span['features']['Bold Status Change From Previous'] = (span['is_bold'] != prev_span['is_bold'])

            if span['font_size'] is not None:
                if span['features']['Space Above'] > span['font_size'] * 1.5:
                    span['features']['Is Previous Line Empty'] = True
                else:
                    span['features']['Is Previous Line Empty'] = False
            else:
                span['features']['Is Previous Line Empty'] = span['features']['Space Above'] > 18
        
        # Next span features
        if i < len(span_data_list) - 1:
            next_span = span_data_list[i+1]
            span['features']['Space Below'] = next_span['bbox']['y0'] - span['bbox']['y1']

            if next_span['font_size'] is not None and span['font_size'] is not None:
                span['features']['Font Size Change To Next'] = next_span['font_size'] - span['font_size']
            else:
                span['features']['Font Size Change To Next'] = 0.0

            span['features']['Bold Status Change To Next'] = (span['is_bold'] != next_span['is_bold'])

            if span['font_size'] is not None:
                if span['features']['Space Below'] > span['font_size'] * 1.5:
                    span['features']['Is Next Line Empty'] = True
                else:
                    span['features']['Is Next Line Empty'] = False
            else:
                span['features']['Is Next Line Empty'] = span['features']['Space Below'] > 18

    processed_spans = []
    for span in span_data_list:
        text = span["text"]
        
        if 'features' not in span:
            span['features'] = {}

        # Basic Features
        span['features']["Characters"] = len(text)
        span['features']["Words"] = len(text.split()) if text.strip() else 0
        span['features']["is_bold"] = span['is_bold']

        if text.islower():
            span['features']["Text Case"] = 0
        elif text.isupper():
            span['features']["Text Case"] = 1
        elif text.istitle():
            span['features']["Text Case"] = 2
        else:
            span['features']["Text Case"] = 3
        
        span['features']["Bold or Not"] = 1 if span["is_bold"] else 0

        if most_frequent_font_size is not None and span['font_size'] is not None:
            span['features']["Font Threshold Flag"] = 1 if span["font_size"] > most_frequent_font_size else 0
        else:
            span['features']["Font Threshold Flag"] = 0

        # Features From Parts of Speech (POS) Tagging
        doc = nlp(text)

        pos_counts = {
            "Verbs": 0, "Nouns": 0, "Adjectives": 0, "Adverbs": 0, "Pronouns": 0,
            "Cardinal Numbers": 0, "Coordinating Conjunctions": 0,
            "Predeterminers": 0, "Interjections": 0
        }

        for token in doc:
            if token.pos_ == "VERB": pos_counts["Verbs"] += 1
            elif token.pos_ in ["NOUN", "PROPN"]: pos_counts["Nouns"] += 1
            elif token.pos_ == "ADJ": pos_counts["Adjectives"] += 1
            elif token.pos_ == "ADV": pos_counts["Adverbs"] += 1
            elif token.pos_ == "PRON": pos_counts["Pronouns"] += 1
            elif token.pos_ == "NUM": pos_counts["Cardinal Numbers"] += 1
            elif token.pos_ == "CCONJ": pos_counts["Coordinating Conjunctions"] += 1
            elif token.pos_ == "DET" and token.dep_ == "predet": pos_counts["Predeterminers"] += 1
            elif token.pos_ == "INTJ": pos_counts["Interjections"] += 1
        
        span['features'].update(pos_counts)

        # Additional Layout Features
        page_width = span.get('page_width', 1)
        page_height = span.get('page_height', 1)
        
        x0, y0, x1, y1 = span['bbox']['x0'], span['bbox']['y0'], span['bbox']['x1'], span['bbox']['y1']

        span['features']["Left Margin Normalized"] = x0 / page_width
        span['features']["Top Margin Normalized"] = y0 / page_height
        span['features']["Width"] = x1 - x0
        span['features']["Height"] = y1 - y0
        span['features']["Aspect Ratio"] = span['features']["Width"] / span['features']["Height"] if span['features']["Height"] else 0
        
        bbox_area = span['features']["Width"] * span['features']["Height"]
        span['features']["Character Density"] = span['features']["Characters"] / bbox_area if bbox_area else 0

        span['features']["All Uppercase Ratio"] = sum(1 for char in text if char.isupper()) / len(text) if len(text) else 0

        span['features']["Ends With Punctuation"] = text.strip().endswith(('.', '!', '?', ':', ';', ','))

        span['features']["Is Numeric"] = text.replace('.', '', 1).isdigit() or text.replace(',', '', 1).isdigit()
        
        # Add remaining features that were in your training script but missing in feature_extractor
        # These need to be properly implemented in extract_text_and_layout or compute_features
        # based on how you calculate them from the PDF.
        # It's critical that these match the features used for training.
        span['features']["Normalized Vertical Position"] = y0 / page_height # Placeholder, you might have a more nuanced calculation
        span['features']["Relative Font Size"] = span['font_size'] / most_frequent_font_size if most_frequent_font_size else 1.0
        span['features']["Space Above Ratio"] = span['features']['Space Above'] / span['font_size'] if span['font_size'] else 0.0
        span['features']["Space Below Ratio"] = span['features']['Space Below'] / span['font_size'] if span['font_size'] else 0.0
        span['features']["Line Height Ratio"] = (span['features']['Height'] / span['font_size']) if span['font_size'] else 1.0
        span['features']["Starts with Roman Numeral"] = bool(text and text.strip().lower().startswith(('i.', 'ii.', 'iii.', 'iv.', 'v.', 'vi.', 'vii.', 'viii.', 'ix.', 'x.')))
        span['features']["Starts with Alphabet Numeral"] = bool(text and text.strip().lower().startswith(('a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.', 'h.', 'i.', 'j.', 'k.', 'l.', 'm.', 'n.', 'o.', 'p.', 'q.', 'r.', 's.', 't.', 'u.', 'v.', 'w.', 'x.', 'y.', 'z.')))
        span['features']["Starts with Arabic Numeral"] = bool(text and text.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.')))
        span['features']["Line is Centered"] = abs((x0 + x1) / 2 - page_width / 2) < (page_width * 0.05) # Heuristic for centered
        span['features']["Indentation Level"] = x0 / page_width # Simple heuristic: normalized x0
        span['features']["Word Character Ratio"] = span['features']["Words"] / span['features']["Characters"] if span['features']["Characters"] else 0.0
        span['features']["Ends With Colon"] = text.strip().endswith(':')
        span['features']["Ends With Question Mark"] = text.strip().endswith('?')
        span['features']["Ends With Period"] = text.strip().endswith('.')

        processed_spans.append(span)

    return processed_spans

# --- Global Model Loading ---
# Load models and scalers/encoders once when the script starts
try:
    is_heading_model = joblib.load('is_heading_classifier_model.pkl')
    is_heading_scaler = joblib.load('is_heading_scaler.pkl')
    is_heading_label_encoder = joblib.load('is_heading_label_encoder.pkl')
    
    heading_level_model = joblib.load('heading_level_classifier_model.pkl')
    heading_level_scaler = joblib.load('heading_level_scaler.pkl')
    heading_level_label_encoder = joblib.load('heading_level_label_encoder.pkl')

except FileNotFoundError as e:
    print(f"Error: Missing model/scaler/encoder file: {e}. "
          "Please ensure all .pkl files from your training script are in the same directory as predict.py.")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred while loading models: {e}")
    exit(1)


# List of all feature keys (MUST match the keys used during training)
FEATURE_KEYS = [
    "Space Above", "Font Size Change From Previous", "Bold Status Change From Previous",
    "Is Previous Line Empty", "Space Below", "Font Size Change To Next",
    "Bold Status Change To Next", "Is Next Line Empty", "Characters", "Words",
    "is_bold", "Text Case", "Bold or Not", "Font Threshold Flag",
    "Verbs", "Nouns", "Adjectives", "Adverbs", "Pronouns", "Cardinal Numbers",
    "Coordinating Conjunctions", "Predeterminers", "Interjections",
    "Left Margin Normalized", "Top Margin Normalized", "Width", "Height",
    "Aspect Ratio", "Character Density", "All Uppercase Ratio",
    "Ends With Punctuation", "Is Numeric",
    "Normalized Vertical Position", "Relative Font Size", "Space Above Ratio",
    "Space Below Ratio", "Line Height Ratio", "Starts with Roman Numeral",
    "Starts with Alphabet Numeral", "Starts with Arabic Numeral",
    "Line is Centered", "Indentation Level", "Word Character Ratio",
    "Ends With Colon", "Ends With Question Mark", "Ends With Period"
]

# Columns that represent boolean-like features that should be treated as integers (0 or 1)
BOOL_COLS_TO_INT = [
    'is_bold', 'Bold Status Change From Previous', 'Bold Status Change To Next',
    'Is Previous Line Empty', 'Is Next Line Empty', 'Ends With Punctuation', 'Is Numeric',
    'Starts with Roman Numeral', 'Starts with Alphabet Numeral', 'Starts with Arabic Numeral',
    'Line is Centered', 'Ends With Colon', 'Ends With Question Mark', 'Ends With Period'
]


def predict_document_outline(pdf_path):
    """
    Processes a single PDF, extracts features, applies the trained models,
    and returns the structured outline in the specified JSON format.
    """
    
    # Step 1: Extract text and basic layout info from PDF
    raw_spans = extract_text_and_layout(pdf_path)
    if not raw_spans:
        return {"title": "", "outline": []}

    # Step 2: Compute rich features for each span
    featured_spans = compute_features(raw_spans)

    # Prepare features for prediction
    features_data = []
    for span in featured_spans:
        current_features = []
        for key in FEATURE_KEYS:
            feature_val = span['features'].get(key)
            if isinstance(feature_val, bool):
                current_features.append(int(feature_val))
            elif feature_val is None:
                current_features.append(0) # Impute None with 0 as done during training
            else:
                current_features.append(feature_val)
        features_data.append(current_features)

    if not features_data:
        return {"title": "", "outline": []}

    X = pd.DataFrame(features_data, columns=FEATURE_KEYS)
    
    # Ensure boolean-like columns are integers, consistent with training
    for col in BOOL_COLS_TO_INT:
        if col in X.columns:
            X[col] = X[col].astype(int)

    # Impute any remaining NaN values with mean, consistent with training
    X = X.fillna(X.mean(numeric_only=True))

    # Step 3: Predict is_heading using Model 1
    X_bin_scaled = is_heading_scaler.transform(X)
    is_heading_predictions_encoded = is_heading_model.predict(X_bin_scaled)
    is_heading_predictions = is_heading_label_encoder.inverse_transform(is_heading_predictions_encoded)
    
    # Use the predictions directly
    is_heading_status = is_heading_predictions

    document_outline = []
    document_title = ""
    title_found = False

    # Filter for headings to apply Model 2
    heading_spans_indices = np.where(is_heading_status)[0]

    if len(heading_spans_indices) > 0:
        X_headings = X.iloc[heading_spans_indices]
        
        # Step 4: Predict heading level using Model 2 for identified headings
        X_level_scaled = heading_level_scaler.transform(X_headings)
        heading_level_predictions_encoded = heading_level_model.predict(X_level_scaled)
        heading_level_predictions = heading_level_label_encoder.inverse_transform(heading_level_predictions_encoded)

        # Iterate through the original spans and assign predicted levels
        level_idx = 0
        for i, span in enumerate(featured_spans):
            if is_heading_status[i]: # If Model 1 predicted it's a heading
                predicted_level = heading_level_predictions[level_idx]
                level_idx += 1
                
                # Check for Title
                if predicted_level == "Title" and not title_found:
                    document_title = span['text']
                    title_found = True
                elif predicted_level in ["H1", "H2", "H3"]:
                    document_outline.append({
                        "level": predicted_level,
                        "text": span['text'],
                        "page": span['page_number']
                    })
    return {
        "title": document_title,
        "outline": document_outline
    }

# --- Main execution block for Docker container ---
if __name__ == "__main__":
    # >>> TEMPORARY LOCAL TESTING PATHS - REMEMBER TO CHANGE BACK FOR DOCKER! <<<
    INPUT_DIR = "/app/input"   # Change to "/app/input" for Docker submission
    OUTPUT_DIR = "/app/output"  # Change to "/app/output" for Docker submission
    # >>> TEMPORARY LOCAL TESTING PATHS - REMEMBER TO CHANGE BACK FOR DOCKER! <<<


    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files_to_process = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]

    if not pdf_files_to_process:
        print(f"No PDF files found in {INPUT_DIR}. Exiting.")
    else:
        for pdf_filename in pdf_files_to_process:
            pdf_path = os.path.join(INPUT_DIR, pdf_filename)
            output_json_filename = pdf_filename.replace(".pdf", ".json")
            output_json_path = os.path.join(OUTPUT_DIR, output_json_filename)

            try:
                result_json = predict_document_outline(pdf_path)
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_json, f, ensure_ascii=False, indent=4)
                print(f"Successfully processed {pdf_filename} and saved output to {output_json_path}")
            except Exception as e:
                print(f"Error processing {pdf_filename}: {e}")
                # Optionally, write an empty or error JSON to output for failed files
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump({"title": "", "outline": []}, f, ensure_ascii=False, indent=4) # Write empty JSON on error