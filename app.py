import os
import logging
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

app = Flask(__name__, static_folder='.', static_url_path='')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
MODEL_DIR = './fine_tuned_model'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class CategoryClassifier:
    def __init__(self, model_path=None):
        """Initialize the Text Category Classifier"""
        self.categories = ['legal', 'business', 'medical', 'sports', 'entertainment']
        
        if model_path and os.path.exists(model_path):
            # Load fine-tuned model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            logging.info(f"Loaded fine-tuned model from {model_path}")
        else:
            # Load base model
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = AutoModelForSequenceClassification.from_pretrained(
                'distilbert-base-uncased', 
                num_labels=len(self.categories)
            )
            logging.info("Loaded base model (not fine-tuned)")
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def classify_text(self, text, max_length=512):
        """
        Classify input text into predefined categories
        
        Args:
            text (str): Input text to classify
            max_length (int): Maximum token length for processing
        
        Returns:
            dict: Classification results with probabilities
        """
        # Truncate text if too long
        text = text[:5000]  # Limit to 5000 characters
        
        # Tokenize the input text
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=max_length, 
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        probabilities = probabilities.cpu().numpy()
        
        # Create probability dictionary
        prob_dict = {}
        for i, category in enumerate(self.categories):
            prob_dict[category] = float(probabilities[0][i])
        
        # Get predicted category and confidence
        predicted_category = self.categories[np.argmax(probabilities)]
        confidence = float(np.max(probabilities))
        
        return {
            'category': predicted_category,
            'confidence': confidence * 100,  # Convert to percentage
            'probabilities': {k: v * 100 for k, v in prob_dict.items()},  # Convert to percentages
            'extracted_text': text[:200] + '...' if len(text) > 200 else text
        }

# Create a global classifier instance
text_classifier = CategoryClassifier(MODEL_DIR if os.path.exists(MODEL_DIR) else None)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file_path):
    """Extract text from different file types"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ' '.join([page.extract_text() for page in reader.pages if page.extract_text()])
            except ImportError:
                logging.error("PyPDF2 is not installed. Install with: pip install PyPDF2")
                return "Error: PyPDF2 library not installed"
        
        elif file_extension == '.docx':
            try:
                import docx
                doc = docx.Document(file_path)
                text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
            except ImportError:
                logging.error("python-docx is not installed. Install with: pip install python-docx")
                return "Error: python-docx library not installed"
        
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return text
    
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {e}")
        return f"Error extracting text: {str(e)}"

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/upload')
def upload():
    return send_from_directory('.', 'upload.html')

@app.route('/result')
def result():
    return send_from_directory('.', 'result.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Check if direct text input is provided
        if 'text' in request.form and request.form['text'].strip():
            text = request.form['text'].strip()
            result = text_classifier.classify_text(text)
            return jsonify(result)
        
        # Check if file is uploaded
        if 'document' not in request.files:
            return jsonify({'error': 'No document part in the request'}), 400
        
        file = request.files['document']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract text from file
            text = extract_text(file_path)
            
            if text.startswith('Error:'):
                return jsonify({'error': text}), 500
            
            # Classify text
            result = text_classifier.classify_text(text)
            
            # Return classification result
            return jsonify(result)
        
        return jsonify({'error': 'File type not allowed'}), 400
    
    except Exception as e:
        logging.error(f"Error during classification: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)