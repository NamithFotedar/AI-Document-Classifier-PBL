# AI-Document-Classifier-PBL

An intelligent document classification system that automatically categorizes text and documents into five categories: legal, business, medical, sports, and entertainment.

## Overview

This project uses natural language processing and deep learning to analyze text content and predict its category. It includes a fine-tuned DistilBERT model to accurately classify documents with high confidence.

## Features

- **Multi-category Classification**: Categorizes documents into 5 distinct categories
- **Multiple File Formats**: Supports TXT, PDF, and DOCX file formats
- **Web Interface**: Clean, responsive UI for easy document upload and result visualization
- **Visual Results**: Displays classification probabilities with interactive charts
- **Text Preview**: Shows extracted text from the document for verification

## Tech Stack

- **Backend**: Flask, PyTorch, Transformers (Hugging Face)
- **Frontend**: HTML, Tailwind CSS, JavaScript, Chart.js
- **ML Model**: Fine-tuned DistilBERT model
- **Document Processing**: PyPDF2, python-docx

## Project Structure

```
.
├── aiclassifier.py         # Core classification module for training
├── app.py                  # Flask web application
├── dataset_gen.py          # Training dataset generation script
├── fine_tuned_model/       # Directory for the fine-tuned model
├── index.html              # Homepage
├── result.html             # Results page with visualization
├── upload.html             # Document upload page
├── text_classification_dataset.csv  # Generated dataset
└── uploads/                # Directory for uploaded documents
```

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install flask torch transformers datasets pandas numpy scikit-learn werkzeug PyPDF2 python-docx
```

3. Generate the training dataset (optional):

```bash
python dataset_gen.py
```

4. Fine-tune the classification model:

```bash
python aiclassifier.py
```

5. Start the web application:

```bash
python app.py
```

6. Open your browser and navigate to `http://localhost:5000`

## Usage

1. **Training the Model**:
   - Run `aiclassifier.py` to fine-tune the model on the dataset
   - The fine-tuned model will be saved in the `fine_tuned_model/` directory

2. **Using the Web Application**:
   - Upload a document through the web interface
   - View the classification results with probability breakdown
   - Verify the extracted text preview

3. **Direct Classification**:
   - Import the `CategoryClassifier` class from `aiclassifier.py`
   - Use the `classify_text()` method to classify text

## Model Training

The model is fine-tuned on a balanced dataset containing examples from all five categories. The `dataset_gen.py` script generates a synthetic dataset with 1,000 examples per category. 

For best results, replace this with real-world examples from each category.

## Performance Metrics

The model is evaluated using the following metrics:
- Accuracy
- F1 Score
- Precision
- Recall

These metrics are calculated during the fine-tuning process and displayed in the console.

## Extending the Project

- **Add New Categories**: Modify the `categories` list in `aiclassifier.py` and retrain the model
- **Support Additional File Types**: Extend the `extract_text()` function in `app.py`
- **Improve Model Performance**: Collect more diverse training data or try different base models

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- Hugging Face for the Transformers library
- The PyTorch team for the deep learning framework
- The Flask team for the web framework
