import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class CategoryClassifier:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=5):
        """
        Initialize the Text Category Classifier
        
        Args:
            model_name (str): Hugging Face model to use as base
            num_labels (int): Number of categories to classify
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.categories = ['legal', 'business', 'medical', 'sports', 'entertainment']
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def fine_tune(self, dataset_path, output_dir='./fine_tuned_model', epochs=3, batch_size=16):
        """
        Fine-tune the model on a custom dataset
        
        Args:
            dataset_path (str): Path to the CSV dataset file
            output_dir (str): Directory to save the fine-tuned model
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Map text categories to numeric labels
        label_map = {cat: i for i, cat in enumerate(self.categories)}
        df['label'] = df['category'].map(label_map)
        
        # Split into train and validation sets
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # Define tokenize function
        def tokenize(batch):
            return self.tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)
        
        # Tokenize datasets
        train_dataset = train_dataset.map(tokenize, batched=True)
        val_dataset = val_dataset.map(tokenize, batched=True)
        
        # Set format for pytorch
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        # Define metrics computation function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
            acc = accuracy_score(labels, predictions)
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train model
        print("Starting fine-tuning...")
        trainer.train()
        
        # Save fine-tuned model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model fine-tuned and saved to {output_dir}")
        
        # Evaluate model
        results = trainer.evaluate()
        print(f"Evaluation results: {results}")
    
    def load_fine_tuned_model(self, model_path):
        """
        Load a fine-tuned model
        
        Args:
            model_path (str): Path to the fine-tuned model
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        print(f"Loaded fine-tuned model from {model_path}")
    
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
        text = text[:5000]  # Limit to 5000 characters for processing
        
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
            'confidence': confidence,
            'probabilities': prob_dict,
            'extracted_text': text[:100] + '...' if len(text) > 100 else text
        }

def main():
    """
    Example usage of the CategoryClassifier
    """
    # Initialize classifier
    classifier = CategoryClassifier()
    
    # Fine-tune on custom dataset
    classifier.fine_tune('text_classification_dataset.csv', epochs=3)
    
    # Test the fine-tuned model
    test_texts = [
        "The court granted the motion to dismiss based on lack of jurisdiction.",
        "Quarterly earnings exceeded analyst expectations by 15%, driving the stock price up.",
        "The patient's lab results showed elevated white blood cell count and abnormal liver enzymes.",
        "The quarterback threw for 300 yards and 3 touchdowns in the championship game.",
        "The new film sequel broke box office records during its opening weekend."
    ]
    
    print("\nTesting the fine-tuned model:")
    for text in test_texts:
        result = classifier.classify_text(text)
        print(f"\nText: {text[:50]}...")
        print(f"Predicted Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("Probabilities:")
        for category, prob in result['probabilities'].items():
            print(f"  {category}: {prob:.2%}")

if __name__ == '__main__':
    main()