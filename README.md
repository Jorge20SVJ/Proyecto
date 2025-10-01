# Proyecto
Machine learning system for detecting phishing attempts in WhatsApp messages using Natural Language Processing techniques. This project was developed as part of my Computer Engineering thesis() at Instituto Politécnico Nacional (IPN).
Note: This repository contains the ML algorithm component of a collaborative project. The complete system includes a mobile application interface developed by my teammate.

WhatsApp users are increasingly targeted by phishing attacks through malicious messages. This system aims to automatically classify text messages to identify potential phishing attempts, helping users avoid security threats.

Technical Approach
Algorithm Selection

Naive Bayes (MultinomialNB): Chosen for its effectiveness with text classification tasks
TF-IDF Vectorization: Converts text into numerical features based on term frequency and importance
Advantages: Fast training, works well with small datasets, handles high-dimensional sparse data effectively

Dataset

Source: SMS Spam Collection dataset
Labels: Binary classification (ham/spam)
Size: 1,672 messages (1,451 ham, 221 spam)
Language: Mixed (primarily English, adapted for Spanish processing)

Technologies Used

Python 3.x
scikit-learn: Machine learning algorithms and metrics
pandas: Data manipulation and analysis
numpy: Numerical computing
NLTK: Natural language processing toolkit
joblib: Model serialization

Installation
Clone the repository:
git clone https://github.com/Jorge20SVJ/whatsapp-phishing-detection-ml.git
cd whatsapp-phishing-detection-ml
Install dependencies:
pip install -r requirements.txt
Download NLTK data:
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
Usage
Run the training script:
python phishing_detector.py
The script will:

Load and preprocess the dataset
Train the Naive Bayes model
Evaluate performance metrics
Save trained model and vectorizer

Results
Model Performance
Overall Accuracy: 97%
Classification Report
                Precision  Recall  F1-Score  Support
Class 0 (Ham)           0.97    1.00      0.98     1451
Class 1 (Spam)          1.00    0.79      0.88      221
Accuracy                                  0.97     1672
Macro Average           0.98    0.90      0.93     1672
Weighted Average        0.97    0.97      0.97     1672
Performance Analysis

Excellent ham detection: 100% recall ensures legitimate messages are never incorrectly flagged
High spam precision: 100% precision means when the model identifies spam, it's always correct
Spam recall at 79%: Detects 8 out of 10 phishing attempts, acceptable trade-off to avoid false positives
Class imbalance consideration: Dataset contains approximately 6.5x more ham than spam messages, which is reflected in the tuned model behavior

Project Structure
whatsapp-phishing-detection-ml/
├── README.md
├── requirements.txt
├── phishing_detector.py       (Main ML algorithm)
├── data/
│   └── SMSSpamCollection.txt  (Training dataset)
└── models/
├── modelo_phishing.pkl    (Trained model)
└── tfidf_vectorizer.pkl   (Fitted vectorizer)
Key Features

Text Preprocessing: Tokenization and Spanish stopword removal
Feature Engineering: TF-IDF vectorization for optimal text representation
Model Persistence: Serialized models for production deployment
Performance Evaluation: Comprehensive metrics reporting in Spanish
Scalable Architecture: Designed for integration with mobile applications

Future Improvements

Cross-validation for more robust evaluation
Hyperparameter tuning for optimal performance
Support for emoji and URL processing (WhatsApp-specific features)
Integration with real-time message processing
Multi-language support
Deep learning approaches (LSTM, BERT)

Collaboration
This ML backend was designed to work with a mobile application frontend. The system architecture allows for:

API integration for real-time predictions
Batch processing capabilities
Model updates and retraining

Technical Decisions
Why Naive Bayes?

Probabilistic approach suitable for text classification
Fast training and prediction times
Good performance on sparse, high-dimensional data
Baseline algorithm recommended for NLP tasks

Why TF-IDF?

Captures both term frequency and document importance
Reduces impact of common words through inverse document frequency
Creates meaningful numerical representations of text

License
This project was developed for academic purposes as part of my thesis requirements.
Contact
Jorge Jair Sanchez Vazquez
Computer Engineering Graduate - IPN
Email: jairsanchez198@gmail.com
GitHub: https://github.com/Jorge20SVJ
Acknowledgments

Instituto Politécnico Nacional (IPN)
Collaborative partner for mobile application development

