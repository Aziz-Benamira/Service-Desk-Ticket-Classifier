# Service Desk Ticket Classifier

Welcome to the **Service Desk Ticket Classifier** project! This repository contains the implementation of a Convolutional Neural Network (CNN) for automatically categorizing customer complaints into predefined categories. This solution is designed to streamline customer service operations by enabling rapid classification of support tickets
The classifier uses a CNN model, trained and evaluated on labeled data, to achieve high accuracy and provide per-class metrics such as precision and recall.

---

## Project Features

### Key Components:
1. **Preprocessing**:
   - Tokenization of text data.
   - Padding or truncating sequences to a fixed length.
   - Mapping words to unique indices.
2. **Model Architecture**:
   - Embedding Layer: Converts word indices to dense vectors.
   - 1D Convolution Layer: Extracts features from text sequences.
   - Linear Layer: Produces class logits for classification.
3. **Metrics**:
   - Accuracy
   - Precision (per class)
   - Recall (per class)

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- scikit-learn
- NLTK
- torchmetrics
- pandas
- NumPy


### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Aziz-BenAmira/service-desk-ticket-classifier.git
   cd service-desk-ticket-classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   ```

---

## Usage

### Data Preparation
Place the following files in the project directory:
- `words.json`: List of unique words in the dataset.
- `text.json`: Tokenized sentences.
- `labels.npy`: NumPy array of labels.

### Training and Evaluation
Run the script:
```bash
python train_and_evaluate.py
```
This script:
- Preprocesses the data.
- Trains the CNN for 3 epochs.
- Evaluates the model on test data and calculates:
  - Accuracy
  - Precision (per class)
  - Recall (per class)

---

## Results
Sample metrics from a trained model:
- **Accuracy**: 77.99%
- **Precision**:
  - Class 0: 65.20%
  - Class 1: 73.68%
  - Class 2: 85.93%
  - Class 3: 89.03%
  - Class 4: 79.43%
- **Recall**:
  - Class 0: 77.08%
  - Class 1: 66.31%
  - Class 2: 79.17%
  - Class 3: 71.88%
  - Class 4: 93.81%

---

## File Structure
```
.
├── train_and_evaluate.py  # Main script for training and evaluation
├── words.json             # Vocabulary file
├── text.json              # Tokenized sentences
├── labels.npy             # Labels for the dataset
├── requirements.txt       # Required Python libraries
└── README.md              # Project documentation
```

---

## Future Work
- Enhance the model with additional layers for better performance.
- Experiment with attention mechanisms to improve classification.
- Deploy the model as an API for real-time classification.

---

## Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

---

