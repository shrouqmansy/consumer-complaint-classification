# Consumer Complaint Classification

Deep learning text classification that predicts the product category of a customer complaint from its written narrative. The project compares three sequence models built from scratch (SimpleRNN, LSTM, GRU) against a fine-tuned HuggingFace transformer (DistilBERT), and serves the best model through a Gradio app.

## Goal

Automatically route consumer complaints to the correct category to reduce manual effort and cost.

## Dataset

Consumer Complaints Dataset for NLP (Kaggle, `complaints_processed.csv`). It has about 162k complaints with a narrative and a product category. The version used here is already preprocessed (lowercase, no punctuation, numbers or stop words, lemmatized).

Classes: `credit_card`, `credit_reporting`, `debt_collection`, `mortgages_and_loans`, `retail_banking`.

The classes are heavily imbalanced (`credit_reporting` is about 91k rows, `retail_banking` about 13k), so the data is balanced by sampling 12,000 rows per class (60,000 total), then split 70/15/15 (42,000 train, 9,000 validation, 9,000 test) with stratification and a fixed seed.

## Pipeline

1. Load and explore the data, plot the class distribution.
2. Balance the classes by sampling.
3. Keras models: tokenizer (20k words), padding to 200 tokens, embedding layer (128 dimensions).
4. Models: SimpleRNN, LSTM, GRU, each bidirectional with dropout (0.5) and early stopping.
5. Transformer: `distilbert-base-uncased` fine-tuned for 3 epochs, max length 256, dynamic padding.
6. Evaluation on the test set: accuracy, precision, recall, F1 and confusion matrix.
7. Deployment with Gradio.

## Results

Test set of 9,000 complaints (macro averages for precision, recall and F1):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SimpleRNN | 0.6908 | 0.6915 | 0.6908 | 0.6753 |
| LSTM | 0.8258 | 0.8257 | 0.8258 | 0.8254 |
| GRU | 0.8444 | 0.8464 | 0.8444 | 0.8447 |
| **DistilBERT** | **0.8638** | **0.8637** | **0.8638** | **0.8637** |

DistilBERT is the best model.

## Findings

- GRU and LSTM clearly beat SimpleRNN, which struggles with long complaints.
- More training data (4,000 to 12,000 rows per class), bidirectional layers, dropout and early stopping reduced overfitting and improved the Keras models.
- Increasing the transformer max length from 128 to 256 tokens did not change accuracy.
- Most errors are between similar categories: `credit_reporting` with `debt_collection`, and `credit_card` with `retail_banking`. These complaints overlap in content, which limits the accuracy for every model.

## Project structure

```
.
├── notebooks
│   ├── 01_rnn_lstm_gru.ipynb   # Keras models
│   └── 02_distilbert.ipynb     # HuggingFace model and comparison
├── results
│   └── results.csv
├── app.py                      # Gradio app
├── requirements.txt
└── README.md
```

## How to run

Install the requirements:

```
pip install -r requirements.txt
```

Run the two notebooks in separate sessions (Keras and PyTorch in the same session can crash the kernel on a shared GPU). The second notebook saves the fine-tuned model to a `complaint_model` folder.

Then start the app:

```
python app.py
```

Enter a complaint narrative in English and the app returns the predicted category with the confidence of each class.

## Model weights

The fine-tuned model is about 270 MB, which is too large for a normal GitHub push, so it is not included in this repository. Generate it with `notebooks/02_distilbert.ipynb`, or upload it to the HuggingFace Hub and load it by name in `app.py`.
