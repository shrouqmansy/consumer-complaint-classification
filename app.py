import re
import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "complaint_model"
labels = ["credit_card", "credit_reporting", "debt_collection", "mortgages_and_loans", "retail_banking"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


def predict(text):
    # the training text was lowercase without punctuation and numbers, so clean the input the same way
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)[0].numpy()
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


demo = gr.Interface(fn=predict, inputs=gr.Textbox(lines=6, label="Complaint"), outputs=gr.Label(num_top_classes=5, label="Prediction"), title="Consumer Complaint Classifier")

if __name__ == "__main__":
    demo.launch()
