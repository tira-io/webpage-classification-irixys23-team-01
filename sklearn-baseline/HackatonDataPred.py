import pandas as pd
from sklearn import pipeline
from sklearn.metrics import classification_report
import joblib

predictions_file = "predictions_3.jsonl"
truth_file = "D1_Validation-truth.jsonl"

predictions = pd.read_json(predictions_file, lines=True)
truth = pd.read_json(truth_file, lines=True)

merged_data = pd.merge(predictions, truth, on='uid')

classification_rep = classification_report(merged_data['label'], merged_data['prediction'])

print(classification_rep)
