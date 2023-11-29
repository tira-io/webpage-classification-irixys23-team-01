import pandas as pd
from sklearn import pipeline
from sklearn.metrics import classification_report
import joblib

# Загрузка предсказаний и правды
predictions_file = "predictions_3.jsonl"
truth_file = "D1_Validation-truth.jsonl"

predictions = pd.read_json(predictions_file, lines=True)
truth = pd.read_json(truth_file, lines=True)

# Слияние данных по полю 'uid'
merged_data = pd.merge(predictions, truth, on='uid')

# Получение отчета по классификации
classification_rep = classification_report(merged_data['label'], merged_data['prediction'])

# Вывод отчета
print(classification_rep)
