from promptClassifier import PromptClassifier
import os
# charger les fichier json
# charger les données
train_data = 'guardrail_dataset_train.json'
test_data ='guardrail_dataset_test.json'
train_data = os.path.join(os.path.dirname(__file__), train_data)
test_data = os.path.join(os.path.dirname(__file__), test_data)
print(train_data)

# entrainer le modèle
classifier = PromptClassifier()

classifier.load_train_and_test_data_from_json(train_data, test_data)
classifier.train_and_evaluate()
classifier.get_best_model()
classifier.export_best_model("best_prompt_model.pkl")

