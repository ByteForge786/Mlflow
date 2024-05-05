import torch
import mlflow.pyfunc

class CustomZeroShotClassifier:
    def __init__(self, model_path, candidate_labels):
        self.model = mlflow.pyfunc.load_model(model_path)

        # If the tokenizer is combined with the model, you might not need to explicitly load it
        
        self.candidate_labels = candidate_labels

    def classify(self, input_text):
        # Tokenize input text
        # If the tokenizer is combined with the model, you might not need to tokenize explicitly
        
        # Perform prediction using MLflow model
        logits = self.model.predict(input_text)["logits"]
        logits_tensor = torch.tensor(logits)
        
        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits_tensor, dim=1)
        
        # Find the label with the highest probability
        predicted_label_index = torch.argmax(probabilities)
        predicted_label = self.candidate_labels[predicted_label_index]
        
        return predicted_label

# Example usage
model_path = "path_to_your_mlflow_model"
candidate_labels = ["politics", "sports", "science", "history"]

classifier = CustomZeroShotClassifier(model_path, candidate_labels)
input_text = "The latest scientific breakthroughs in medicine."
predicted_label = classifier.classify(input_text)
print("Predicted Label:", predicted_label)
