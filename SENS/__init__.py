from .classifier import EmotionClassifier, load_model

classifier = None

def load(model_path):
    global classifier
    classifier = load_model(model_path)

def classify_emotion(texts):
    if classifier is None:
        raise ValueError("Classifier model is not loaded. Please call `SENS.load(model_path)` first.")
    return classifier.classify(texts)
