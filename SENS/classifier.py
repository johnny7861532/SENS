import time
import joblib
import numpy as np
from .utils import EmbeddingGenerator

# 情緒標籤解釋
emotion_labels = {
    0: "其它 (Other)",
    1: "喜 (Like)",
    2: "哀 (Sadness)",
    3: "怖 (Disgust)",
    4: "怒 (Anger)",
    5: "楽 (Happiness)"
}

class EmotionClassifier:
    def __init__(self, model_path):
        self.embedding_generator = EmbeddingGenerator()
        self.classifier = joblib.load(model_path)

    def classify(self, texts):
        start_time = time.time()
        embeddings = self.embedding_generator.get_embeddings(texts)
        predictions = self.classifier.predict(embeddings)
        elapsed_time = time.time() - start_time
        results = []
        for i, label in enumerate(predictions):
            results.append({
                'text': texts[i],
                'predicted_emotion': emotion_labels[label]
            })
        return results, elapsed_time

def load_model(model_path):
    return EmotionClassifier(model_path)
