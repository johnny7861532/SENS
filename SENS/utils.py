import torch
from transformers import BertTokenizer, BertModel

class EmbeddingGenerator:
    def __init__(self, model_name='bert-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def get_embeddings(self, texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=128)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings.append(outputs.last_hidden_state.mean(dim=1))
        if len(embeddings) == 0:
            raise ValueError("未生成任何嵌入。請檢查您的數據和處理。")
        return torch.cat(embeddings, dim=0).cpu().numpy()
