# This script is revised based on the codes shared for attendees of 2021 Korea University Computaional Linguistics Workshop.
# Reference 1: https://github.com/kiyoungkim1/LMkor
# Reference 2: https://github.com/Seongtae-Kim/WinterSchool_BERT

from typing import overload

class Sentiment_Analysis(FineTuning):
    def __init__(self, tokenizer_path, train_ratio=None, batch_size=None, epoch=None):
        super().__init__(tokenizer_path, train_ratio, batch_size, epoch)
        self.build()

    def build(self):
        from transformers import BertForSequenceClassification
        # self.bert = BertForSequenceClassification.from_pretrained(
        #    self.bert_model_path, config=self.bert_config_path)
        self.bert = BertForSequenceClassification.from_pretrained(
            "snunlp/KR-Medium")

        print("BERT loaded")

    def predict(self, sentence):
        import numpy as np
        import torch

        if not self.trained:
            print("Training is required. Testing with an untrained model...")

        self.get_max_length([sentence])
        self.encode([sentence])

        self.bert.eval()
        with torch.no_grad():
            self.bert.to(self.device)
            self.input_ids = self.input_ids.to("cuda")
            self.attention_masks = self.attention_masks.to("cuda")
            logit = self.bert(self.input_ids,
                              attention_mask=self.attention_masks)
        return logit
        return "positive" if np.argmax(logit[0].detach().cpu().numpy()) else "negative"
