# This script is revised based on the codes shared for attendees of 2021 Korea University Computaional Linguistics Workshop.
# Reference 1: https://github.com/kiyoungkim1/LMkor
# Reference 2: https://github.com/Seongtae-Kim/WinterSchool_BERT

from typing import overload


class FineTuning:
    def __init__(self, tokenizer_path, train_ratio=None, batch_size=None, epoch=None):
        self.epoch = epoch
        self.batch = batch_size
        self.train_ratio = train_ratio
        self.set_device()
        self.build_BERT(tokenizer_path)
        self.trained = False

    def set_device(self):
        import torch
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        if not torch.cuda.is_available():
            print(
                "Change default hardware (CPU to GPU)!")
        else:
            print("GPU is running now. {}".format(torch.cuda.get_device_name(0)))

    def get_max_length(self, corpus, verbose=False) -> int:
        mxlen = 0
        for sent in corpus:
            if type(sent) is str:
                input_ids = self.tokenizer.tokenize(sent)
                mxlen = max(mxlen, len(input_ids))
        if verbose:
            print("max length is... ", mxlen)
        return mxlen

    def encode(self, corpus, labels=None, _tqdm=True, verbose=False):
        from tqdm.notebook import tqdm
        import torch

        self.corpus = corpus

        input_ids = []
        attention_masks = []
        if labels is not None:
            assert len(corpus) == len(labels)
        mxlen = self.get_max_length(corpus, verbose)
        if _tqdm:
            for sent in tqdm(corpus):
                encoded = self.tokenizer.encode_plus(
                    sent,
                    add_special_tokens=True,
                    max_length=mxlen,
                    truncation=True,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt')
                input_ids.append(encoded['input_ids'])
                attention_masks.append(encoded['attention_mask'])
        else:
            for sent in corpus:
                encoded = self.tokenizer.encode_plus(
                    sent,
                    add_special_tokens=True,
                    max_length=mxlen,
                    truncation=True,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt')
                input_ids.append(encoded['input_ids'])
                attention_masks.append(encoded['attention_mask'])

        self.input_ids = torch.cat(input_ids, dim=0)
        self.attention_masks = torch.cat(attention_masks, dim=0)

        if labels is not None:
            self.labels = torch.tensor(labels)

    def get_corpus_specifications(self):
        from Korpora import Korpora
        for name, desc in Korpora.corpus_list().items():
            print("{:<40}  {:<}".format(name, desc))

    def build_corpus(self, corpus_name):
        from Korpora import Korpora
        return Korpora.load(corpus_name)

    def build_BERT(self, tokenizer_path):
        from transformers import BertConfig, BertTokenizer
        self.bert_tokenizer_path = tokenizer_path
        self.tokenizer = BertTokenizer.from_pretrained(
            self.bert_tokenizer_path)

    def prepare(self, verbose=False):
        self.build_dataset(verbose)
        self.build_dataloader()
        self.build_optimizer()
        self.build_scheduler()

    def build_scheduler(self):
        from transformers import get_linear_schedule_with_warmup
        self.total_steps = len(self.train_dataloader) * self.epoch
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value in run_glue.py
                                                         num_training_steps=self.total_steps)

    def build_optimizer(self):
        from transformers import AdamW
        self.optimizer = AdamW(self.bert.parameters(), lr=2e-5, eps=1e-8)

    def build_dataset(self, verbose):
        from torch.utils.data import TensorDataset, random_split
        assert self.input_ids != [] and self.attention_masks != []

        if self.labels is not None:
            self.dataset = TensorDataset(
                self.input_ids, self.attention_masks, self.labels)
        else:
            self.dataset = TensorDataset(self.input_ids, self.attention_masks)

        self.train_size = int(self.train_ratio*len(self.dataset))
        self.val_size = len(self.dataset) - self.train_size

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [self.train_size, self.val_size])
        if verbose:
            print('{:>5,} training samples'.format(self.train_size))
            print('{:>5} validation samples'.format(self.val_size))

    def build_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        assert self.train_dataset is not None and self.val_dataset is not None

        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.batch,
        )
        self.validation_dataloader = DataLoader(
            self.val_dataset,
            sampler=SequentialSampler(self.val_dataset),
            batch_size=self.batch)

    def flat_accuracy(self, preds, labels):
        import numpy as np
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def train(self, verbose=True):
        from tqdm.notebook import tqdm
        import random
        import torch
        import numpy as np

        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        training_log = []
        desc_training_loss = None
        self.bert.train()
        self.bert.to(self.device)
        with tqdm(range(0, self.epoch), leave=False, bar_format="{percentage:2.2f}% {bar} {desc} | {elapsed}>{remaining}") as t:
            for epoch_i in range(0, self.epoch):
                t.update()
                total_train_loss = 0
                train_accs=[]

                for step, batch in enumerate(self.train_dataloader):
                    desc = "epoch: {:,}/{:,} | step: {:,}/{:,}".format(
                        epoch_i+1, len(range(0, self.epoch)), step+1, len(self.train_dataloader))

                    if desc_training_loss is not None:
                        t.set_description_str(desc+" | "+desc_training_loss)
                    else:
                        t.set_description_str(desc)

                    b_input_ids, b_input_mask, b_labels = map(
                        lambda e: e.to(self.device), batch)

                    self.bert.zero_grad()

                    output = self.bert(b_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_input_mask,
                                       labels=b_labels)
                    loss = output[0]
                    logits = output[1]

                    total_train_loss += loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.bert.parameters(), 1.0)

                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()
                    acc = self.flat_accuracy(logits, label_ids)
                    train_accs.append(acc)
                    self.optimizer.step()
                    self.scheduler.step()
                avg_train_acc = sum(train_accs) / len(train_accs)
                avg_train_loss = total_train_loss / \
                    len(self.train_dataloader)
                desc_training_loss = "mean training loss: {:.2f} / average accuracies:{}".format(
                    avg_train_loss, round(avg_train_acc, 2))
                training_log.append(
                            "{:<50}{}".format(desc, desc_training_loss))

        if verbose:
            for log in training_log:
                print(log)
        self.trained = True

    def validate(self):
        import torch
        self.bert.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        for batch in self.validation_dataloader:
            b_input_ids, b_input_mask, b_labels = map(
                lambda e: e.to(self.device), batch)

            with torch.no_grad():
                self.bert.to(self.device)
                output = self.bert(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            loss = output[0]
            logits = output[1]

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += self.flat_accuracy(logits, label_ids)
        avg_val_accuracy = total_eval_accuracy / \
            len(self.validation_dataloader)
        avg_val_loss = total_eval_loss / len(self.validation_dataloader)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation Accuracy: {0:.2f}".format(avg_val_accuracy))


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
        return "positive" if np.argmax(logit[0].detach().cpu().numpy()) else "negative"
