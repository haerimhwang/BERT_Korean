# This script is revised based on the codes shared for attendees of 2021 Korea University Computaional Linguistics Workshop.
# Reference 1: https://github.com/kiyoungkim1/LMkor
# Reference 2: https://github.com/Seongtae-Kim/WinterSchool_BERT

class Models:
    def __init__(self) -> None:
        self.lists = {}

        # M-BERT
        from transformers import BertTokenizerFast, BertForMaskedLM
        self.bert_multilingual_tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-multilingual-cased')
        self.bert_multilingual_model = BertForMaskedLM.from_pretrained(
            'bert-base-multilingual-cased').eval()
        self.lists["M-BERT"] = {"Tokenizer": self.bert_multilingual_tokenizer,
                                "Model": self.bert_multilingual_model}
        print("====================================")
        print("[BERT] Google Multilingual BERT loaded")
        print("====================================")

        # KR-BERT
        from transformers import BertTokenizerFast, BertForMaskedLM
        self.krbert_tokenizer = BertTokenizerFast.from_pretrained(
            'snunlp/KR-Medium')
        self.krbert_model = BertForMaskedLM.from_pretrained(
            'snunlp/KR-Medium').eval()
        self.lists["KR-Medium"] = {"Tokenizer": self.krbert_tokenizer,
                                   "Model": self.krbert_model}
        print("====================================")
        print("[BERT] KR-BERT loaded")
        print("====================================")

        # BERT
        from transformers import BertTokenizerFast, BertForMaskedLM
        self.bert_kor_tokenizer = BertTokenizerFast.from_pretrained(
            'kykim/bert-kor-base')
        self.bert_kor_model = BertForMaskedLM.from_pretrained(
            'kykim/bert-kor-base').eval()
        self.lists["bert-kor-base"] = {"Tokenizer": self.bert_kor_tokenizer,
                                       "Model": self.bert_kor_model}
        print("====================================")
        print("[BERT] BERT-kor-base loaded")
        print("====================================")

        # ALBERT
        from transformers import AlbertForMaskedLM
        self.albert_tokenizer = BertTokenizerFast.from_pretrained(
            'kykim/albert-kor-base')
        self.albert_model = AlbertForMaskedLM.from_pretrained(
            'kykim/albert-kor-base').eval()
        self.lists["albert-kor-base"] = {"Tokenizer": self.albert_tokenizer,
                                         "Model": self.albert_model}
        print("====================================")
        print("[BERT] ALBERT-kor-base loaded")
        print("====================================")

        # XLM-Roberta
        from transformers import XLMRobertaTokenizerFast, XLMRobertaForMaskedLM
        self.xlmroberta_tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            'xlm-roberta-base')
        self.xlmroberta_model = XLMRobertaForMaskedLM.from_pretrained(
            'xlm-roberta-base').eval()
        self.lists["xlm-roberta-base"] = {"Tokenizer": self.xlmroberta_tokenizer,
                                          "Model": self.xlmroberta_model}
        print("====================================")
        print("[BERT] XLM-Roberta-kor loaded")
        print("====================================")

        from transformers import BertTokenizerFast, EncoderDecoderModel
        self.tokenizer_bertshared = BertTokenizerFast.from_pretrained(
            "kykim/bertshared-kor-base")
        self.bertshared_model = EncoderDecoderModel.from_pretrained(
            "kykim/bertshared-kor-base")
        self.lists["bertshared-kor-base"] = {"Tokenizer": self.tokenizer_bertshared,
                                             "Model": self.bertshared_model}
        print("====================================")
        print("[Seq2seq + BERT] bertshared-kor-base loaded")
        print("====================================")

        # gpt3-kor-small_based_on_gpt2
        from transformers import BertTokenizerFast, GPT2LMHeadModel
        self.tokenizer_gpt3 = BertTokenizerFast.from_pretrained(
            "kykim/gpt3-kor-small_based_on_gpt2")
        self.model_gpt3 = GPT2LMHeadModel.from_pretrained(
            "kykim/gpt3-kor-small_based_on_gpt2")
        self.lists["gpt3-kor-small_based_on_gpt2"] = {"Tokenizer": self.tokenizer_gpt3,
                                                      "Model": self.model_gpt3}
        print("====================================")
        print("[GPT3] gpt3-small-based-on-gpt2 loaded")
        print("====================================")

        # electra-base-kor
        from transformers import ElectraTokenizerFast, ElectraModel
        self.tokenizer_electra = ElectraTokenizerFast.from_pretrained(
            "kykim/electra-kor-base")
        self.electra_model = ElectraModel.from_pretrained(
            "kykim/electra-kor-base")
        self.lists["electra-kor-base"] = {"Tokenizer": self.tokenizer_electra,
                                          "Model": self.electra_model}
        print("====================================")
        print("[ELECTRA] electra-kor-base loaded")
        print("====================================")

        from transformers import ElectraTokenizerFast, ElectraForQuestionAnswering
        self.electra_tokenizer_QA = ElectraTokenizerFast.from_pretrained(
            "monologg/koelectra-base-v3-finetuned-korquad")
        self.electra_model_QA = ElectraForQuestionAnswering.from_pretrained(
            "monologg/koelectra-base-v3-finetuned-korquad")
        self.lists["electra-kor-QA"] = {"Tokenizer": self.electra_tokenizer_QA,
                                        "Model": self.electra_model_QA}
        print("====================================")
        print("[ELECTRA] koelectra-base-v3-finetuned-korquad loaded")
        print("====================================")

    def summarize(self, text):
        input_ids = self.lists['bertshared-kor-base']["Tokenizer"].encode(
            text, return_tensors="pt")

        sent_len = len(input_ids[0])
        min_length = max(10, int(0.1*sent_len))
        max_length = min(128, int(0.3*sent_len))

        outputs = self.bertshared_model.generate(
            input_ids,
            min_length=min_length,
            max_length=max_length
        )
        print(self.lists['bertshared-kor-base']["Tokenizer"].decode(
            outputs[0], skip_special_tokens=True))

    def generate_text(self, text, num):
        input_ids = self.lists["gpt3-kor-small_based_on_gpt2"]["Tokenizer"].encode(
            text, return_tensors='pt')
        input_ids = input_ids[:, 1:]  # remove cls token

        outputs = self.lists["gpt3-kor-small_based_on_gpt2"]["Model"].generate(
            input_ids,
            min_length=30,
            max_length=50,
            do_sample=True,
            top_k=10,
            top_p=0.95,
            no_repeat_ngram_size=2,
            num_return_sequences=num
        )

        for idx, generated in enumerate(
                [self.lists["gpt3-kor-small_based_on_gpt2"]["Tokenizer"].decode(sentence, skip_special_tokens=True) for sentence in outputs]):
            print('{0}: {1}'.format(idx, generated))

    def answer_question(self, question, context):
        from transformers import pipeline
        qa = pipeline("question-answering",
                      "monologg/koelectra-base-v3-finetuned-korquad")
        return qa(question=question, context=context)["answer"]

    def mask_predict(self, text_sentence, top_k=10, top_clean=3):
        import torch
        if '<mask>' not in text_sentence:
            print('Input <mask>. e.g., 이거 <mask> 재밌네')
            return

        # ========================= BERT =================================
        input_ids, mask_idx = self.encode_mask(
            self.lists['bert-kor-base']["Tokenizer"], text_sentence)
        with torch.no_grad():
            predict = self.lists['bert-kor-base']["Model"](input_ids)[0]
        bert = self.decode_mask(self.lists['bert-kor-base']["Tokenizer"], predict[0, mask_idx, :].topk(
            top_k).indices.tolist(), top_clean)

        # ========================= ALBERT =================================
        input_ids, mask_idx = self.encode_mask(
            self.lists['albert-kor-base']["Tokenizer"], text_sentence)
        with torch.no_grad():
            predict = self.lists['albert-kor-base']["Model"](input_ids)[0]
        albert = self.decode_mask(self.lists['albert-kor-base']["Tokenizer"], predict[0, mask_idx, :].topk(
            top_k).indices.tolist(), top_clean)

        # ========================= BERT MULTILINGUAL =================================
        input_ids, mask_idx = self.encode_mask(self.lists['M-BERT']["Tokenizer"], text_sentence,
                                               mask_token=self.lists['M-BERT']["Tokenizer"].mask_token,
                                               mask_token_id=self.lists['M-BERT']["Tokenizer"].mask_token_id)
        with torch.no_grad():
            predict = self.lists['M-BERT']["Model"](input_ids)[0]
        bert_multilingual = self.decode_mask(self.lists['M-BERT']["Tokenizer"], predict[0, mask_idx, :].topk(
            top_k).indices.tolist(), top_clean)

        # ========================= XLM ROBERTA BASE =================================
        input_ids, mask_idx = self.encode_mask(self.lists['xlm-roberta-base']["Tokenizer"], text_sentence,
                                               mask_token=self.lists['xlm-roberta-base']["Tokenizer"].mask_token,
                                               mask_token_id=self.lists['xlm-roberta-base']["Tokenizer"].mask_token_id)
        with torch.no_grad():
            predict = self.lists['xlm-roberta-base']["Model"](input_ids)[0]
        xlm = self.decode_mask(self.lists['xlm-roberta-base']["Tokenizer"], predict[0, mask_idx, :].topk(
            top_k).indices.tolist(), top_clean)

        results = {'kykim/bert-kor-base': bert,
                   'kykim/albert-kor-base': albert,
                   'bert_multilingual': bert_multilingual,
                   'xlm': xlm}

        for model, tokens in results.items():
            print('{0}: {1}'.format(model, tokens))

    def decode_mask(self, tokenizer, pred_idx, top_clean):
        import string
        ignore_tokens = string.punctuation + '[PAD][UNK]<pad><unk> '
        tokens = []
        for w in pred_idx:
            token = ''.join(tokenizer.decode(w).split())
            if token not in ignore_tokens:
                tokens.append(token.replace('##', ''))
        return ' / '.join(tokens[:top_clean])

    def encode_mask(self, tokenizer, text_sentence, add_special_tokens=True, mask_token='[MASK]', mask_token_id=4):
        import torch
        text_sentence = text_sentence.replace('<mask>', mask_token)
        # if <mask> is the last token, append a "." so that models dont predict punctuation.
        if mask_token == text_sentence.split()[-1]:
            text_sentence += ' .'

        input_ids = torch.tensor(
            [tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
        mask_idx = torch.where(input_ids == mask_token_id)[1].tolist()[0]
        return input_ids, mask_idx
