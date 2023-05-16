import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertTokenizer


def tokenize_text(tokenizer, text):

    tokens = tokenizer.tokenize(text)         
    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, ids


def padding(data, padding_mark=0):

    max_length = max([len(line) for line in data])
    for i in range(len(data)):
        fillup_num = max_length - len(data[i])
        data[i] = data[i] + [padding_mark for _ in range(fillup_num)]

    return data


def get_distribution(input_ids, gpt2_lm_model):

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    moedel_output = gpt2_lm_model(
        input_ids,
        attention_mask=input_ids.ne(0)
    )

    logits = moedel_output.logits   # batch_size x seq_len x vocab_size

    # logits: batch_size x seq_len x vocab_size

    probs = F.softmax(logits, dim=-1)

    return probs


def calc_bias(text, tokenizer):
    tokens, ids = tokenize_text(tokenizer, text)
    return len(text) - len(tokens) + 2


def sentence_probability(text, bert_tokenizer, gpt2_lm_model):
    tokens = []
    ids = []
    for _ in range(len(text)):
        tokens_, ids_ = tokenize_text(bert_tokenizer, text[_])
        tokens.append(tokens_)
        ids.append(ids_)
    ids = padding(ids)
    input_ids = torch.LongTensor(ids)
    with torch.no_grad():
        distribution = get_distribution(input_ids=input_ids, gpt2_lm_model=gpt2_lm_model)
    distribution = distribution.tolist()
    sent_probability = [1 for _ in range(len(distribution))]

    sent_ids = []
    for _ in range(len(distribution)):
        sent_ids.append(ids[_][1:])
    for _ in range(len(distribution)):
        tok_cnt = 0
        for i in range(len(sent_ids[_])):
            if sent_ids[_][i] == 0:
                break
            tok_cnt += 1
            sent_probability[_] *= distribution[_][i][sent_ids[_][i]]
        sent_probability[_] = sent_probability[_] ** (1 / (tok_cnt))
        sent = bert_tokenizer.convert_ids_to_tokens(sent_ids[_])
    return sent_probability


def masked_probability(sentences, spans, tokenizer, lm_model):
    assert len(sentences) == len(spans)

    tokens = []
    ids = []
    masked_tokens = []
    masked_ids = []

    for _ in range(len(sentences)):
        tokens_, ids_ = tokenize_text(tokenizer, sentences[_])

        start, end = spans[_]
        start += 1
        end += 1
        start -= calc_bias(sentences[_][:(start-1)], tokenizer)
        end -= calc_bias(sentences[_][:(end-1)], tokenizer)
        spans[_] = (start, end)
        masked_tokens_ = tokens_[(start):(end)]
        masked_ids_ = ids_[(start):(end)]

        for __ in range(start, end):
            tokens_[__] = "[MASK]"
            ids_[__] = 103

        tokens.append(tokens_)
        ids.append(ids_)
        masked_tokens.append(masked_tokens_)
        masked_ids.append(masked_ids_)

    ids = padding(ids)
    input_ids = torch.LongTensor(ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    lm_model = lm_model.to(device)
    distribution = get_distribution(input_ids=input_ids, gpt2_lm_model=lm_model)
    distribution = distribution.tolist()

    probs = []

    for _ in range(len(sentences)):
        distr = distribution[_]
        start, end = spans[_]
        prob = 1
        cnt = 0
        for __ in range(start, end):
            prob *= distr[__][masked_ids[_][cnt]]
            cnt += 1
        if cnt <= 0:
            probs.append(0)
            continue
        probs.append(prob ** (1 / cnt))

    return probs, masked_tokens


if __name__ == '__main__':
    model = GPT2LMHeadModel.from_pretrained("./gpt2-chinese/")
    tokenizer = BertTokenizer(vocab_file="./gpt2-chinese/vocab.txt")
    sent = "我们要好好学习，天天向上！"
    print(sentence_probability([sent], tokenizer, model))