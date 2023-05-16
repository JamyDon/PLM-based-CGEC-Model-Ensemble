import os
import sys
import torch
import torch.nn.functional as F
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


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


def get_embedding(input_ids, bert_model):

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    encoded, pooler_output = bert_model(
        input_ids,
        output_all_encoded_layers=False,
        attention_mask=input_ids.ne(0)
    )

    return encoded


def get_distribution(input_ids, bert_lm_model):

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    prediction_scores = bert_lm_model(
        input_ids,
        attention_mask=input_ids.ne(0)
    )

    probs = torch.softmax(prediction_scores, dim=-1)    # 利用softmax将生成概率归一化

    return probs


def calc_bias(text, tokenizer):
    tokens, ids = tokenize_text(tokenizer, text)
    return len(text) - len(tokens) + 2


def sentence_probability(text, bert_tokenizer, bert_lm_model):
    tokens = []
    ids = []
    for _ in range(len(text)):
        tokens_, ids_ = tokenize_text(bert_tokenizer, text[_])
        tokens.append(tokens_)
        ids.append(ids_)
    ids = padding(ids)
    input_ids = torch.LongTensor(ids)
    with torch.no_grad():
        distribution = get_distribution(input_ids=input_ids, bert_lm_model=bert_lm_model)
    distribution = distribution.tolist()
    sent_probability = [1 for _ in range(len(distribution))]

    for _ in range(len(distribution)):
        tok_cnt = 0
        for i in range(1, len(ids[_]) - 1):
            if ids[_][i] == 102:
                break
            tok_cnt += 1
            sent_probability[_] *= distribution[_][i][ids[_][i]]
        sent_probability[_] = sent_probability[_] ** (1 / (tok_cnt))
        sent = bert_tokenizer.convert_ids_to_tokens(ids[_])
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
    distribution = get_distribution(input_ids=input_ids, bert_lm_model=lm_model)
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

    bert_model_dir = "./macbert-base"

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
    bert_model = BertModel.from_pretrained(bert_model_dir)
    bert_lm_model = BertForMaskedLM.from_pretrained(bert_model_dir)
    bert_lm_model.eval()

    corpus = [
        "今天天气真好！",
        "我要好好学习！"
    ]

    ids_list = []
    for text in corpus:
        tokens, ids = tokenize_text(bert_tokenizer, text)
        ids_list.append(ids)

    input_ids = padding(ids_list, padding_mark=0)
    input_ids = torch.LongTensor(input_ids)

    distribution = get_distribution(input_ids, bert_lm_model)
    values, indices = torch.topk(distribution, 1, dim=-1)
    indices = indices.squeeze(-1)[:, 1: -1].tolist()
    for line in indices:
        print(">>> Highest probability tokens:", bert_tokenizer.convert_ids_to_tokens(line))
    print("\n")

    text = "再见，我还有很多功课要做。"
    tokens, ids = tokenize_text(bert_tokenizer, text)
    input_ids = torch.LongTensor(ids)
    distribution = get_distribution(input_ids=input_ids, bert_lm_model=bert_lm_model)
    distribution = distribution[0].tolist()
    sent_probability = 1

    for i in range(1, len(ids) - 1):
        print(">>> Token: %s   Probability: %.2f" % (tokens[i], distribution[i][ids[i]]))
        sent_probability *= distribution[i][ids[i]]
    sent_probability = sent_probability ** (1 / (len(ids) - 2))
    print(">>> Sentence Probability: %.2f" % (sent_probability))
    print("\n")

    text = "还有一个办法就是研究怎样才能在相等的土地上收获更多的农作品。"
    tokens, ids = tokenize_text(bert_tokenizer, text)


    for i in range(1, len(tokens) - 1):
        new_tokens, new_ids = tokens[:], ids[:]
        new_tokens[i] = "[MASK]"
        new_ids[i] = 103

        input_ids = torch.LongTensor(new_ids)
        distribution = get_distribution(input_ids, bert_lm_model)   # batch_size x seq_len x vocab_size
        distribution = distribution[0]      # seq_len x vocab_size

        values, indices = torch.topk(distribution, 1, dim=-1)
        values = values.squeeze(-1).tolist()
        indices = indices.squeeze(-1).tolist()
        pred_tokens = bert_tokenizer.convert_ids_to_tokens(indices)

        print(">>> Input Sentence: %s" % " ".join(new_tokens))
        print(">>> Highest word probability: %s - %.2f" % (pred_tokens[i], values[i]))
        print(">>> Golden word probability: %s - %.2f" % (tokens[i], distribution[i][ids[i]].item()))
        print()
