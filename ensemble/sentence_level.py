import sys
import os
import time
import argparse
import torch
from collections import Counter
from tqdm import tqdm
sys.path.append('./ensemble/')
# import merge_edit
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from PLM.bert_scorer import sentence_probability
# from pytorch_pretrained_bert import BertTokenizer
# from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel
# from PLM.gpt2_scorer import sentence_probability
from traditional import get_models
from rule_ensemble import parse_m2, validate


def get_models(filename):
    models = ["./data/nlpcc/nlpcc_test.src"]
    with open(filename, "r") as f:
        for line in f:
            models.append("./data/nlpcc/models/" + line.rstrip("\n") + ".tgt")
    return models


def get_tgts(filenames):
    all_tgts = []
    for filename in filenames:
        this_tgts = []
        with open(filename, "r") as f:
            for line in f:
                this_tgts.append(line.rstrip("\n"))
        all_tgts.append(this_tgts)
    
    tgts = []
    for _ in range(len(all_tgts[0])):
        sentence_tgts = []
        for __ in range(len(all_tgts)):
            if(all_tgts[__][_] not in sentence_tgts):
                sentence_tgts.append(all_tgts[__][_])
        tgts.append(sentence_tgts)
    
    return tgts


def main(args):
    total_edits = []
    models = get_models(args.models_path)
    tgts = get_tgts(models)
    model_candidates = []

    with open(args.output_path, "w", encoding="utf-8") as o:
        bert_tokenizer = BertTokenizer.from_pretrained(args.plm_path)
        plm = BertForMaskedLM.from_pretrained(args.plm_path)
        # plm = GPT2LMHeadModel.from_pretrained(args.plm_path)
        plm.eval()

        cnt = 0;

        for i in tqdm(range(len(tgts))):
            tgt_sentences = tgts[i]
            probabilities = sentence_probability(tgt_sentences, bert_tokenizer, plm)
            max_probability = 0
            max_probability_idx = 0
            for j in range(len(probabilities)):
                cnt += 1
                if probabilities[j] > max_probability:
                    max_probability = probabilities[j]
                    max_probability_idx = j

            cnt += 1
            o.write(tgt_sentences[max_probability_idx] + "\n")


if __name__ == "__main__":
    begin_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--models_path',
                        required=True)
    parser.add_argument('--output_path',
                        required=True)
    parser.add_argument('--plm_path',
                        required=True)
    # parser.add_argument('--log_path',
    #                     required=True)        
    args = parser.parse_args()
    main(args)

    end_time = time.time()
    run_time = round(end_time - begin_time)
    hour = run_time // 3600
    minute = (run_time - 3600*hour) // 60
    second = run_time - 3600 * hour - 60 * minute
    print(f"Executed in {hour}H {minute}Min {second}Sec")
