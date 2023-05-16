import sys
import os
import time
import argparse
from collections import Counter
from modules.classifier import check_spell_error
from tqdm import tqdm
sys.path.append('./ensemble/')
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# from PLM.bert_scorer import sentence_probability, padding
from pytorch_pretrained_bert import BertTokenizer
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel
from PLM.gpt2_scorer import sentence_probability, padding
from m2convertor import generate_tgt
from traditional import get_models, parse_m2, validate
import torch


class Error:
    def __init__(self, start, end, type):
        self.start = start
        self.end = end
        self.type = type
        self.edit_candidates = []


class Edit:
    def __init__(self, edit):
        self.edit = edit
        self.type = self.get_type()
        self.prob = 0
    
    def get_type(self):
        if self.edit == "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0":
            return "N"
        elif "|||W|||" in self.edit:
            return "W"
        elif "|||M|||" in self.edit:
            return "M"
        elif "|||R|||" in self.edit:
            return "R"
        elif "|||S|||" in self.edit:
            return "S"
        elif "|||SP|||" in self.edit:
            return "SP"
        else:
            print("Undefined edit type")
            print(self.edit)
            exit()
    
    def to_error(self):
        edit = self.edit.split("|||")
        spans = edit[0].split(" ")
        start = spans[1]
        end = spans[2]
        return Error(start, end, self.type)


def parse_m2(filename):
    sources = []
    edits = []
    with open(filename, "r") as f:
        chunk = []
        for line in f:
            if line == "\n":
                sources.append(chunk[0])
                edit_list = []
                for s in chunk[1:]:
                    if s[0] == "A":
                        edit_list.append(s)
                edits.append(edit_list)
                chunk = []
            else:
                chunk.append(line.rstrip("\n"))
        if chunk:
            sources.append(chunk[0])
            edit_list = []
            for s in chunk[2:]:
                if s[0] == "A":
                    edit_list.append(s)
            edits.append(edit_list)
    return sources, edits


def main(args):
    total_edits = []
    models = get_models(args.models_path)

    for model in models:
        sources, edits = parse_m2(model)
        total_edits.append(edits)

    with open(args.output_path, "w", encoding="utf-8") as o:
        bert_tokenizer = BertTokenizer.from_pretrained(args.plm_path)
        plm = GPT2LMHeadModel.from_pretrained(args.plm_path)
        # plm = BertForMaskedLM.from_pretrained(args.plm_path)
        plm.eval()

        candidate_cnt = 0
        large_cnt = 0

        with open("./logs/nlpcc/edit_combination.log", "w") as log:
            for i in tqdm(range(len(sources))):
                src_src = sources[i]
                src_tokens = src_src.split(" ")[1:]
                src = "".join(src_tokens)
                error_candidates = []

                for edits in total_edits:
                    for e in edits[i]:
                        err_exist = False
                        if "|||NA|||" in e:
                            continue
                        edit = Edit(e)
                        error = edit.to_error()
                        
                        for err in error_candidates:
                            if err_exist:
                                break
                            if err.start == error.start and err.end == error.end and err.type == error.type:
                                err_exist = True
                                if edit not in err.edit_candidates:
                                    err.edit_candidates.append(e.replace("A ", "", 1))
                        
                        if err_exist == False:
                            error.edit_candidates.append(e.replace("A ", "", 1))
                            error_candidates.append(error)
                
                edits_cnt = 1
                for error in error_candidates:
                    error.edit_candidates.append("-1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0")
                    error.edit_candidates = list(set(error.edit_candidates))
                    edits_cnt *= len(error.edit_candidates)

                if edits_cnt > 300:
                    large_cnt += 1
                    out = src_src + "\n" + "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0" + "\n\n"
                    o.write(out)
                    continue

                edits_candidates = []
                for error in error_candidates:
                    if edits_candidates:
                        new_edits_candidates = []
                        for edit in error.edit_candidates:
                            for edits in edits_candidates:
                                new_edits = edits[:]
                                new_edits.append(edit)
                                new_edits_candidates.append(new_edits)
                        edits_candidates = new_edits_candidates[:]
                    else:
                        edits_candidates = [[edit] for edit in error.edit_candidates]
                final_edits_candidates = []
                for edits in edits_candidates:
                    edits = list(set(edits))
                    edits.sort()
                    if "-1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0" in edits:
                        edits.remove("-1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0")
                    if not edits:
                        edits = ["-1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0"]
                    final_edits_candidates.append(edits)
                if not final_edits_candidates:
                    final_edits_candidates.append(["-1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0"])
                edits_candidates = final_edits_candidates
                edits_candidates.sort()
                candidate_cnt += len(edits_candidates)

                max_sent_prob = 0
                max_prob_idx = -1

                tgt_sentences = generate_tgt(src_src.replace("S ", "", 1), edits_candidates)
                tgt_probs = sentence_probability(tgt_sentences, bert_tokenizer, plm)
                
                for j in range(len(tgt_sentences)):
                    prob = tgt_probs[j]
                    if prob > max_sent_prob:
                        max_sent_prob = prob
                        max_prob_idx = j
                
                final_edits = []
                for edit in edits_candidates[max_prob_idx]:
                    edit = "A " + edit
                    final_edits.append((edit, max_sent_prob))

                if final_edits:
                    final_edits = validate(final_edits)
                    out = src_src + "\n" + "\n".join(final_edits) + "\n\n"
                else:
                    out = src_src + "\n" + "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0" + "\n\n"
                o.write(out)

        print("Percentage: " + str(100-100*large_cnt/len(sources)) + "%\n")
        print("Candidates: " + str(candidate_cnt) + "%\n")


if __name__ == "__main__":
    begin_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--models_path',
                        required=True)
    parser.add_argument('--output_path',
                        required=True)
    parser.add_argument('--plm_path',
                        required=True)
    args = parser.parse_args()
    main(args)

    end_time = time.time()
    run_time = round(end_time - begin_time)
    hour = run_time // 3600
    minute = (run_time - 3600*hour) // 60
    second = run_time - 3600 * hour - 60 * minute
    print(f"Executed in {hour}H {minute}Min {second}Sec")
