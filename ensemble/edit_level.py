import sys
import os
import time
import argparse
import torch
from collections import Counter
from tqdm import tqdm
sys.path.append('./ensemble/')
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from PLM.bert_scorer import sentence_probability
# from pytorch_pretrained_bert import BertTokenizer
# from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel
# from PLM.gpt2_scorer import sentence_probability
from utils import generate_tgt
from traditional import get_models
from rule_ensemble import parse_m2, validate


class Error:
    def __init__(self, start, end, type_):
        self.start = start
        self.end = end
        self.type = type_
        self.edit_candidates = []


class Edit:
    def __init__(self, edit, src):
        self.edit = edit.replace("A ", "", 1)
        type_, start, end, tokens = self.get_info(edit, src)
        self.type = type_
        self.tokens = tokens
        self.span = (start, end)
    
    def get_info(self, edit, src):
        ed = edit.split("|||")
        span = ed[0].split(" ")
        start = eval(span[1])
        end = eval(span[2])
        tokens = ed[2].replace(" ", "")
        if edit == "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0":
            type_ = "N"
        elif "|||W|||" in edit:
            type_ = "W"
            end = start + len(tokens)
        elif "|||M|||" in edit:
            assert start == end
            type_ = "M"
            end_tok = "[SEP]"
            if start < len(src):
                end_tok = src[start]
            # tokens = tokens + end_tok
            start = start
            end = start + len(tokens) + 1#！！！
        elif "|||R|||" in edit:
            type_ = "R"
            if end >= len(src):
                tokens = "[SEP]"
            else:
                tokens = src[end]
            end = start + 1
        elif "|||S|||" in edit:
            type_ = "S"
            end = start + len(tokens)
        elif "|||SP|||" in edit:
            type_ = "SP"
            end = start + len(tokens)
        else:
            print("Undefined edit type")
            print(edit)
            exit()
        
        return type_, start, end, tokens

    
    def to_error(self, src):
        edit = self.edit.split("|||")
        spans = edit[0].split(" ")
        start = eval(spans[0])
        end = eval(spans[1])

        if self.type == "M" or self.type == "R":
            end += 1

        return Error(start, end, self.type)


def main(args):
    total_edits = []
    models = get_models(args.models_path)

    for model in models:
        sources, edits = parse_m2(model)
        total_edits.append(edits)

    with open(args.output_path, "w", encoding="utf-8") as o:
        bert_tokenizer = BertTokenizer.from_pretrained(args.plm_path)
        plm = BertForMaskedLM.from_pretrained(args.plm_path)
        # plm = GPT2LMHeadModel.from_pretrained(args.plm_path)
        plm.eval()

        for i in tqdm(range(len(sources))):
            src_src = sources[i]
            src_tokens = src_src.split(" ")[1:]
            src = "".join(src_tokens)
            error_candidates = []

            for edits in total_edits:
                for e in edits[i]:
                    err_exist = False
                    if "|||NA|||" in e or "|||noop|||" in e:
                        continue
                    edit = Edit(e, src)
                    error = edit.to_error(src)
                    
                    for err in error_candidates:
                        if err_exist:
                            break
                        if err.start == error.start and err.end == error.end and err.type == error.type:
                            err_exist = True
                            ed_exist = False
                            for ed in err.edit_candidates:
                                if ed.edit == edit.edit:
                                    ed_exist = True
                                    break
                            if not ed_exist:
                                err.edit_candidates.append(edit)
                    
                    if err_exist == False:
                        error.edit_candidates.append(edit)
                        error_candidates.append(error)

            final_edits = []
            for error in error_candidates:
                max_prob = 0
                max_prob_idx = 0

                tgt_sentences = [src]
                for edit in error.edit_candidates:
                    tgt_sentences.append(generate_tgt(src_src.replace("S ", "", 1), [[edit.edit]])[0])
                probs = sentence_probability(tgt_sentences, bert_tokenizer, plm)

                for j in range(len(tgt_sentences)):
                    prob = probs[j]
                    if prob > max_prob:
                        max_prob = prob
                        max_prob_idx = j
                
                if max_prob_idx > 0:
                    final_edits.append(("A " + error.edit_candidates[max_prob_idx-1].edit, max_prob))
            
            if final_edits:
                final_edits = validate(final_edits)
                out = src_src + "\n" + "\n".join(final_edits) + "\n\n"
            else:
                out = src_src + "\n" + "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0" + "\n\n"
            o.write(out)


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
