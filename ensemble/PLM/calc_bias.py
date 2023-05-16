def calc_bias(text, tokenizer):
    tokens, ids = tokenize_text(tokenizer, text)
    return len(text) - len(tokens)