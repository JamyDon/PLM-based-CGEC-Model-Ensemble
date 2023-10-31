from m2convertor import M2Processor

def generate_tgt(src, edits):
    tgts = []
    for edit in edits:
        m2_item = M2Processor(src, edit)
        tgt_list = m2_item.get_para()
        tgt = tgt_list[0].replace(" ", "")
        tgts.append(tgt)
    return tgts
