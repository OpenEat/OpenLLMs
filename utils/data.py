from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index # -100

def get_labels_gen(pad_token_id):
    def get_labels(data):
        input_ids = data["input_ids"]
        labels = input_ids.clone()
        labels[labels == pad_token_id] = IGNORE_TOKEN_ID
        return {"labels": labels}
    return get_labels

def get_pad_len(pad_token_id):
    """ get_seq_len """
    def pad_length(data):
        input_ids = data["input_ids"]
        padded_len = len(input_ids)
        pad_len = 0
        for id in input_ids:
            if id == pad_token_id:
                pad_len += 1
            else:
                break
        return {"pad_len": pad_len}
    return pad_length

def sft2pretrain(data):
    """ sft2pretrain """
    if "sft" in data:
        texts = []
        chats = data["conversations"]
        if chats[0]["user"] != "user":
            chats = chats[1:]
        for i in range(len(chats) // 2):
            prompt = chats[2 * i]
            completion = chats[2 * i + 1]
            if not (prompt["from"] == "user" and completion["from"] == "assistant"):
                continue
            prompt = prompt["value"].strip()
            completion = completion["value"].strip()
            chat = "user:{}\nsystem:{}".format(prompt, completion)
            texts.append(chat)
        text = "\n".join(texts)
        data["text"] = text
    return data

def sft2pair(data):
    """ data """
    if "sft" in data:
        texts = []
        chats = data["conversations"]
        if chats[0]["user"] != "user":
            chats = chats[1:]
        for i in range(len(chats) // 2):
            prompt = chats[2 * i]
            completion = chats[2 * i + 1]
            if not (prompt["from"] == "user" and completion["from"] == "assistant"):
                continue
            prompt = prompt["value"].strip()
            completion = completion["value"].strip()
            chat = {"user": prompt, "assistant": completion}
            texts.append(chat)
    return data


