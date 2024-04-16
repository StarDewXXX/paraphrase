import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Adam
import json
import json
import os
import pickle
import copy
import sys

from torch.nn import functional as F
import torch.optim as optim
# 加载数据集
class SelfParaphraseDataset(Dataset):
    def __init__(self, data_path, num_data):
        self.input_texts = []
        self.target_texts = []
        self.generate_texts = []
        with open(data_path) as f:
            data = json.load(f)
        
        self.num_data = min(num_data, len(data))
        for i in range(self.num_data):
            item = data[i]
            self.input_texts.append(item['input'])
            self.target_texts.append(item['target'])
            self.generate_texts.append(item['generate'])


    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]
        generate_text = self.generate_texts[idx]

        return {'input':input_text,'target':target_text, 'generate':generate_text}


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    pad_token_id,
    average_log_prob: bool = False,):
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
    labels[labels == pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    # print(per_token_logps)
    # print("---------------------")

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


# 训练循环
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化tokenizer和模型
model_path = "t5"
tokenizer = T5Tokenizer.from_pretrained(model_path)
ref_model = T5ForConditionalGeneration.from_pretrained(model_path).to(device).eva;()
policy_model = T5ForConditionalGeneration.from_pretrained(model_path).to(device).train()

# 准备数据集

max_length = 64
batch_size = 8
accum = 4
beta = 0.2
lr = 5e-5
dataset_name = "qqp"
iter = 0
model_id = iter
train_path = f"./data/{dataset_name}_self_{iter}.json"
num_data = 2048
train_dataset = SelfParaphraseDataset(train_path, num_data, tokenizer=tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

optimizer = Adam(policy_model.parameters(), lr=lr)

for epoch in range(3):  # 3个epoch
    for index,batch in enumerate(train_loader):
        input_batch = batch['input']
        target_batch = batch['target']
        generate_batch = batch['generate']

        input_encoding = tokenizer(input_batch, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
        target_encoding = tokenizer(target_batch, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
        generate_encoding = tokenizer(generate_batch, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")

        input_ids = input_encoding["input_ids"].to(device)
        attention_mask = input_encoding["attention_mask"].to(device)

        targets = target_encoding["labels"].to(device)
        decoder_attention_mask_target = target_encoding["decoder_attention_mask"].to(device)

        generates = target_encoding["labels"].to(device)
        decoder_attention_mask_generate = target_encoding["decoder_attention_mask"].to(device)

        with torch.no_grad():
            ref_logits_target = ref_model(input_ids=input_ids, attention_mask=attention_mask, labels=targets, decoder_attention_mask=decoder_attention_mask_target).logits
            ref_logits_generate = ref_model(input_ids=input_ids, attention_mask=attention_mask, labels=generates, decoder_attention_mask=decoder_attention_mask_generate).logits
        
        policy_logits_target = policy_model(input_ids=input_ids, attention_mask=attention_mask, labels=targets, decoder_attention_mask=decoder_attention_mask_target).logits
        policy_logits_generate = policy_model(input_ids=input_ids, attention_mask=attention_mask, labels=generates, decoder_attention_mask=decoder_attention_mask_generate).logits
        
        real_logps = get_batch_logps(policy_logits_target, targets, tokenizer.pad_token_id)
        ref_real_logps = get_batch_logps(ref_logits_target, targets, tokenizer.pad_token_id)
        dispreferred_logps = get_batch_logps(policy_logits_generate, generates, tokenizer.pad_token_id)
        ref_dispreferred_logps = get_batch_logps(ref_logits_generate, generates, tokenizer.pad_token_id)
        
        losses = - F.logsigmoid(beta*((real_logps - ref_real_logps) - (dispreferred_logps - ref_dispreferred_logps))) / accum
        loss = torch.mean(losses)

        loss.backward()
        if (index+1) % accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            print("Epoch:", epoch, "Loss:", loss.item()*accum)

torch.save(policy_model.state_dict(),f"./ckpts/{dataset_name}_{model_id}_self")
