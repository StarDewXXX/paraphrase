import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Model,T5ForConditionalGeneration, T5Tokenizer
from torch.optim import Adam
import json
from tqdm import tqdm
# 加载数据集
class ParaphraseDataset(Dataset):
    def __init__(self, data_path, num_data):
        self.input_texts = []
        self.target_texts = []
        with open(data_path) as f:
            data = json.load(f)
        
        self.num_data = min(num_data, len(data))
        for i in range(self.num_data):
            item = data[i]
            self.input_texts.append(item['input'])
            self.target_texts.append(item['target'])


    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]

        return {'input':input_text,'target':target_text}

# 初始化tokenizer和模型
model_path = "google/flan-t5-base" #"google-t5/t5-base"#"google/t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# 准备数据集


# 训练循环
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
max_length = 64
batch_size = 32
accum = 1
lr = 5e-5
dataset_name = "qqp"
train_path = f"./data/{dataset_name}_train.json"
num_data = 204800
train_dataset = ParaphraseDataset(train_path, num_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

optimizer = Adam(model.parameters(), lr=lr)

for epoch in range(3):  # 3个epoch
    for index,batch in enumerate(tqdm(train_loader)):
        input_batch = batch['input']
        target_batch = batch['target']

        input_encoding = tokenizer(input_batch, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
        target_encoding = tokenizer(target_batch, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = input_encoding["input_ids"].to(device)
        attention_mask = input_encoding["attention_mask"].to(device)
        labels = target_encoding["input_ids"].to(device)
        decoder_attention_mask = target_encoding["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, decoder_attention_mask=decoder_attention_mask)
        loss = outputs.loss / accum

        # 反向传播并更新参数
        loss.backward()
        if (index+1) % accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            print("Epoch:", epoch, "Loss:", loss.item()*accum)

torch.save(model.state_dict(),f"./ckpts/{dataset_name}_sft")
