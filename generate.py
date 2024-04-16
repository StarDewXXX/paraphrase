import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json

class ParaphraseDataset(Dataset):
    def __init__(self, data_path, num_data, tokenizer):
        self.input_texts = []
        self.target_texts = []
        with open(data_path) as f:
            data = json.load(f)
        
        self.num_data = min(num_data, len(data))
        for i in range(self.num_data):
            item = data[i]
            self.input_texts.append(item['input'])
            self.target_texts.append(item['target'])

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]

        return {'input': input_text, 'target': target_text}

def generate_paraphrases_batched(model, tokenizer, dataset, batch_size, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    paraphrases = []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in dataloader:
            input_texts = batch['input']
            target_texts = batch['target']
            input_encodings = tokenizer(input_texts, padding="longest", truncation=True, max_length=64, return_tensors="pt").to(device)

            output_ids = model.generate(input_ids=input_encodings.input_ids, attention_mask=input_encodings.attention_mask, max_length=64, num_return_sequences=1, early_stopping=True)
            output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for input_text, target_text, generated_text in zip(input_texts, target_texts, output_texts):
                paraphrases.append({'input': input_text, 'target': target_text, 'generate': generated_text})

    with open(output_file, 'w') as f:
        json.dump(paraphrases, f, indent=4)

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Load trained model weights
model_path = "./ckpts/qqp_sft"
model.load_state_dict(torch.load(model_path))

# Load inference dataset
dataset_name = "qqp"
train_path = f"./data/{dataset_name}_train.json"
iter = 0
num_data = 1000
test_dataset = ParaphraseDataset(train_path, num_data, tokenizer=tokenizer)

# Generate paraphrases in batches and save to JSON file
output_file = train_path = f"./data/{dataset_name}_self_{iter}.json"
batch_size = 16
generate_paraphrases_batched(model, tokenizer, test_dataset, batch_size, output_file)

print("Paraphrases generated and saved successfully!")
