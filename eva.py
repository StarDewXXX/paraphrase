import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Model,T5ForConditionalGeneration, T5Tokenizer
from torch.optim import Adam
import json
from tqdm import tqdm

def evaluate(model,inputs,val_batch_size,):
    for i in range(0, int(num_val / val_batch_size)):

        input_sentences = inputs[i * val_batch_size:(i + 1) * val_batch_size]
        output_sentences = model.generate(
                inputs_ids=input_embeds_augmented,
                attention_mask=input_attention_mask,
                do_sample=False,
                max_length=max_length
            )
        # print(output_sentences)
        for j in range(0, val_batch_size):
            group = output_sentences[j * num_return_sequences:(j + 1) * num_return_sequences]
            self_bleu_list = []
            # print(i * val_batch_size + j)
            # print("[Input] ", inputs[i * val_batch_size + j])
            for sent in group:
                self_bleu = sacrebleu.corpus_bleu([sent], [[inputs[i * val_batch_size + j]]], lowercase=True).score
                self_bleu_list.append(self_bleu)
                # print("[Output] ", sent,"[Bleu]:",self_bleu)
            best_index = torch.topk(torch.tensor(self_bleu_list), num_return_sequences, largest=False).indices[low - 1]
            # print('best:',best_index+1)
            outputs.append(group[best_index])
            # print(i * val_batch_size + j)
            # print("[Input] ", inputs[i * val_batch_size + j])
            # print("[Output]",group[best_index])
            # print("[Ref]   ",refs_padded[i * val_batch_size + j][0])

    alpha = 0.8
    # tgt_bleu = sacrebleu.corpus_bleu(outputs,refs_padded[0:1], lowercase=True).score
    tgt_bleu = sacrebleu.corpus_bleu(outputs, list(zip(*refs_padded)), lowercase=True).score
    self_bleu = sacrebleu.corpus_bleu(outputs, [inputs], lowercase=True).score
    ibleu = alpha * tgt_bleu - (1 - alpha) * self_bleu

    print("num_return:",num_return_sequences)
    print("low:",low)
    print("num_beam:",num_beams)
    print("top_p:",top_p)
    print("top_k",top_k)
    print("target-bleu:", tgt_bleu)
    print("self-bleu:", self_bleu)
    print("ibleu:", ibleu)
    print("_____________________________________________")