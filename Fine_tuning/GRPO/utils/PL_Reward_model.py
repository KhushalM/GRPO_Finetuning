# train_and_push_pl_reward.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoTokenizer,
    AdamW, get_scheduler
)
from datasets import load_dataset
from huggingface_hub import HfApi, login
from tqdm import tqdm
import argparse

class ListwiseRankingDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=256):
        """
        hf_dataset: a Hugging Face Dataset where each row has:
           - "prompt": str
           - "completions": List[str]
           - "ranking": List[int]   (0 = best)
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = item["prompt"]
        comps  = item["completions"]
        ranks  = torch.tensor(item["ranking"], dtype=torch.long)

        # tokenize each completion
        encs = [
            self.tokenizer(
                prompt + comp,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            for comp in comps
        ]
        input_ids     = torch.stack([e["input_ids"].squeeze(0)     for e in encs], dim=0)  # (N, T)
        attention_mask= torch.stack([e["attention_mask"].squeeze(0) for e in encs], dim=0)  # (N, T)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "ranking":        ranks
        }

def collate_fn(batch):
    # batch: List[{"input_ids":(N,T), "attention_mask":(N,T), "ranking":(N,)}]
    input_ids      = torch.stack([b["input_ids"]      for b in batch], dim=0)  # (B, N, T)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    rankings       = torch.stack([b["ranking"]        for b in batch], dim=0)  # (B, N)

    B, N, T = input_ids.shape 
    # flatten prompt/completion dim for model
    input_ids      = input_ids.view(B * N, T)
    attention_mask = attention_mask.view(B * N, T)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "rankings":       rankings
    }

class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        H = self.encoder.config.hidden_size
        self.v_head = nn.Linear(H, 1)

    def forward(self, input_ids, attention_mask):
        out          = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden  = out.last_hidden_state     # (B*N, T, H)
        pooled       = last_hidden[:, -1, :]      # take final token: (B*N, H)
        reward_score = self.v_head(pooled).squeeze(-1)  # (B*N,)
        return reward_score

def plackett_luce_loss(scores, rankings):
    """
    scores: Tensor (B, N)
    rankings: LongTensor (B, N) where 0 is best
    """
    B, N = scores.size()
    loss = 0.0
    for b in range(B):
        order = rankings[b]              # e.g. tensor([2,0,3,1])
        ordered_scores = scores[b][order]  # permute into true ranking
        for t in range(N):
            num = ordered_scores[t]
            den = torch.logsumexp(ordered_scores[t:], dim=0)
            loss += -(num - den)
    return loss / B

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default = 'KhushalM/Qwen2.5-rlaif-Feynman-Dataset', type=str)
    parser.add_argument("--base_model", default = 'KhushalM/Qwen2.5-1.5-SFT-Merged', type=str)
    parser.add_argument("--reward_model_repo", default = 'KhushalM/Qwen2.5-PlRewardModel', type=str)
    parser.add_argument("--output_dir",      type=str, default="./pl_reward")
    parser.add_argument("--batch_size",      type=int, default=4)
    parser.add_argument("--epochs",          type=int, default=3)
    parser.add_argument("--lr",              type=float, default=2e-5)
    parser.add_argument("--max_length",      type=int, default=256)
    parser.add_argument("--warmup_steps",    type=int, default=100)
    parser.add_argument("--device",          type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) load HF dataset
    raw = load_dataset(args.dataset_name, use_auth_token=True)
    # assume 'train' split
    ds  = raw["train"] if "train" in raw else raw

    # 2) prepare tokenizer + dataset + dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_auth_token=True)
    ds_pt     = ListwiseRankingDataset(ds, tokenizer, max_length=args.max_length)
    loader    = DataLoader(
        ds_pt, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn
    )

    # 3) build model & optimizer & scheduler
    model     = RewardModel(args.base_model).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(loader)
    scheduler   = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # 4) training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        bar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in bar:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            rank  = batch["rankings"].to(device)            # (B, N)

            raw_scores = model(ids, mask)                   # (B*N,)
            B, N       = rank.size()
            scores     = raw_scores.view(B, N)              # (B, N)

            loss = plackett_luce_loss(scores, rank)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            bar.set_postfix(loss=loss.item())

        # save epoch checkpoint
        torch.save(
            model.state_dict(),
            os.path.join(args.output_dir, f"reward_model_epoch{epoch}.pt")
        )
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # 5) create & push to HF repo
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)
    model.push_to_hub(args.reward_model_repo)
    tokenizer.push_to_hub(args.reward_model_repo)
    print(f"\n✅ Reward model pushed to https://huggingface.co/{args.reward_model_repo}")
    return

if __name__ == "__main__":
    main()
