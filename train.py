import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import json
import argparse
from tqdm import tqdm
from src.config import NeuraCoderConfig
from src.model import NeuraCoder
from src.tokenizer import NeuraCoderTokenizer

class CodeDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(json_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        code = self.data[idx]["code"]
        tokens = self.tokenizer.encode(code, add_special_tokens=True)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def train_epoch(model, loader, optimizer, scaler, criterion, epoch, device, grad_accum, config, clip_threshold):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=dist.get_rank() != 0)
    
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        with autocast(enabled=config.use_mixed_precision):
            logits, moe_loss = model(x)
            loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))
            loss = loss + moe_loss
            loss = loss / grad_accum
        
        scaler.scale(loss).backward()
        
        if (step + 1) % grad_accum == 0:
            if config.clip_threshold_dynamic:
                # Gradient clipping پویا: norm کل گرادیان‌ها
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                clip = max(config.gradient_clipping, total_norm * 0.5)  # adaptive
            else:
                clip = config.gradient_clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum
        pbar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})
    return total_loss / len(loader)

def main(rank, world_size, args):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    config = NeuraCoderConfig()
    config.num_layers = args.num_layers
    config.use_moe = args.use_moe
    config.use_flash_attention = args.flash
    
    model = NeuraCoder(config).to(device)
    model = DDP(model, device_ids=[rank])
    
    tokenizer = NeuraCoderTokenizer()
    dataset = CodeDataset(args.data_path, tokenizer, max_length=config.max_seq_len)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=config.weight_decay)
    
    # Warmup + Cosine scheduler
    total_steps = len(loader) * args.epochs // args.gradient_accumulation
    warmup_steps = config.warmup_steps
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    scaler = GradScaler(enabled=config.use_mixed_precision)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        loss = train_epoch(model, loader, optimizer, scaler, criterion, epoch, device, args.gradient_accumulation, config, config.gradient_clipping)
        scheduler.step()
        if rank == 0:
            print(f"Epoch {epoch} done. Loss: {loss:.4f}")
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(args.save_dir, f"epoch_{epoch}.pt"))
            config.save_pretrained(args.save_dir)
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--flash", action="store_true", default=True)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No GPU found")
    torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size)