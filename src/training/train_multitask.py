#!/usr/bin/env python3
"""
train_multitask.py  —  NepPlural Step 4 (training)

Local, seed-looping version of finetune_nlu.ipynb. Preserves the original
multi-task architecture (shared encoder + per-axis linear heads, mean pooling,
class-weighted CE, best-val-checkpoint). Two things change vs the notebook:

  1. Splits are FIXED and read from data/splits/{train,val,test}.csv
     (test = the 300 human-gold comments). No random re-splitting.
  2. Label maps are built from the UNION of the three splits, so every class
     — including No-Persona — is covered even if absent from one split.

Runs the same model over several seeds and writes per-seed metrics + test
predictions to Results_v2/<model_tag>/seed<k>_*. Aggregate across seeds with
aggregate_results.py.

Example (one model, 5 seeds):
  python src/training/train_multitask.py --model xlm-roberta-base
Loop over all five encoders in the shell:
  for m in xlm-roberta-base bert-base-multilingual-cased \
           IRIIS-RESEARCH/BERT_Nepali_110M IRIIS-RESEARCH/RoBERTa_Nepali_125M \
           NepBERTa/NepBERTa ; do
    python src/training/train_multitask.py --model "$m"
  done
"""

import os, json, random, copy, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

TASKS = ["intent", "primary_driver", "value_orientation", "affect"]
TEXT_COL = "comment"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_label_maps(frames):
    """Union label maps across all splits so no class (incl No-Persona) is missed."""
    alldf = pd.concat(frames, ignore_index=True)
    label2id = {t: {lab: i for i, lab in enumerate(sorted(alldf[t].dropna().unique()))} for t in TASKS}
    id2label = {t: {i: lab for lab, i in m.items()} for t, m in label2id.items()}
    return label2id, id2label


class CommentDataset(Dataset):
    def __init__(self, frame, label2id):
        self.texts = frame[TEXT_COL].astype(str).tolist()
        self.labels = {t: frame[t].map(label2id[t]).tolist() for t in TASKS}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        item = {"text": self.texts[i]}
        for t in TASKS:
            item[t] = self.labels[t][i]
        return item


class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({t: nn.Linear(hidden, n) for t, n in num_labels.items()})

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return {t: head(self.dropout(pooled)) for t, head in self.heads.items()}


def run_seed(seed, args, train_df, val_df, test_df, label2id, id2label, device):
    set_seed(seed)
    use_amp = device.type == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def collate(batch):
        enc = tokenizer([b["text"] for b in batch], truncation=True,
                        max_length=args.max_length, padding=True, return_tensors="pt")
        labels = {t: torch.tensor([b[t] for b in batch], dtype=torch.long) for t in TASKS}
        return enc, labels

    train_loader = DataLoader(CommentDataset(train_df, label2id), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(CommentDataset(val_df, label2id), batch_size=args.batch_size * 2,
                            shuffle=False, collate_fn=collate)
    test_loader = DataLoader(CommentDataset(test_df, label2id), batch_size=args.batch_size * 2,
                             shuffle=False, collate_fn=collate)

    # class weights from train only (inverse-freq, damped) — matches original
    class_weights = {}
    for t in TASKS:
        counts = train_df[t].map(label2id[t]).value_counts().sort_index()
        counts = counts.reindex(range(len(label2id[t])), fill_value=1)
        w = (len(train_df) / (len(counts) * counts.values)) ** args.weight_damp
        class_weights[t] = torch.tensor(w, dtype=torch.float32, device=device)

    model = MultiTaskClassifier(args.model, {t: len(label2id[t]) for t in TASKS},
                                dropout=args.dropout).to(device)

    total_steps = len(train_loader) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_frac * total_steps), total_steps)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    loss_fns = {t: nn.CrossEntropyLoss(weight=class_weights[t]) for t in TASKS}

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        preds = {t: [] for t in TASKS}; golds = {t: [] for t in TASKS}
        for enc, labels in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(**enc)
            for t in TASKS:
                preds[t].extend(logits[t].argmax(-1).cpu().tolist())
                golds[t].extend(labels[t].tolist())
        macro = {t: f1_score(golds[t], preds[t], average="macro", zero_division=0) for t in TASKS}
        return macro, preds, golds

    best_score, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for enc, labels in train_loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = {t: v.to(device) for t, v in labels.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(**enc)
                loss = sum(loss_fns[t](logits[t], labels[t]) for t in TASKS)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            running += loss.item()
        val_f1, _, _ = evaluate(val_loader)
        mean_f1 = float(np.mean(list(val_f1.values())))
        if mean_f1 > best_score:
            best_score, best_state = mean_f1, copy.deepcopy(model.state_dict())
        print(f"  seed {seed} epoch {epoch}/{args.epochs} loss={running/len(train_loader):.4f} "
              f"val mean-F1={mean_f1:.3f}")

    model.load_state_dict(best_state)
    test_f1, test_preds, test_golds = evaluate(test_loader)

    tag = args.model.replace("/", "__")
    out = f"{args.out_dir}/{tag}"; os.makedirs(out, exist_ok=True)
    summary = {}
    for t in TASKS:
        summary[t] = {"macro_f1": round(test_f1[t], 4),
                      "accuracy": round(accuracy_score(test_golds[t], test_preds[t]), 4)}
    summary["mean_macro_f1"] = round(float(np.mean(list(test_f1.values()))), 4)
    with open(f"{out}/seed{seed}_metrics.json", "w") as f:
        json.dump({"model": args.model, "seed": seed, "best_val_mean_macro_f1": round(best_score, 4),
                   "test": summary}, f, indent=2)
    pred = test_df[[TEXT_COL] + TASKS].copy()
    for t in TASKS:
        pred[f"{t}_pred"] = [id2label[t][p] for p in test_preds[t]]
    pred.to_csv(f"{out}/seed{seed}_test_predictions.csv", index=False)
    with open(f"{out}/label_maps.json", "w") as f:
        json.dump(label2id, f, indent=2, ensure_ascii=False)
    print(f"  seed {seed} DONE  test mean macro-F1 = {summary['mean_macro_f1']:.4f}")
    return summary["mean_macro_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--out_dir", default="src/training/Results_v2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight_damp", type=float, default=0.5)
    ap.add_argument("--dry_run", action="store_true", help="validate data/label maps and exit (no model)")
    args = ap.parse_args()

    train_df = pd.read_csv(f"{args.splits_dir}/train.csv")
    val_df = pd.read_csv(f"{args.splits_dir}/val.csv")
    test_df = pd.read_csv(f"{args.splits_dir}/test.csv")
    label2id, id2label = build_label_maps([train_df, val_df, test_df])

    print(f"splits: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    for t in TASKS:
        classes = list(label2id[t])
        print(f"  {t}: {len(classes)} classes -> {classes}")
        assert "No-Persona" in classes, f"No-Persona missing from {t} label map!"
    if args.dry_run:
        print("\nDRY RUN OK — label maps built, No-Persona present in every axis. Exiting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nModel: {args.model}\nSeeds: {args.seeds}\n")
    scores = [run_seed(s, args, train_df, val_df, test_df, label2id, id2label, device)
              for s in args.seeds]
    print(f"\n{args.model}: mean test macro-F1 over {len(scores)} seeds = "
          f"{np.mean(scores):.4f} ± {np.std(scores):.4f}")


if __name__ == "__main__":
    main()
