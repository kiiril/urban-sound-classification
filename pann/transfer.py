import torch
import torch.nn as nn
from models import Cnn14, Cnn14_16k, init_layer
from dataset import UrbanSound8KWav
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt

class TransferCnn14(nn.Module):
    def __init__(self,
                 base_model: str,
                 sample_rate: int,
                 window_size: int,
                 hop_size: int,
                 mel_bins: int,
                 fmin: int,
                 fmax: int,
                 classes_num: int,
                 freeze_base: bool,
                 dropout_rate: float = 0.3):
        super().__init__()
        audioset_classes = 527

        Base = Cnn14 if base_model == 'Cnn14' else Cnn14_16k
        self.base = Base(sample_rate=sample_rate, window_size=window_size,
                         hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                         classes_num=audioset_classes)

        # no Dropout layer, since implemented in the base model
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        self.init_weights()
        
    def init_weights(self):
        init_layer(self.fc_transfer)
        
    def load_from_pretrain(self, pretrained_checkpoint_path, map_location='cuda'):
        checkpoint = torch.load(pretrained_checkpoint_path, map_location=map_location)
        # checkpoints have a 'model' key
        self.base.load_state_dict(checkpoint['model'], strict=False)
        
    def forward(self, x):
        out = self.base(x)
        emb = out['embedding']
        logits = self.fc_transfer(emb)
        probs = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs}
    
    
def load_pann(num_classes: int,
              checkpoint_path: str,
              mode: str = 'fixed_feature',
              variant: str = 'Cnn14_16k',
              dropout_rate: float = 0.3):
    
    if variant == 'Cnn14':
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax = 32000, 1024, 320, 64, 50, 14000
    elif variant == "Cnn14_16k":
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax = 16000, 512, 160, 64, 50, 8000
    else:
        raise ValueError("variant must be 'Cnn14' or 'Cnn14_16k'")
    
    freeze = (mode == "fixed_feature")
    model = TransferCnn14(
        base_model=variant,
        sample_rate=sample_rate, window_size=window_size, hop_size=hop_size,
        mel_bins=mel_bins, fmin=fmin, fmax=fmax,
        classes_num=num_classes,
        freeze_base=freeze,
        dropout_rate=dropout_rate
    )
    model.load_from_pretrain(checkpoint_path)
    
    # Param groups—simple and non-overlapping
    if mode == "fixed_feature":
        param_groups = [
            {"params": model.fc_transfer.parameters(), "lr": 1e-3},
        ]
    elif mode == "fine_tuning":
        param_groups = [
            {"params": model.base.parameters(),       "lr": 1e-4},
            {"params": model.fc_transfer.parameters(),"lr": 5e-4},
        ]
    else:
        raise ValueError("mode must be 'fixed_feature' or 'fine_tuning'.")

    return model, param_groups, sample_rate


def make_collate_fixed(target_samples: int, training: bool):
    def collate(batch):
        wavs, labels = zip(*batch)
        out = []
        for w in wavs:
            n = w.numel()
            if n < target_samples:
                pad = torch.zeros(target_samples, dtype=w.dtype)
                pad[:n] = w
                w = pad
            elif n > target_samples:
                if training:
                    start = random.randint(0, n - target_samples)
                else:
                    start = (n - target_samples) // 2
                w = w[start:start+target_samples]
            out.append(w)
        return torch.stack(out, 0), torch.tensor(labels, dtype=torch.long)
    return collate


def run(mode='fixed_feature', variant='Cnn14_16k', num_of_epochs=5, patience=3):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    checkpoint_path = 'pretrained_models/Cnn14_16k_mAP=0.438.pth'
    model, pg, target_sr = load_pann(10, checkpoint_path, mode, variant)
    model.to(device)
    
    optimizer = torch.optim.Adam(pg, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    target_samples = int(target_sr * 4.0)   # 4 s for US8K
    collate_train = make_collate_fixed(target_samples, training=True)
    collate_eval  = make_collate_fixed(target_samples, training=False)
    
    train_dataset = UrbanSound8KWav('../datasets', folds=range(1, 9), target_sr=16000)
    val_dataset = UrbanSound8KWav('../datasets', folds=[9], target_sr=16000)
    test_dataset = UrbanSound8KWav('../datasets', folds=[10], target_sr=16000)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                          num_workers=2, pin_memory=True, collate_fn=collate_train)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False,
                          num_workers=2, pin_memory=True, collate_fn=collate_eval)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False,
                          num_workers=2, pin_memory=True, collate_fn=collate_eval)
    
    def loop(model, loader, optimizer=None, criterion=None):
        training = optimizer is not None
        model.train(training)
        total = correct = 0
        loss_sum = 0.0

        for wav, y in loader:
            wav = wav.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if training:
                out = model(wav)
                logits = out["logits"]
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    logits = model(wav)["logits"]
                    loss = criterion(logits, y)

            loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return loss_sum / total, correct / total
    
    print('Training started')
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_path = f"pann_{variant}_{mode}_best.pth"
    early_stop_counter = 0
    
    for epoch in range(1, num_of_epochs + 1):
        train_loss, train_acc = loop(model, train_loader, optimizer, criterion)
        val_loss, val_acc = loop(model, val_loader, None, criterion)
        
        # 3. Store metrics for each epoch
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.3f}, Acc: {train_acc:.3f} | Val Loss: {val_loss:.3f}, Acc: {val_acc:.3f}")
        
        # 4. Save model based on best validation LOSS
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved with validation loss: {best_val_loss:.3f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break
            
    # 5. Plotting the results and saving to a file
    epochs_range = range(1, num_of_epochs + 1)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'PANN ({variant}) Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title(f'PANN ({variant}) Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    output_filename = f'pann_{variant}_{mode}_training_plot.png'
    plt.savefig(output_filename)
    print(f"\nPlot saved to {output_filename}")
    plt.close()
            
    # 6. Load best model and run final test evaluation
    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc = loop(model, test_loader, None, criterion)
    print(f"\n*** {variant} | {mode} | test acc: {test_acc:.3f} (from model with val_loss: {best_val_loss:.3f}) ***")


if __name__ == '__main__':
    run(mode='fixed_feature', num_of_epochs=3)
    run(mode='fine_tuning', num_of_epochs=3)   

        
