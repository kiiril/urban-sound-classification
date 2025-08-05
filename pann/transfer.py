import torch
import torch.nn as nn
from models import Cnn14, Cnn14_16k, init_layer
from dataset import UrbanSound8KWav
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import time

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
    

def count_trainable_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

def measure_inference_time(model, loader, device, num_batches=10):
    """Measure average inference time per sample"""
    model.eval()
    times = []
    total_samples = 0
    
    with torch.no_grad():
        for i, (wav, y) in enumerate(loader):
            if i >= num_batches:
                break
                
            wav = wav.to(device)
            batch_size = wav.size(0)
            
            # Time inference
            start_time = time.time()
            _ = model(wav)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            batch_time = end_time - start_time
            times.append(batch_time)
            total_samples += batch_size
    
    total_inference_time = sum(times)
    avg_time_per_sample = total_inference_time / total_samples
    
    return avg_time_per_sample * 1000  # Convert to milliseconds


def evaluate_with_metrics(model, loader, criterion, device, class_names=None):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for wav, y in loader:
            wav, y = wav.to(device), y.to(device)
            out = model(wav)
            logits = out["logits"]  # PANN returns dict with "logits" key
            loss = criterion(logits, y)
            
            loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    # calculate accuracy and loss
    accuracy = correct / total
    avg_loss = loss_sum / total
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate precision, recall, F1-score
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }
    
    return metrics


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_path=None):
    """
    Plot and save confusion matrix using seaborn
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def print_detailed_metrics(metrics, class_names, dataset_name=""):
    """
    Print comprehensive evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"DETAILED EVALUATION METRICS{' - ' + dataset_name if dataset_name else ''}")
    print(f"{'='*60}")
    
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {metrics['recall_macro']:.4f}")
    print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
    
    print(f"\n{'Per-Class Metrics:':<20}")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {metrics['precision_per_class'][i]:<12.4f} "
              f"{metrics['recall_per_class'][i]:<12.4f} {metrics['f1_per_class'][i]:<12.4f} "
              f"{metrics['support_per_class'][i]:<10}")
    

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

    # Count trainable parameters
    trainable_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params:,}")
    
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
    
    # UrbanSound8K class names (same as AST)
    class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 
                   'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 
                   'siren', 'street_music']
    
    
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
    training_start_time = time.time()
    
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

    # Calculate total training time
    total_training_time = time.time() - training_start_time
    print(f"Total training time: {total_training_time:.1f} seconds ({total_training_time/60:.1f} minutes)")
            
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
    print(f"Training plot saved to {output_filename}")
    plt.close()
            
    # 6. Load best model and run final test evaluation
    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(best_path, map_location=device))

    # Measure inference time per sample
    inference_time_per_sample = measure_inference_time(model, test_loader, device)
    print(f"Inference time per sample: {inference_time_per_sample:.2f} ms")

    # COMPREHENSIVE EVALUATION
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)

    # Test set evaluation  
    test_metrics = evaluate_with_metrics(model, test_loader, criterion, device, class_names)
    print_detailed_metrics(test_metrics, class_names, "TEST SET")

    # Plot test confusion matrix
    plot_confusion_matrix(
        test_metrics['confusion_matrix'], 
        class_names, 
        title=f'PANN {variant} {mode.title()} - Test Confusion Matrix',
        save_path=f'pann_{variant}_{mode}_test_confusion_matrix.png'
    )

    # Save detailed results to file
    results_file = f'pann_{variant}_{mode}_detailed_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"PANN {variant} {mode.upper()} MODE - COMPREHENSIVE EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        
        # Add the 3 key metrics
        f.write("KEY PERFORMANCE METRICS:\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Total Training Time: {total_training_time/60:.1f} minutes\n")
        f.write(f"Inference Time per Sample: {inference_time_per_sample:.2f} ms\n\n")
        
        f.write("TEST SET RESULTS:\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Macro Precision: {test_metrics['precision_macro']:.4f}\n")
        f.write(f"Macro Recall: {test_metrics['recall_macro']:.4f}\n")
        f.write(f"Macro F1-Score: {test_metrics['f1_macro']:.4f}\n\n")
        
        f.write("PER-CLASS TEST SET METRICS:\n")
        f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 70 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<15} {test_metrics['precision_per_class'][i]:<12.4f} "
                   f"{test_metrics['recall_per_class'][i]:<12.4f} {test_metrics['f1_per_class'][i]:<12.4f} "
                   f"{test_metrics['support_per_class'][i]:<10}\n")
    
    print(f"\nDetailed results saved to {results_file}")
    
    # Final summary with the 3 key metrics
    print(f"\n*** PANN {variant} {mode} FINAL RESULTS ***")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total training time: {total_training_time/60:.1f} minutes")
    print(f"Inference time per sample: {inference_time_per_sample:.2f} ms")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1-Score (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"Test Precision (Macro): {test_metrics['precision_macro']:.4f}")
    print(f"Test Recall (Macro): {test_metrics['recall_macro']:.4f}")


if __name__ == '__main__':
    run(mode='fixed_feature', num_of_epochs=3)
    run(mode='fine_tuning', num_of_epochs=3)   

        
