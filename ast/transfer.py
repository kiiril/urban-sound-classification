from dataset import UrbanSound8K
from model import ASTModel
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import time, random

import torchaudio
from urban_audio_pipeline.pann.dataset import UrbanSound8KWav

def load_ast(num_classes, checkpoint_path, mode='fixed_feature', dropout_rate=0.3):
    ast = ASTModel(label_dim=num_classes, input_tdim=400, imagenet_pretrain=True, audioset_pretrain=True)
    
    # Adding Dropout layer to avoid overfitting
    original_layernorm = ast.mlp_head[0]
    original_linear = ast.mlp_head[1]
    
    in_features = original_linear.in_features
    
    ast.mlp_head = nn.Sequential(
        original_layernorm,
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    ast.load_state_dict(checkpoint, strict=False)
    
    if mode == 'fixed_feature':
        for n, p in ast.named_parameters():
            if not n.startswith('mlp_head.'):
                p.requires_grad = False
        
        param_groups = [{'params': ast.mlp_head.parameters(), 'lr': 1e-3}]
    elif mode == 'fine_tuning':
        head_params = [p for n, p in ast.named_parameters() if n.startswith('mlp_head.')]
        backbone_params = [p for n, p in ast.named_parameters() if not n.startswith('mlp_head.')]
        param_groups = [
            {'params': backbone_params, 'lr': 1e-4},
            {'params': head_params, 'lr': 5e-4}
        ]
    else:
        raise ValueError("Wrong mode parameter passed!")
    
    return ast, param_groups

# FIXME: part of the trick for inference measurements
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


def count_trainable_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, loader, device, num_batches=10):
    """Measure average inference time per sample"""
    model.eval()
    times = []
    total_samples = 0
    
    # FIX: this is just a trick, better to correct it!
    # Helper function to process a BATCH of waveforms correctly
    def extract_fbank_batch(waveforms, target_length=400):
        fbanks = []
        for wav in waveforms:
            # Calculate fbank for a single waveform
            fbank = torchaudio.compliance.kaldi.fbank(
                wav.unsqueeze(0), htk_compat=True, sample_frequency=16000,
                use_energy=False, window_type='hanning', num_mel_bins=128,
                dither=0.0, frame_shift=10)

            # Pad or truncate the fbank
            n_frames = fbank.shape[0]
            p = target_length - n_frames
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)
            elif p < 0:
                fbank = fbank[0:target_length, :]

            # Normalize and add to list
            fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
            fbanks.append(fbank)
        
        # Stack individual fbanks into a batch
        return torch.stack(fbanks, dim=0)
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
                
            x = x.to(device)
            batch_size = x.size(0)
            
            # Time inference
            start_time = time.time()
            
            # FIXME: part of the inference time measurement trick
            fbank_batch = extract_fbank_batch(x)
            _ = model(fbank_batch)
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
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            
            loss_sum += loss.item() * y.size(0)
            preds = out.argmax(1)
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


def run(mode='fixed_feature', num_of_epochs=5, patience=3):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    train_dataset = UrbanSound8K('../datasets', folds=range(1, 9), training=True)
    val_dataset = UrbanSound8K('../datasets', folds=[9], training=False)
    test_dataset = UrbanSound8K('../datasets', folds=[10], training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    
    checkpoint_path = 'pretrained_models/audio_mdl.pth'
    model, pg = load_ast(10, checkpoint_path, mode)
    model.to(device)
    
    # Count trainable parameters
    trainable_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params:,}")
    
    opt = torch.optim.Adam(pg, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # TODO: replace with method from class
    class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 
                   'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 
                   'siren', 'street_music']
    
    def loop(loader, train=False):
        model.train(train)
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # reshape for AST: [B, T, 128]
            if train:
                out = model(x)
                loss = criterion(out, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                with torch.no_grad():
                    out = model(x)
                    loss = criterion(out, y)
                
            loss_sum += loss.item() * y.size(0)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
        return loss_sum / total, correct / total
    
    print('Training started')
    training_start_time = time.time()
    
    # 2. Initialize lists to store metrics over epochs
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # 3. Track best validation loss to save the best model
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(1, num_of_epochs + 1):
        train_loss, train_acc = loop(train_loader, train=True)
        val_loss,   val_acc   = loop(val_loader,   train=False)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.3f}, Acc: {train_acc:.3f} | Val Loss: {val_loss:.3f}, Acc: {val_acc:.3f}")
        
        # 4. Save model based on best validation LOSS
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"ast_{mode}_best.pth")
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
            
    # 5. Plotting the results after the training loop
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.title(f'AST ({mode}) Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True) # Added grid for better readability

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.title(f'AST ({mode}) Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True) # Added grid for better readability

    plt.tight_layout()
    
    output_filename = f'ast_{mode}_training_plot.png'
    plt.savefig(output_filename)
    print(f"Training plot saved to {output_filename}")        
            
    # Final test evaluation
    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(f"ast_{mode}_best.pth"))
    
    # FIXME: part of the trick with measuring inference
    target_samples = int(16000 * 4.0)   # 4 s for US8K
    collate_inf  = make_collate_fixed(target_samples, training=False)
    inference_ds = UrbanSound8KWav('../datasets', folds=[10], target_sr=16000)
    inference_loader = DataLoader(inference_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_inf)
    # Measure inference time per sample
    inference_time_per_sample = measure_inference_time(model, inference_loader, device)
    print(f"Inference time per sample: {inference_time_per_sample:.2f} ms")
    
    # COMPREHENSIVE EVALUATION
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    
    # Validation set evaluation
    val_metrics = evaluate_with_metrics(model, val_loader, criterion, device, class_names)
    print_detailed_metrics(val_metrics, class_names, "VALIDATION SET")
    
    # Test set evaluation  
    test_metrics = evaluate_with_metrics(model, test_loader, criterion, device, class_names)
    print_detailed_metrics(test_metrics, class_names, "TEST SET")
    
    # Plot confusion matrices
    plot_confusion_matrix(
        val_metrics['confusion_matrix'], 
        class_names, 
        title=f'AST {mode.title()} - Validation Confusion Matrix',
        save_path=f'ast_{mode}_val_confusion_matrix.png'
    )
    
    plot_confusion_matrix(
        test_metrics['confusion_matrix'], 
        class_names, 
        title=f'AST {mode.title()} - Test Confusion Matrix',
        save_path=f'ast_{mode}_test_confusion_matrix.png'
    )
    
    # Save detailed results to file
    results_file = f'ast_{mode}_detailed_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"AST {mode.upper()} MODE - COMPREHENSIVE EVALUATION RESULTS\n")
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
    print(f"\n*** AST {mode} FINAL RESULTS ***")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total training time: {total_training_time/60:.1f} minutes")
    print(f"Inference time per sample: {inference_time_per_sample:.2f} ms")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1-Score (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"Test Precision (Macro): {test_metrics['precision_macro']:.4f}")
    print(f"Test Recall (Macro): {test_metrics['recall_macro']:.4f}")

if __name__ == '__main__':
    run(mode='fixed_feature', num_of_epochs=10)
    run(mode='fine_tuning', num_of_epochs=20)
    
    
    