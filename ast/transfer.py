from dataset import UrbanSound8K
from model import ASTModel
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def load_ast(num_classes, checkpoint_path, mode='fixed_feature', dropout_rate=0.3):
    ast = ASTModel(label_dim=num_classes, input_tdim=512, imagenet_pretrain=True, audioset_pretrain=True)
    
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


def run(mode='fixed_feature', num_of_epochs=5, patience=3):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    train_dataset = UrbanSound8K('../datasets', folds=range(1, 9))
    val_dataset = UrbanSound8K('../datasets', folds=[9])
    test_dataset = UrbanSound8K('../datasets', folds=[10])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    checkpoint_path = 'pretrained_models/audio_mdl.pth'
    model, pg = load_ast(10, checkpoint_path, mode)
    model.to(device)
    
    opt = torch.optim.AdamW(pg, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
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
            
    # 5. Plotting the results after the training loop
    epochs_range = range(1, num_of_epochs + 1)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True) # Added grid for better readability

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True) # Added grid for better readability

    plt.tight_layout()
    
    output_filename = f'ast_{mode}_training_plot.png'
    plt.savefig(output_filename)
            
    # Final test evaluation
    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(f"ast_{mode}_best.pth"))
    test_loss, test_acc = loop(test_loader, train=False)
    print(f"\n*** {mode} test accuracy: {test_acc:.3f} (from model with val_loss: {best_val_loss:.3f}) ***")

if __name__ == '__main__':
    run(mode='fixed_feature', num_of_epochs=3)
    run(mode='fine_tuning', num_of_epochs=3)
    
    
    