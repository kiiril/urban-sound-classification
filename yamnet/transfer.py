import tensorflow as tf
from tf_keras import layers, Model, callbacks as KC
import features as features_lib
import params as yamnet_params
from yamnet import yamnet, yamnet_frames_model
from dataset import build_lists, make_dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import time

class Transfer(Model):
    def __init__(self, num_classes: int,
                 backbone_trainable: bool,
                 base_lr: float = 1e-4,
                 head_lr: float = 5e-4,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.params = yamnet_params.Params()
        # Important: keep num_classes at the original 521 *inside the base* to load weights.
        # The frames model builds logits with params.num_classes, but we will ignore those
        # scores and only use embeddings. So leave params.num_classes = 521.
        self.base = yamnet_frames_model(self.params)
        self.base.trainable = backbone_trainable

        # New transfer head on top of embeddings (1024 -> num_classes)
        self.dropout = layers.Dropout(dropout_rate)
        self.head = layers.Dense(num_classes, name="transfer_head")

        # learning rates for param groups
        self.base_lr = base_lr
        self.head_lr = head_lr

        # will be set in compile()
        self.loss_fn = None
        self.base_opt = None
        self.head_opt = None
        self.acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        
    def call(self, wav, training=False):
        """
        wav: [B, T] float32 (padded per batch)
        Returns: logits [B, num_classes]
        """
        # per-example function: 1-D waveform -> [1024] embedding
        def per_example(w):
            scores, embeddings, _ = self.base(w, training=False)   # must be false to not update BatchNorm
            clip_emb = tf.reduce_mean(embeddings, axis=0)             # [1024]
            return clip_emb

        clip_embs = tf.map_fn(
            per_example,
            wav,
            fn_output_signature=tf.TensorSpec(shape=(1024,), dtype=tf.float32)
        )  # [B, 1024]

        x = self.dropout(clip_embs, training=training)
        logits = self.head(x)                                         # [B, C]
        return logits
    
    def compile(self, optimizer=None, loss=None, **kwargs):
        # We ignore 'optimizer' argument and create two internal Adam optimizers.
        super().compile(**kwargs)
        self.loss_fn = loss or tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # Use separate opt instances so we can set different learning rates.
        self.base_opt = tf.keras.optimizers.Adam(self.base_lr, weight_decay=1e-4)
        self.head_opt = tf.keras.optimizers.Adam(self.head_lr, weight_decay=1e-4)
        
    @property
    def metrics(self):
        return [self.loss_metric, self.acc_metric]

    def train_step(self, data):
        wav, y = data

        with tf.GradientTape(persistent=True) as tape:
            logits = self(wav, training=True)
            loss = self.loss_fn(y, logits)

        # split variables once, directly from submodules
        base_vars = self.base.trainable_variables
        head_vars = self.head.trainable_variables + self.dropout.trainable_variables

        if base_vars:  # fixed_feature mode will have base_vars = []
            base_grads = tape.gradient(loss, base_vars)
            self.base_opt.apply_gradients(zip(base_grads, base_vars))

        head_grads = tape.gradient(loss, head_vars)
        self.head_opt.apply_gradients(zip(head_grads, head_vars))

        self.loss_metric.update_state(loss)
        self.acc_metric.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        wav, y = data
        logits = self(wav, training=False)
        loss = self.loss_fn(y, logits)
        self.loss_metric.update_state(loss)
        self.acc_metric.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}
    
    
def count_trainable_parameters(model):
    """Count number of trainable parameters in TensorFlow model"""
    return sum([tf.size(var).numpy() for var in model.trainable_variables])


def measure_inference_time(model, dataset, num_batches=10):
    """Measure average inference time per sample for TensorFlow model"""
    times = []
    total_samples = 0
    
    batch_count = 0
    for wav_batch, _ in dataset:
        if batch_count >= num_batches:
            break
            
        batch_size = tf.shape(wav_batch)[0].numpy()
        
        # Time inference
        start_time = time.time()
        _ = model(wav_batch, training=False)
        if tf.config.list_physical_devices('GPU'):
            tf.experimental.sync_devices()
        end_time = time.time()
        
        batch_time = end_time - start_time
        times.append(batch_time)
        total_samples += batch_size
        batch_count += 1
    
    total_inference_time = sum(times)
    avg_time_per_sample = total_inference_time / total_samples
    
    return avg_time_per_sample * 1000  # Convert to milliseconds


def evaluate_with_metrics(model, dataset, class_names=None):
    """Comprehensive evaluation function for TensorFlow model"""
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0
    
    # Reset metrics
    model.loss_metric.reset_state()
    model.acc_metric.reset_state()
    
    for wav_batch, label_batch in dataset:
        logits = model(wav_batch, training=False)
        loss = model.loss_fn(label_batch, logits)
        
        # Update metrics
        model.loss_metric.update_state(loss)
        model.acc_metric.update_state(label_batch, logits)
        
        # Collect predictions and labels
        preds = tf.argmax(logits, axis=1).numpy()
        labels = label_batch.numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
        
        total_loss += loss.numpy() * tf.shape(label_batch)[0].numpy()
        total_samples += tf.shape(label_batch)[0].numpy()
    
    # Calculate accuracy and loss
    accuracy = model.acc_metric.result().numpy()
    avg_loss = total_loss / total_samples
    
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
    

def _get_lr(opt):
    lr = opt.learning_rate
    try:
        # Variable / Tensor
        return float(tf.convert_to_tensor(lr).numpy())
    except Exception:
        # schedule or python scalar
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            return float(lr(0).numpy())
        return float(lr)

def _set_lr(opt, val: float):
    lr = opt.learning_rate
    try:
        # Variable-like
        lr.assign(val)
    except Exception:
        # Fallback to attribute set (works for python float or 'auto')
        opt.learning_rate = val

class DualLRPlateau(KC.Callback):
    def __init__(self, model, monitor="val_loss",
                 factor=0.5, patience=2, min_lr=1e-6, cooldown=0):
        super().__init__()
        self.m = model
        self.monitor = monitor
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_lr = float(min_lr)
        self.cooldown = int(cooldown)
        self.best = float("inf")
        self.wait = 0
        self.cool = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        cur = logs.get(self.monitor)
        if cur is None:
            return

        if cur < self.best - 1e-8:
            self.best = float(cur)
            self.wait = 0
            return

        if self.cool > 0:
            self.cool -= 1
            self.wait = 0
            return

        self.wait += 1
        if self.wait >= self.patience:
            for opt in (self.m.base_opt, self.m.head_opt):
                if opt is None:
                    continue
                old = _get_lr(opt)
                new = max(self.min_lr, old * self.factor)
                _set_lr(opt, new)
            self.wait = 0
            self.cool = self.cooldown
            b = _get_lr(self.m.base_opt)
            h = _get_lr(self.m.head_opt)
            print(f"\n[DualLRPlateau] epoch {epoch+1}: base_lr={b:.3e}, head_lr={h:.3e}")
    
    
def load_base_weights(model: Transfer, weights_path: str):
    """
    Load pretrained YAMNet weights into model.base.
    If you exported weights as a Keras .h5 for the frames model:
        model.base.load_weights(weights_path)
    If your weights are a TF checkpoint, use tf.train.Checkpoint.
    """
    # Try Keras weights first
    try:
        model.base.load_weights(weights_path)
        return
    except Exception:
        pass
    # Checkpoint fallback
    ckpt = tf.train.Checkpoint(model=model.base)
    ckpt.restore(weights_path).expect_partial()
    

def run(mode='fixed_feature', num_of_epochs=5, patience=3):
    # Configure GPU usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Using GPU: {len(gpus)} GPU(s) detected")
        except RuntimeError as e:
            print(f"❌ GPU error: {e}")
    else:
        print("⚠️  No GPU detected, using CPU")
    
    train_files, train_labels = build_lists('../datasets', folds=range(1, 9))
    val_files,   val_labels   = build_lists('../datasets', folds=[9])
    test_files,  test_labels  = build_lists('../datasets', folds=[10])
    
    train_ds = make_dataset(train_files, train_labels, 8, training=True)
    val_ds   = make_dataset(val_files,   val_labels,   16,  training=False)
    test_ds  = make_dataset(test_files,  test_labels,  16,  training=False)
    
    if mode == "fixed_feature":
        model = Transfer(num_classes=10, backbone_trainable=False, head_lr=1e-3)
    elif mode == "fine_tuning":
        model = Transfer(num_classes=10, backbone_trainable=True,
                               base_lr=1e-4, head_lr=5e-4)
    else:
        raise ValueError("mode must be 'fixed_feature' or 'fine_tuning'.")
    
    load_base_weights(model, 'yamnet.h5')
    
    # BUILD THE MODEL with a dummy input to initialize all layers
    # Get a sample batch to build the model
    for sample_wav, sample_labels in train_ds.take(1):
        # This forward pass builds all layers
        _ = model(sample_wav, training=False)
        break
    
    # Count trainable parameters
    trainable_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params:,}")
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['acc']) # It's good practice to specify metrics here
    
    callbacks = [
        # DualLRPlateau(model, monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        KC.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        KC.ModelCheckpoint(
        f"yamnet_{mode}_best.weights.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode='min'
        ),
    ]
    
    class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 
                   'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 
                   'siren', 'street_music']
    
    print('Training started')
    training_start_time = time.time()
    
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=num_of_epochs, callbacks=callbacks)
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    print(f"Total training time: {total_training_time:.1f} seconds ({total_training_time/60:.1f} minutes)")
    
    # 3. Add plotting right after training using the history object
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['loss'], label='Training Loss')
    plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
    plt.title(f'YAMNet ({mode}) Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['acc'], label='Training Accuracy')
    plt.plot(epochs_range, history.history['val_acc'], label='Validation Accuracy')
    plt.title(f'YAMNet ({mode}) Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    output_filename = f'yamnet_{mode}_training_plot.png'
    plt.savefig(output_filename)
    print(f"\nPlot saved to {output_filename}")
    plt.close()

    # Because restore_best_weights=True, the model already has the best weights.
    # The load_weights line is technically redundant but confirms the saved file is used.
    model.load_weights(f"yamnet_{mode}_best.weights.h5")
    
    # Measure inference time per sample
    inference_time_per_sample = measure_inference_time(model, test_ds)
    print(f"Inference time per sample: {inference_time_per_sample:.2f} ms")
    
    # COMPREHENSIVE EVALUATION
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    
    # Test set evaluation  
    test_metrics = evaluate_with_metrics(model, test_ds, class_names)
    print_detailed_metrics(test_metrics, class_names, "TEST SET")
    
    # Plot test confusion matrix
    plot_confusion_matrix(
        test_metrics['confusion_matrix'], 
        class_names, 
        title=f'YAMNet {mode.title()} - Test Confusion Matrix',
        save_path=f'yamnet_{mode}_test_confusion_matrix.png'
    )
    
    # Save detailed results to file
    results_file = f'yamnet_{mode}_detailed_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"YAMNET {mode.upper()} MODE - COMPREHENSIVE EVALUATION RESULTS\n")
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
    print(f"\n*** YAMNet {mode} FINAL RESULTS ***")
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
    
    
