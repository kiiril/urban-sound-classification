import tensorflow as tf
from tf_keras import layers, Model, callbacks as KC
import features as features_lib
import params as yamnet_params
from yamnet import yamnet, yamnet_frames_model
from dataset import build_lists, make_dataset
import matplotlib.pyplot as plt

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
            scores, embeddings, _ = self.base(w, training=training)   # embeddings: [Np, 1024]
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
        self.base_opt = tf.keras.optimizers.Adam(self.base_lr)
        self.head_opt = tf.keras.optimizers.Adam(self.head_lr)
        
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
    train_files, train_labels = build_lists('../datasets', folds=range(1, 9))
    val_files,   val_labels   = build_lists('../datasets', folds=[9])
    test_files,  test_labels  = build_lists('../datasets', folds=[10])
    
    train_ds = make_dataset(train_files, train_labels, 8, training=True)
    val_ds   = make_dataset(val_files,   val_labels,   16,  training=False)
    test_ds  = make_dataset(test_files,  test_labels,  16,  training=False)
    
    if mode == "fixed_feature":
        model = Transfer(num_classes=10, backbone_trainable=False)
    elif mode == "fine_tuning":
        model = Transfer(num_classes=10, backbone_trainable=True,
                               base_lr=1e-4, head_lr=5e-4)
    else:
        raise ValueError("mode must be 'fixed_feature' or 'fine_tuning'.")
    
    load_base_weights(model, 'yamnet.h5')
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['acc']) # It's good practice to specify metrics here
    
    callbacks = [
        DualLRPlateau(model, monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        KC.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        KC.ModelCheckpoint(
        f"yamnet_{mode}_best.weights.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode='min'
        ),
    ]
    
    print('Training started')
    
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=num_of_epochs, callbacks=callbacks)
    
    # 3. Add plotting right after training using the history object
    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'YAMNet ({mode}) Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
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
    
    print("\nEvaluating with best model weights...")
    test_metrics = model.evaluate(test_ds, return_dict=True)
    print(f"*** {mode} test: acc={test_metrics['acc']:.3f} loss={test_metrics['loss']:.4f} ***")
    

if __name__ == '__main__':
    run(mode='fixed_feature', num_of_epochs=3)
    run(mode='fine_tuning', num_of_epochs=3)
    
    
