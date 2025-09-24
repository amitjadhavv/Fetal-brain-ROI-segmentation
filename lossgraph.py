import re
import matplotlib.pyplot as plt

log_file = "run_avnet_noDNN_1122983.log"

# Regex patterns
epoch_pattern = re.compile(r"Epoch\s+(\d+)/(\d+)")
train_loss_pattern = re.compile(r"Train Loss:\s*([\d\.]+)")
val_loss_pattern = re.compile(r"Val Loss:\s*([\d\.]+)")
train_iou_pattern = re.compile(r"Train IoU Score:\s*([\d\.]+)")
val_iou_pattern = re.compile(r"Val IoU:\s*([\d\.]+)")
early_stop_pattern = re.compile(r"Early stopping triggered at epoch\s+(\d+)")

epochs, train_loss, val_loss, train_iou, val_iou = [], [], [], [], []
current_epoch = None
early_stop_epoch = None

with open(log_file, "r") as f:
    for line in f:
        # Skip "Saved new best model"
        if line.strip().startswith("Saved new best model"):
            continue

        # Detect early stopping epoch
        es_match = early_stop_pattern.search(line)
        if es_match:
            early_stop_epoch = int(es_match.group(1))
            continue
        # Match epoch
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epochs.append(current_epoch)

        # Training loss
        train_match = train_loss_pattern.search(line)
        if train_match:
            train_loss.append(float(train_match.group(1)))

        # Validation loss
        val_match = val_loss_pattern.search(line)
        if val_match:
            val_loss.append(float(val_match.group(1)))

        # Training IoU
        train_iou_match = train_iou_pattern.search(line)
        if train_iou_match:
            train_iou.append(float(train_iou_match.group(1)))

        # Validation IoU
        val_iou_match = val_iou_pattern.search(line)
        if val_iou_match:
            val_iou.append(float(val_iou_match.group(1)))


# --- Plot Loss Curves ---
plt.figure(figsize=(10,6))  # rectangular shape
plt.plot(epochs[:len(train_loss)], train_loss, label="Train Loss")
plt.plot(epochs[:len(val_loss)], val_loss, label="Val Loss")

if early_stop_epoch:
    plt.axvline(x=early_stop_epoch, color="red", linestyle="--", label=f"Early Stopping (Epoch {early_stop_epoch})")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("loss_curves.png", dpi=300)

# --- Plot IoU Curves (Train + Val) ---
plt.figure(figsize=(10,6))  # rectangular shape
plt.plot(epochs[:len(train_iou)], train_iou, label="Train IoU", color="blue")
plt.plot(epochs[:len(val_iou)], val_iou, label="Val IoU", color="green")

if early_stop_epoch:
    plt.axvline(x=early_stop_epoch, color="red", linestyle="--", label=f"Early Stopping (Epoch {early_stop_epoch})")

plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("Training and Validation IoU Across Epochs")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("iou_curves.png", dpi=300)

print("✅ Saved 'loss_curves.png' and 'iou_curves.png'")
