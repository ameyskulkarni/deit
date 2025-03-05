import json
import matplotlib.pyplot as plt

# Read the log file
log_file = "models/log.txt"  # Change this to your actual log file path

# Parse the JSON lines
data = []
with open(log_file, "r") as f:
    for line in f:
        data.append(json.loads(line.strip()))

# Extract values
epochs = [entry["epoch"] for entry in data]
train_loss = [entry["train_loss"] for entry in data]
test_loss = [entry["test_loss"] for entry in data]
test_acc1 = [entry["test_acc1"] for entry in data]
test_acc5 = [entry["test_acc5"] for entry in data]
train_lr = [entry["train_lr"] for entry in data]

# Plot the metrics
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(epochs, train_loss, marker='o', label='Train Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, test_loss, marker='o', label='Test Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Loss')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs, test_acc1, marker='o', label='Test Acc@1', color='green')
plt.plot(epochs, test_acc5, marker='o', label='Test Acc@5', color='purple')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs, train_lr, marker='o', label='Learning Rate', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.legend()

plt.tight_layout()
plt.show()
