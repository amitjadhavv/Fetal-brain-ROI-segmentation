import json
import matplotlib.pyplot as plt

# Read the JSON file
with open('loss_history_roi.json', 'r') as file:
    data = json.load(file)
print(data)
# Extract loss values and calculate epoch indices
loss_values = data['train_loss']
epochs = range(len(loss_values))

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, label='Loss',  linestyle='-')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Save the graph as an image
plt.savefig('loss_vs_epoch.png')
plt.show()
