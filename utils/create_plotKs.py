import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x_values = ["2k", "2.5k", "3k", "3.5k", "4k", "4.5k", "5k"]
dev_topk_values = [65.35, 65.68, 65.43, 65.39, 64.47, 64.48, 65.69]
test_topk_values = [66.08, 66.53, 65.95, 65.97, 64.56, 65.06, 66.86]
dev_hybk_values = [62.8, 65.81, 66.48, 65.15, 65.56, 65.98, 66.81]
test_hybk_values = [63.39, 65.99, 66.97, 65.59, 66.48, 66.12, 67.22]

# Create the plot
plt.figure(figsize=(8, 6))

# plt.plot(x_values, dev_topk_values, label='dev mar-topk', color='blue', linestyle='-')
plt.plot(x_values, dev_hybk_values, label='dev mar-hybk', color='red', linestyle='--')
# plt.plot(x_values, test_topk_values, label='test mar-topk', color='green', linestyle='-.')
plt.plot(x_values, test_hybk_values, label='test mar-hybk', color='purple', linestyle=':')

# Add labels and title
plt.xlabel('Augmented data size')
plt.ylabel('Accuracies')
plt.title('Marathi sentiment accuracies with mar-hybk students')

# Add a legend
plt.legend()

plt.savefig("different_ks.png")

# # Show the plot
# plt.show()