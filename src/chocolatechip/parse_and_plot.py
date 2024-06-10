import re
import pandas as pd
import matplotlib.pyplot as plt

# Path to the log file
log_file_path = 'training_output.log'

# Regular expression to match the relevant lines
line_pattern = re.compile(r'(\d+): loss=([\d.]+), avg loss=([\d.]+), rate=([\d.]+), ([\d.]+) seconds, (\d+) images, time remaining=\d+')

# Lists to store extracted data
batches = []
losses = []
avg_losses = []
rates = []
seconds = []
images = []

# Read and parse the log file
with open(log_file_path, 'r') as log_file:
    for line in log_file:
        match = line_pattern.search(line)
        if match:
            batches.append(int(match.group(1)))
            losses.append(float(match.group(2)))
            avg_losses.append(float(match.group(3)))
            rates.append(float(match.group(4)))
            seconds.append(float(match.group(5)))
            images.append(int(match.group(6)))

# Create a DataFrame
df = pd.DataFrame({
    'batch': batches,
    'loss': losses,
    'avg_loss': avg_losses,
    'rate': rates,
    'seconds': seconds,
    'images': images
})

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(df['batch'], df['loss'], label='Loss')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.title('Loss over Batches')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('loss_over_batches.png')

# Display the plot
plt.show()

# Save DataFrame to a CSV (optional)
df.to_csv('training_metrics.csv', index=False)
