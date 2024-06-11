import re
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

command = "darknet detector -map -dont_show -verbose -nocolor train /home/beto/nn/LegoGears_v2/LegoGears.data /home/beto/nn/LegoGears_v2/LegoGears.cfg 2>&1"

# Regular expression to match the relevant lines
line_pattern = re.compile(r'(\d+): loss=([\d.]+), avg loss=([\d.]+), rate=([\d.]+), ([\d.]+) seconds, (\d+) images, time remaining=\d+')

# Lists to store extracted data
batches = []
losses = []
avg_losses = []
rates = []
seconds = []
images = []

def run_command_and_capture_output(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            # Parse the line using the regular expression
            match = line_pattern.search(output)
            if match:
                batches.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                avg_losses.append(float(match.group(3)))
                rates.append(float(match.group(4)))
                seconds.append(float(match.group(5)))
                images.append(int(match.group(6)))
    rc = process.poll()
    return rc

return_code = run_command_and_capture_output(command)
print(f"Command exited with return code {return_code}")

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
