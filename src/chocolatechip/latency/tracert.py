import re
import numpy as np
import matplotlib.pyplot as plt

# Function to parse the traceroute data
def parse_traceroute(file_path):
    hop_latencies = {}
    hostname_pattern = re.compile(r"\d+\s+(\d+ ms)\s+(\d+ ms)\s+(\d+ ms)\s+([\w.-]+)")

    with open(file_path, 'r') as file:
        for line in file:
            match = hostname_pattern.search(line)
            if match:
                # Remove ' ms' from the latency values and convert to integers
                latencies = [int(latency.split()[0]) for latency in match.groups()[:3]]
                hostname = match.group(4)
                
                # Store the latencies per hostname
                if hostname not in hop_latencies:
                    hop_latencies[hostname] = []
                hop_latencies[hostname].extend(latencies)
    
    return hop_latencies

# Function to plot the latency data
def plot_latency(hop_latencies):
    hostnames = list(hop_latencies.keys())
    avg_latencies = [np.mean(hop_latencies[host]) for host in hostnames]
    min_latencies = [np.min(hop_latencies[host]) for host in hostnames]
    max_latencies = [np.max(hop_latencies[host]) for host in hostnames]
    
    # Error bars: min-max range
    yerr = [np.subtract(avg_latencies, min_latencies), np.subtract(max_latencies, avg_latencies)]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.errorbar(hostnames, avg_latencies, yerr=yerr, fmt='o', capsize=5, label='Latency with Error Bars')
    
    x_positions = np.arange(len(hostnames))
    plt.xticks(ticks=x_positions, labels=hostnames, rotation=45, ha='right')
    
    plt.xlabel('Hostnames')
    plt.ylabel('Latency (ms)')
    plt.title('Traceroute Latency per Hop')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig('traceroute_latency.png')

# Main execution
file_path = 'traceroute.txt'
hop_latencies = parse_traceroute(file_path)
plot_latency(hop_latencies)