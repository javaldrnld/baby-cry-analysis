import numpy as np
import matplotlib.pyplot as plt

# Sample data
time = np.linspace(0, 10, 1000)
rms = 0.8 * np.exp(-time / 2) + 0.2 * np.sin(2 * np.pi * time)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the RMS values
ax.plot(time, rms, color='#4c78a8', linewidth=2)

# Set axis labels and title
ax.set_xlabel('Time (s)')
ax.set_ylabel('RMS Amplitude')
ax.set_title('RMS Plot for Audio Signal')

# Set tick labels and grid
ax.set_xticks(np.arange(0, 11, 2))
ax.set_yticks(np.arange(0, 0.9, 0.2))
ax.grid(True, color='#cccccc')

# Display the plot
plt.show()