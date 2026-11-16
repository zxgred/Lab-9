import numpy as np

np.random.seed(7)
readings = np.random.randint(20, 41, size=(6, 3))
print(readings)

# (1) mean of each sensor
mean = readings.mean(axis=0)
print("Mean per sensor:", mean)

# (2) standardize with broadcasting
normalized = (readings - mean) / readings.std(axis=0)
print("Normalized column means:", normalized.mean(axis=0).round(2))

# (3) hour of peak per sensor
print("Hour of max per sensor:", readings.argmax(axis=0))

# (4) append the mean row
with_mean = np.vstack([readings, mean])
print(with_mean.shape)

# (5) reproducibility
np.random.seed(7)
readings_again = np.random.randint(20, 41, size=(6, 3))
print("Reproducible:", np.array_equal(readings, readings_again))