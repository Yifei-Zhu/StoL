import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


file_path = './data/raw_data/props.pkl'
feat_dict = load_pickle_file(file_path)

data=feat_dict['Etot']

minimum = min(data)
maximum = max(data)
mean = sum(data) / len(data)

import numpy as np
arr = np.array(data)
median = np.median(arr)
std_dev = np.std(arr)

print(f"Min: {minimum}, Max: {maximum}, Mean: {mean}, Median: {median}, Std Dev: {std_dev}")

plt.hist(data, bins=100)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data')
plt.show()

sns.kdeplot(data)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('KDE Plot of Data')
plt.show()