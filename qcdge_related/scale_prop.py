import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA

class PropDataScale:
    def __init__(self, data):
        """
        Initialize the PropDataScale with a list of numerical data.

        Args:
            data (list): List of numerical values.
        """
        self.data = np.array(data, dtype=np.float64).reshape(-1, 1)
        self.scaled_data = None

    def min_max_normalization(self, feature_range=(0, 1)):
        """
        Apply Min-Max normalization to scale data to the given range.

        Args:
            feature_range (tuple): Desired range of transformed data (default is (0, 1)).

        Returns:
            numpy.ndarray: Scaled data.
        """
        scaler = MinMaxScaler(feature_range=feature_range)
        self.scaled_data = scaler.fit_transform(self.data)
        return self.scaled_data

    def standardization(self):
        """
        Standardize data to have a mean of 0 and standard deviation of 1.

        Returns:
            numpy.ndarray: Standardized data.
        """
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.data)
        return self.scaled_data

    def robust_scaling(self):
        """
        Apply robust scaling to data (handles outliers by using median and IQR).

        Returns:
            numpy.ndarray: Scaled data.
        """
        scaler = RobustScaler()
        self.scaled_data = scaler.fit_transform(self.data)
        return self.scaled_data

    def log_transformation(self):
        """
        Apply log transformation to reduce skewness.

        Returns:
            numpy.ndarray: Log-transformed data.
        """
        if np.any(self.data <= 0):
            raise ValueError("Log transformation requires all data to be positive.")
        return np.log(self.data)

    def inverse_transform(self):
        """
        Inverse transformation of the scaled data to return to the original space.

        Returns:
            numpy.ndarray: Original data.
        """
        if self.scaled_data is None:
            raise ValueError("No scaled data to inverse transform.")
        if isinstance(self.scaler, (MinMaxScaler, StandardScaler, RobustScaler)):
            return self.scaler.inverse_transform(self.scaled_data)
        else:
            raise ValueError("Scaler type not recognized or inverse transformation not supported.")


    def describe_data(self):
        """
        Describe the data distribution with basic statistics and plots.

        Prints:
            Minimum, maximum, mean, median, standard deviation.
        Plots:
            Histogram and KDE plot of the data.
        """
        arr = self.data.flatten()
        minimum = np.min(arr)
        maximum = np.max(arr)
        mean = np.mean(arr)
        median = np.median(arr)
        std_dev = np.std(arr)

        print(f"Min: {minimum}, Max: {maximum}, Mean: {mean}, Median: {median}, Std Dev: {std_dev}")

        # Plot histogram
        plt.hist(arr, bins=100, alpha=0.7, color='blue')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Data')
        plt.show()

        # Plot KDE
        sns.kdeplot(arr, color='red', fill=True)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('KDE Plot of Data')
        plt.show()

# Example usage
if __name__ == "__main__":
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    preprocessor = PropDataScale(data)

    print("Min-Max Normalization:", preprocessor.min_max_normalization())
    print("Standardization:", preprocessor.standardization())
    print("Robust Scaling:", preprocessor.robust_scaling())
    print("Log Transformation:", preprocessor.log_transformation())

    # Describe the data distribution
    preprocessor.describe_data()
