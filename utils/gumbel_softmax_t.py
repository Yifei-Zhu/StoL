import numpy as np

class TemperatureScheduler:
    def __init__(self, initial_temperature=0.5, min_temperature=0.1, max_steps=1000, strategy="linear", decay_rate=0.01, power=2.0):
        """
        A class for managing temperature schedules.

        Parameters:
        initial_temperature (float): Starting temperature.
        min_temperature (float): Minimum temperature.
        max_steps (int): Total number of steps.
        strategy (str): Decay strategy ("linear", "exponential", "power").
        decay_rate (float): Rate of decay (used in exponential decay).
        power (float): Power factor for power decay (used in power strategy).
        """
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_steps = max_steps
        self.strategy = strategy.lower()
        self.decay_rate = decay_rate
        self.power = power

    def get_temperature(self, step):
        """
        Get the temperature at a given step based on the strategy.

        Parameters:
        step (int): The current step.

        Returns:
        float: Adjusted temperature.
        """
        if self.strategy == "linear":
            return max(
                self.min_temperature,
                self.initial_temperature - (self.initial_temperature - self.min_temperature) * (step / self.max_steps),
            )

        elif self.strategy == "exponential":
            return max(
                self.min_temperature,
                self.initial_temperature * np.exp(-self.decay_rate * step),
            )

        elif self.strategy == "power":
            return max(
                self.min_temperature,
                self.min_temperature + (self.initial_temperature - self.min_temperature) * ((1 - step / self.max_steps) ** self.power),
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

# Example usage
if __name__ == "__main__":
    scheduler_linear = TemperatureScheduler(initial_temperature=1.0, min_temperature=0.1, max_steps=1000, strategy="linear")
    scheduler_exponential = TemperatureScheduler(initial_temperature=1.0, min_temperature=0.1, max_steps=1000, strategy="exponential", decay_rate=0.01)
    scheduler_power = TemperatureScheduler(initial_temperature=1.0, min_temperature=0.1, max_steps=1000, strategy="power", power=2.0)

    steps = np.linspace(0, 1000, 100)
    temperatures_linear = [scheduler_linear.get_temperature(step) for step in steps]
    temperatures_exponential = [scheduler_exponential.get_temperature(step) for step in steps]
    temperatures_power = [scheduler_power.get_temperature(step) for step in steps]

    import matplotlib.pyplot as plt

    plt.plot(steps, temperatures_linear, label="Linear Decay")
    plt.plot(steps, temperatures_exponential, label="Exponential Decay")
    plt.plot(steps, temperatures_power, label="Power Decay (Fast Early)")
    plt.xlabel("Steps")
    plt.ylabel("Temperature")
    plt.legend()
    plt.title("Temperature Decay Strategies")
    plt.show()