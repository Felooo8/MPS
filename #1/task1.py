import numpy as np
import matplotlib.pyplot as plt

def get_mean(s):
    return np.mean(s)

def get_variance(s):
    return np.std(s, ddof=1)

def check_sigma(variance, seed, mean):
    sm = 0
    for xi in seed:
        sm += np.square(xi-mean)
    s2 = 1/(seed.size-1)*sm
    print(s2)
    print(variance)
    if not np.isclose(s2, variance, rtol=1e-05, atol=1e-08):
        print("Different variances values")
    else:
        print("Similar variances values")

def create_plot(display_data):
    data = 100

    mu, sigma = 1, 0.1 # mean and standard deviation
    seed = np.random.normal(mu, sigma, data)

    mean = get_mean(seed)
    s = get_variance(seed)

    variance = np.square(s)

    mean_diff = abs(mu-mean)
    variance_diff = abs(np.square(sigma)-variance)

    n1 = mean-2*s
    n2 = mean+2*s

    total_within_range = ((n1 < seed) & (seed < n2)).sum()
    percentage_within_range = total_within_range/data*100
    
    if display_data:
        print(f"Difference in mean value: {mean_diff}")
        print(f"Difference in variance value: {variance_diff}")

        check_sigma(variance, seed, mean)

        plt.hist(seed, density=True, bins=30)
        plt.ylabel('Probability density')
        plt.xlabel('x')
        plt.show()

    return percentage_within_range

if __name__ == "__main__":
    samples = 50
    averages = []
    display_data = True

    for i in range(1, samples):
        within_range = create_plot(display_data)
        averages.append(within_range)
        display_data = False

    average = np.average(averages)
    print(f"For {samples} samples on average: {average}% of generated numbers were within range")

