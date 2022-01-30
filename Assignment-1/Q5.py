import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    alpha = np.linspace(0,1,num=1000)
    optimal_bayes = []
    expected = []

    for i in range(len(alpha)):
        if(alpha[i] >= 0.5):
            optimal_bayes.append(1-alpha[i])
        else:
            optimal_bayes.append(alpha[i])
        expected.append(2*alpha[i]*(1-alpha[i]))

    plt.plot(alpha, optimal_bayes, color='red', label='bayes optimal')
    plt.plot(alpha, expected, color='blue', label='expected')
    plt.xlabel("Alpha")
    plt.ylabel("Error")
    plt.title(f"Error as a function of alpha")
    plt.legend()
    plt.show()