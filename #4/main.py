import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


T=[0,0.5,1,5,10,20]
P=[760,625,528,85,14,0.16]

def get_colleration(y_actual, y_predicted):
    correlation_matrix = np.corrcoef(y_actual,y_predicted)
    correlation = correlation_matrix[0][1]
    return correlation

def get_y(coefs, x):
    return np.exp(coefs[1]) * np.exp(coefs[0]*x)

if __name__ == "__main__":
    correlation = get_colleration(T,P)
    x = np.linspace(-2,22,100)

    coefs = np.polyfit(T,np.log(P),1, w=np.sqrt(P))

    y = get_y(coefs, x)
    plt.plot(x,y)

    r2 = get_colleration(P,get_y(coefs, np.array(T)))**2


    print(f"Równanie funkcji regresji: {round(np.exp(coefs[1]), 2)} · e^({round(coefs[0], 2)}x)")
    print(f"Współczynnik determinacji: {round(r2, 4)}")

    plt.scatter(T, P, c='r')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()
