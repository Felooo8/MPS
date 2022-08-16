import numpy as np
import matplotlib.pyplot as plt

X=[15,20,25,30,35,40,50,57]
Y=[9.5,8.5,8,7.5,6.5,6,5,5]

if __name__ == "__main__":
    correlation_matrix = np.corrcoef(X,Y)
    correlation = correlation_matrix[0][1]

    coefs = np.polyfit(X,Y,1)
    fx = np.poly1d(coefs) 
    time_at_45 = fx(45)

    r_squared = np.corrcoef(Y,fx(X))[0][1]**2

    print(f"Współczynnik koleracji: {round(correlation, 2)}")
    print(f"Równanie funckji regresji liniowe: {round(coefs[0], 2)}x + {round(coefs[1], 2)}")
    print(f"Czas bezawaryjnej pracy, która pracuje 45 miesięcy: {round(time_at_45, 2)}")
    print(f"Współczynnik determinacji: {round(r_squared, 2)}")

    plt.plot(X,fx(X))

    plt.scatter(X, Y, c='r')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()
