import numpy as np
import matplotlib.pyplot as plt

X=[0,1,2,1.5,2.5,3]
Y=[1,2,4,3,7,9]

def get_colleration(y_actual, y_predicted):
    correlation_matrix = np.corrcoef(y_actual,y_predicted)
    correlation = correlation_matrix[0][1]
    return correlation

if __name__ == "__main__":
    correlation = get_colleration(X,Y)
    x = np.linspace(0,3,100)

    if correlation > 0.6:
        coefs = np.polyfit(X,Y,1)
        fx = np.poly1d(coefs) 
        plt.plot(x,fx(x))
        r_squared_lin = get_colleration(Y,fx(X))**2

    coefs_2 = np.polyfit(X,Y,2)
    fx_2 = np.poly1d(coefs_2) 
    r_squared_quadratic = get_colleration(Y,fx_2(X))**2

    plt.plot(x,fx_2(x), 'g')


    print(f"Współczynnik koleracji: {round(correlation, 2)}")
    print(f"Równanie funckji regresji liniowej: {round(coefs[0], 2)}x + {round(coefs[1], 2)}")
    print(f"Równanie funckji regresji kwadratowej: {round(coefs_2[0], 2)}x^2 + {round(coefs_2[1], 2)}x + {round(coefs_2[2], 2)}")
    print(f"Współczynnik determinacji funkcji liniowej: {round(r_squared_lin, 2)}")
    print(f"Współczynnik determinacji funkcji kwadratowej: {round(r_squared_quadratic, 2)}")

    plt.scatter(X, Y, c='r')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()
