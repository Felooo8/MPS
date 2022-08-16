import numpy as np
import matplotlib.pyplot as plt


X1=[990,1150,1080,1100,1280,990,1000,1200,1000,1150]
X2=[75,87,88,79,92,80,84,91,74,75]
Y=[2.2,3.2,2.6,3.3,3.8,2.0,2.2,3.6,2.1,2.8]

def get_colleration(y_actual, y_predicted):
    correlation_matrix = np.corrcoef(y_actual,y_predicted)
    correlation = correlation_matrix[0][1]
    return correlation

if __name__ == "__main__":
    correlation1 = get_colleration(X1,Y)
    x1 = np.linspace(980,1300,100)

    if correlation1 > 0.6:
        coefs1 = np.polyfit(X1,Y,1)
        fx1 = np.poly1d(coefs1) 
        plt.figure(1)
        plt.scatter(X1, Y, c='r')
        plt.plot(x1,fx1(x1))


    correlation2 = get_colleration(X2,Y)
    x2 = np.linspace(72,95,100)

    if correlation2 > 0.6:
        coefs2 = np.polyfit(X2,Y,1)
        fx2 = np.poly1d(coefs2) 
        plt.figure(2)
        plt.scatter(X2, Y, c='b')
        plt.plot(x2,fx2(x2))

    plt.ylabel('y')
    plt.xlabel('x')
    
    ones=[1,1,1,1,1,1,1,1,1,1]
    X = np.column_stack((ones,X1,X2))
    Xt=np.matrix.transpose(X)
    XX = np.matmul(Xt,X)
    X_minus1 = np.linalg.inv(XX)

    xx = np.matmul(X_minus1,Xt)
    A = np.matmul(xx,np.array(Y))
    fx3 = np.poly1d(A) 
    a=np.multiply(A[1],X1)
    b=np.multiply(A[2],X2)
    c=A[0]
    y2=a+b+c

    correlation3 = get_colleration(Y,y2)
    r_2 = correlation3**2

    ax = plt.axes(projection='3d')
    x = np.linspace(-10,10,100)
    xline = np.linspace(980,1300,100)
    yline = np.linspace(72,95,100)
    zline = np.multiply(A[1],xline) + np.multiply(A[2],yline) + c
    ax.plot3D(xline, yline, zline, 'gray')

    zdata = Y
    xdata = X1
    ydata = X2
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')

    print(f"Równanie funckji regresji liniowej X1: {round(coefs1[0], 2)}x + {round(coefs1[1], 2)}")
    print(f"Współczynnik koleracji X1: {round(correlation1, 2)}")
    print(f"Równanie funckji regresji liniowej X2: {round(coefs2[0], 2)}x + {round(coefs2[1], 2)}")
    print(f"Współczynnik koleracji X2: {round(correlation2, 2)}")
    print(f"Równanie funckji regresji liniowej X1 i X2: {round(A[1], 3)}x_1 + {round(A[2], 3)}x_2 + {round(A[0], 3)}")
    print(f"Współczynnik koleracji X1 i X2: {round(correlation3, 2)}")
    print(f"Współczynnik determinacji X1 i X2: {round(r_2, 3)}")

    plt.show()
