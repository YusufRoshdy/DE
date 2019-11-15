import tkinter as tk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


root = tk.Tk()
root.wm_title("Embedding in Tk")


class equation:
    def __init__(self, x0, y0, X, N):
        self.x0 = x0
        self.y0 = y0
        self.X = X
        self.N = N
    def draw(self):
        x = np.linspace(self.x0, self.X, self.N)
        c = 1.0/(self.y0-np.exp(self.x0))+self.x0
        y = 1.0/(c-x) + np.exp(x)

        plt.ion()

        fig = plt.figure()
        fig.add_subplot(111).plot(x, y, '-r')
        fig.canvas.draw()
        fig.canvas.flush_events()

class EulerApproximations(equation):
    def __init__(self, x0, y0, X, N):
        super().__init__(x0, y0, X, N)

    def draw(self):
        x = [self.x0]
        y = [self.y0]
        dy = [2]  # dy(0)
        step = (self.X-self.x0)/self.N
        for i in range(1, int(self.N)):
            x.append(x[i-1]+step)
            y.append(y[i-1] + step*dy[i-1])
            dy.append(np.exp(2*x[i]) + np.exp(x[i]) +
                    y[i]*y[i] - 2*y[i]*np.exp(x[i]))

        tx = np.linspace(self.x0, self.X, self.N)
        tc = 1.0/(self.y0-np.exp(self.x0))+self.x0
        ty = 1.0/(tc-tx) + np.exp(tx)
        le = np.abs(ty-y)


        plt.ion()

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x, y)
        axs[0, 0].set_title('euler')

        axs[0, 1].plot(x, le, 'tab:orange')
        axs[0, 1].set_title('local error')

        axs[1, 0].plot(x, y, 'tab:green')
        axs[1, 0].set_title('global error')

        fig.canvas.draw()
        fig.canvas.flush_events()

def f(x, y):
    return np.exp(2*x) + np.exp(x) + y**2 - 2*y*np.exp(x)
    
def improvedEulers(x0, y0, X, N):

    print(x0, y0, X, N)

    x = [x0]
    y = [y0]
    dy = [np.exp(2*x0) + np.exp(x0) + y0*y0 - 2*y0*np.exp(x0)]  # dy(x0,y0)
    dy1 = [np.exp(2*x0) + np.exp(x0) + y0*y0 - 2*y0*np.exp(x0)]
    step = (X-x0)/N
    for i in range(1, int(N)):
        x.append(x[i-1]+step)
        y.append(y[i-1] + step/2*(dy[i-1] + dy1[i-1]))
        dy.append(f(x[i], y[i]))
        x1 = x[i]+step
        y1 = y[i-1]+step*dy[i-1]
        dy1.append(np.exp(2*x1) + np.exp(x1) + y1*y1 - 2*y1*np.exp(x1))
        print(x[i], y[i], dy[i])

    tx = np.linspace(x0, X, N)
    tc = 1.0/(y0-np.exp(x0))+x0
    ty = 1.0/(tc-tx) + np.exp(tx)
    le = np.abs(ty-y)

    plt.ion()

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x, y)
    axs[0, 0].set_title('improved Eulers')

    axs[0, 1].plot(x, le, 'tab:orange')
    axs[0, 1].set_title('local error')

    axs[1, 0].plot(x, y, 'tab:green')
    axs[1, 0].set_title('global error')

    fig.canvas.draw()
    fig.canvas.flush_events()


def rungeKutta(x0, y0, X, N):

    X1 = np.linspace(x0, X, N+1)
    Y = []
    x_cur = x0
    y_cur = x0
    Y.append(y_cur)
    h = (X-x0)/(N)
    for i in range(int(N)):
        k1 = f(x_cur, y_cur)
        k2 = f(x_cur+h/2, y_cur+h/2*k1)
        k3 = f(x_cur+h/2, y_cur+h/2*k2)
        k4 = f(x_cur+h, y_cur+h*k3)
        y_cur = y_cur+h/6*(k1+2*k2+2*k3+k4)
        Y.append(y_cur)
        x_cur += h
    Y = np.array(Y)
    
    tx = np.linspace(x0, X, N+1)
    tc = 1.0/(y0-np.exp(x0))+x0
    ty = 1.0/(tc-tx) + np.exp(tx)
    le = np.abs(ty-Y)
    
    plt.ion()

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(X1, Y)
    axs[0, 0].set_title('rungeKutta')

    axs[0, 1].plot(X1, le, 'tab:orange')
    axs[0, 1].set_title('local error')

    axs[1, 0].plot(X1, Y, 'tab:green')
    axs[1, 0].set_title('global error')

    fig.canvas.draw()
    fig.canvas.flush_events()


def _update(_class):
    x0 = -1
    y0 = -1
    X = -1
    N = -1
    try:
        if e_x0.get():
            x0 = float(e_x0.get())
        if e_y0.get():
            y0 = float(e_y0.get())
        if e_X.get():
            X = float(e_X.get())
        if e_N.get():
            N = float(e_N.get())
        print("x=%d fd=%d sf=%d fsd=%d", x0, y0, X, N)
    except ValueError:
        l_status.config(text="Error")
        return False
    e = _class(x0, y0, X, N)
    e.draw()
    l_status.config(text=" ")


def _exact():
    _update(equation)


def _euler():
    _update(EulerApproximations)


def _improvedEulers():
    _update(improvedEulers)


def _rungeKutta():
    _update(rungeKutta)


b_quit = tk.Button(master=root, text="Exact", command=_exact)
b_quit.pack(side=tk.RIGHT)
b_update = tk.Button(master=root, text="Euler", command=_euler)
b_update.pack(side=tk.RIGHT)
b_update = tk.Button(master=root, text="ImprovedEulers",
                     command=_improvedEulers)
b_update.pack(side=tk.RIGHT)
b_update = tk.Button(master=root, text="RungeKutta", command=_rungeKutta)
b_update.pack(side=tk.RIGHT)

l_x0 = tk.Label(root, text="x0")
l_x0.pack(side=tk.LEFT)
e_x0 = tk.Entry(root)
e_x0.pack(side=tk.LEFT)

l_y0 = tk.Label(root, text="y0")
l_y0.pack(side=tk.LEFT)
e_y0 = tk.Entry(root)
e_y0.pack(side=tk.LEFT)

l_X = tk.Label(root, text="X")
l_X.pack(side=tk.LEFT)
e_X = tk.Entry(root)
e_X.pack(side=tk.LEFT)

l_N = tk.Label(root, text="N")
l_N.pack(side=tk.LEFT)
e_N = tk.Entry(root)
e_N.pack(side=tk.LEFT)

l_status = tk.Label(root, text="     ")
l_status.pack(side=tk.LEFT)


tk.mainloop()
