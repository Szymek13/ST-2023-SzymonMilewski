import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.animation as animation

X = np.array([[2000], [2002], [2005], [2007], [2010]])
y = np.array([6.5, 7.0, 7.4, 8.2, 9.0])

model = LinearRegression()

model.fit(X, y)

rok = [[2021]]
przewidywany_procent = model.predict(rok)
rok = [[2023]]
przewidywany_procent2 = model.predict(rok)

print("Przewidywany procent bezrobotnych w 2021 roku: {:.3f}%".format(przewidywany_procent[0]))
print("Wynik przekroczy 12% w 2023 roku: {:.3f}%".format(przewidywany_procent2[0]))

fig, ax = plt.subplots()
ax.scatter(X, y)

line, = ax.plot([], [])

def animate(i):
    model.fit(X, y)

    a = model.coef_[0]
    b = model.intercept_

    y_pred = a * X + b

    line.set_data(X, y_pred)

    return line,

ani = animation.FuncAnimation(fig, animate, frames=10, blit=True)

plt.show()
