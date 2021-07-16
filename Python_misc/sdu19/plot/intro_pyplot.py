import matplotlib.pyplot as plt
import numpy as np

a = np.array([1,4,2,36,3,77,3,4,12])
b = np.array([2,4,25,6,3,6,3,6,22])

epochs = range(1,10)

plt.plot(epochs, a, label="Training accuracy")
plt.plot(epochs, b, label="Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()