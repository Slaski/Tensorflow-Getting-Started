#
#   house_price_prediction.py
#
#   This is a very simple prediction of house prices based on house size, implemented
#   in Tensorflow. This code is part of Pluralsight's course "Tensorflow: Getting Started"
#

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Generate some house sizes between 1000 and 3500 (typical sq ft of houses)
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)


# Generate house prices from the sizes with a random noise added
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)


# Plot generated hours and size
plt.plot(house_size, house_price, "bx") # bx = blue x
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()
