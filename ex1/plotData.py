import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.read_csv('ex1data1.txt').plot(x='Profit', y='Population', style='rx')
plt.show()
