import numpy as np
from matplotlib import pyplot as plt

arr = np.load('np.npy')
arr2 = np.load('trans_dir.npy')
plt.plot(arr[0,:],arr[2,:])
plt.plot(-1*arr2[0,:],arr2[2,:])
plt.show()