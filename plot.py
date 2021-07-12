import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

path = '/home/yagnesh/Desktop/GitHub-Repos/VisualOdom/Trajectory.npy'
traj = np.load(path)

plt.plot(traj[0,:],traj[1,:])

plt.show()
plt.pause(0)