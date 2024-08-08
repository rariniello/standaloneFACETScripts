import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

####################################################################
# ROAR
####################################################################
cutoff = 0.005
cdict = {
    "red": (
        (0.0, 1, 1),
        (cutoff, 1, 1),
        (0.125 * 0.9 + 0.1, 0, 0),
        (0.35 * 0.9 + 0.1, 0, 0),
        (0.66 * 0.9 + 0.1, 1, 1),
        (0.89 * 0.9 + 0.1, 1, 1),
        (1.0, 0.5, 0.5),
    ),
    "green": (
        (0.0, 1, 1),
        (cutoff, 1, 1),
        (0.125 * 0.9 + 0.1, 0, 0),
        (0.375 * 0.9 + 0.1, 1, 1),
        (0.64 * 0.9 + 0.1, 1, 1),
        (0.91 * 0.9 + 0.1, 0, 0),
        (1.0, 0, 0),
    ),
    "blue": (
        (0.0, 1, 1),
        (0.11 * 0.9 + 0.1, 1, 1),
        (0.34 * 0.9 + 0.1, 1, 1),
        (0.65 * 0.9 + 0.1, 0, 0),
        (1.0, 0, 0),
    ),
}

roar = LinearSegmentedColormap("roar", segmentdata=cdict, N=512)

####################################################################
# W_Viridis
####################################################################
cmap = mpl.cm.get_cmap("viridis")
# cmap = plt.get_cmap('cividis')
color_list = cmap(np.linspace(0, 1, 1024))

K = 64
L = 4
white = np.ones((L, 4))
begin = np.zeros((K, 4))
begin[:, 0] = np.linspace(1, color_list[0, 0], K)
begin[:, 1] = np.linspace(1, color_list[0, 1], K)
begin[:, 2] = np.linspace(1, color_list[0, 2], K)
begin[:, 3] = 1.0
color_list = np.vstack((white, begin, color_list))

# test = colors.LinearSegmentedColormap.from_list("test", color_list)
cmap_W_Viridis = mpl.colors.ListedColormap(
    color_list, name="w_viridis", N=color_list.shape[0]
)

####################################################################
# W_plasma
####################################################################
cmap = mpl.cm.get_cmap("plasma")
# cmap = plt.get_cmap('cividis')
color_list = cmap(np.linspace(0, 1, 1024))

K = 64
L = 4
white = np.ones((L, 4))
begin = np.zeros((K, 4))
begin[:, 0] = np.linspace(1, color_list[0, 0], K)
begin[:, 1] = np.linspace(1, color_list[0, 1], K)
begin[:, 2] = np.linspace(1, color_list[0, 2], K)
begin[:, 3] = 1.0
color_list = np.vstack((white, begin, color_list))

# test = colors.LinearSegmentedColormap.from_list("test", color_list)
cmap_W_Plasma = mpl.colors.ListedColormap(
    color_list, name="w_plasma", N=color_list.shape[0]
)

RdBu = mpl.cm.get_cmap("RdBu")
cl = np.zeros((11, 3))
for i in range(11):
    cl[i, :] = [
        RdBu._segmentdata["red"][i, 1],
        RdBu._segmentdata["green"][i, 1],
        RdBu._segmentdata["blue"][i, 1],
    ]
cl[5, :] = [1.0, 1.0, 1.0]
cmap_RdWBu = LinearSegmentedColormap.from_list("RdWBu", cl)
cmap_BuW = LinearSegmentedColormap.from_list("RdWBu", cl[5:])
cmap_RdW = LinearSegmentedColormap.from_list("RdWBu", np.flip(cl[:6], axis=0))
