import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.manifold import TSNE
from .pca import map_pca, smt_pca


def add_colorbar(axis, data_len):
    axins = inset_axes(
        axis,
        width="5%",  # width = 5% of parent_bbox width
        height="50%",  # height : 50%
        loc="lower left",
        bbox_to_anchor=(1.05, 0.0, 1, 1),
        bbox_transform=axis.transAxes,
        borderpad=0,
    )
    cbar = plt.colorbar(
        cm.ScalarMappable(norm=plt.Normalize(0, data_len), cmap="coolwarm"),
        ax=axis,
        cax=axins,
    )
    cbar.ax.set_ylabel(" Time (ms)")


def fit_tsne(X):

    tsne_model = TSNE(n_components=2, perplexity=30, random_state=2020)
    embed = tsne_model.fit_transform(X)

    return embed


def map_and_smooth(data, W, V):
    """
        Map PCA weights to data and smooth
    """
    # map pc
    pc_10ms = map_pca(W, V, data)

    # smooth the first two PC
    n = 2500
    pc_smt_ = np.zeros((pc_10ms.shape[0], n))
    for i in range(pc_10ms.shape[0]):
        pc_smt = smt_pca(pc_10ms[i].mean(axis=0), n)
        pc_smt_[i, :] = pc_smt

    return pc_smt_


def traj_viz_anim(
    x, y, idx_1, idx_2, name_x="PC 1", name_y="PC 2", color_type="continuous"
):
    """
    Visualize animated neural trajectories with continous or discrete color coding

    Use np.take_along_axis with spike data and returned indices

    Arguments:
    x -- N x 1 array of data to be plotted along the x axis
    y -- N x 1 array of data to be plotted along the y axis.
    idx_1 -- index (along axis=0) position of the first color change. e.g. stimulus onset
    idx_2 -- index (along axis=0) position of the first color change. e.g. action onset

    Keyword Arguments:
    name_x -- label for x axis
    name_y -- label for y axis
    color_type -- 'discrete' or 'continuous'
    """

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()

    dat_len = len(x)

    # 3rd variable for the color
    t = np.arange(0, dat_len)

    # reshape to be numlines x points per line x 2 (x and y)
    points = np.array([x, y]).T.reshape(
        -1, 1, 2
    )  # reshape into  numlines x points per line x 2 (x and y)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if color_type == "continuous":
        col_precision = (
            2  # color precision: num of points that is assigned to the same color
        )

        # Create the line collection object, setting the colormapping parameters.
        # Set values used for colormapping separately.
        line = LineCollection(
            segments,
            cmap=plt.get_cmap("coolwarm"),
            norm=plt.Normalize(0, dat_len / col_precision),
        )
    elif color_type == "discrete":
        # Create a colormap for red, green and blue and a norm to color
        # f' < idx_1 red, f' > idx2 blue, and the rest green
        cmap = ListedColormap(["r", "g", "b"])
        norm = BoundaryNorm([0, idx_1, idx_2, len(x)], cmap.N)

        # Create the line collection object, setting the colormapping parameters.
        # Have to set the actual values used for colormapping separately.
        line = LineCollection(segments, cmap=cmap, norm=norm)

    line.set_array(t)
    line.set_linewidth(3)

    # plot the line
    ax.add_collection(line)

    # initialization function: plot the background of each frame
    def init():
        line.set_segments([])
        return (line,)

    # animation function.  This is called sequentially
    def animate(i):
        segments_ani = segments[:i, :, :]
        line.set_segments(segments_ani)
        return (line,)

    # call the animator.  blit=True means only re-draw the parts that have changed.
    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=segments.shape[0], interval=10, blit=True
    )

    # figure style
    plt.xlim(np.min(x) - np.std(x) / 2, np.max(x) + np.std(x) / 2)
    plt.xlabel(f"{name_x}")
    plt.ylim(np.min(y) - np.std(y) / 2, np.max(y) + np.std(y) / 2)
    plt.ylabel(f"{name_y}")
    plt.title("Trajectory Viz")

    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,
        labelbottom=False,  # labels along the bottom edge are off
        labelleft=False,
    )
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.draw()
    plt.show()
    return ani


def traj_viz_continous(
    x, y, axis, name_x="PC 1", name_y="PC 2", title="Neural Trajectory"
):
    """
    Visualize neural trajectories with continous color coding

    Use np.take_along_axis with spike data and returned indices

    Arguments:
    x -- N x 1 array of data to be plotted along the x axis
    y -- N x 1 array of data to be plotted along the y axis.
    axis -- the axis (of subplots) to draw the figure

    Keyword Arguments:
    name_x -- label for x axis
    name_y -- label for y axis
    title -- title
    """

    dat_len = len(x)
    t = np.arange(0, dat_len)
    col_precision = (
        2  # color precision: num of points that is assigned to the same color
    )

    # Create a set of line segments so that we can color them individually
    points = np.array([x, y]).T.reshape(
        -1, 1, 2
    )  # reshape into  numlines x points per line x 2 (x and y)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
    lc = LineCollection(
        segments,
        cmap=plt.get_cmap("coolwarm"),
        norm=plt.Normalize(0, dat_len / col_precision),
    )
    lc.set_array(t)
    lc.set_linewidth(3)

    # plot the line
    axis.add_collection(lc)

    # make it a little nicer
    axis.set_xlim(np.min(x) - np.std(x) / 2, np.max(x) + np.std(x) / 2)
    axis.set_xlabel(f"{name_x}")
    axis.set_ylim(np.min(y) - np.std(y) / 2, np.max(y) + np.std(y) / 2)
    axis.set_ylabel(f"{name_y}")
    axis.set_title(f"{title}")

    axis.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,
        labelbottom=False,  # labels along the bottom edge are off
        labelleft=False,
    )


def traj_viz_discrete(x, y, idx_1, idx_2, name_x="PC 1", name_y="PC 2"):
    """
    Visualize neural trajectories with discrete color coding
    based on specified indices

    Use np.take_along_axis with spike data and returned indices

    Arguments:
    x -- N x 1 array of data to be plotted along the x axis
    y -- N x 1 array of data to be plotted along the y axis
    idx_1 -- index position of the first color change. e.g. stimulus onset
    idx_2 -- index position of the first color change. e.g. action onset

    Keyword Arguments:
    name_x -- label for x axis
    name_y -- label for y axis
    """
    # 3rd variable for the color
    z = np.arange(0, len(x))

    plt.figure(figsize=(6, 6))

    # Create a colormap for red, green and blue and a norm to color
    # f' < idx_1 red, f' > idx2 blue, and the rest green
    cmap = ListedColormap(["r", "g", "b"])
    norm = BoundaryNorm([0, idx_1, idx_2, len(x)], cmap.N)

    # reshape to be numlines x points per line x 2 (x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(z)
    lc.set_linewidth(3)
    plt.gca().add_collection(lc)

    plt.xlim(np.min(x) - np.std(x) / 2, np.max(x) + np.std(x) / 2)
    plt.xlabel(f"{name_x}")
    plt.ylim(np.min(y) - np.std(y) / 2, np.max(y) + np.std(y) / 2)
    plt.ylabel(f"{name_y}")
    plt.title("Trajectory Viz")

    # make it a little nicer
    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,
        labelbottom=False,  # labels along the bottom edge are off
        labelleft=False,
    )
    ax = plt.gca()
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # plt.arrow(x[-1], y[-1], -(x[-2]-x[-1]), -(y[-2]-y[-1]),
    # width = 0.04, shape='full', lw=0, length_includes_head=True,
    # head_width=.04, color='k')


def update_limits(pc_xlim, pc_ylim, xs, ys):
    pc_ylim[0] = ys[0] if pc_ylim[0] > ys[0] else pc_ylim[0]
    pc_ylim[1] = ys[1] if pc_ylim[1] < ys[1] else pc_ylim[1]
    pc_xlim[0] = xs[0] if pc_xlim[0] > xs[0] else pc_xlim[0]
    pc_xlim[1] = xs[1] if pc_xlim[1] < xs[1] else pc_xlim[1]

    return pc_xlim, pc_ylim
