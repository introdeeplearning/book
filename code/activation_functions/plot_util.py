import matplotlib.pyplot as plt

def setup_axis(xlim, ylim):
    _, ax = plt.subplots()

    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    for s in ax.spines.values():
        s.set_zorder(0)

    return ax
