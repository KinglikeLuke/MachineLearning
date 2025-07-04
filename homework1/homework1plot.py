import typing
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib import rcParams
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

fig, ax = plt.subplots()
ax:Axes
ax.set_xlim(-0.5, 3)
ax.set_ylim(-2.5, 2.5)
spacing = 1
node_radius = 0.2
n_samples = 2
node_style = {"facecolor":"crimson", "edgecolor":"black", "alpha":0.5}
input_style = {"facecolor":"gray", "edgecolor":"black", "alpha":0.5}
output_style = {"facecolor":"navy", "edgecolor":"black", "alpha":0.5}

def labels_to_nodes(patches:typing.List[Circle]|Circle, labels, _ax:Axes):
    if isinstance(patches, list):
        for label, patch in zip(labels, patches):
            _ax.text(*patch.center, label, ha="center", va="center")
    else:
        _ax.text(*patches.center, labels, ha="center", va="center")

def annotate_edge_to_center(ax, circle_from, circle_to, text=None, text_offset = 0.07, **arrowprops):
    """
    Draws an arrow from the edge of `circle_from` (pointing toward circle_to)
    to the center of `circle_to`.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to draw on.
        circle_from (matplotlib.patches.Circle): The source circle (arrow starts from its edge).
        circle_to (matplotlib.patches.Circle): The target circle (arrow points to its center).
        **arrowprops: Additional kwargs for arrow styling (passed to ax.annotate).
    """
    # Get centers and radius
    x1, y1 = circle_from.center
    r1 = circle_from.get_radius()
    x2, y2 = circle_to.center
    r2 = circle_to.get_radius()

    # Direction vector from circle_from to circle_to
    dx, dy = x2 - x1, y2 - y1
    dist = np.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist
    
    if dist == 0:
        # Avoid division by zero
        return

    # Compute edge point on circle_from
    edge_x_1 = x1 + ux * r1
    edge_y_1 = y1 + uy * r1

    edge_x_2 = x2 - ux * r1
    edge_y_2 = y2 - uy * r2
    # Draw annotation
    ax.annotate(
        '', 
        xy=(edge_x_2, edge_y_2),
        xytext=(edge_x_1, edge_y_1),
        arrowprops=arrowprops or dict(arrowstyle='->', lw=2)
    )
    if text:
        # Midpoint of arrow
        mid_x = (edge_x_1 + edge_x_2) / 2
        mid_y = (edge_y_1 + edge_y_2) / 2

        # Perpendicular offset for text
        perp_x = -uy * text_offset
        perp_y = ux * text_offset

        # Position of text
        text_x = mid_x + perp_x
        text_y = mid_y + perp_y

        # Angle for text rotation (in degrees)
        angle = np.degrees(np.arctan2(dy, dx))

        ax.text(
            text_x, text_y, text,
            ha='center', va='center',
            rotation=angle,
            rotation_mode='anchor',
            fontsize=10
        )

x_input:Circle = ax.add_patch(Circle((0,0), node_radius, **input_style))
hidden_top = [ax.add_patch(Circle((spacing, n_samples*spacing - i*spacing), node_radius, **node_style)) for i in range(n_samples)]
hidden_bottom = [ax.add_patch(Circle((spacing, -n_samples*spacing + i*spacing), node_radius, **node_style)) for i in range(n_samples)]

labels_to_nodes(x_input, r"$x$", ax)
labels_to_nodes(hidden_top, [fr"$h_{i}$" for i in range(n_samples)], ax)
labels_to_nodes(hidden_bottom, [fr"$h_{{N-{i}}}$" if i>0 else r"$h_N$" for i in range(n_samples)], ax)

for i, circle in enumerate(hidden_top):
    annotate_edge_to_center(ax, x_input, circle, text=fr"$h_{{\mathbf{{A_2}}_{i+1},\mathbf{{b_2}}_{i+1}}}(x)$", **{"arrowstyle":'->', "lw":2})
for i, circle in enumerate(hidden_bottom):
    annotate_edge_to_center(ax, x_input, circle, text=fr"$h_{{\mathbf{{A_2}}_{{N-{i}}},\mathbf{{b_2}}_{{N-{i}}}}}(x)$" if i > 0 
                            else r"$h_{\mathbf{A_2}_{N},\mathbf{b_2}_{N}}(x)$", **{"arrowstyle":'->', "lw":2})
ax.text(spacing, 0, r"$\dots$", ha="center", va="center", rotation="vertical")    

output = [ax.add_patch(Circle((2.6*spacing, n_samples*spacing*i), node_radius, **output_style)) for i in [-0.5, 0.5]]
labels_to_nodes(output, [fr"$\hat{{y}}_{i}$" for i in [1,2]], ax)
for j, output_circle in enumerate(output):
    for i, hidden_circle in enumerate(hidden_top):
        annotate_edge_to_center(ax, hidden_circle, output_circle, text=None if i>0 
                                else fr"$o_{{\mathbf{{A_1}}_{{{j+1}{i+1}}},\mathbf{{b_2}}_{{{j+1}{i+1}}}}}(h_{i})$", **{"arrowstyle":'->', "lw":2})
    for i, hidden_circle in enumerate(hidden_bottom):
        annotate_edge_to_center(ax, hidden_circle, output_circle, text=None if i>0
                                else fr"$o_{{\mathbf{{A_1}}_{{{j+1}N}},\mathbf{{b_2}}_{{{j+1}N}}}}(h_{i})$", **{"arrowstyle":'->', "lw":2})

ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
for spine in ax.spines.values():
    spine.set_visible(False)
plt.show()