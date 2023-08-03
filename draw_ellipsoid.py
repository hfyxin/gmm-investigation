from matplotlib import pyplot as plt
import numpy as np

fig = plt.figure()  # Square figure
ax = fig.add_subplot(projection='3d')

def draw_ellipsoid(ax, radii, centroid, color='g'):
    # radii: np.array [a, b, c], Coefficients in x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
    assert radii.shape == (3,)
    assert centroid.shape == (3,)
    rx, ry, rz = np.sqrt(radii) # Radii corresponding to the coefficients

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    # move ellpsoid to given position
    x += centroid[0]
    y += centroid[1]
    z += centroid[2]

    # Plot:
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=color, alpha=0.2)

    # Adjustment of the axes, so that they all have the same span:
    # max_radius = max(rx, ry, rz)
    # for axis in 'xyz':
    #     getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

draw_ellipsoid(ax, np.array([1,1,1]), np.array([1,1,0]), color='g')
draw_ellipsoid(ax, np.array([1,1,4]), np.array([-1,-1,0]), color='r')
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.set_zlim([-2,2])
plt.show()
