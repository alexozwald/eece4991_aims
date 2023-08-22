def generate_helix(res=25, R=0.15, r=4, N=3, a=3, b=2):
    """
Generate the helical curve and surrounding wire surface.

Parameters:
    res (int): Plot resolution. Determines the # of points in the helix
               A higher value leads to a finer resolution
    R (float): Wire radius. 
    r (float): Helix radius to the wire centerline
    t (int): Number of coils or turns in the helix.
    a (int): Scalar for resolution. Adjusts the resolution of the
             helical path, influencing how smooth the helix appears.
             Multiplying factor for 'res'.
    b (int): Determines how much of the wire surface is generated. A
             value of 2 generates both upper and lower surfaces of the
             wire, while a value of 1 generates only the upper surface.

Returns:
    surface: Dictionary containing x, y, z arrays for the wire surface.
    line: Dictionary containing x, y, z arrays for the centerline of the wire.
    """

    # Surface of the wire (path, wire surface => mesh)
    surface_u = np.linspace(0, N*2*np.pi, int(a*res))
    surface_v = np.linspace(0, b*np.pi, res)
    surface_u, surface_v = np.meshgrid(surface_u, surface_v)

    surface_x = (r + R*np.cos(surface_v)) * np.cos(surface_u)
    surface_y = (r + R*np.cos(surface_v)) * np.sin(surface_u)
    surface_z = R*np.sin(surface_v) + surface_u/np.pi

    # Center line of the wire
    c = a  # Scalar for resolution for line_u to make it look nicer
    line_u = np.linspace(0, N*2*np.pi, int(c*res))
    line_x = r * np.cos(line_u)
    line_y = r * np.sin(line_u)
    line_z = line_u/np.pi

    surface = {'x': surface_x, 'y': surface_y, 'z': surface_z}
    line = {'x': line_x, 'y': line_y, 'z': line_z}

    return surface, line


def plot_helix3(surface, line, r, pitch):
    """
Produce a 3D plot of the helical curve and surrounding wire surface.

Parameters:
    surface: Dict containing x, y, z arrays for the wire surface
    line: Dict containing x, y, z arrays for the centerline of the wire
    r: Radius of the helix
    pitch: Pitch of the helix
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(True)

    # Compute the tangent's x and y components
    tangent_x = -r * np.sin(np.arctan2(surface['y'], surface['x']))
    tangent_y = r * np.cos(np.arctan2(surface['y'], surface['x']))

    # Compute color values based on the tangent
    colors = np.sqrt(tangent_x**2 + tangent_y**2)

    ax.plot(line['x'], line['y'], line['z'], 'r', linewidth=2)

    surf = ax.plot_surface(surface['x'], surface['y'], surface['z'],
        cmap=cm.jet, facecolors=cm.jet(colors), linewidth=0,
        antialiased=True, alpha=0.5)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    ax.view_init(elev=12, azim=45)
    ax.axis('equal')
    plt.show()
