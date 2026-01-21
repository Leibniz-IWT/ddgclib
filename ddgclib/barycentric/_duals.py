import numpy as np
import matplotlib.pyplot as plt
import collections

from scipy.spatial import Delaunay, QhullError
import scipy
import polyscope as ps

from ._complex import Complex
from ._vertex import VertexCacheField
from ddgclib._misc import coldict

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


################################
# New code for volumes:
################################
def _signed_volume_parallelepiped(u, v, w):
    u, v, w = map(np.array, (u, v, w))
    v_para = np.dot(u, v)*u
    v_ortho = v - v_para
    w_prime = w - 2*v_para
    return (np.cross(u, v_ortho)).dot(w_prime)/6

def _volume_parallelepiped(u, v, w):
    vol = np.abs(_signed_volume_parallelepiped(u, v, w))
    print(f"Volume of Parallelepiped={vol}")
    return vol
################################


# Example Usage:
if 0:
    u = np.array((1, 0, 0))
    v = np.array((0, 1, 0))
    w = np.array((0, 0, 1))
    _volume_parallelepiped(u, v, w)


class _PlanePoints:
    """
    A special helper class for plot_dual to define attributes when tri is known, but can't be found with QHull due to
    too few points
    """
    def __init__(self, simplices, points):
        self.simplices = simplices
        self.points = points


def plot_dual(vd, HC, vector_field=None, scalar_field=None, fn='', up="x_up"
              , stl=False, length_scale=1.0, point_radii=0.005):
    # Reset the indices for plotting:
    for i, v in enumerate(HC.V):
        v.index = i
    v1 = vd
    # Initialize polyscope
    ps.init()
    ps.set_up_dir('z_up')
    do = coldict['do']
    lo = coldict['lo']
    db = coldict['db']
    lb = coldict['lb']
    tg = coldict['tg']  # Tab:green colour
    # %% Plot Barycentric dual mesh
    # Loop over primary edges
    dual_points_set = set()
    ssets = []  # Sets of simplices
    v1 = vd
    for i, v2 in enumerate(v1.nn):
        # Find the dual vertex of e12:
        vc_12 = 0.5 * (v2.x_a - v1.x_a) + v1.x_a  # TODO: Should be done in the compute_vd function
        vc_12 = HC.Vd[tuple(vc_12)]

        # Find local dual points intersecting vertices terminating edge:
        dset = v2.vd.intersection(v1.vd)  # Always 5 for boundaries
        # Start with the first vertex and then build triangles, loop back to it:
        vd_i = list(dset)[0]
        if v1.boundary and v2.boundary:
            # print(f'len(dset) = {len(dset)}')
            # Remove the boundary edge which should already be in the set:
            if not (len(vd_i.nn.intersection(dset)) == 1):
                for vd in dset:
                    vd_i = vd
                    if len(vd_i.nn.intersection(dset)) == 1:
                        break
            # iter_len = 3
            # The set length much be different because all interior planes
            # are counted minus two boudary vertices which do not form triangles
            # such as the flux planes in the bulk
            iter_len = len(list(dset)) - 2
        else:
            iter_len = len(list(dset))

        # Main loop
        dsetnn = vd_i.nn.intersection(dset)  # Always 1 internal dual vertices
        vd_j = list(dsetnn)[0]
        # NOTE: In the boundary edges the last triangle does not have
        #      a final vd_j
        # print(f'dset = {dset}')
        for _ in range(iter_len):  # For boundaries should be length 2?
            ssets.append([vc_12, vd_i, vd_j])
            dsetnn_k = vd_j.nn.intersection(dset)  # Always 2 internal dual vertices in interior
            # print(f'dsetnn_k = {dsetnn_k}')
            dsetnn_k.remove(vd_i)  # Should now be size 1
            vd_i = vd_j
            try:
                # Alternatively it should produce an IndexError only when
                # _ = 2 (end of range(3) and we are on a boundary edge
                # so that (v1.boundary and v2.boundary) is true
                vd_j = list(dsetnn_k)[0]  # Retrieve the next vertex
            except IndexError:
                pass  # Should only happen for boundary edges

        # Find local dual points intersecting vertices terminating edge:
        dset = v2.vd.intersection(v1.vd)
        pi = []
        for vd in dset:
            # pi.append(vd.x + 1e-9 * np.random.rand())
            pi.append(vd.x)
            dual_points_set.add(vd.x)
        pi = np.array(pi)
        pi_2d = pi[:, :2] + 1e-9 * np.random.rand()

        # Plot dual points:
        dual_points = []
        for vd in dual_points_set:
            dual_points.append(vd)

        dual_points = np.array(dual_points)
        ps_cloud = ps.register_point_cloud("Dual points", dual_points)
        ps_cloud.set_color(do)
        ps_cloud.set_radius(point_radii)

    # Build the simplices for plotting
    faces = []
    vdict = collections.OrderedDict()  # Ordered cache of vertices to plot
    ind = 0
    # Now iterate through all the constructed simplices and find indexes
    for s in ssets:
        f = []
        for vd in s:
            if not (vd.x in vdict):
                vdict[vd.x] = ind
                ind += 1

            f.append(vdict[vd.x])
        faces.append(f)

    verts = np.array(list(vdict.keys()))
    faces = np.array(faces)

    print(f'verts = {verts}')
    dsurface = ps.register_surface_mesh(f"Dual face", verts, faces,
                                        color=do,
                                        edge_width=0.0,
                                        edge_color=(0.0, 0.0, 0.0),
                                        smooth_shade=False)

    dsurface.set_transparency(0.5)
    # Plot primary mesh
    HC.dim = 2  # The dimension has changed to 2 (boundary surface)
    HC.vertex_face_mesh()
    HC.dim = 3  # Reset the dimension to 3
    points = np.array(HC.vertices_fm)
    triangles = np.array(HC.simplices_fm_i)

    # %% Register the primary vertices as a point cloud
    # `my_points` is a Nx3 numpy array
    my_points = points
    ps_cloud = ps.register_point_cloud("Primary points", my_points)
    ps_cloud.set_color(tuple(db))
    ps_cloud.set_radius(point_radii)
    # ps_cloud.set_color((0.0, 0.0, 0.0))
    verts = my_points
    faces = triangles
    if stl:
        #  msh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                pass
                # msh.vectors[i][j] = verts[f[j], :]

        # msh.save(f'{fn}.stl')

    ### Plot the primary mesh
    # `verts` is a Nx3 numpy array of vertex positions
    # `faces` is a Fx3 array of indices, or a nested list
    if 1:
        surface = ps.register_surface_mesh("Primary surface", verts, faces,
                                           color=db,
                                           edge_width=1.0,
                                           edge_color=(0.0, 0.0, 0.0),
                                           smooth_shade=False)

        surface.set_transparency(0.3)
        # Add a scalar function and a vector function defined on the mesh
        # vertex_scalar is a length V numpy array of values
        # face_vectors is an Fx3 array of vectors per face

        # Scene options (New, not working for scaling
        # NOTE: VERY BROKEN AS IT SCALES THE DIFFERENT MESHES RELATIVELY: NEVER USE THIS:
        #ps.set_autocenter_structures(True)
        #ps.set_autoscale_structures(True)

        # View the point cloud and mesh we just registered in the 3D UI
        # ps.show()
        # Plot particles
        # Ground plane options
        ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
        ps.set_ground_plane_height_factor(0.1)  # adjust the plane height
        ps.set_shadow_darkness(0.2)  # lighter shadows
        ps.set_shadow_blur_iters(2)  # lighter shadows
        ps.set_transparency_mode('pretty')
        ps.set_length_scale(length_scale)
        #ps.set_length_scale(length_scale)
     #   ps.set_length_scale(length_scale)
        # ps.look_at((0., -10., 0.), (0., 0., 0.))
       # ps.look_at((1., -8., -8.), (0., 0., 0.))
        # ps.set_ground_plane_height_factor(x, is_relative=True)
        ps.set_screenshot_extension(".png")
        # Take a screenshot
        # It will be written to your current directory as screenshot_000000.jpg, etc
        ps.screenshot(fn)

    return ps, du


def plot_dual_oldY(v, HC, vector_field=None, scalar_field=None, fn='', up="x_up"
              , stl=False):
    v1 = v
    # Initialize polyscope
    ps.init()
    ps.set_up_dir('z_up')
    do = coldict['do']
    lo = coldict['lo']
    db = coldict['db']
    lb = coldict['lb']
    tg = coldict['tg']  # Tab:green colour
    # %% Plot Barycentric dual mesh
    # Loop over primary edges
    dual_points_set = set()
    ssets = []  # Sets of simplices
    v1 = v
    for i, v2 in enumerate(v1.nn):
        # Find the dual vertex of e12:
        vc_12 = 0.5 * (v2.x_a - v1.x_a) + v1.x_a  # TODO: Should be done in the compute_vd function
        vc_12 = HC.Vd[tuple(vc_12)]

        # Find local dual points intersecting vertices terminating edge:
        dset = v2.vd.intersection(v1.vd)  # Always 5 for boundaries
        # Start with the first vertex and then build triangles, loop back to it:
        vd_i = list(dset)[0]
        if v1.boundary and v2.boundary:
            # Remove the boundary edge which should already be in the set:
            if not (len(vd_i.nn.intersection(dset)) == 1):
                for vd in dset:
                    vd_i = vd
                    if len(vd_i.nn.intersection(dset)) == 1:
                        break
            iter_len = 3
        else:
            iter_len = len(list(dset))

        # Main loop
        dsetnn = vd_i.nn.intersection(dset)  # Always 1 internal dual vertices
        vd_j = list(dsetnn)[0]
        # NOTE: In the boundary edges the last triangle does not have
        #      a final vd_j
        for _ in range(iter_len):  # For boundaries should be length 2?
            ssets.append([vc_12, vd_i, vd_j])
            dsetnn_k = vd_j.nn.intersection(dset)  # Always 2 internal dual vertices in interior
            #print(f'dsetnn_k = {dsetnn_k}')
            dsetnn_k.remove(vd_i)  # Should now be size 1
            vd_i = vd_j
            try:
                # Alternatively it should produce an IndexError only when
                # _ = 2 (end of range(3) and we are on a boundary edge
                # so that (v1.boundary and v2.boundary) is true
                vd_j = list(dsetnn_k)[0]  # Retrieve the next vertex
            except IndexError:
                pass  # Should only happen for boundary edges

        # Find local dual points intersecting vertices terminating edge:
        dset = v2.vd.intersection(v1.vd)
        pi = []
        for vd in dset:
            # pi.append(vd.x + 1e-9 * np.random.rand())
            pi.append(vd.x)
            dual_points_set.add(vd.x)
        pi = np.array(pi)
        pi_2d = pi[:, :2] + 1e-9 * np.random.rand()

        # Plot dual points:
        dual_points = []
        for vd in dual_points_set:
            dual_points.append(vd)

        dual_points = np.array(dual_points)
        ps_cloud = ps.register_point_cloud("Dual points", dual_points)
        ps_cloud.set_color(do)

    # Build the simplices for plotting
    faces = []
    vdict = collections.OrderedDict()  # Ordered cache of vertices to plot
    ind = 0
    # Now iterate through all the constructed simplices and find indexes
    for s in ssets:
        f = []
        for v in s:
            if not (v.x in vdict):
                vdict[v.x] = ind
                ind += 1

            f.append(vdict[v.x])
        faces.append(f)

    verts = np.array(list(vdict.keys()))
    faces = np.array(faces)

    dsurface = ps.register_surface_mesh(f"Dual face {i}", verts, faces,
                                        color=do,
                                        edge_width=0.0,
                                        edge_color=(0.0, 0.0, 0.0),
                                        smooth_shade=False)

    # Plot primary mesh
    HC.dim = 2  # The dimension has changed to 2 (boundary surface)
    HC.vertex_face_mesh()
    points = np.array(HC.vertices_fm)
    triangles = np.array(HC.simplices_fm_i)

    # %% Register the primary vertices as a point cloud
    # `my_points` is a Nx3 numpy array
    my_points = points
    ps_cloud = ps.register_point_cloud("Primary points", my_points)
    ps_cloud.set_color(tuple(db))
    # ps_cloud.set_color((0.0, 0.0, 0.0))
    verts = my_points
    faces = triangles
    if stl:
        #  msh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                pass
                # msh.vectors[i][j] = verts[f[j], :]

        # msh.save(f'{fn}.stl')

    ### Plot the primary mesh
    # `verts` is a Nx3 numpy array of vertex positions
    # `faces` is a Fx3 array of indices, or a nested list
    if 1:
        surface = ps.register_surface_mesh("Primary surface", verts, faces,
                                           color=db,
                                           edge_width=1.0,
                                           edge_color=(0.0, 0.0, 0.0),
                                           smooth_shade=False)

        surface.set_transparency(0.3)
        # Add a scalar function and a vector function defined on the mesh
        # vertex_scalar is a length V numpy array of values
        # face_vectors is an Fx3 array of vectors per face

        # View the point cloud and mesh we just registered in the 3D UI
        # ps.show()
        # Plot particles
        # Ground plane options
        ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
        ps.set_ground_plane_height_factor(0.1)  # adjust the plane height
        ps.set_shadow_darkness(0.2)  # lighter shadows
        ps.set_shadow_blur_iters(2)  # lighter shadows
        ps.set_transparency_mode('pretty')
        # ps.look_at((0., -10., 0.), (0., 0., 0.))
        ps.look_at((1., -8., -8.), (0., 0., 0.))
        # ps.set_ground_plane_height_factor(x, is_relative=True)
        ps.set_screenshot_extension(".png")
        # Take a screenshot
        # It will be written to your current directory as screenshot_000000.jpg, etc
        ps.screenshot(fn)
    return ps


def plot_dual_oldZ(v, HC, vector_field=None, scalar_field=None, fn='', up="x_up"
              , stl=False):
    v1 = v
    # Initialize polyscope
    ps.init()
    ps.set_up_dir('z_up')
    do = coldict['do']
    lo = coldict['lo']
    db = coldict['db']
    lb = coldict['lb']
    tg = coldict['tg']  # Tab:green colour
    #%% Plot Barycentric dual mesh
    # Loop over primary edges
    dual_points_set = set()
    ssets = []  # Sets of simplices
    v1 = v
    for i, v2 in enumerate(v1.nn):
        print('-')
        # Find the dual vertex of e12:
        vc_12 = 0.5 * (v2.x_a - v1.x_a) + v1.x_a  #TODO: Should be done in the compute_vd function
        vc_12 = HC.Vd[tuple(vc_12)]
        print(f'vc_12.x = {vc_12.x}')
        # Find local dual points intersecting vertices terminating edge:
        dset = v2.vd.intersection(v1.vd)  # Always 5 for boundaries
        print(f'len(dset) = {len(dset)}')
        # Start with the first vertex and then build triangles, loop back to it:
        vd_i = list(dset)[0]
        if v1.boundary and v2.boundary:
            # Remove the boundary edge which should already be in the set:
            if not (len(vd_i.nn.intersection(dset)) == 1):
                for vd in dset:
                    vd_i = vd
                    if len(vd_i.nn.intersection(dset)) == 1:
                        break

            # Main loop
            dsetnn = vd_i.nn.intersection(dset)  # Always 1 internal dual vertices
            print(f'vd_i.nn.intersection(dset) = {vd_i.nn.intersection(dset)}')
            print(f'vd_i.x = {vd_i.x}')
            print(f' len(vd_i.nn.intersection(dset)) = { len(vd_i.nn.intersection(dset))}')
            vd_j = list(dsetnn)[0]
            print(f'vd_j init = {vd_j.x}')
            #NOTE: In the boundary edges the last triangle does not have
            #      a final vd_j
            for _ in range(3):  # For boundaries should be length 2?
                ssets.append([vc_12, vd_i, vd_j])
                dsetnn_k = vd_j.nn.intersection(dset)  # Always 2 internal dual vertices in interior
                print(f'len(dsetnn_k) = {len(dsetnn_k)}')
                for v in dsetnn_k:
                    print(f'v.x in dsetnn_k = {v.x}')
                #if not (v1.boundary and v2.boundary):
                dsetnn_k.remove(vd_i)  # Should now be size 1
                print(f'len(dsetnn_k) post remove = {len(dsetnn_k)}')
                vd_i = vd_j
                try:
                    vd_j = list(dsetnn_k)[0]  # Retrieve the next vertex
                    print(f'vd_j after for loop = {vd_j.x}')
                except IndexError:
                    pass  # Should only happen for boundary edges

        else:
            # Main loop
            dsetnn = vd_i.nn.intersection(dset)  # Always 2 internal dual vertices
            vd_j = list(dsetnn)[0]
            for _ in list(dset):
                ssets.append([vc_12, vd_i, vd_j])
                dsetnn_j = vd_j.nn.intersection(dset)  # Always 2 internal dual vertices in interior
                if not (v1.boundary and v2.boundary):
                    dsetnn_j.remove(vd_i)  # Should now be size 1
                vd_i = vd_j
                vd_j = list(dsetnn_j)[0]  # Retrieve the next vertex


        # Find local dual points intersecting vertices terminating edge:
        dset = v2.vd.intersection(v1.vd)
        pi = []
        for vd in dset:
            # pi.append(vd.x + 1e-9 * np.random.rand())
            pi.append(vd.x)
            dual_points_set.add(vd.x)
        pi = np.array(pi)
        pi_2d = pi[:, :2] + 1e-9 * np.random.rand()

        # Plot dual points:
        dual_points = []
        for vd in dual_points_set:
            dual_points.append(vd)

        dual_points = np.array(dual_points)
        ps_cloud = ps.register_point_cloud("Dual points", dual_points)
        ps_cloud.set_color(do)


    # Build the simplices for plotting
    faces = []
    vdict = collections.OrderedDict()  # Ordered cache of vertices to plot
    ind = 0
    # Now iterate through all the constructed simplices and find indexes
    for s in ssets:
        f = []
        for v in s:
            if not (v.x in vdict):
                vdict[v.x] = ind
                ind += 1

            f.append(vdict[v.x])
        faces.append(f)

    verts = np.array(list(vdict.keys()))
    faces = np.array(faces)

    dsurface = ps.register_surface_mesh(f"Dual face {i}", verts, faces,
                                        color=do,
                                        edge_width=0.0,
                                        edge_color=(0.0, 0.0, 0.0),
                                        smooth_shade=False)

    # Plot primary mesh
    HC.dim = 2  # The dimension has changed to 2 (boundary surface)
    HC.vertex_face_mesh()
    points = np.array(HC.vertices_fm)
    triangles = np.array(HC.simplices_fm_i)

    #%% Register the primary vertices as a point cloud
    # `my_points` is a Nx3 numpy array
    my_points = points
    ps_cloud = ps.register_point_cloud("Primary points", my_points)
    ps_cloud.set_color(tuple(db))
    # ps_cloud.set_color((0.0, 0.0, 0.0))
    verts = my_points
    faces = triangles
    if stl:
        #  msh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                pass
                # msh.vectors[i][j] = verts[f[j], :]

        # msh.save(f'{fn}.stl')

    ### Plot the primary mesh
    # `verts` is a Nx3 numpy array of vertex positions
    # `faces` is a Fx3 array of indices, or a nested list
    if 1:
        surface = ps.register_surface_mesh("Primary surface", verts, faces,
                                           color=db,
                                           edge_width=1.0,
                                           edge_color=(0.0, 0.0, 0.0),
                                           smooth_shade=False)

        surface.set_transparency(0.3)
        # Add a scalar function and a vector function defined on the mesh
        # vertex_scalar is a length V numpy array of values
        # face_vectors is an Fx3 array of vectors per face

        # View the point cloud and mesh we just registered in the 3D UI
        # ps.show()
        # Plot particles
        # Ground plane options
        ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
        ps.set_ground_plane_height_factor(0.1)  # adjust the plane height
        ps.set_shadow_darkness(0.2)  # lighter shadows
        ps.set_shadow_blur_iters(2)  # lighter shadows
        ps.set_transparency_mode('pretty')
        # ps.look_at((0., -10., 0.), (0., 0., 0.))
        ps.look_at((1., -8., -8.), (0., 0., 0.))
        # ps.set_ground_plane_height_factor(x, is_relative=True)
        ps.set_screenshot_extension(".png")
        # Take a screenshot
        # It will be written to your current directory as screenshot_000000.jpg, etc
        ps.screenshot(fn)
    return ps

def plot_dual_old(v, HC, vector_field=None, scalar_field=None, fn='', up="x_up"
              , stl=False):
    # Initialize polyscope
    ps.init()
    ps.set_up_dir('z_up')

    do = coldict['do']
    lo = coldict['lo']
    db = coldict['db']
    lb = coldict['lb']
    tg = coldict['tg']  # Tab:green colour
    ## Plot dual mesh
    # Loop over primary edges
    dual_points_set = set()
    for i, v2 in enumerate(v.nn):
        # For each primary edge:
        # find local dual points intersecting vertices terminating edge:
        dset = v2.vd.intersection(v.vd)
        pi = []
        for vd in dset:
            # pi.append(vd.x + 1e-9 * np.random.rand())
            pi.append(vd.x)
            dual_points_set.add(vd.x)
        pi = np.array(pi)
        pi_2d = pi[:, :2] + 1e-9 * np.random.rand()
        # Find the (delaunay) simplices from scipy:
        try:
            tri = scipy.spatial.Delaunay(pi_2d, qhull_options='QJ')
        except QhullError as e:
            #NOTE: This should not be needed in the future, but during development some
            #      boundary primary edges only have 3 vertices associated with them. There-
            #      a quick hack is to do the manual triangulation below:
            print(f'e = {e}')
            simplices = [[0, 1, 2]]
            points = pi
            tri = _PlanePoints(simplices, points)
            pass  # print(e)
        verts = pi
        faces = tri.simplices
        dsurface = ps.register_surface_mesh(f"Dual face {i}", verts, faces,
                                            color=do,
                                            edge_width=0.0,
                                            edge_color=(0.0, 0.0, 0.0),
                                            smooth_shade=False)

        dsurface.set_transparency(0.7)

    # Plot dual points:
    dual_points = []
    for vd in dual_points_set:
        dual_points.append(vd)

    dual_points = np.array(dual_points)
    ps_cloud = ps.register_point_cloud("Dual points", dual_points)
    ps_cloud.set_color(do)

    # Plot primary mesh
    HC.dim = 2  # The dimension has changed to 2 (boundary surface)
    HC.vertex_face_mesh()
    points = np.array(HC.vertices_fm)
    triangles = np.array(HC.simplices_fm_i)
    ### Register a point cloud
    # `my_points` is a Nx3 numpy array
    my_points = points
    ps_cloud = ps.register_point_cloud("Primary points", my_points)
    ps_cloud.set_color(tuple(db))
    # ps_cloud.set_color((0.0, 0.0, 0.0))
    verts = my_points
    faces = triangles
    if stl:
        print(f'verts = {verts}')
        print(f'faces = {faces}')
        #  msh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                pass
                # msh.vectors[i][j] = verts[f[j], :]

        # msh.save(f'{fn}.stl')
    ### Register a mesh
    # `verts` is a Nx3 numpy array of vertex positions
    # `faces` is a Fx3 array of indices, or a nested list
    surface = ps.register_surface_mesh("Primary surface", verts, faces,
                                       color=db,
                                       edge_width=1.0,
                                       edge_color=(0.0, 0.0, 0.0),
                                       smooth_shade=False)

    surface.set_transparency(0.3)
    # Add a scalar function and a vector function defined on the mesh
    # vertex_scalar is a length V numpy array of values
    # face_vectors is an Fx3 array of vectors per face

    # View the point cloud and mesh we just registered in the 3D UI
    # ps.show()
    # Plot particles
    # Ground plane options
    ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
    ps.set_ground_plane_height_factor(0.1)  # adjust the plane height
    ps.set_shadow_darkness(0.2)  # lighter shadows
    ps.set_shadow_blur_iters(2)  # lighter shadows
    ps.set_transparency_mode('pretty')
    # ps.look_at((0., -10., 0.), (0., 0., 0.))
    #ps.look_at((1., -8., -8.), (0., 0., 0.))
    # ps.set_ground_plane_height_factor(x, is_relative=True)
    ps.set_screenshot_extension(".png")
    # Take a screenshot
    # It will be written to your current directory as screenshot_000000.jpg, etc
    ps.screenshot(fn)
    return ps


def _set_boundary(v, val=True):
    """
    small helper fuction to set the boundary value property for the supplied vertex.
    :param v:
    :return:
    """
    v.boundary = val

# Dual complex computation functions:
def _merge_local_duals_vector(x_a_l, Vd_cache, cdist=1e-10):
    """
    For a proposed new vertex position, first check the local dual cache
    of vertices for a similar position, if one is found, use that exact
    position instead to avoid generating duplicate dual vetices.

    This is needed due to overflow errors giving slightly different results
    and therefore producing multiple keys for the same dual.

    :param x_a_l: List of vectors of new vertex position
    :param Vd_cache: iterable object of local dual vertices
    :param cdist: scalar, tolerance of identifying dual vertices
    :return: x_a_l: The modified list of vertex, newly merged and unmerged
    """
    for vd_i in Vd_cache:
        for i, x_a in enumerate(x_a_l):
            dist = np.linalg.norm(vd_i.x_a - x_a)
            if dist < cdist:
                x_a_l[i] = vd_i.x_a

    return x_a_l


def compute_vd(HC, cdist=1e-10):
    """
    Computes the dual vertices of a primal vertex cache HC.V on
    each dim - 1 simplex.

    Currently only dim = 2, 3 is supported

    cdist: float, tolerance for where a unique dual vertex can exist

    """
    # TODO: Merging the dual vertices is probably inefficient, it might
    # actually be more efficient to do one global merge of the vertices
    # after the routine has finished instead.
    # Construct dual cache
    HC.Vd = VertexCacheField()

    # Construct dual neighbour sets
    for v in HC.V:
        v.vd = set()

    # hcv = copy.copy(HC.V)
    if HC.dim == 2:
        for v1 in HC.V:

            for v2 in v1.nn:
                # Compute the local dual neighbourhood to current v2:
                # NOTE: This should be updated in every v3 for loop because dual vertices are
                #      being added to this nn cache every loop:
                v1_d_nn = list(v1.vd)
                # If boundary vertex, we stop and generate a new vertex on the boundary edge.
                try:
                    if v1.boundary and v2.boundary:
                        cd = v1.x_a + 0.5 * (v2.x_a - v1.x_a)
                        cd = _merge_local_duals_vector([cd], v1_d_nn, cdist=cdist)[0]
                        vd = HC.Vd[tuple(cd)]
                        v1.vd.add(vd)
                        v2.vd.add(vd)
                        # Connect to dual simplex
                        v1nn_u_v2nn = v1.nn.intersection(v2.nn)  # Should always be length 1
                        v3 = list(v1nn_u_v2nn)[0]
                        verts = np.zeros([3, HC.dim])
                        verts[0] = v1.x_a
                        verts[1] = v2.x_a
                        verts[2] = v3.x_a
                        cd1 = np.mean(verts, axis=0)
                        vd1 = HC.Vd[tuple(cd1)]
                        # Connect the two dual vertices forming the boundary dual edge:
                        vd.connect(vd1)
                        continue
                except AttributeError:
                    pass
                # Find all v2.nn also connected to v1:
                v1nn_u_v2nn = v1.nn.intersection(v2.nn)  # Should always be length 2
                # In 2D there are only two
                v3_1 = list(v1nn_u_v2nn)[0]
                v3_2 = list(v1nn_u_v2nn)[1]
                if (v3_1 is v1) or (v3_2 is v1):
                    continue
                verts = np.zeros([3, HC.dim])
                verts[0] = v1.x_a
                verts[1] = v2.x_a
                verts[2] = v3_1.x_a
                # Compute the circumcentre:
                # cd = circumcenter(verts)
                # Compute the barycentre of the first connected triangle sharing primary edge/face e_1e:
                cd1 = np.mean(verts, axis=0)

                # Compute the barycentre of the first connected triangle sharing primary edge/face e_1e:
                verts[2] = v3_2.x_a
                cd2 = np.mean(verts, axis=0)

                # Ensure that floating point errors are not generating a unique vertex
                # NOTE: In the future cdist should be selected dynamically based on the local
                #       distance between v2 and its dual vertices / primary edge connections
                (cd1, cd2) = _merge_local_duals_vector([cd1, cd2], v1_d_nn, cdist=cdist)

                vd1 = HC.Vd[tuple(cd1)]
                vd2 = HC.Vd[tuple(cd2)]
                # Connect the two dual vertices:
                vd1.connect(vd2)

                # Connect to all primal vertices of v3_1 dual
                for v in [v1, v2, v3_1]:
                    v.vd.add(vd1)

                # Connect to all primal vertices of v3_2 dual
                for v in [v1, v2, v3_2]:
                    v.vd.add(vd2)

    elif HC.dim == 3:
        for v1 in HC.V:
            for v2 in v1.nn:
                # Note: every boundary primary edge only has two boundary tetrahedra connected
                # and therefore only two barycentric dual points. We do not need to connect with
                # other duals therefore simply connect to the primary edges.

                # Find all v2.nn also connected to v1:
                v1nn_u_v2nn = v1.nn.intersection(v2.nn)
                for v3 in v1nn_u_v2nn:
                    # Compute the local dual neighbourhood to current v2:
                    # NOTE: This should be updated in every v3 for loop because dual vertices are
                    #      being added to this nn cache every loop:
                    v1_d_nn = list(v1.vd)
                    # print('-')
                    if (v3 is v1):
                        continue

                    v1nn_u_v2nn_u_v3nn = v1nn_u_v2nn.intersection(
                        v3.nn)  # Should be length 2, unless the triangle is on the boundary
                    # print(f'v1.x = {v1.x}')
                    # print(f'v2.x = {v2.x}')
                    # print(f'v3.x = {v3.x}')
                    v4_1 = list(v1nn_u_v2nn_u_v3nn)[0]
                    # v4_2 = list(v1nn_u_v2nn_u_v3nn)[1]

                    # if (v4_1 is v1) or (v4_1 is v2) or (v4_2 is v1) or (v4_2 is v2):
                    #    continue

                    # debug above, should never occur?:
                    if 1:
                        if (v4_1 is v1) or (v4_1 is v2):
                            print(f'WARNING (v4_1 is v1) or (v4_1 is v2)')

                    # Compute the two duals of tetrahedra connected by face f_123 of triangle [v1, v2, v3]
                    verts = np.zeros([HC.dim + 1, HC.dim])
                    verts[0] = v1.x_a  # TODO: Added 08.03.24, investigate accidental deletion?
                    verts[1] = v2.x_a
                    verts[2] = v3.x_a
                    verts[3] = v4_1.x_a
                    #  Compute the barycentre of the first connected simplex sharing primary face f_123:
                    cd1 = np.mean(verts, axis=0)

                    # If v123 is on the boundary then we instead want to generate the barycenter
                    # dual vd123 and then connect it to edge dual vd12 and cd1
                    if (v1.boundary and v2.boundary) and v3.boundary:
                        # debug print:
                        if len(list(v1nn_u_v2nn_u_v3nn)) > 1:
                            print(
                                f'WARNING: len(list(v1nn_u_v2nn_u_v3nn)) = {len(list(v1nn_u_v2nn_u_v3nn))} which is > expected 1')

                        # verts_b = np.zeros([3, HC.dim])
                        verts_b = verts[:3]
                        cd2 = np.mean(verts_b, axis=0)

                        # Connect the dual of e_12 primal edge vertices
                        cd12 = v1.x_a + 0.5 * (v2.x_a - v1.x_a)
                        cd12 = _merge_local_duals_vector([cd12], v1_d_nn, cdist=cdist)[0]
                        vd12 = HC.Vd[tuple(cd12)]
                        v1.vd.add(vd12)
                        v2.vd.add(vd12)

                    # Compute the barycentre of the second connected simplex sharing primary face f_123:
                    else:
                        v4_2 = list(v1nn_u_v2nn_u_v3nn)[1]
                        verts[3] = v4_2.x_a
                        cd2 = np.mean(verts, axis=0)

                        # debug above, should never occur?:
                        if 1:
                            if (v4_2 is v1) or (v4_2 is v2):
                                print(f'WARNING (v4_1 is v1) or (v4_1 is v2)')

                    (cd1, cd2) = _merge_local_duals_vector([cd1, cd2], v1_d_nn, cdist=cdist)

                    #  Define the new dual vertices
                    vd1 = HC.Vd[tuple(cd1)]
                    vd2 = HC.Vd[tuple(cd2)]
                    # Connect the two dual vertices:
                    vd1.connect(vd2)

                    # Connect to all primal vertices of v3_1 dual
                    for v in [v1, v2, v3, v4_1]:
                        v.vd.add(vd1)

                    # Connect to all primal vertices of v3_2 dual
                    if (v1.boundary and v2.boundary) and v3.boundary:
                        for v in [v1, v2, v3]:  # v4_2 doesn't exist on boundary face
                            v.vd.add(vd2)
                    else:
                        for v in [v1, v2, v3, v4_2]:
                            v.vd.add(vd2)

    return HC  # self


def compute_vd_old3(HC, cdist=1e-10):
    """
    Computes the dual vertices of a primal vertex cache HC.V on
    each dim - 1 simplex.

    Currently only dim = 2, 3 is supported

    cdist: float, tolerance for where a unique dual vertex can exist

    """
    # Construct dual cache
    HC.Vd = VertexCacheField()

    # Construct dual neighbour sets
    for v in HC.V:
        v.vd = set()

    # hcv = copy.copy(HC.V)
    if HC.dim == 2:
        for v1 in HC.V:
            for v2 in v1.nn:
                # If boundary vertex, we stop and generate a new vertex on the boundary edge.
                try:
                    if v1.boundary and v2.boundary:
                        cd = v1.x_a + 0.5 * (v2.x_a - v1.x_a)
                        vd = HC.Vd[tuple(cd)]
                        v1.vd.add(vd)
                        v2.vd.add(vd)
                        # Connect to dual simplex
                        v1nn_u_v2nn = v1.nn.intersection(v2.nn)  # Should always be length 1
                        v3 = list(v1nn_u_v2nn)[0]
                        verts = np.zeros([3, HC.dim])
                        verts[0] = v1.x_a
                        verts[1] = v2.x_a
                        verts[2] = v3.x_a
                        cd1 = np.mean(verts, axis=0)
                        vd1 = HC.Vd[tuple(cd1)]
                        # Connect the two dual vertices forming the boundary dual edge:
                        vd.connect(vd1)
                        continue
                except AttributeError:
                    pass
                # Find all v2.nn also connected to v1:
                v1nn_u_v2nn = v1.nn.intersection(v2.nn)  # Should always be length 2
                # In 2D there are only two
                v3_1 = list(v1nn_u_v2nn)[0]
                v3_2 = list(v1nn_u_v2nn)[1]
                if (v3_1 is v1) or (v3_2 is v1):
                    continue
                verts = np.zeros([3, HC.dim])
                verts[0] = v1.x_a
                verts[1] = v2.x_a
                verts[2] = v3_1.x_a
                # Compute the circumcentre:
                # cd = circumcenter(verts)
                # Compute the barycentre of the first connected triangle sharing primary edge/face e_1e:
                cd1 = np.mean(verts, axis=0)

                # Compute the barycentre of the first connected triangle sharing primary edge/face e_1e:
                verts[2] = v3_2.x_a
                cd2 = np.mean(verts, axis=0)
                # Note instead of below, could round off cd in general to say nearest 1e-12
                # Check for uniqueness first (new, expensive, could
                # be improved by checking duals of neighbours only?):
                for vd_i in HC.Vd:
                    dist1 = np.linalg.norm(vd_i.x_a - cd1)
                    dist2 = np.linalg.norm(vd_i.x_a - cd2)
                    if dist1 < cdist:
                        cd1 = vd_i.x_a
                    if dist2 < cdist:
                        cd2 = vd_i.x_a

                vd1 = HC.Vd[tuple(cd1)]
                vd2 = HC.Vd[tuple(cd2)]
                # Connect the two dual vertices:
                vd1.connect(vd2)

                # Connect to all primal vertices of v3_1 dual
                for v in [v1, v2, v3_1]:
                    v.vd.add(vd1)

                # Connect to all primal vertices of v3_2 dual
                for v in [v1, v2, v3_2]:
                    v.vd.add(vd2)

    elif HC.dim == 3:
        for v1 in HC.V:
            for v2 in v1.nn:
                # Note: every boundary primary edge only has two boundary tetrahedra connected
                # and therefore only two barycentric dual points. We do not need to connect with
                # other duals therefore simply connect to the primary edges.
                if v1.boundary and v2.boundary:
                    # Find all v2.nn also connected to v1:
                    # Find the other two primary edges
                    v1nn_u_v2nn = v1.nn.intersection(v2.nn)
                    for v3 in v1nn_u_v2nn:
                        if (v3 is v1):
                            continue
                        if v3.boundary:
                            # Find the barycentre of the triangle
                            verts = np.zeros([3, HC.dim])
                            verts[0] = v1.x_a  # TODO: Added 08.03.24, investigate accidental deletion?
                            verts[1] = v2.x_a
                            verts[2] = v3.x_a
                            # Compute the circumcentre:
                            # cd = circumcenter(verts)
                            # Compute the barycentre:
                            cd = np.mean(verts, axis=0)
                            for vd_i in HC.Vd:
                                dist = np.linalg.norm(vd_i.x_a - cd)
                                if dist < cdist:
                                    cd = vd_i.x_a

                            # Define the new dual vertex on the face/triangle [v1, v2, v3]:
                            vd123 = HC.Vd[tuple(cd)]

                            # Find the simplex that connects to boundary face/triangle [v1, v2, v3]:
                            v1nn_u_v2nn_u_v3nn = v1.nn.intersection(v2.nn).intersection(
                                v3.nn)  # Always length 1
                            v4 = list(v1nn_u_v2nn_u_v3nn)[0]

                            verts = np.zeros([4, HC.dim])
                            verts[0] = v1.x_a
                            verts[1] = v2.x_a
                            verts[2] = v3.x_a
                            verts[3] = v4.x_a
                            cd1234 = np.mean(verts, axis=0)
                            vd1234 = HC.Vd[tuple(cd1234)]
                            # Connect the two dual vertices forming the boundary dual edge:
                            vd123.connect(vd1234)

                            # Connect to all primal vertices on the boundary:
                            # TODO: CHECK IF ALL THE CONNECTIONS BELOW ARE CORRECT:
                            for v in [v1, v2, v3, v4]:
                                if not v.boundary:
                                    continue
                                v.vd.add(vd123)
                                vd123.nn.add(v)  #TODO: THis seems broken, should not be done?

                            # Connect the dual of e_12 to vd1234
                            if 1:
                                cd12 = v1.x_a + 0.5 * (v2.x_a - v1.x_a)
                                vd12 = HC.Vd[tuple(cd12)]
                            if 0:
                                vd1234.connect(vd12)
                            # Connect vd123 with duals of primary edges e_12 and e_13
                            if 1:
                               # cd12 = v1.x_a + 0.5 * (v2.x_a - v1.x_a)
                               # vd12 = HC.Vd[tuple(cd12)]

                                # cd13 = v1.x_a + 0.5 * (v3.x_a - v1.x_a)
                                # vd13 = HC.Vd[tuple(cd13)]

                                # Close out on primary border edges:
                                # vd123.connect(vd12)
                                # vd123.connect(vd13)

                                # Also connect to primary edges
                                v1.vd.add(vd12)
                                v2.vd.add(vd12)
                            # v1.vd.add(vd13)
                            # v2.vd.add(vd13)

                    continue


                # Find all v2.nn also connected to v1:
                v1nn_u_v2nn = v1.nn.intersection(v2.nn)
                for v3 in v1nn_u_v2nn:
                    if (v3 is v1):
                        continue
                    v1nn_u_v2nn_u_v3nn = v1nn_u_v2nn.intersection(v3.nn)  # Should be length 2
                    # print(f'len(v1nn_u_v2nn_u_v3nn) = {len(v1nn_u_v2nn_u_v3nn)}')
                    # In 3D there are only two
                    v4_1 = list(v1nn_u_v2nn_u_v3nn)[0]
                    v4_2 = list(v1nn_u_v2nn_u_v3nn)[1]
                    if (v4_1 is v1) or (v4_1 is v2) or (v4_2 is v1) or (v4_2 is v2):
                        continue

                    # Compute the two duals of tetrahedra connected by face f_123 of triangle [v1, v2, v3]
                    verts = np.zeros([HC.dim + 1, HC.dim])
                    verts[0] = v1.x_a  # TODO: Added 08.03.24, investigate accidental deletion?
                    verts[1] = v2.x_a
                    verts[2] = v3.x_a
                    verts[3] = v4_1.x_a
                    #  Compute the barycentre of the first connected simplex sharing primary face f_123:
                    cd1 = np.mean(verts, axis=0)

                    #  Compute the barycentre of the first connected simplex sharing primary face f_123:
                    verts[3] = v4_2.x_a
                    cd2 = np.mean(verts, axis=0)

                    # Note instead of below, could round off cd in general to say nearest 1e-12
                    # Check for uniqueness first (new, expensive, could
                    # be improved by checking duals of neighbours only?):
                    if 0:
                        for vd_i in HC.Vd:
                            dist1 = np.linalg.norm(vd_i.x_a - cd1)
                            dist2 = np.linalg.norm(vd_i.x_a - cd2)
                            if dist1 < cdist:
                                cd1 = vd_i.x_a
                            if dist2 < cdist:
                                cd2 = vd_i.x_a

                    #  Define the new dual vertices
                    vd1 = HC.Vd[tuple(cd1)]
                    vd2 = HC.Vd[tuple(cd2)]
                    # Connect the two dual vertices:
                    vd1.connect(vd2)

                    # Connect to all primal vertices of v3_1 dual
                    for v in [v1, v2, v3, v4_1]:
                        v.vd.add(vd1)

                    # Connect to all primal vertices of v3_2 dual
                    for v in [v1, v2, v3, v4_2]:
                        v.vd.add(vd2)

                    if 0:
                        for v4 in v1nn_u_v2nn_u_v3nn:
                            if (v4 is v1) or (v4 is v2):
                                continue
                            # TODO: Re-implement cache:
                            verts = np.zeros([HC.dim + 1, HC.dim])
                            verts[0] = v1.x_a  # TODO: Added 08.03.24, investigate accidental deletion?
                            verts[1] = v2.x_a
                            verts[2] = v3.x_a
                            verts[3] = v4.x_a

                            # Compute the circumcentre:
                            # cd = circumcenter(verts)
                            # Compute the barycentre:
                            cd = np.mean(verts, axis=0)
                            # Note instead of below, could round off cd in general to say nearest 1e-12
                            # Check for uniqueness first (new, expensive, could
                            # be improved by checking duals of neighbours only?):
                            for vd_i in HC.Vd:
                                dist = np.linalg.norm(vd_i.x_a - cd)
                                if dist < cdist:
                                    cd = vd_i.x_a

                            #  Define the new dual vertex:
                            vd = HC.Vd[tuple(cd)]
                            # Connect to all primal vertices
                            for v in [v1, v2, v3]:  # TODO: WHAT ABOUT v4?!?
                                v.vd.add(vd)
                                # vd.nn.add(v)  #TODO: Investigate; I removed this 03.10.24
    return HC  # self


def compute_vd_old(HC, cdist=1e-10):
    """
    Computes the dual vertices of a primal vertex cache HC.V on
    each dim - 1 simplex.

    Currently only dim = 2, 3 is supported

    cdist: float, tolerance for where a unique dual vertex can exist

    """
    # Construct dual cache
    HC.Vd = VertexCacheField()

    # Construct dual neighbour sets
    for v in HC.V:
        v.vd = set()

    # hcv = copy.copy(HC.V)
    if HC.dim == 2:
        for v1 in HC.V:
            for v2 in v1.nn:
                # If boundary vertex, we stop and generate a new vertex on the boundary edge.
                try:
                    if v1.boundary and v2.boundary:
                        cd = v1.x_a + 0.5 * (v2.x_a - v1.x_a)
                        vd = HC.Vd[tuple(cd)]
                        v1.vd.add(vd)
                        v2.vd.add(vd)
                        # vd.nn.add(v1)
                        # vd.nn.add(v2)
                        continue
                except AttributeError:
                    pass
                # Find all v2.nn also connected to v1:
                v1nn_u_v2nn = v1.nn.intersection(v2.nn)  # Should always be length 2
                print(f'len(v1nn_u_v2nn) = {len(v1nn_u_v2nn)}')
                print(f'v1nn_u_v2nn[0] = {list(v1nn_u_v2nn)[0]}')
               # print(f'v1nn_u_v2nn[1] = {list(v1nn_u_v2nn)[1]}')
                # In 2D there are only two
                v3_1 = list(v1nn_u_v2nn)[0]
                v3_2 = list(v1nn_u_v2nn)[1]
                if (v3_1 is v1) or (v3_2 is v1):
                    continue
                verts = np.zeros([3, HC.dim])
                verts[0] = v1.x_a
                verts[1] = v2.x_a
                verts[2] = v3_1.x_a
                # Compute the circumcentre:
                # cd = circumcenter(verts)
                # Compute the barycentre of the first connected triangle sharing primary edge/face e_1e:
                cd1 = np.mean(verts, axis=0)
                verts[2] = v3_2.x_a
                # Compute the barycentre of the first connected triangle sharing primary edge/face e_1e:
                cd2 = np.mean(verts, axis=0)
                # Note instead of below, could round off cd in general to say nearest 1e-12
                # Check for uniqueness first (new, expensive, could
                # be improved by checking duals of neighbours only?):
                for vd_i in HC.Vd:
                    dist1 = np.linalg.norm(vd_i.x_a - cd1)
                    dist2 = np.linalg.norm(vd_i.x_a - cd2)
                    if dist1 < cdist:
                        cd1 = vd_i.x_a
                    if dist2 < cdist:
                        cd2 = vd_i.x_a

                vd1 = HC.Vd[tuple(cd1)]
                vd2 = HC.Vd[tuple(cd2)]
                # Connect the two dual vertices:
                # vd1.connect(vd2)

                # Connect to all primal vertices of v3_1 dual
                for v in [v1, v2, v3_1]:
                    v.vd.add(vd1)

                # Connect to all primal vertices of v3_2 dual
                for v in [v1, v2, v3_2]:
                    v.vd.add(vd2)
                if 0:
                    for v3 in v1nn_u_v2nn:
                        if (v3 is v1):
                            continue
                        # TODO: Re-implement cache:
                        verts = np.zeros([3, HC.dim])
                        verts[0] = v1.x_a
                        verts[1] = v2.x_a
                        verts[2] = v3.x_a

                        # Compute the circumcentre:
                        # cd = circumcenter(verts)
                        # Compute the barycentre:
                        cd = np.mean(verts, axis=0)
                        # Note instead of below, could round off cd in general to say nearest 1e-12
                        # Check for uniqueness first (new, expensive, could
                        # be improved by checking duals of neighbours only?):
                        for vd_i in HC.Vd:
                            dist = np.linalg.norm(vd_i.x_a - cd)
                            if dist < cdist:
                                cd = vd_i.x_a

                        vd = HC.Vd[tuple(cd)]
                        # Connect to all primal vertices
                        for v in [v1, v2, v3]:
                            v.vd.add(vd)
                            # vd.nn.add(v)

    elif HC.dim == 3:
        for v1 in HC.V:
            for v2 in v1.nn:
                try:
                    # Note: every boundary primary edge only has two boundary tetrahedra connected
                    # and therfore only two barycentric dual points. We do not need to connect with
                    # other duals therefore simply connect to the primary edges.
                    if v1.boundary and v2.boundary:
                        # Find all v2.nn also connected to v1:
                        # Find the other two primary edges
                        v1nn_u_v2nn = v1.nn.intersection(v2.nn)
                        for v3 in v1nn_u_v2nn:
                            if (v3 is v1):
                                continue
                            try:
                                if v3.boundary:
                                    # Find the barycentre of the triangle
                                    if 1:
                                        verts = np.zeros([3, HC.dim])
                                        verts[0] = v1.x_a  # TODO: Added 08.03.24, investigate accidental deletion?
                                        verts[1] = v2.x_a
                                        verts[2] = v3.x_a

                                        # Compute the circumcentre:
                                        # cd = circumcenter(verts)
                                        # Compute the barycentre:
                                        cd = np.mean(verts, axis=0)

                                        for vd_i in HC.Vd:
                                            dist = np.linalg.norm(vd_i.x_a - cd)
                                            if dist < cdist:
                                                cd = vd_i.x_a

                                        # Define the new dual vertex:
                                        vd = HC.Vd[tuple(cd)]
                                        # Connect to all primal vertices
                                        for v in [v1, v2, v3]:
                                            v.vd.add(vd)
                                            vd.nn.add(v)

                                    # Add a dual on the primary edge e_12 = v2 - v1
                                    if 1:
                                        def project_vector(vector1, vector2):
                                            """
                                            Project vector1 onto vector2
                                            :param vector1:
                                            :param vector2:
                                            :return:
                                            """
                                            # Calculate the dot product of vector1 and vector2
                                            dot_product = np.dot(vector1, vector2)

                                            # Calculate the square of the magnitude of vector2
                                            mag_squared = np.linalg.norm(vector2) ** 2

                                            # Calculate the projection using the formula given above
                                            projection = (dot_product / mag_squared) * vector2

                                            return projection

                                    if 0:
                                        # Define the original vectors  #
                                        vector1 = vd.x_a - v1.x_a
                                        vector2 = v2.x_a - v1.x_a

                                        # Project vector1 onto vector2
                                        projected_vector = project_vector(vector1, vector2)

                                        # Find the terminal point of the projected vector by adding its components to the origin
                                        vd_12a = projected_vector + v1.x_a
                                        vd_12 = HC.Vd[tuple(vd_12a)]
                                        for v in [v1, v2]:
                                            v.vd.add(vd_12)
                                            vd_12.nn.add(v)

                                    if 0:
                                        vd_12a = (2 / 3.0) * (v2.x_a - v1.x_a) + v1.x_a
                                        vd_12 = HC.Vd[tuple(vd_12a)]
                                        for v in [v1, v2]:
                                            v.vd.add(vd_12)
                                            vd_12.nn.add(v)
                            except AttributeError:
                                pass

                        continue
                except AttributeError:
                    pass

                # Find all v2.nn also connected to v1:
                v1nn_u_v2nn = v1.nn.intersection(v2.nn)
                for v3 in v1nn_u_v2nn:
                    if (v3 is v1):
                        continue
                    v1nn_u_v2nn_u_v3nn = v1nn_u_v2nn.intersection(v3.nn)
                    # for v4 in v1nn_u_v2nn_u_v3nn:
                    for v4 in v1nn_u_v2nn_u_v3nn:
                        if (v4 is v1) or (v4 is v2):
                            continue
                        # TODO: Re-implement cache:
                        verts = np.zeros([HC.dim + 1, HC.dim])
                        verts[0] = v1.x_a  # TODO: Added 08.03.24, investigate accidental deletion?
                        verts[1] = v2.x_a
                        verts[2] = v3.x_a
                        verts[3] = v4.x_a

                        # Compute the circumcentre:
                        # cd = circumcenter(verts)
                        # Compute the barycentre:
                        cd = np.mean(verts, axis=0)
                        # Note instead of below, could round off cd in general to say nearest 1e-12
                        # Check for uniqueness first (new, expensive, could
                        # be improved by checking duals of neighbours only?):
                        for vd_i in HC.Vd:
                            dist = np.linalg.norm(vd_i.x_a - cd)
                            if dist < cdist:
                                cd = vd_i.x_a

                        #  Define the new dual vertex:
                        vd = HC.Vd[tuple(cd)]
                        # Connect to all primal vertices
                        for v in [v1, v2, v3]:  # TODO: WHAT ABOUT v4?!?
                            v.vd.add(vd)
                            # vd.nn.add(v)  #TODO: Investigate; I removed this 03.10.24
    return HC  # self


def compute_vd_old2(HC, cdist =1e-10):
    """
    Computes the dual vertices of a primal vertex cache HC.V on
    each dim - 1 simplex.

    Currently only dim = 2, 3 is supported

    cdist: float, tolerance for where a unique dual vertex can exist

    """
    # Construct dual cache
    HC.Vd = VertexCacheField()

    # Construct dual neighbour sets
    for v in HC.V:
        v.vd = set()

    # hcv = copy.copy(HC.V)
    if HC.dim == 2:
        for v1 in HC.V:
            for v2 in v1.nn:
                # If boundary vertex, we stop and generate a new vertex on the boundary edge.
                try:
                    if v1.boundary and v2.boundary:
                        cd = v1.x_a + 0.5 * (v2.x_a - v1.x_a)
                        vd = HC.Vd[tuple(cd)]
                        v1.vd.add(vd)
                        v2.vd.add(vd)
                        #vd.nn.add(v1)
                        #vd.nn.add(v2)
                        continue
                except AttributeError:
                    pass
                # Find all v2.nn also connected to v1:
                v1nn_u_v2nn = v1.nn.intersection(v2.nn)
                for v3 in v1nn_u_v2nn:
                    if (v3 is v1):
                        continue
                    # TODO: Re-implement cache:
                    verts = np.zeros([3, HC.dim])
                    verts[0] = v1.x_a
                    verts[1] = v2.x_a
                    verts[2] = v3.x_a

                    # Compute the circumcentre:
                    # cd = circumcenter(verts)
                    # Compute the barycentre:
                    cd = np.mean(verts, axis=0)
                    # Note instead of below, could round off cd in general to say nearest 1e-12
                    # Check for uniqueness first (new, expensive, could
                    # be improved by checking duals of neighbours only?):
                    for vd_i in HC.Vd:
                        dist = np.linalg.norm(vd_i.x_a - cd)
                        if dist < cdist:
                            cd = vd_i.x_a

                    vd = HC.Vd[tuple(cd)]
                    # Connect to all primal vertices
                    for v in [v1, v2, v3]:
                        v.vd.add(vd)
                        #vd.nn.add(v)
    elif HC.dim == 3:
        for v1 in HC.V:
            for v2 in v1.nn:
                try:
                    # Note: every boundary primary edge only has two boundary tetrahedra connected
                    # and therfore only two barycentric dual points. We do not need to connect with
                    # other duals therefore simply connect to the primary edges.
                    if v1.boundary and v2.boundary:
                        # Find all v2.nn also connected to v1:
                        # Find the other two primary edges
                        v1nn_u_v2nn = v1.nn.intersection(v2.nn)
                        for v3 in v1nn_u_v2nn:
                            if (v3 is v1):
                                continue
                            try:
                                if v3.boundary:
                                    # Find the barycentre of the triangle
                                    if 1:
                                        verts = np.zeros([3, HC.dim])
                                        verts[0] = v1.x_a  # TODO: Added 08.03.24, investigate accidental deletion?
                                        verts[1] = v2.x_a
                                        verts[2] = v3.x_a

                                        # Compute the circumcentre:
                                        # cd = circumcenter(verts)
                                        # Compute the barycentre:
                                        cd = np.mean(verts, axis=0)

                                        for vd_i in HC.Vd:
                                            dist = np.linalg.norm(vd_i.x_a - cd)
                                            if dist < cdist:
                                                cd = vd_i.x_a

                                        # Define the new dual vertex:
                                        vd = HC.Vd[tuple(cd)]
                                        # Connect to all primal vertices
                                        for v in [v1, v2, v3]:
                                            v.vd.add(vd)
                                            vd.nn.add(v)

                                    # Add a dual on the primary edge e_12 = v2 - v1
                                    if 1:
                                        def project_vector(vector1, vector2):
                                            """
                                            Project vector1 onto vector2
                                            :param vector1:
                                            :param vector2:
                                            :return:
                                            """
                                            # Calculate the dot product of vector1 and vector2
                                            dot_product = np.dot(vector1, vector2)

                                            # Calculate the square of the magnitude of vector2
                                            mag_squared = np.linalg.norm(vector2) ** 2

                                            # Calculate the projection using the formula given above
                                            projection = (dot_product / mag_squared) * vector2

                                            return projection

                                    if 0:
                                        # Define the original vectors  #
                                        vector1 = vd.x_a - v1.x_a
                                        vector2 = v2.x_a - v1.x_a

                                        # Project vector1 onto vector2
                                        projected_vector = project_vector(vector1, vector2)

                                        # Find the terminal point of the projected vector by adding its components to the origin
                                        vd_12a = projected_vector + v1.x_a
                                        vd_12 = HC.Vd[tuple(vd_12a)]
                                        for v in [v1, v2]:
                                            v.vd.add(vd_12)
                                            vd_12.nn.add(v)

                                    if 0:
                                        vd_12a = (2/3.0)*(v2.x_a - v1.x_a) + v1.x_a
                                        vd_12 = HC.Vd[tuple(vd_12a)]
                                        for v in [v1, v2]:
                                            v.vd.add(vd_12)
                                            vd_12.nn.add(v)
                            except AttributeError:
                                pass

                        continue
                except AttributeError:
                    pass

                # Find all v2.nn also connected to v1:
                v1nn_u_v2nn = v1.nn.intersection(v2.nn)
                for v3 in v1nn_u_v2nn:
                    if (v3 is v1):
                        continue
                    v1nn_u_v2nn_u_v3nn = v1nn_u_v2nn.intersection(v3.nn)
                    # for v4 in v1nn_u_v2nn_u_v3nn:
                    for v4 in v1nn_u_v2nn_u_v3nn:
                        if (v4 is v1) or (v4 is v2):
                            continue
                        # TODO: Re-implement cache:
                        verts = np.zeros([HC.dim + 1, HC.dim])
                        verts[0] = v1.x_a  #TODO: Added 08.03.24, investigate accidental deletion?
                        verts[1] = v2.x_a
                        verts[2] = v3.x_a
                        verts[3] = v4.x_a

                        # Compute the circumcentre:
                        # cd = circumcenter(verts)
                        # Compute the barycentre:
                        cd = np.mean(verts, axis=0)
                        # Note instead of below, could round off cd in general to say nearest 1e-12
                        # Check for uniqueness first (new, expensive, could
                        # be improved by checking duals of neighbours only?):
                        for vd_i in HC.Vd:
                            dist = np.linalg.norm(vd_i.x_a - cd)
                            if dist < cdist:
                                cd = vd_i.x_a

                        #  Define the new dual vertex:
                        vd = HC.Vd[tuple(cd)]
                        # Connect to all primal vertices
                        for v in [v1, v2, v3]:  #TODO: WHAT ABOUT v4?!?
                            v.vd.add(vd)
                            #vd.nn.add(v)  #TODO: Investigate; I removed this 03.10.24
    return HC  # self


def compute_vd_old(HC, cdist =1e-10):
    """
    Computes the dual vertices of a primal vertex cache HC.V on
    each dim - 1 simplex.

    Currently only dim = 2, 3 is supported

    cdist: float, tolerance for where a unique dual vertex can exist

    """
    # Construct dual cache
    HC.Vd = VertexCacheField()

    # Construct dual neighbour sets
    for v in HC.V:
        v.vd = set()

    # hcv = copy.copy(HC.V)
    if HC.dim == 2:
        for v1 in HC.V:
            for v2 in v1.nn:
                # If boundary vertex, we stop and generate a new vertex on the boundary edge.
                try:
                    if v1.boundary and v2.boundary:
                        cd = v1.x_a + 0.5 * (v2.x_a - v1.x_a)
                        vd = HC.Vd[tuple(cd)]
                        v1.vd.add(vd)
                        v2.vd.add(vd)
                        #vd.nn.add(v1)
                        #vd.nn.add(v2)
                        continue
                except AttributeError:
                    pass
                # Find all v2.nn also connected to v1:
                v1nn_u_v2nn = v1.nn.intersection(v2.nn)
                for v3 in v1nn_u_v2nn:
                    if (v3 is v1):
                        continue
                    # TODO: Re-implement cache:
                    verts = np.zeros([3, HC.dim])
                    verts[0] = v1.x_a
                    verts[1] = v2.x_a
                    verts[2] = v3.x_a

                    # Compute the circumcentre:
                    # cd = circumcenter(verts)
                    # Compute the barycentre:
                    cd = np.mean(verts, axis=0)
                    # Note instead of below, could round off cd in general to say nearest 1e-12
                    # Check for uniqueness first (new, expensive, could
                    # be improved by checking duals of neighbours only?):
                    for vd_i in HC.Vd:
                        dist = np.linalg.norm(vd_i.x_a - cd)
                        if dist < cdist:
                            cd = vd_i.x_a

                    vd = HC.Vd[tuple(cd)]
                    # Connect to all primal vertices
                    for v in [v1, v2, v3]:
                        v.vd.add(vd)
                        #vd.nn.add(v)
    elif HC.dim == 3:
        for v1 in HC.V:
            for v2 in v1.nn:
                try:
                    # Note: every boundary primary edge only has two boundary tetrahedra connected
                    # and therfore only two barycentric dual points. We do not need to connect with
                    # other duals therefore simply connect to the primary edges.
                    if v1.boundary and v2.boundary:

                        #TODO: Below not needed if using new function `compute_dual_on_boundary_face`
                        cd = v1.x_a + 0.5 * (v2.x_a - v1.x_a)
                        vd = HC.Vd[tuple(cd)]
                        v1.vd.add(vd)
                        v2.vd.add(vd)
                        vd.nn.add(v1)
                        vd.nn.add(v2)
                        #TODO: We need at least two more dual vertices which are on the two boundary
                        #      faces connected. These vertices must in the same plane as vd and the
                        #      the two dual vertices computed for the connected tetrahedra.

                        # Find all v2.nn also connected to v1:
                        # Find the other two primary edges
                        v1nn_u_v2nn = v1.nn.intersection(v2.nn)
                        for v3 in v1nn_u_v2nn:
                            if (v3 is v1):
                                continue
                            try:
                                if v3.boundary:
                                    # Find the barycentre of the triangle (doesn't work well)
                                    if 0:
                                        verts = np.zeros([3, HC.dim])
                                        verts[0] = v1.x_a  # TODO: Added 08.03.24, investigate accidental deletion?
                                        verts[1] = v2.x_a
                                        verts[2] = v3.x_a

                                        # Compute the circumcentre:
                                        # cd = circumcenter(verts)
                                        # Compute the barycentre:
                                        cd = np.mean(verts, axis=0)

                                        for vd_i in HC.Vd:
                                            dist = np.linalg.norm(vd_i.x_a - cd)
                                            if dist < cdist:
                                                cd = vd_i.x_a

                                        # Define the new dual vertex:
                                        vd = HC.Vd[tuple(cd)]
                                        # Connect to all primal vertices
                                        for v in [v1, v2, v3]:
                                            v.vd.add(vd)
                                            vd.nn.add(v)

                                    # Project a fourth dual from the dual triangle to the primary triangle
                                    if 1:
                                        # Find the dual vertex connect to all edges in this boundary triangle
                                        #TODO: Should always be a single vertex, unfortunately sometimes empty
                                        vd12s = v1.vd.intersection(v2.vd)
                                        vd12s = vd12s.intersection(v3.vd)

                                        #try:
                                       #     for vd12 in vd12s:
                                        #        pass
                                            # vd12
                                        #    print(f'vd12s = {vd12s}')
                                        #    print(f'vd12 = {vd12}')
                                       # except UnboundLocalError:
                                        #    pass
                            except AttributeError:
                                pass

                        continue
                except AttributeError:
                    pass

                # Find all v2.nn also connected to v1:
                v1nn_u_v2nn = v1.nn.intersection(v2.nn)
                for v3 in v1nn_u_v2nn:
                    if (v3 is v1):
                        continue
                    v1nn_u_v2nn_u_v3nn = v1nn_u_v2nn.intersection(v3.nn)
                    # for v4 in v1nn_u_v2nn_u_v3nn:
                    for v4 in v1nn_u_v2nn_u_v3nn:
                        if (v4 is v1) or (v4 is v2):
                            continue
                        # TODO: Re-implement cache:
                        verts = np.zeros([HC.dim + 1, HC.dim])
                        verts[0] = v1.x_a  #TODO: Added 08.03.24, investigate accidental deletion?
                        verts[1] = v2.x_a
                        verts[2] = v3.x_a
                        verts[3] = v4.x_a

                        # Compute the circumcentre:
                        # cd = circumcenter(verts)
                        # Compute the barycentre:
                        cd = np.mean(verts, axis=0)
                        # Note instead of below, could round off cd in general to say nearest 1e-12
                        # Check for uniqueness first (new, expensive, could
                        # be improved by checking duals of neighbours only?):
                        for vd_i in HC.Vd:
                            dist = np.linalg.norm(vd_i.x_a - cd)
                            if dist < cdist:
                                cd = vd_i.x_a

                        #  Define the new dual vertex:
                        vd = HC.Vd[tuple(cd)]
                        # Connect to all primal vertices
                        for v in [v1, v2, v3]:  #TODO: WHAT ABOUT v4?!?
                            v.vd.add(vd)
                            #vd.nn.add(v)  #TODO: Investigate; I removed this 03.10.24
    return HC  # self

# Find the Delaunay dual
def triang_dual(points, plot_delaunay=False):
    """
    Compute the Delaunay triangulation plus the dual points. Put into hyperct complex object.
    #TODO: We neeed to comput boundaries before compute_vd
    """
    dim = points.shape[1]
    tri = Delaunay(points)
    if plot_delaunay:  # Plot Delaunay complex
        import matplotlib.pyplot as plt
        plt.triplot(points[: ,0], points[: ,1], tri.simplices)
        plt.plot(points[: ,0], points[: ,1], 'o')
        plt.show()

    # Put Delaunay back into hyperct Complex object:
    HC = Complex(dim)
    for s in tri.simplices:
        for v1i in s:
            for v2i in s:
                if v1i is v2i:
                    continue
                else:
                    v1 = tuple(points[v1i])
                    v2 = tuple(points[v2i])
                    HC.V[v1].connect(HC.V[v2])

    #compute_vd(HC, cdist =1e-10)
    return HC, tri

# Plot duals
def plot_dual_mesh_2D(HC, tri, points):
    """
    Plot the dual mesh and show edge connectivity. Blue is the primary mesh. Orange is the dual mesh.

    Note: Original points need to be fed that was used in the order to build `tri`
    """
    import matplotlib.pyplot as plt

    # Find the dual points
    dual_points = []
    for vd in HC.Vd:
        dual_points.append(vd.x_a)
    dual_points = np.array(dual_points)
    # Primal points
    # points = []
    # for v in HC.V:
    #    points.append(v.x_a)

    # points = np.array(points)

    for v in HC.V:
        # "Connect duals":
        for v2 in v.nn:
            v1vdv2vd = v.vd.intersection(v2.vd)  # Cardinality always 1 or 2?
            if len(v1vdv2vd) == 1:
                continue
            v1vdv2vd = list(v1vdv2vd)
            x = [v1vdv2vd[0].x[0], v1vdv2vd[1].x[0]]
            y = [v1vdv2vd[0].x[1], v1vdv2vd[1].x[1]]
            plt.plot(x, y, color='orange')

        for vd in v.vd:
            x = [v.x[0], vd.x[0]]
            y = [v.x[1], vd.x[1]]
            plt.plot(x, y, '--', color='tab:green')
    plt.triplot(points[: ,0], points[: ,1], tri.simplices, color='tab:blue')
    plt.plot(points[: ,0], points[: ,1],  'o', color='tab:blue')
    plt.plot(dual_points[: ,0], dual_points[: ,1], 'o', color='tab:orange')
    plt.show()

def plot_dual_mesh_3D(HC, dual_points):
    """
    NOTE: The bug remains where it can only be plotted once, therefore rebuild the complex.
    """
    hcfig, hcaxes, _, _ = HC.plot_complex()
    hcaxes.scatter3D(dual_points[:, 0], dual_points[:, 1], dual_points[:, 2], color = "green")
    for v in HC.V:
        for vd in v.vd:
            x = [v.x[0], vd.x[0]]
            y = [v.x[1], vd.x[1]]
            z = [v.x[2], vd.x[2]]
            hcaxes.plot(x, y, zs=z, linestyle='--',  color='tab:green')
        # ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i],VecEnd_y[i]],zs=[VecStart_z[i],VecEnd_z[i]])

    plt.show()
    return


# Geometry and dual computations
def area_of_polygon(points):
    """Calculates the area of a polygon in 3D space.

    Args:
      points: A numpy array of shape (n, 3), where each row represents a point in
        3D space.

    Returns:
      The area of the polygon.
    """

    # Calculate the cross product of each pair of adjacent edges.
    edges = points[1:] - points[:-1]
    cross_products = np.cross(edges[:-1], edges[1:])

    # Calculate the area of the triangle formed by each pair of adjacent edges and
    # the origin.
    triangle_areas = 0.5 * np.linalg.norm(cross_products, axis=1)

    # Sum the areas of all the triangles to get the total area of the polygon.
    return np.sum(triangle_areas)

def volume_of_geometric_object(points, extra_point):
    """Calculates the volume of a geometric object defined by adding an extra
    point away from the plane and connecting all points in the plane to it.

    Args:
    points: A numpy array of shape (n, 3), where each row represents a point in
      3D space.
    extra_point: A numpy array of shape (3,), representing the extra point away
      from the plane.

    Returns:
    The volume of the geometric object.
    """

    # Calculate the normal vector to the plane that contains the base polygon.
    normal_vector = np.cross(points[1] - points[0], points[2] - points[0])

    # Calculate the projection of the extra point onto the plane.
    projected_extra_point = extra_point - np.dot(extra_point - points[0], normal_vector) / np.linalg.norm(normal_vector)**2 * normal_vector

    # Calculate the distance between the extra point and its projection onto the plane.
    distance = np.linalg.norm(extra_point - projected_extra_point)

    # Calculate the area of the base polygon.
    base_area = area_of_polygon(points)

    # Calculate the volume of the geometric object.
    volume = 1/3 * base_area * distance

    return volume


def v_star(v_i, v_j, HC, n=None, dim=2):
    """
    Compute the dual flux planes and volume of the primary edge e_ij.
    It's needed to specify the dimension dim.

    n is a directional vector

    return : e_ij_star
    """
    if dim == 2:
        # e_ij_star = 0  # Initialize total dual area to zero
        # Find the shared dual vertices between vp1 and vp2
        vdnn = v_i.vd.intersection(v_j.vd)  # Should always be 2 for dim=2

        vd1 = list(vdnn)[0]
        vd2 = list(vdnn)[1]
        e_ij_star = np.linalg.norm(vd1.x_a - vd2.x_a)

    elif dim == 3:
        # Set irectional vector if None:
        if n is None:
            n = np.array([0, 0, 0])
        e_ij_star = 0
        # Find the dual vertex of e12:
        vc_12 = 0.5 * (v_j.x_a - v_i.x_a) + v_i.x_a  # TODO: Should be done in the compute_vd function
        vc_12 = HC.Vd[tuple(vc_12)]
        # Find local dual points intersecting vertices terminating edge:
        dset = v_j.vd.intersection(v_i.vd)  # Always 5 for boundaries
        # Start with the first vertex and then build triangles, loop back to it:
        vd_i = list(dset)[0]
        if v_i.boundary and v_j.boundary:
            # Remove the boundary edge which should already be in the set:
            if not (len(vd_i.nn.intersection(dset)) == 1):
                for vd in dset:
                    vd_i = vd
                    if len(vd_i.nn.intersection(dset)) == 1:
                        break
            iter_len = 3
        else:
            iter_len = len(list(dset))

        # Main loop
        dsetnn = vd_i.nn.intersection(dset)  # Always 1 internal dual vertices
        vd_j = list(dsetnn)[0]
        # NOTE: In the boundary edges the last triangle does not have
        #      a final vd_j
        ssets = []  # Sets of simplices
        A_ij = []  # list of triangle vector areas
        V_ij = []  # list of signed tetrahedral vector volumes
        for _ in range(iter_len):  # For boundaries should be length 2?
            # Compute the discrete vector area of the local triangle
            wedge_dij_ik = np.cross(vc_12.x_a - vd_i.x_a, vd_j.x_a - vd_i.x_a)
            if np.dot(normalized(wedge_dij_ik), n) < 0:  #TODO: Should just reverse sign
                wedge_dij_ik = np.cross(vd_j.x_a - vd_i.x_a, vc_12.x_a - vd_i.x_a)

            A_ij.append(wedge_dij_ik/2.0)

            # Compute the local signed volume
            verts = np.zeros([3, 3])
            verts[0] = vc_12.x_a
            verts[1] = vd_i.x_a
            verts[2] = vd_j.x_a
            #v_dij_i = volume_of_geometric_object(points, extra_point)
            v_dij_i = volume_of_geometric_object(verts, v_i.x_a)
            V_ij.append(v_dij_i)

            # Add to the set of simplces (undeeded here?)
            ssets.append([vc_12, vd_i, vd_j])
            dsetnn_k = vd_j.nn.intersection(dset)  # Always 2 internal dual vertices in interior
            dsetnn_k.remove(vd_i)  # Should now be size 1
            vd_i = vd_j
            try:
                # Alternatively it should produce an IndexError only when
                # _ = 2 (end of range(3) and we are on a boundary edge
                # so that (v_i.boundary and v_j.boundary) is true
                vd_j = list(dsetnn_k)[0]  # Retrieve the next vertex
            except IndexError:
                pass  # Should only happen for boundary edges
        A_ij = np.array(A_ij)
        #A_ij = np.sum(A_ij, axis=0)
        V_ij = np.array(V_ij)
    else:
        print("WARNING: Not implemented yet from dim > 3")

    #return e_ij_star
    return A_ij, V_ij


def e_star(v_i, v_j, HC, n=None, dim=2):
    """
    Compute the dual of the primary edge e_ij. It's needed to specify the dimension dim.

    n is a directional vector

    return : e_ij_star
    """
    if dim == 2:
        # e_ij_star = 0  # Initialize total dual area to zero
        # Find the shared dual vertices between vp1 and vp2
        vdnn = v_i.vd.intersection(v_j.vd)  # Should always be 2 for dim=2

        vd1 = list(vdnn)[0]
        vd2 = list(vdnn)[1]
        e_ij_star = np.linalg.norm(vd1.x_a - vd2.x_a)

    elif dim == 3:
        # Set irectional vector if None:
        if n is None:
            n = np.array([0, 0, 0])
        e_ij_star = 0
        # Find the dual vertex of e12:
        vc_12 = 0.5 * (v_j.x_a - v_i.x_a) + v_i.x_a  # TODO: Should be done in the compute_vd function
        vc_12 = HC.Vd[tuple(vc_12)]
        # Find local dual points intersecting vertices terminating edge:
        dset = v_j.vd.intersection(v_i.vd)  # Always 5 for boundaries
        # Start with the first vertex and then build triangles, loop back to it:
        vd_i = list(dset)[0]
        if v_i.boundary and v_j.boundary:
            # Remove the boundary edge which should already be in the set:
            if not (len(vd_i.nn.intersection(dset)) == 1):
                for vd in dset:
                    vd_i = vd
                    if len(vd_i.nn.intersection(dset)) == 1:
                        break
            iter_len = 3
        else:
            iter_len = len(list(dset))

        # Main loop
        dsetnn = vd_i.nn.intersection(dset)  # Always 1 internal dual vertices
        vd_j = list(dsetnn)[0]
        # NOTE: In the boundary edges the last triangle does not have
        #      a final vd_j
        ssets = []  # Sets of simplices
        A_ij = []  # list of triangle vector areas
        for _ in range(iter_len):  # For boundaries should be length 2?
            # Compute the discrete vector area of the local triangle
            wedge_ij_ik = np.cross(vc_12.x_a - vd_i.x_a, vd_j.x_a - vd_i.x_a)
            if np.dot(normalized(wedge_ij_ik), n) < 0:  #TODO: Should just reverse sign
                wedge_ij_ik = np.cross(vd_j.x_a - vd_i.x_a, vc_12.x_a - vd_i.x_a)

            A_ij.append(wedge_ij_ik/2.0)
            # Add to the set of simplces (undeeded here?)
            ssets.append([vc_12, vd_i, vd_j])
            dsetnn_k = vd_j.nn.intersection(dset)  # Always 2 internal dual vertices in interior
            dsetnn_k.remove(vd_i)  # Should now be size 1
            vd_i = vd_j

            try:
                # Alternatively it should produce an IndexError only when
                # _ = 2 (end of range(3) and we are on a boundary edge
                # so that (v_i.boundary and v_j.boundary) is true
                vd_j = list(dsetnn_k)[0]  # Retrieve the next vertex
            except IndexError:
                pass  # Should only happen for boundary edges

    else:
        print("WARNING: Not implemented yet from dim > 3")

    return e_ij_star


def e_star_old(v_i, v_j, dim=2):
    """
    Compute the dual of the primary edge e_ij. It's needed to specify the dimension dim.

    return : e_ij_star
    """
    if dim == 2:
        # e_ij_star = 0  # Initialize total dual area to zero
        # Find the shared dual vertices between vp1 and vp2
        vdnn = v_i.vd.intersection(v_j.vd)  # Should always be 2 for dim=2

        vd1 = list(vdnn)[0]
        vd2 = list(vdnn)[1]
        e_ij_star = np.linalg.norm(vd1.x_a - vd2.x_a)

    elif dim == 3:
        # e_ij_star = 0  # Initialize total dual area to zero
        vdnn = v_i.vd.intersection(v_j.vd)
        local_dual_points = []
        for vd in vdnn:
            local_dual_points.append(vd.x)

        # Added 02.20.2024
        local_dual_points.append(vd.x)
        print(f'local_dual_points = {local_dual_points}')
        local_dual_points = np.array(local_dual_points)
        e_ij_star = area_of_polygon(local_dual_points)
    else:
        print("WARNING: Not implemented yet from dim > 3")

    return e_ij_star


# Area computations
def d_area(vp1):
    """
    Compute the dual area of a vertex object vp1, which is the sum of the areas
    of the local dual triangles formed between vp1, its neighbouring vertices,
    and their shared dual vertices.

    Parameters:
    -----------
    vp1 : object
        A vertex object containing the following attributes:
        - vp1.nn: a list of neighboring vertex objects
        - vp1.vd: a set of dual vertex objects
        - vp1.x_a: a numpy array representing the position of vp1

    Returns:
    --------
    darea : float
        The total dual area of the vertex object vp1
    """

    darea = 0  # Initialize total dual area to zero
    for vp2 in vp1.nn:  # Iterate over neighboring vertex objects
        # Find the shared dual vertices between vp1 and vp2
        vdnn = vp1.vd.intersection(vp2.vd)
        # Compute the midpoint between vp1 and vp2
        mp = (vp1.x_a + vp2.x_a) / 2
        # Compute the height of the dual triangle between vp1, vp2, and a dual vertex
        h = np.linalg.norm(mp - vp1.x_a)
        for vdi in vdnn:  # Iterate over shared dual vertices
            # Compute the base of the dual triangle between vp1, vp2, and vdi
            b = np.linalg.norm(vdi.x_a - mp)
            # Add the area of the dual triangle to the total dual area
            darea += 0.5 * b * h

    return darea

# Volume computations (Note: in DDG the Hodge dual of scalar points in 3D)


## Boundary computations
import numpy as np

def _reflect_vertex_over_edge(triangle, target_index=0):
    """
    Reflect a given vertex of a triangle  (passed as array) over
    the opposing edge of the target vertex, maintaining the same plane.

    :param triangle: np.ndarray, input triangle
    :param target_index: int, target index
    :return: np.ndarray: Updated triangle with the reflected vertex.
    """
    p_o = triangle[target_index]
    p_1 = triangle[(target_index + 1) % 3]
    p_2 = triangle[(target_index + 2) % 3]
    p_midpoint = (p_1 + p_2) / 2
    # Move along the direction of (p_midpoint-p0) for twice the distance
    p_ref = p_o + 2 * (p_midpoint - p_o)
    triangle[target_index] = p_ref
    return triangle


def _find_intersection(plane1, plane2, plane3):
    """
    Find the intersection of 3 planes, the planes,
    the arguments are supplied as coeeficients of the planes
    e.g. for the first plane:

    a_1 x_1 + b_1 x_2 + c_1 x_3 + d_1 = 0

    :param plane1: np.ndarray, vector of 4 elements
    :param plane2: np.ndarray, vector of 4 elements
    :param plane3: np.ndarray, vector of 4 elements
    :return: interesection_point, np.ndarray, vector 3 elements
    """
    # Extract coefficients from each plane equation
    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2
    a3, b3, c3, d3 = plane3

    # Coefficients matrix (A)
    A = np.array([[a1, b1, c1],
                  [a2, b2, c2],
                  [a3, b3, c3]])

    # Check if the matrix is singular
    if np.linalg.det(A) == 0:
        raise ValueError("The planes are parallel or nearly parallel. No unique solution.")

    # Right-hand side vector (b)
    b = np.array([-d1, -d2, -d3])

    # Solve the system of linear equations
    intersection_point = np.linalg.solve(A, b)

    return intersection_point


def _find_plane_equation(v_1, v_2, v_3):
    """
    Find the plane equation of a given plane spanned by vectors e_21 = v_2 - v_1
    and e_31 = v_3 - v_1
    """
    # Compute vectors lying in the plane
    vector1 = np.array(v_2) - np.array(v_1)
    vector2 = np.array(v_3) - np.array(v_1)

    # Compute the normal vector using the cross product
    normal_vector = np.cross(vector1, vector2)

    # Extract coefficients for the plane equation ax + by + cz + d = 0
    a, b, c = normal_vector
    d = -np.dot(normal_vector, np.array(v_1))

    # Return the coefficients [a, b, c, d]
    return [a, b, c, d]

# DDG gradient operations on primary edges (for continuum)
def dP_old(vp1):
    """
    Compute the integrated pressure differential for vertex v
    """
    dP_i = 0  # Total integrated Laplacian for each vertex
    for vp2 in vp1.nn:  # Iterate over neighboring vertex objects
        #  l_ij = np.linalg.norm(vp2.x_a - vp1.x_a)  # Not used here?
        e_dual = 0  # Initialize total edge dual length to zero
        # Find the shared dual vertices between vp1 and vp2
        vdnn = vp1.vd.intersection(vp2.vd)
        # Compute the midpoint between vp1 and vp2
        mp = (vp1.x_a + vp2.x_a) / 2
        # Compute the height of the dual triangle between vp1, vp2, and a dual vertex
        # Compute the height of the dual triangle between vp1, vp2, and a dual vertex
        # h = np.linalg.norm(mp - vp1.x_a)
        # print(f'vdnn = {vdnn})
        for vdi in vdnn:  # Iterate over shared dual vertices
            # Compute the base of the dual triangle between vp1, vp2, and vdi
            b = np.linalg.norm(vdi.x_a - mp)  # dihedral weight
            # Add the area of the dual triangle to the total dual area
            e_dual += b
            if e_dual <= 1e-12:  # TODO: Set tolerance
                e_dual = 0

            # w_ij = l_ij/e_dual  # Weight
            # if (w_ij is np.inf) or (e_dual == 0):

            continue
        # Compute the area flux for the pressure differential:
        Area = e_dual * 1  # m2, Chosen height was 1 for our 2D test case
        # Compute the dual
        dP_ij = Area * (vp2.P - vp1.P)
        dP_i += dP_ij

    return dP_i


def dP(vp1, dim=3, z=1):
    """
    Compute the integrated pressure differential for vertex vp1

    dim=3
    z=1

    Note the routine is the same for 2D and 3D
    """
    dP_i = 0  # Total integrated Laplacian for each vertex
    for vp2 in vp1.nn:  # Iterate over neighboring vertex objects
        # Compute the dual length of e_ij
        e_dual = e_star(vp1, vp2, dim=dim)

        # Compute the area flux for the pressure differential:
        Area = e_dual * 1  # m2, Chosen height was 1 for our 2D test case
        # Compute the dual
        dP_ij = Area * (vp2.P - vp1.P)
        print('-')
        print(f'vp2.x, vp1.x = {vp2.x, vp1.x}')
        print(f'vp2.P - vp1.P = {vp2.P - vp1.P}')
        print(f'Area= {Area}')
        dP_i += dP_ij
    print(f'dP_i = {dP_i}')
    return dP_i


def du_old(vp1):
    """
    Compute the Laplacian of the velocity field for vertex v

    TODO: Compare this with the
    """
    du_i = 0  # Total integrated Laplacian for each vertex
    for vp2 in vp1.nn:  # Iterate over neighboring vertex objects
        l_ij = np.linalg.norm(vp2.x_a - vp1.x_a)
        e_dual = 0  # Initialize total edge dual length to zero
        # Find the shared dual vertices between vp1 and vp2
        vdnn = vp1.vd.intersection(vp2.vd)
        # Compute the midpoint between vp1 and vp2
        mp = (vp1.x_a + vp2.x_a) / 2
        # Compute the height of the dual triangle between vp1, vp2, and a dual vertex
        # h = np.linalg.norm(mp - vp1.x_a)
        # print(f'vdnn = {vdnn})
        for vdi in vdnn:  # Iterate over shared dual vertices
            # Compute the base of the dual triangle between vp1, vp2, and vdi
            b = np.linalg.norm(vdi.x_a - mp)  # dihedral weight
            # Add the area of the dual triangle to the total dual area
            e_dual += b
            if e_dual <= 1e-12:  # TODO: Set tolerance
                e_dual = 0

        w_ij = l_ij / e_dual  # Weight
        # print('.')
        # print(f'l_ij = {l_ij}')
        # print(f'e_dual, len(vdnn)  = {e_dual, len(vdnn)}')
        # print(f'w_ij = { w_ij}')
        # print(f'w_ij is np.inf = {w_ij is np.inf}')
        # print(f'w_ij == np.inf = {w_ij == np.inf}')
        if (w_ij is np.inf) or (e_dual == 0):
            continue
        # Compute the dual
        du_ij = w_ij * (vp2.u - vp1.u)
        du_i += du_ij

    return du_i


def du(vp1, dim=3):
    """
    Compute the Laplacian of the velocity field for vertex v

    """
    du_i = 0  # Total integrated Laplacian for each vertex
    for vp2 in vp1.nn:  # Iterate over neighboring vertex objects
        l_ij = np.linalg.norm(vp2.x_a - vp1.x_a)
        e_dual = e_star(vp1, vp2, dim=dim)

        w_ij = l_ij / e_dual  # Weight
        if (w_ij is np.inf) or (e_dual == 0):
            continue
        # Compute the dual
        du_ij = np.abs(w_ij) * (vp2.u - vp1.u)
        du_i += du_ij
    print(f'du_i = {du_i}')
    return du_i


def dudt_old(v, mu=8.90 * 1e-4):
    # Equal to the acceleration at a vertex (RHS of equation)
    dudt = -dP(v) + mu * du(v)
    dudt = dudt / v.m  # normalize by mass
    return dudt

def dudt(v, dim=3, mu=8.90 * 1e-4):
    # Equal to the acceleration at a vertex (RHS of equation)
    dudt = -dP(v, dim=dim) + mu * du(v, dim=dim)
    dudt = dudt/v.m  # normalize by mass
    return dudt