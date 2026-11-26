import numpy as np


def _load_vertices_obj(path):
    """Return (N,3) float array of vertices from an OBJ file."""
    verts, _ = _load_mesh(path)
    return verts


def _load_mesh(path):
    """Return (vertices, faces) from an OBJ file.

    vertices: (N,3) float64 array.
    faces: list of tuples of 0-based vertex indices (triangles or polygons).
    Face tokens may be 'v', 'v/vt', 'v//vn', or 'v/vt/vn'; only vertex index is used.
    """
    verts = []
    faces = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idxs = tuple(int(p.split("/")[0]) - 1 for p in parts)
                faces.append(idxs)
    return np.array(verts, dtype=np.float64), faces


def _faces_to_edges(faces):
    """Return deduplicated undirected edges as (E,2) int32 array from face list."""
    edge_set = set()
    for face in faces:
        n = len(face)
        for k in range(n):
            a, b = face[k], face[(k + 1) % n]
            edge_set.add((min(a, b), max(a, b)))
    if not edge_set:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(sorted(edge_set), dtype=np.int32)


def _subsample_edges(edges, max_edges=3000):
    """Return at most max_edges rows from edges array by uniform stride."""
    stride = max(1, len(edges) // max_edges)
    return edges[::stride]


def find_extremes(mesh_path):
    """Load mesh and print the vertices at min/max X, Y, Z."""
    verts = _load_vertices_obj(mesh_path)
    if len(verts) == 0:
        print("No vertices found.")
        return

    axes = ["X", "Y", "Z"]
    for i, ax in enumerate(axes):
        imin = np.argmin(verts[:, i])
        imax = np.argmax(verts[:, i])
        print(f"Min {ax}: {verts[imin]}  (vertex #{imin})")
        print(f"Max {ax}: {verts[imax]}  (vertex #{imax})")


def main():
    find_extremes("/home/borisef/data/3D_models/orlan/orlan.obj")


if __name__ == "__main__":
    main()
