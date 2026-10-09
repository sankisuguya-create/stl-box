"""
Sphere cut into 5 parallel slabs, joined by embedded magnets.

  python gen_slab_sphere.py 150   # Ø150, 30mm slabs -> ../sphere150/
  python gen_slab_sphere.py 200   # Ø200, 40mm slabs -> ../sphere200/

Cut planes at z = ±T/2, ±3T/2 (sphere centered at origin). THREE distinct
printable shapes (5 pieces total):
  - cap       : slabs 1 & 5 (flat face on bed, dome up)   x2
  - edge_slab : slabs 2 & 4 (smaller face on bed)         x2
  - mid_slab  : slab 3 (barrel)                           x1

Magnet pockets (identical pattern on every mating face):
  - 3 pockets on an outer ring at 0/120/240 deg + 1 at 180 deg on an inner ring
  - mirror-symmetric set -> mating faces align
  - mid/edge slabs carry pockets on BOTH faces -> two pauses per print

Requires: pip install trimesh manifold3d numpy
"""

import os
import sys

import numpy as np
import trimesh
from manifold3d import Manifold

# size -> (radius, slab thickness, pocket layout [(r, deg), ...])
CONFIGS = {
    150: (75.0, 30.0, [(45.0, 0.0), (45.0, 120.0), (45.0, 240.0), (25.0, 180.0)]),
    200: (100.0, 40.0, [(60.0, 0.0), (60.0, 120.0), (60.0, 240.0), (30.0, 180.0)]),
}

FLOOR = 0.8
CHAMFER = 0.3
SEG = 720

MD, MT = 10.0, 3.0           # magnet dia x thickness
EMBED_DCLEAR, EMBED_TCLEAR = 0.5, 0.2
OPEN_DCLEAR, OPEN_TCLEAR = 0.3, 0.15


def slab(R, z0, z1):
    """Slab of the sphere between z0 and z1, re-based so its bottom face
    sits at z=0. Caps are produced in print orientation (flat face down)."""
    box = Manifold.cube([2 * R + 8, 2 * R + 8, z1 - z0]).translate(
        [-(R + 4), -(R + 4), z0])
    m = Manifold.sphere(R, SEG) ^ box
    return m.translate([0, 0, -z0])


def chamfer(m, R):
    keep_cone = Manifold.cylinder(CHAMFER, R - CHAMFER, R, 256)
    keep_above = Manifold.cube([4 * R, 4 * R, R]).translate(
        [-2 * R, -2 * R, CHAMFER])
    return m ^ (keep_cone + keep_above)


def add_pockets(m, T, pockets, faces, embed):
    """faces: iterable of 'bed' and/or 'top' (top = z=T face)."""
    for f in faces:
        if embed:
            d = (MD + EMBED_DCLEAR) / 2
            z0, z1 = (FLOOR, FLOOR + MT + EMBED_TCLEAR) if f == "bed" \
                else (T - MT - EMBED_TCLEAR - FLOOR, T - FLOOR)
        else:
            d = (MD + OPEN_DCLEAR) / 2
            z0, z1 = (-0.5, MT + OPEN_TCLEAR) if f == "bed" \
                else (T - MT - OPEN_TCLEAR, T + 0.5)
        for r, deg in pockets:
            x, y = r * np.cos(np.radians(deg)), r * np.sin(np.radians(deg))
            p = Manifold.cylinder(z1 - z0, d, d, 96).translate([x, y, z0])
            m = m - p
    return m


def to_trimesh(mani):
    mesh = mani.to_mesh()
    return trimesh.Trimesh(
        np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3],
        np.asarray(mesh.tri_verts, dtype=np.int64))


def report(tm, name):
    b = tm.bounds
    print(f"{name}: faces={len(tm.faces)} watertight={tm.is_watertight} "
          f"vol={tm.volume/1000:.0f}cm3 "
          f"X[{b[0][0]:.1f},{b[1][0]:.1f}] Y[{b[0][1]:.1f},{b[1][1]:.1f}] "
          f"Z[{b[0][2]:.1f},{b[1][2]:.1f}]")


def build(size, kind, embed):
    R, T, pockets = CONFIGS[size]
    if kind == "cap":            # slabs 1,5
        m = slab(R, 1.5 * T, R)
        faces = ["bed"]
    elif kind == "edge_slab":    # slabs 2,4
        m = slab(R, -1.5 * T, -0.5 * T)
        faces = ["bed", "top"]
    elif kind == "mid_slab":     # slab 3
        m = slab(R, -0.5 * T, 0.5 * T)
        faces = ["bed", "top"]
    m = add_pockets(m, T, pockets, faces, embed)
    return to_trimesh(chamfer(m, R))


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", f"sphere{size}")
    os.makedirs(out, exist_ok=True)
    T = CONFIGS[size][1]
    for kind in ("cap", "edge_slab", "mid_slab"):
        for variant, embed in (("embed", True), ("openface", False)):
            tm = build(size, kind, embed)
            fn = os.path.join(out, f"sphere{size}_{kind}_m10x3_{variant}.stl")
            tm.export(fn)
            report(tm, os.path.basename(fn))
    print(f"\npause ceilings: bed pockets Z={FLOOR + MT + EMBED_TCLEAR:.1f}mm"
          f" | top pockets Z={T - FLOOR:.1f}mm"
          " -> pause before the first layer where the holes are gone"
          " (check in slicer preview)")
