"""
Ø200mm sphere cut into 5 parallel 40mm slabs, joined by embedded magnets.

Cut planes at z = -60, -20, +20, +60 (sphere centered at origin, R=100).
This yields only THREE distinct printable shapes:
  - cap        : slabs 1 & 5 (printed flat-face down; dome up)   x2
  - edge_slab  : slabs 2 & 4 (Ø160 face on bed, Ø196 top)        x2
  - mid_slab   : slab 3 (Ø196 disc, slight barrel edge)          x1

Magnet pockets (identical pattern on every mating face):
  - 3 pockets on r=60mm ring at 0/120/240 deg + 1 key at r=30mm, 180 deg
  - mirror-symmetric set -> matching faces always align; key breaks
    rotational symmetry -> each interface mates in exactly one orientation
  - for 10x3mm neodymium: pocket dia 10.5, depth 3.2, floor 0.8
  - mid slabs carry pockets on BOTH faces -> two pauses per print
"""

import numpy as np
import trimesh
from manifold3d import Manifold

R = 100.0
T = 40.0           # slab thickness
FLOOR = 0.8
CHAMFER = 0.3
SEG = 720          # sphere resolution
POCKETS = [(60.0, 0.0), (60.0, 120.0), (60.0, 240.0), (30.0, 180.0)]

MD, MT = 10.0, 3.0           # magnet dia x thickness
EMBED_DCLEAR, EMBED_TCLEAR = 0.5, 0.2
OPEN_DCLEAR, OPEN_TCLEAR = 0.3, 0.15

CUTS = [-100, -60, -20, 20, 60, 100]


def slab(z0, z1):
    """Slab of the sphere between z0 and z1, re-based so its bottom face
    sits at z=0. Caps are produced in print orientation (flat face down)."""
    box = Manifold.cube([2 * R + 8, 2 * R + 8, z1 - z0]).translate(
        [-(R + 4), -(R + 4), z0])
    m = Manifold.sphere(R, SEG) ^ box
    return m.translate([0, 0, -z0])


def chamfer(m):
    keep_cone = Manifold.cylinder(CHAMFER, R - CHAMFER, R, 256)
    keep_above = Manifold.cube([4 * R, 4 * R, R]).translate(
        [-2 * R, -2 * R, CHAMFER])
    return m ^ (keep_cone + keep_above)


def add_pockets(m, faces, embed):
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
        for r, deg in POCKETS:
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


def build(kind, embed):
    if kind == "cap":          # slabs 1,5: cut face r=80 -> Ø160 on bed
        m = slab(60, 100)
        m = add_pockets(m, ["bed"], embed)
    elif kind == "edge_slab":  # slabs 2,4: bed face Ø160, top Ø196
        m = slab(-60, -20)
        m = add_pockets(m, ["bed", "top"], embed)
    elif kind == "mid_slab":   # slab 3: Ø196 barrel, faces both Ø196
        m = slab(-20, 20)
        m = add_pockets(m, ["bed", "top"], embed)
    return to_trimesh(chamfer(m))


if __name__ == "__main__":
    import os
    out = "/home/ubuntu/sphere_magnet/out200"
    os.makedirs(out, exist_ok=True)
    for kind in ("cap", "edge_slab", "mid_slab"):
        for variant, embed in (("embed", True), ("openface", False)):
            tm = build(kind, embed)
            fn = f"{out}/sphere200_{kind}_m10x3_{variant}.stl"
            tm.export(fn)
            report(tm, fn.split("/")[-1])
    print("\npauses: bed pockets ceiling Z=4.0mm | top pockets ceiling Z=39.2mm")
