"""
Diameter-50mm split sphere with embedded magnet pockets.
Two identical hemispheres (print the same STL twice), mated by 4 magnets each.

Pocket layout on the flat face:
  - 3 pockets at r=16mm, angles 0/120/240 deg
  - 1 pocket at r=9mm, angle 180 deg
The pattern is mirror-symmetric (required for identical halves). Flat faces
close at any angle; at 120/240 deg 3 of the 4 magnet pairs still engage.

embed variant: pockets are fully internal, 0.8mm of material between the flat
face and the magnet. Insert magnets via a mid-print pause just before the
pocket ceiling prints -> flat face stays perfectly smooth.
openface variant: pockets are recesses on the flat face, magnets inserted
after printing (stronger hold, visible magnet circles).

Requires: pip install trimesh manifold3d numpy
"""

import numpy as np
import trimesh
from manifold3d import Manifold

R = 25.0            # sphere radius (mm)
FLOOR = 0.8         # embed variant: material between flat face and pocket
CHAMFER = 0.25      # 45 deg chamfer on the flat-face rim (avoids lip at seam)
SEG = 512           # sphere circular segments (surface smoothness)
POCKETS = [(16.0, 0.0), (16.0, 120.0), (16.0, 240.0), (9.0, 180.0)]

EMBED_DCLEAR = 0.5   # pocket diameter clearance for mid-print drop-in
EMBED_TCLEAR = 0.2   # pocket depth headroom
OPEN_DCLEAR = 0.3    # tighter fit for post-print insertion
OPEN_TCLEAR = 0.15

SIZES = {
    "m6x3": (6.0, 3.0),   # most common neodymium size
    "m5x2": (5.0, 2.0),
    "m8x3": (8.0, 3.0),
    "m10x3": (10.0, 3.0),  # same magnets as the dia-200 kit
}


def hemi():
    """Upper half of a sphere, flat face on the z=0 plane."""
    s = Manifold.sphere(R, SEG)
    top = Manifold.cube([2 * R + 8, 2 * R + 8, R + 4]).translate(
        [-(R + 4), -(R + 4), 0])
    return s ^ top


def chamfer(m):
    """45 deg x CHAMFER chamfer on the rim of the flat face."""
    keep_cone = Manifold.cylinder(CHAMFER, R - CHAMFER, R, 256)
    keep_above = Manifold.cube([4 * R, 4 * R, R + 4]).translate(
        [-2 * R, -2 * R, CHAMFER])
    return m ^ (keep_cone + keep_above)


def pockets(md, mt, embed):
    solids = []
    for r, deg in POCKETS:
        x, y = r * np.cos(np.radians(deg)), r * np.sin(np.radians(deg))
        if embed:
            d = (md + EMBED_DCLEAR) / 2
            z0, z1 = FLOOR, FLOOR + mt + EMBED_TCLEAR
        else:
            d = (md + OPEN_DCLEAR) / 2
            z0, z1 = -0.5, mt + OPEN_TCLEAR
        solids.append(
            Manifold.cylinder(z1 - z0, d, d, 96).translate([x, y, z0]))
    return solids, (z1 if embed else z1 + 0.5)


def to_trimesh(mani):
    mesh = mani.to_mesh()
    v = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    f = np.asarray(mesh.tri_verts, dtype=np.int64)
    return trimesh.Trimesh(v, f)


def report(tm, name):
    b = tm.bounds
    watertight = tm.is_watertight
    print(f"{name}: faces={len(tm.faces)} watertight={watertight} "
          f"vol={tm.volume:.0f}mm3 "
          f"X[{b[0][0]:.2f},{b[1][0]:.2f}] Y[{b[0][1]:.2f},{b[1][1]:.2f}] "
          f"Z[{b[0][2]:.2f},{b[1][2]:.2f}]")


def build(md, mt, embed):
    h = hemi()
    for p in pockets(md, mt, embed)[0]:
        h = h - p
    h = chamfer(h)
    return to_trimesh(h)


if __name__ == "__main__":
    import os, sys
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "sphere50")
    os.makedirs(out, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "embed"):
        for tag, (md, mt) in SIZES.items():
            tm = build(md, mt, embed=True)
            fn = f"{out}/hemisphere50_{tag}_embed.stl"
            tm.export(fn)
            ceil_z = FLOOR + mt + EMBED_TCLEAR
            report(tm, fn.split("/")[-1])
            print(f"   pocket ceiling Z={ceil_z:.2f}mm -> pause before the "
                  f"first layer where the holes are gone (slicer preview)")

    if which in ("all", "open"):
        md, mt = SIZES["m6x3"]
        tm = build(md, mt, embed=False)
        fn = f"{out}/hemisphere50_m6x3_openface.stl"
        tm.export(fn)
        report(tm, fn.split("/")[-1])
