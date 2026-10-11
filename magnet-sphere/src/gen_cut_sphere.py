"""
Ø150mm sphere cut at arbitrary heights, joined by embedded magnets.

  python gen_cut_sphere.py            # -> ../sphere150_cut5/

Cuts are measured from the TOP of the sphere (mm): 32.5, 75, 100, 125.
Pieces are numbered from the top (P1 = top cap ... P5 = bottom cap).
Every piece is printed with its LARGER cut face on the bed, so no piece has
an overhang (the sphere only narrows upward from the bed).

Pockets: 4 per mating face, 90 deg apart on one ring (point-symmetric).
The ring radius is chosen per cut so the pocket keeps >= WALL mm of material
to the sphere surface on the narrower side of that cut.

Requires: pip install trimesh manifold3d numpy
"""

import os
import sys

import numpy as np
from manifold3d import Manifold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_slab_sphere import (  # noqa: E402
    FLOOR, MD, MT, EMBED_DCLEAR, EMBED_TCLEAR, SEG, CHAMFER, to_trimesh, report)

R = 75.0
CUTS_FROM_TOP = [32.5, 75.0, 100.0, 125.0]
WALL = 4.0                      # min material between pocket and sphere surface
ANGLES = (0.0, 90.0, 180.0, 270.0)

POCKET_DEPTH = FLOOR + MT + EMBED_TCLEAR   # 4.0mm from the face
PD = (MD + EMBED_DCLEAR) / 2               # pocket radius


def r_at(z):
    return np.sqrt(max(R * R - z * z, 0.0))


def ring_radius(zc):
    """Pocket ring radius for the cut at height zc (sphere-centred coords).
    Checked on the side where the sphere narrows, at full pocket depth."""
    z_in = abs(zc) + POCKET_DEPTH
    r = r_at(z_in) - PD - WALL
    return float(np.floor(r))


def piece(z_lo, z_hi):
    """Piece between z_lo and z_hi, oriented with the larger face down and
    re-based to z=0. Returns (manifold, bed_cut_z, top_cut_z) where the cut
    z values are in sphere coords (None if that side is the sphere pole)."""
    box = Manifold.cube([2 * R + 8, 2 * R + 8, z_hi - z_lo]).translate(
        [-(R + 4), -(R + 4), z_lo])
    m = Manifold.sphere(R, SEG) ^ box
    lo_cut = z_lo if z_lo > -R else None
    hi_cut = z_hi if z_hi < R else None
    r_lo = r_at(z_lo) if lo_cut is not None else -1
    r_hi = r_at(z_hi) if hi_cut is not None else -1
    if r_lo >= r_hi:                       # bottom side is larger: as is
        return m.translate([0, 0, -z_lo]), lo_cut, hi_cut
    # flip upside down so the upper (larger) face sits on the bed
    m = m.mirror([0, 0, 1]).translate([0, 0, z_hi])
    return m, hi_cut, lo_cut


def chamfer(m, face_r):
    """45 deg chamfer on the bed-face rim (elephant-foot relief)."""
    keep_cone = Manifold.cylinder(CHAMFER, face_r - CHAMFER, face_r, 256)
    keep_above = Manifold.cube([4 * R, 4 * R, 2 * R]).translate(
        [-2 * R, -2 * R, CHAMFER])
    return m ^ (keep_cone + keep_above)


def pockets(m, T, bed_cut, top_cut):
    for cut, side in ((bed_cut, "bed"), (top_cut, "top")):
        if cut is None:
            continue
        rr = ring_radius(cut)
        z0, z1 = (FLOOR, POCKET_DEPTH) if side == "bed" \
            else (T - POCKET_DEPTH, T - FLOOR)
        for deg in ANGLES:
            x, y = rr * np.cos(np.radians(deg)), rr * np.sin(np.radians(deg))
            m = m - Manifold.cylinder(z1 - z0, PD, PD, 96).translate([x, y, z0])
    return m


def build_all():
    zs = [R] + [R - c for c in CUTS_FROM_TOP] + [-R]   # top -> bottom
    out = []
    for i in range(len(zs) - 1):
        z_hi, z_lo = zs[i], zs[i + 1]
        m, bed_cut, top_cut = piece(z_lo, z_hi)
        T = z_hi - z_lo
        m = pockets(m, T, bed_cut, top_cut)
        m = chamfer(m, r_at(bed_cut))
        out.append((i + 1, T, bed_cut, top_cut, to_trimesh(m)))
    return out


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "sphere150_cut5")
    os.makedirs(out, exist_ok=True)
    for n, T, bed_cut, top_cut, tm in build_all():
        fn = os.path.join(out, f"sphere150cut5_P{n}_m10x3_embed.stl")
        tm.export(fn)
        report(tm, os.path.basename(fn))
        msg = f"   T={T:.1f} bed face Ø{2*r_at(bed_cut):.1f} ring r={ring_radius(bed_cut):.0f}" \
              f" | pause Z={POCKET_DEPTH:.1f}"
        if top_cut is not None:
            msg += f" & Z={T - FLOOR:.1f} (top face Ø{2*r_at(top_cut):.1f}," \
                   f" ring r={ring_radius(top_cut):.0f})"
        print(msg)
