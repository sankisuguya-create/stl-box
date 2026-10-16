"""
Attraction between two coaxial disc magnets vs. gap (magnetic surface-charge model).

  python mag_force.py              # φ10×3 N35, gaps 0.8–1.8mm
  python mag_force.py 1.4 1.2      # specific gaps (mm)

Assumes ideal, uniformly magnetised magnets (Br=1.2T ~ N35) facing each other
on the same axis. Ratios between gaps are reliable; absolute kgf can be off by
20–30% (grade, coating, misalignment). Requires numpy.
"""

import sys

import numpy as np

BR = 1.2          # remanence [T] (N35 ~1.17–1.22)
DIA, THK = 10.0, 3.0   # magnet [mm]
REF_GAP = 1.6     # reference gap [mm] (= 0.8mm floor on each side)

MU0 = 4e-7 * np.pi


def _grid(a, nr=30, nt=60):
    r = (np.arange(nr) + 0.5) / nr * a
    th = (np.arange(nt) + 0.5) / nt * 2 * np.pi
    R, T = np.meshgrid(r, th)
    dA = (a / nr) * R * (2 * np.pi / nt)
    pts = np.c_[(R * np.cos(T)).ravel(), (R * np.sin(T)).ravel()]
    return pts, (BR / MU0 * dA).ravel()


P, Q = _grid(DIA / 2 * 1e-3)
DX = P[:, None, 0] - P[None, :, 0]
DY = P[:, None, 1] - P[None, :, 1]
QQ = Q[:, None] * Q[None, :]


def _disc(z):
    return MU0 / (4 * np.pi) * np.sum(QQ * z / (DX**2 + DY**2 + z**2) ** 1.5)


def force_n(gap_mm):
    g, t = gap_mm * 1e-3, THK * 1e-3
    return _disc(g) - 2 * _disc(g + t) + _disc(g + 2 * t)


if __name__ == "__main__":
    gaps = [float(x) for x in sys.argv[1:]] or [1.8, 1.6, 1.4, 1.2, 1.0, 0.8]
    ref = force_n(REF_GAP)
    print(f"φ{DIA:g}×{THK:g}mm Br={BR}T  (100 = gap {REF_GAP}mm)")
    for g in gaps:
        f = force_n(g)
        print(f"gap {g:.1f}mm: {100 * f / ref:5.0f}   {f / 9.81:.2f} kgf/pair")
