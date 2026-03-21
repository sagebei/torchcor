"""
torchcor/ecg/torso_tags.py
===========================
Single source of truth for CARP region tags used across all three
torchcor ECG pipelines:

  1. KCL anatomic torso    (KCL_torso3 VTU / CARP files)
  2. Synthetic torso       (gmsh-generated CARP files)
  3. CT-scan torso         (future: CT segmentation → CARP)

KCL CARP tag convention  (KCL_Info.pdf, Qian et al. 2022)
-----------------------------------------------------------
  1 – 25 : anatomy tissue (isotropic conductivity)
  26 – 33: ICD hardware   (excluded from anatomy meshes)
  34      : RV wall       (bidomain)
  35      : LV wall       (bidomain)

  NOTE — LV/RV blood tag assignment (CANONICAL):
    Tag 24 = RV blood pool  (deoxygenated, dark red)
    Tag 25 = LV blood pool  (oxygenated, bright red)
  Any code that has LV_BLOOD=24 or RV_BLOOD=25 contains a swap bug.

Synthetic / CT extra tags  (50 – 99, reserved)
-----------------------------------------------
  50 : synthetic lung L
  51 : synthetic lung R
  52 : synthetic spine
  53 : synthetic sternum
  54 : synthetic ribs
  55 : synthetic muscle layer
  56 : synthetic fat layer
"""

from __future__ import annotations

from typing import NamedTuple


# ==============================================================================
#  Data structure
# ==============================================================================

class TissueTag(NamedTuple):
    """Per-tissue metadata used by viewers and solvers."""
    tag:          int
    name:         str
    conductivity: float         # S/m  (isotropic; 0.0 = bidomain or non-conductive)
    color:        tuple         # RGB 0-1 (3-tuple of float)
    opacity:      float         # default display opacity (0 = hidden, 1 = solid)


# ==============================================================================
#  Full tag table  (CARP native numbering)
# ==============================================================================

_TABLE: list[TissueTag] = [
    # ── KCL anatomy tags 1–25 (isotropic conductivity, S/m) ─────────────────
    TissueTag( 1, "Inner body",            0.247, (0.88, 0.72, 0.65), 0.0),
    TissueTag( 2, "Skin",                  0.117, (0.91, 0.76, 0.60), 0.2),
    TissueTag( 3, "Bones",                 0.050, (0.96, 0.94, 0.84), 1.0),
    TissueTag( 4, "Kidneys",               0.167, (0.68, 0.28, 0.14), 1.0),
    TissueTag( 5, "Liver",                 0.167, (0.50, 0.16, 0.08), 1.0),
    TissueTag( 6, "Stomach",               0.100, (0.84, 0.76, 0.60), 1.0),
    TissueTag( 7, "Spleen",                0.100, (0.42, 0.12, 0.22), 1.0),
    TissueTag( 8, "Lungs",                 0.071, (0.93, 0.68, 0.62), 0.5),
    TissueTag( 9, "SVC blood pool",        0.667, (0.52, 0.04, 0.04), 0.8),
    TissueTag(10, "Superior vena cava",    0.667, (0.58, 0.06, 0.18), 0.8),
    TissueTag(11, "Pulm. artery pool",     0.667, (0.72, 0.08, 0.08), 0.8),
    TissueTag(12, "Pulmonary artery",      0.667, (0.76, 0.12, 0.12), 0.8),
    TissueTag(13, "Aorta",                 0.667, (0.75, 0.18, 0.12), 0.8),
    TissueTag(14, "Aorta blood pool",      0.667, (0.58, 0.06, 0.06), 0.8),
    TissueTag(15, "SVC plane",             0.667, (0.48, 0.16, 0.42), 0.3),
    TissueTag(16, "Mitral valve",          0.667, (0.92, 0.88, 0.76), 1.0),
    TissueTag(17, "Pulmonary valve",       0.667, (0.90, 0.86, 0.74), 1.0),
    TissueTag(18, "Aortic valve",          0.667, (0.94, 0.90, 0.78), 1.0),
    TissueTag(19, "Tricuspid valve",       0.667, (0.91, 0.87, 0.75), 1.0),
    TissueTag(20, "L. atrial wall",        0.250, (0.76, 0.14, 0.14), 1.0),
    TissueTag(21, "L. atrial blood pool",  0.667, (0.54, 0.06, 0.06), 0.5),
    TissueTag(22, "R. atrial blood pool",  0.667, (0.50, 0.04, 0.04), 0.5),
    TissueTag(23, "R. atrial wall",        0.250, (0.72, 0.14, 0.14), 1.0),
    # Tags 24/25: CANONICAL assignment — do NOT swap these.
    TissueTag(24, "RV blood pool",         0.667, (0.48, 0.04, 0.04), 0.5),  # deoxygenated
    TissueTag(25, "LV blood pool",         0.667, (0.56, 0.05, 0.05), 0.5),  # oxygenated

    # ── ICD hardware tags 26–33 (excluded from anatomy meshes) ──────────────
    TissueTag(26, "Coronary sinus coil",   0.0,   (0.72, 0.74, 0.80), 0.6),
    TissueTag(27, "SVC coil",              0.0,   (0.68, 0.70, 0.78), 0.6),
    TissueTag(28, "RV apical coil",        0.0,   (0.64, 0.66, 0.74), 0.6),
    TissueTag(29, "RV septal coil",        0.0,   (0.60, 0.62, 0.70), 0.6),
    TissueTag(30, "Subcutaneous ICD lead", 0.0,   (0.56, 0.58, 0.66), 0.6),
    TissueTag(31, "Left can",              0.0,   (0.80, 0.82, 0.86), 0.8),
    TissueTag(32, "Right can",             0.0,   (0.78, 0.80, 0.84), 0.8),
    TissueTag(33, "Subcutaneous ICD can",  0.0,   (0.76, 0.78, 0.82), 0.8),

    # ── Ventricular myocardium (bidomain — conductivity is extracellular iso avg) ──
    TissueTag(34, "RV wall",               0.136, (0.88, 0.10, 0.10), 1.0),
    TissueTag(35, "LV wall",               0.136, (0.82, 0.06, 0.06), 1.0),

    # ── Synthetic / CT extra tags (50–99 reserved) ───────────────────────────
    TissueTag(50, "Synth. lung L",         0.071, (0.93, 0.68, 0.62), 0.4),
    TissueTag(51, "Synth. lung R",         0.071, (0.93, 0.68, 0.62), 0.4),
    TissueTag(52, "Synth. spine",          0.050, (0.96, 0.94, 0.84), 1.0),
    TissueTag(53, "Synth. sternum",        0.050, (0.94, 0.92, 0.82), 1.0),
    TissueTag(54, "Synth. ribs",           0.050, (0.92, 0.90, 0.80), 1.0),
    TissueTag(55, "Synth. muscle",         0.150, (0.70, 0.20, 0.20), 0.5),
    TissueTag(56, "Synth. fat",            0.040, (0.95, 0.88, 0.60), 0.3),
]

# Fast lookup by tag number
TAG_INFO: dict[int, TissueTag] = {t.tag: t for t in _TABLE}


# ==============================================================================
#  Named tag constants  (CARP native numbering)
# ==============================================================================

BODY_TAG: int = 1   # Inner body / bulk soft tissue
SKIN_TAG: int = 2
BONE_TAG: int = 3

# Ventricular tags — CANONICAL (do NOT swap 24 and 25)
RV_BLOOD: int = 24   # RV blood pool  (deoxygenated)
LV_BLOOD: int = 25   # LV blood pool  (oxygenated)
RV_WALL:  int = 34   # RV myocardial wall (bidomain)
LV_WALL:  int = 35   # LV myocardial wall (bidomain)

# Synthetic extra tags
SYNTH_LUNG_L:   int = 50
SYNTH_LUNG_R:   int = 51
SYNTH_SPINE:    int = 52
SYNTH_STERNUM:  int = 53
SYNTH_RIBS:     int = 54
SYNTH_MUSCLE:   int = 55
SYNTH_FAT:      int = 56


# ==============================================================================
#  Tag group frozensets
# ==============================================================================

ICD_TAGS:        frozenset[int] = frozenset(range(26, 34))         # hardware, excluded from anatomy
VENT_TAGS:       frozenset[int] = frozenset({RV_BLOOD, LV_BLOOD, RV_WALL, LV_WALL})
BLOOD_TAGS:      frozenset[int] = frozenset({RV_BLOOD, LV_BLOOD})  # isotropic blood pools
MYO_TAGS:        frozenset[int] = frozenset({RV_WALL, LV_WALL})    # bidomain myocardium
ATRIA_TAGS:      frozenset[int] = frozenset({20, 21, 22, 23})
ALL_KCL_TAGS:    frozenset[int] = frozenset(range(1, 36))
SYNTH_STRUCT_TAGS: frozenset[int] = frozenset(range(50, 57))


# ==============================================================================
#  Isotropic conductivity table  (S/m)
#
#  Covers tags 1–25 only.  Tags 34/35 (ventricular myocardium) use anisotropic
#  bidomain conductivities and are handled separately by the pipeline.
#  ICD hardware tags 26–33 are excluded from anatomy meshes.
# ==============================================================================

CONDUCTIVITY: dict[int, float] = {
    t.tag: t.conductivity
    for t in _TABLE
    if 1 <= t.tag <= 25
}
# Add synthetic structure conductivities
CONDUCTIVITY.update({
    t.tag: t.conductivity
    for t in _TABLE
    if t.tag >= 50
})
