SKIP_SECONDS    = 4  #62
MAX_DIST        = 60

# Regions are tuned against this source resolution and scaled per input video.
REFERENCE_FRAME_SIZE = (1366, 768)
CROP_REF             = (0, 90, 1356, 491)
CROP_REF_SIZE        = (CROP_REF[2] - CROP_REF[0], CROP_REF[3] - CROP_REF[1])
HOLE_REF             = (445, 0, 1356 - 445, 491)
MAX_TRAIL            = 38
TRAIL_DECAY          = 0.95

SCORE_POLYGON_REF_BY_SIDE = {
    "red": [
        (342, 136),
        (385, 136),
        (405, 165),
        (385, 195),
        (342, 195),
        (322, 165),
    ],
    "blue": [
        (974, 126),
        (1022, 126),
        (1046, 155),
        (1022, 185),
        (974, 185),
        (950, 155),
    ],
}

ACTIVE_REGION_REF_BY_SIDE = {
    "red": (0, 0, CROP_REF_SIZE[0] // 2, CROP_REF_SIZE[1]),
    "blue": (CROP_REF_SIZE[0] // 2, 0, CROP_REF_SIZE[0], CROP_REF_SIZE[1]),
}

PARABOLA_MIN_POINTS      = 8
PARABOLA_A_MAX           = 0.003
PARABOLA_R2_MIN          = 0.50
SCORE_TRAIL_WINDOW       = 8
SCORE_MIN_DESCENT        = 4
SCORE_MIN_INSIDE_POINTS  = 2
BOUNCE_OUT_RISE          = 4
SCORE_COOLDOWN_FRAMES    = 0
ID_SCORE_COOLDOWN_FRAMES = 10

GHOST_FRAMES = 4
FRAME_SKIP   = 2

