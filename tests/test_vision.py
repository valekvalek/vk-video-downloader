import numpy as np

from matchlens.schemas import OPP, OURS, UNKNOWN, BBox
from matchlens.vision.jersey import NumberVoter
from matchlens.vision.pitch import KEYPOINTS_M, apply_homography, estimate_homography, inside_pitch
from matchlens.vision.team_split import assign_teams, dominant_color, torso_crop
from matchlens.vision.track import IouTracker


def _frame_with(color, box=(20, 20, 60, 100), size=(120, 100)):
    f = np.zeros((size[0], size[1], 3), np.uint8)
    f[:] = (30, 160, 40)  # трава
    x1, y1, x2, y2 = box
    f[y1:y2, x1:x2] = color
    return f


def test_dominant_color_ignores_grass():
    f = _frame_with((220, 30, 30))
    c = dominant_color(torso_crop(f, BBox(20, 20, 60, 100)))
    assert c is not None and c[0] > 200 and c[2] < 60


def test_assign_teams_splits_red_and_blue():
    red = [np.array([210, 30, 30]) + i for i in range(5)]
    blue = [np.array([30, 40, 200]) + i for i in range(5)]
    labels = assign_teams(red + blue + [None], our_color=(200, 30, 30))
    assert labels[:5] == [OURS] * 5
    assert labels[5:10] == [OPP] * 5
    assert labels[10] == UNKNOWN


def test_tracker_keeps_ids_and_creates_new():
    t = IouTracker(iou_threshold=0.3, max_lost=2)
    a = t.update([BBox(0, 0, 10, 20), BBox(100, 0, 110, 20)])
    b = t.update([BBox(1, 0, 11, 20), BBox(101, 0, 111, 20)])
    assert [i for i, _ in a] == [i for i, _ in b]
    c = t.update([BBox(1, 0, 11, 20), BBox(300, 0, 310, 20)])
    assert c[0][0] == a[0][0] and c[1][0] not in (a[0][0], a[1][0])


def test_voter_needs_votes_and_margin():
    v = NumberVoter(min_votes=3, min_margin=1.5)
    v.add(1, 7, 0.9)
    v.add(1, 7, 0.8)
    assert v.result(1).number is None  # мало голосов
    v.add(1, 7, 0.9)
    assert v.result(1).number == 7
    for _ in range(3):
        v.add(2, 4, 0.5)
        v.add(2, 9, 0.5)
    assert v.result(2).number is None  # спорный номер не приписываем
    v.add(3, 5, 0.1)  # ниже min_conf — не считается
    assert v.result(3).votes == 0


def test_homography_recovers_pitch_coordinates():
    names = ["corner_tl", "corner_tr", "corner_bl", "corner_br", "center"]
    dst = np.array([KEYPOINTS_M[n] for n in names])
    # «камера»: масштаб + сдвиг + лёгкая перспектива
    true_h = np.array([[8.0, 1.0, 100.0], [0.5, 6.0, 50.0], [0.0004, 0.0002, 1.0]])
    inv = np.linalg.inv(true_h)
    src = np.array([(inv @ [x, y, 1])[:2] / (inv @ [x, y, 1])[2] for x, y in dst])
    h = estimate_homography(src, dst)
    got = apply_homography(h, src)
    assert np.abs(got - dst).max() < 1e-6


def test_inside_pitch_mask():
    m = inside_pitch(np.array([[50, 30], [-10, 30], [50, 90]]))
    assert m.tolist() == [True, False, False]
