from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from k1score.core.paths import Paths

W, H, FPS = 640, 480, 30


SKIN = (120, 160, 215)                 # BGR, inside the skin HSV range
SHORTS_A, SHORTS_B = (15, 15, 15), (30, 40, 200)   # fighter A black shorts, fighter B red shorts
SEG = 40                               # frames per shot
BOXES = np.array([[100, 120, 220, 420], [380, 120, 500, 420], [290, 220, 340, 400]], dtype=np.float32)


def _paint_fighter(img, box, shorts):
    x1, y1 = int(box[0]), int(box[1])
    cv2.rectangle(img, (x1 + 25, y1 + 95), (x1 + 105, y1 + 214), SKIN, -1)        # bare torso
    cv2.rectangle(img, (x1 + 25, y1 + 216), (x1 + 105, y1 + 285), shorts, -1)


def _paint_referee(img, box):
    x1, y1 = int(box[0]), int(box[1])
    cv2.rectangle(img, (x1 + 8, y1 + 58), (x1 + 44, y1 + 128), (20, 20, 20), -1)    # dark shirt
    cv2.rectangle(img, (x1 + 8, y1 + 130), (x1 + 44, y1 + 172), (25, 25, 25), -1)   # dark trousers


def make_video(path, segments=((SEG, (200, 40, 40)), (SEG, (40, 200, 40)), (SEG, (40, 40, 200))), people=False):
    """Solid-colour shots (camera cuts). With people=True: fighter A (black shorts) and B (red shorts) plus a
    referee are painted; A and B swap sides in the middle shot."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for seg, (n, color) in enumerate(segments):
        for _ in range(n):
            img = np.full((H, W, 3), color, dtype=np.uint8)
            if people:
                left, right = (SHORTS_A, SHORTS_B) if seg != 1 else (SHORTS_B, SHORTS_A)
                _paint_fighter(img, BOXES[0], left)
                _paint_fighter(img, BOXES[1], right)
                _paint_referee(img, BOXES[2])
            w.write(img)
    w.release()
    return path


@pytest.fixture
def video(tmp_path):
    return make_video(tmp_path / "clip.mp4")


@pytest.fixture
def people_video(tmp_path):
    return make_video(tmp_path / "people.mp4", people=True)


@pytest.fixture
def paths(tmp_path):
    (tmp_path / "proj" / "configs").mkdir(parents=True)
    return Paths(tmp_path / "proj")


class FakeModel:
    """Mimics the ultralytics API used by the pipeline: two fighters (ids 1,2) + a small referee (id 3)."""

    predictor = None
    calls = 0

    def _people(self):
        boxes = BOXES.copy()
        # real people bob around; a perfectly static box would be (correctly) treated as an overlay graphic
        boxes[:, [0, 2]] += 10 * np.sin(self.calls / 3)
        boxes[:, [1, 3]] += 8 * np.cos(self.calls / 5)
        kp = np.zeros((3, 17, 3), dtype=np.float32)
        for i, b in enumerate(boxes):
            kp[i, :, 0] = np.linspace(b[0], b[2], 17)
            kp[i, :, 1] = np.linspace(b[1], b[3], 17)
            kp[i, :, 2] = 0.9
        return boxes, kp

    def _result(self, with_ids):
        boxes, kp = self._people()

        class B:
            xyxy = torch.tensor(boxes)
            id = torch.tensor([1.0, 2.0, 3.0]) if with_ids else None

            def __len__(self):
                return 3

        return SimpleNamespace(boxes=B(), keypoints=SimpleNamespace(data=torch.tensor(kp)))

    def predict(self, frames, **kw):
        return [self._result(False) for _ in frames]

    def track(self, frame, **kw):
        self.calls += 1
        return [self._result(True)]
