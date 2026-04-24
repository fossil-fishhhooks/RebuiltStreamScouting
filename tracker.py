import cv2
import numpy as np

from config import GHOST_FRAMES, MAX_DIST


def make_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], dtype=np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost        = np.eye(4, dtype=np.float32)
    return kf


class Track:
    def __init__(self, track_id, x, y):
        self.id = track_id
        self.kf = make_kalman()
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.ghost_count = 0

    def predict(self):
        pred = self.kf.predict()
        return int(pred[0][0]), int(pred[1][0])

    def update(self, x, y):
        self.kf.correct(np.array([[x], [y]], dtype=np.float32))
        self.ghost_count = 0

    def position(self):
        s = self.kf.statePost
        return int(s[0][0]), int(s[1][0])


class Tracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, detections):
        predictions = {tid: t.predict() for tid, t in self.tracks.items()}

        pairs = []
        for di, (dx, dy) in enumerate(detections):
            best_id, best_dist = None, MAX_DIST
            for tid, (px, py) in predictions.items():
                dist = ((px - dx) ** 2 + (py - dy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is not None:
                pairs.append((di, best_id))

        used_tracks = {}
        for di, tid in pairs:
            if tid in used_tracks:
                prev_di = used_tracks[tid]
                px, py = predictions[tid]
                dx1, dy1 = detections[prev_di]
                dx2, dy2 = detections[di]
                if ((px - dx2) ** 2 + (py - dy2) ** 2) < ((px - dx1) ** 2 + (py - dy1) ** 2):
                    used_tracks[tid] = di
            else:
                used_tracks[tid] = di

        matched_det_indices = set(used_tracks.values())
        matched_track_ids = set()

        for tid, di in used_tracks.items():
            dx, dy = detections[di]
            self.tracks[tid].update(dx, dy)
            matched_track_ids.add(tid)

        for di, (dx, dy) in enumerate(detections):
            if di not in matched_det_indices:
                self.tracks[self.next_id] = Track(self.next_id, dx, dy)
                self.next_id += 1

        for tid, track in list(self.tracks.items()):
            if tid not in matched_track_ids:
                track.ghost_count += 1
                if track.ghost_count > GHOST_FRAMES:
                    del self.tracks[tid]

        return {tid: t.position() for tid, t in self.tracks.items()}

