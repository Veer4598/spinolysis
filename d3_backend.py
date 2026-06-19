import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np


class PoseQuality(Enum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    PARTIAL = "PARTIAL"
    WAITING = "WAITING"
    UNKNOWN = "UNKNOWN"


@dataclass
class PoseMetrics:
    pose_status: str = PoseQuality.WAITING.value
    tilt_angle: float = 0.0
    arm_angle: float = 0.0
    feedback: str = "Hold position"
    keypoints_detected: int = 0


class PoseAnalyzer:
    """
    Backend-safe pose analyzer aligned to d3.py logic.
    """

    def __init__(self):
        # Keep ids aligned with current website flow.
        self.exercises = {
            1: {"name": "NECK_SIDE_TILT", "reps": 3, "hold_time": 0, "side": "both", "display_name": "Neck Tilt"},
            2: {"name": "ARM_RAISE_OVERHEAD", "reps": 0, "hold_time": 10, "side": "both", "display_name": "Hand Raise"},
            3: {"name": "ARM_ROTATION", "reps": 0, "hold_time": 10, "side": "both", "display_name": "Arm Rotation"},
            4: {"name": "CAT_POSE", "reps": 0, "hold_time": 15, "side": "both", "display_name": "Cat Pose"},
            5: {"name": "CHILD_POSE", "reps": 0, "hold_time": 15, "side": "both", "display_name": "Child Pose"},
            6: {"name": "BRIDGE_POSE", "reps": 0, "hold_time": 15, "side": "both", "display_name": "Bridge Pose"},
            7: {"name": "MOUNTAIN_POSE", "reps": 0, "hold_time": 15, "side": "both", "display_name": "Mountain Pose"},
        }

        # Accept both new names and legacy d3.py names.
        self.exercise_aliases = {
            "MARJARYASANA_CAT_POSE": "CAT_POSE",
            "BALASANA_CHILD_POSE": "CHILD_POSE",
            "SETU_BANDHA_BRIDGE_POSE": "BRIDGE_POSE",
            "PARVATASANA_DOWNWARD_DOG": "DOWNWARD_DOG",
        }

    def _resolve_exercise_name(self, exercise_type: Union[str, int]) -> str:
        if isinstance(exercise_type, int):
            return self.exercises.get(exercise_type, {}).get("name", "UNKNOWN")

        raw = str(exercise_type)
        if raw.isdigit():
            return self.exercises.get(int(raw), {}).get("name", "UNKNOWN")

        return self.exercise_aliases.get(raw, raw)

    @staticmethod
    def _kp_visible(kp) -> bool:
        return kp is not None and len(kp) >= 3 and kp[2] > 0.03

    def _count_visible_keypoints(self, keypoints) -> int:
        return int(sum(1 for kp in keypoints if self._kp_visible(kp)))

    @staticmethod
    def derive_virtual_neck(keypoints) -> Optional[tuple]:
        if keypoints is None or len(keypoints) < 7:
            return None
        if keypoints[5][2] > 0.1 and keypoints[6][2] > 0.1:
            return (
                (keypoints[5][0] + keypoints[6][0]) / 2,
                (keypoints[5][1] + keypoints[6][1]) / 2,
            )
        if keypoints[5][2] > 0.1:
            return (keypoints[5][0], keypoints[5][1])
        if keypoints[6][2] > 0.1:
            return (keypoints[6][0], keypoints[6][1])
        return None

    @staticmethod
    def derive_virtual_midhip(keypoints) -> Optional[tuple]:
        if keypoints is None or len(keypoints) < 13:
            return None
        if keypoints[11][2] > 0.1 and keypoints[12][2] > 0.1:
            return (
                (keypoints[11][0] + keypoints[12][0]) / 2,
                (keypoints[11][1] + keypoints[12][1]) / 2,
            )
        if keypoints[11][2] > 0.1:
            return (keypoints[11][0], keypoints[11][1])
        if keypoints[12][2] > 0.1:
            return (keypoints[12][0], keypoints[12][1])
        return None

    @staticmethod
    def calculate_angle(point1, point2, point3) -> float:
        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3

        v1 = (x1 - x2, y1 - y2)
        v2 = (x3 - x2, y3 - y2)

        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 * mag2 == 0:
            return 0.0

        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.degrees(math.acos(cos_angle))

    @staticmethod
    def calculate_vertical_angle(point1, point2) -> float:
        if point1 is None or point2 is None:
            return 0.0
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        if dy == 0:
            return 0.0
        return abs(math.degrees(math.atan2(dx, -dy)))

    def analyze_pose(self, keypoints, exercise_type="unknown"):
        if keypoints is None or len(keypoints) == 0:
            return {
                "pose_status": PoseQuality.WAITING.value,
                "quality": PoseQuality.WAITING.value,
                "feedback": "No pose detected",
                "score": 0.0,
                "keypoints_detected": 0,
                "tilt_angle": 0.0,
                "arm_angle": 0.0,
                "angle": 0.0,
                "confidence": 0.0,
                "alignment": 0.0,
                "symmetry": 0.0,
            }

        name = self._resolve_exercise_name(exercise_type)
        metrics = PoseMetrics()
        metrics.keypoints_detected = self._count_visible_keypoints(keypoints)

        neck_virtual = self.derive_virtual_neck(keypoints)
        midhip_virtual = self.derive_virtual_midhip(keypoints)

        if metrics.keypoints_detected < 2:
            metrics.feedback = "Move into full camera view"
            return self._as_result(metrics)

        if name == "NECK_SIDE_TILT":
            self._analyze_neck_tilt(keypoints, neck_virtual, metrics)
        elif name == "ARM_RAISE_OVERHEAD":
            self._analyze_arm_raise(keypoints, metrics)
        elif name == "ARM_ROTATION":
            self._analyze_arm_rotation(keypoints, metrics)
        elif name == "CAT_POSE":
            self._analyze_cat_pose(keypoints, neck_virtual, midhip_virtual, metrics)
        elif name == "DOWNWARD_DOG":
            self._analyze_downward_dog(keypoints, metrics)
        elif name == "CHILD_POSE":
            self._analyze_child_pose(keypoints, neck_virtual, midhip_virtual, metrics)
        elif name == "BRIDGE_POSE":
            self._analyze_bridge_pose(keypoints, metrics)
        elif name == "MOUNTAIN_POSE":
            self._analyze_mountain_pose(keypoints, metrics)
        else:
            metrics.pose_status = PoseQuality.UNKNOWN.value
            metrics.feedback = f"Unknown exercise: {name}"

        return self._as_result(metrics)

    def _as_result(self, metrics: PoseMetrics):
        score_map = {
            PoseQuality.CORRECT.value: 0.95,
            PoseQuality.PARTIAL.value: 0.7,
            PoseQuality.INCORRECT.value: 0.35,
            PoseQuality.WAITING.value: 0.0,
            PoseQuality.UNKNOWN.value: 0.0,
        }
        score = float(np.clip(score_map.get(metrics.pose_status, 0.0), 0.0, 1.0))
        angle_value = metrics.tilt_angle if metrics.tilt_angle > 0 else metrics.arm_angle

        return {
            "pose_status": metrics.pose_status,
            "quality": metrics.pose_status,
            "feedback": metrics.feedback,
            "score": score,
            "keypoints_detected": metrics.keypoints_detected,
            "tilt_angle": float(metrics.tilt_angle),
            "arm_angle": float(metrics.arm_angle),
            "angle": float(angle_value),
            "confidence": score,
            "alignment": 0.0,
            "symmetry": 0.0,
        }

    def _analyze_neck_tilt(self, keypoints, neck_virtual, metrics: PoseMetrics):
        if neck_virtual is None or not self._kp_visible(keypoints[0]):
            metrics.feedback = "Keep head and shoulders visible"
            return

        nose = (keypoints[0][0], keypoints[0][1])
        neck = (neck_virtual[0], neck_virtual[1])
        metrics.tilt_angle = self.calculate_vertical_angle(neck, nose)

        if 15 <= metrics.tilt_angle <= 45:
            metrics.pose_status = PoseQuality.CORRECT.value
            metrics.feedback = f"Good neck tilt: {metrics.tilt_angle:.1f}°"
            if self._kp_visible(keypoints[5]) and self._kp_visible(keypoints[6]):
                shoulder_height_diff = abs(keypoints[5][1] - keypoints[6][1])
                if shoulder_height_diff > 30:
                    metrics.pose_status = PoseQuality.PARTIAL.value
                    metrics.feedback = "Good tilt, keep shoulders level"
        elif 5 <= metrics.tilt_angle < 15:
            metrics.pose_status = PoseQuality.PARTIAL.value
            metrics.feedback = f"Tilt slightly more ({metrics.tilt_angle:.1f}°)"
        else:
            metrics.pose_status = PoseQuality.INCORRECT.value
            metrics.feedback = "Adjust neck tilt angle"

    def _analyze_arm_raise(self, keypoints, metrics: PoseMetrics):
        left_arm_angle = 0.0
        right_arm_angle = 0.0

        if self._kp_visible(keypoints[5]) and self._kp_visible(keypoints[7]) and self._kp_visible(keypoints[9]):
            left_arm_angle = self.calculate_angle(
                (keypoints[5][0], keypoints[5][1]),
                (keypoints[7][0], keypoints[7][1]),
                (keypoints[9][0], keypoints[9][1]),
            )

        if self._kp_visible(keypoints[6]) and self._kp_visible(keypoints[8]) and self._kp_visible(keypoints[10]):
            right_arm_angle = self.calculate_angle(
                (keypoints[6][0], keypoints[6][1]),
                (keypoints[8][0], keypoints[8][1]),
                (keypoints[10][0], keypoints[10][1]),
            )

        metrics.arm_angle = (left_arm_angle + right_arm_angle) / 2 if left_arm_angle and right_arm_angle else max(left_arm_angle, right_arm_angle)

        if metrics.arm_angle <= 0:
            metrics.pose_status = PoseQuality.PARTIAL.value
            metrics.feedback = "Show at least one full arm for analysis"
        elif metrics.arm_angle > 160:
            metrics.pose_status = PoseQuality.CORRECT.value
            metrics.feedback = "Arms fully raised"
        elif metrics.arm_angle > 120:
            metrics.pose_status = PoseQuality.PARTIAL.value
            metrics.feedback = "Raise arms higher"
        else:
            metrics.pose_status = PoseQuality.INCORRECT.value
            metrics.feedback = "Lift arms overhead"

    def _analyze_arm_rotation(self, keypoints, metrics: PoseMetrics):
        valid = False
        wrist_height = 0.0
        shoulder_height = 0.0
        if self._kp_visible(keypoints[9]) and self._kp_visible(keypoints[5]):
            wrist_height = keypoints[9][1]
            shoulder_height = keypoints[5][1]
            valid = True
        elif self._kp_visible(keypoints[10]) and self._kp_visible(keypoints[6]):
            wrist_height = keypoints[10][1]
            shoulder_height = keypoints[6][1]
            valid = True

        if valid:
            if abs(wrist_height - shoulder_height) < 50:
                metrics.pose_status = PoseQuality.CORRECT.value
                metrics.feedback = "Good arm rotation height"
            else:
                metrics.pose_status = PoseQuality.INCORRECT.value
                metrics.feedback = "Keep wrist around shoulder level"
        else:
            metrics.pose_status = PoseQuality.PARTIAL.value
            metrics.feedback = "Show at least one arm clearly"

    def _analyze_cat_pose(self, keypoints, neck_virtual, midhip_virtual, metrics: PoseMetrics):
        if neck_virtual is None or midhip_virtual is None:
            metrics.pose_status = PoseQuality.PARTIAL.value
            metrics.feedback = "Keep torso fully visible"
            return

        nose = (keypoints[0][0], keypoints[0][1]) if self._kp_visible(keypoints[0]) else None
        if nose is None:
            metrics.pose_status = PoseQuality.PARTIAL.value
            metrics.feedback = "Keep head visible"
            return

        neck = (neck_virtual[0], neck_virtual[1])
        midhip = (midhip_virtual[0], midhip_virtual[1])
        spine_angle = self.calculate_angle(nose, neck, midhip)

        if spine_angle < 150:
            metrics.pose_status = PoseQuality.CORRECT.value
            metrics.feedback = "Good spinal flexion"
            if self._kp_visible(keypoints[5]) and nose[1] > keypoints[5][1] + 20:
                metrics.pose_status = PoseQuality.PARTIAL.value
                metrics.feedback = "Good curve, lower head slightly"
        else:
            metrics.pose_status = PoseQuality.INCORRECT.value
            metrics.feedback = "Round upper back more"

    def _analyze_downward_dog(self, keypoints, metrics: PoseMetrics):
        if self._kp_visible(keypoints[5]) and self._kp_visible(keypoints[11]) and self._kp_visible(keypoints[13]):
            if keypoints[11][1] < keypoints[5][1] - 50:
                if self._kp_visible(keypoints[15]):
                    leg_angle = self.calculate_angle(
                        (keypoints[11][0], keypoints[11][1]),
                        (keypoints[13][0], keypoints[13][1]),
                        (keypoints[15][0], keypoints[15][1]),
                    )
                    if leg_angle > 150:
                        metrics.pose_status = PoseQuality.CORRECT.value
                        metrics.feedback = "Good downward dog alignment"
                    else:
                        metrics.pose_status = PoseQuality.PARTIAL.value
                        metrics.feedback = "Straighten legs more"
                else:
                    metrics.pose_status = PoseQuality.PARTIAL.value
                    metrics.feedback = "Keep legs visible"
            else:
                metrics.pose_status = PoseQuality.INCORRECT.value
                metrics.feedback = "Lift hips higher"
        else:
            metrics.pose_status = PoseQuality.PARTIAL.value
            metrics.feedback = "Keep hips, knees and shoulders visible"

    def _analyze_child_pose(self, keypoints, neck_virtual, midhip_virtual, metrics: PoseMetrics):
        if self._kp_visible(keypoints[11]) and self._kp_visible(keypoints[15]):
            hip_ankle_dist = math.sqrt(
                (keypoints[11][0] - keypoints[15][0]) ** 2 +
                (keypoints[11][1] - keypoints[15][1]) ** 2
            )
            if hip_ankle_dist < 100:
                if neck_virtual and midhip_virtual:
                    torso_angle = self.calculate_vertical_angle(
                        (neck_virtual[0], neck_virtual[1]),
                        (midhip_virtual[0], midhip_virtual[1]),
                    )
                    if torso_angle < 30:
                        metrics.pose_status = PoseQuality.CORRECT.value
                        metrics.feedback = "Good child pose"
                    else:
                        metrics.pose_status = PoseQuality.PARTIAL.value
                        metrics.feedback = "Relax torso forward"
                else:
                    metrics.pose_status = PoseQuality.CORRECT.value
                    metrics.feedback = "Good child pose"
            else:
                metrics.pose_status = PoseQuality.INCORRECT.value
                metrics.feedback = "Bring hips closer to heels"
        else:
            metrics.pose_status = PoseQuality.PARTIAL.value
            metrics.feedback = "Keep hips and ankles visible"

    def _analyze_bridge_pose(self, keypoints, metrics: PoseMetrics):
        if self._kp_visible(keypoints[5]) and self._kp_visible(keypoints[11]):
            if keypoints[11][1] < keypoints[5][1] - 50:
                if self._kp_visible(keypoints[13]) and self._kp_visible(keypoints[15]):
                    knee_angle = self.calculate_angle(
                        (keypoints[11][0], keypoints[11][1]),
                        (keypoints[13][0], keypoints[13][1]),
                        (keypoints[15][0], keypoints[15][1]),
                    )
                    if knee_angle < 120:
                        metrics.pose_status = PoseQuality.CORRECT.value
                        metrics.feedback = "Strong bridge position"
                    else:
                        metrics.pose_status = PoseQuality.PARTIAL.value
                        metrics.feedback = "Bend knees a bit more"
                else:
                    metrics.pose_status = PoseQuality.CORRECT.value
                    metrics.feedback = "Good bridge lift"
            else:
                metrics.pose_status = PoseQuality.INCORRECT.value
                metrics.feedback = "Lift hips higher"
        else:
            metrics.pose_status = PoseQuality.PARTIAL.value
            metrics.feedback = "Keep shoulders and hips visible"

    def _analyze_mountain_pose(self, keypoints, metrics: PoseMetrics):
        if self._kp_visible(keypoints[3]) and self._kp_visible(keypoints[5]) and self._kp_visible(keypoints[11]) and self._kp_visible(keypoints[15]):
            deviation = abs(keypoints[3][0] - keypoints[5][0]) + abs(keypoints[5][0] - keypoints[11][0]) + abs(keypoints[11][0] - keypoints[15][0])
            if deviation < 30:
                metrics.pose_status = PoseQuality.CORRECT.value
                metrics.feedback = "Excellent mountain alignment"
            elif deviation < 60:
                metrics.pose_status = PoseQuality.PARTIAL.value
                metrics.feedback = "Good posture, align a bit more"
            else:
                metrics.pose_status = PoseQuality.INCORRECT.value
                metrics.feedback = "Stack ear, shoulder, hip, ankle"
        else:
            metrics.pose_status = PoseQuality.PARTIAL.value
            metrics.feedback = "Stand fully in frame"
