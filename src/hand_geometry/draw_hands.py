from __future__ import annotations

import math

import cv2
import numpy as np


def _hand_endpoint(
	center_x: int,
	center_y: int,
	length: float,
	angle_degrees: float,
) -> tuple[int, int]:
	angle_radians = math.radians(angle_degrees - 90.0)
	end_x = center_x + int(round(length * math.cos(angle_radians)))
	end_y = center_y + int(round(length * math.sin(angle_radians)))
	return end_x, end_y


def draw_hands_on_image(
	image: np.ndarray,
	hour: int,
	minute: int,
	second: int,
) -> np.ndarray:
	if image is None or image.size == 0:
		raise ValueError("image must be a non-empty numpy array")

	output = image.copy()
	height, width = output.shape[:2]
	center_x = width // 2
	center_y = height // 2
	radius = int(min(width, height) * 0.42)

	hour_angle = ((hour % 12) + (minute / 60.0) + (second / 3600.0)) * 30.0
	minute_angle = (minute + second / 60.0) * 6.0
	second_angle = second * 6.0

	hour_end = _hand_endpoint(center_x, center_y, radius * 0.5, hour_angle)
	minute_end = _hand_endpoint(center_x, center_y, radius * 0.78, minute_angle)
	second_end = _hand_endpoint(center_x, center_y, radius * 0.9, second_angle)

	cv2.line(output, (center_x, center_y), hour_end, (30, 30, 30), 8, cv2.LINE_AA)
	cv2.line(output, (center_x, center_y), minute_end, (40, 40, 40), 5, cv2.LINE_AA)
	cv2.line(output, (center_x, center_y), second_end, (0, 0, 220), 2, cv2.LINE_AA)
	cv2.circle(output, (center_x, center_y), 8, (20, 20, 20), -1, cv2.LINE_AA)

	return output
