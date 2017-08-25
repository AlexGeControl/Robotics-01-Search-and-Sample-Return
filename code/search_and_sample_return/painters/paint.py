# Set up session:
import argparse

import numpy as np
import cv2

class BirdEyeViewPainter:
    def __init__(
        self,
        frame_size,
        ground_color = (255, 0, 0),
        obstacle_color = (0, 0, 255),
        rock_color = (255, 255, 255),
        heading_color = (0, 255, 0)
    ):
        self.W, self.H = frame_size

        self.ground_color = ground_color
        self.obstacle_color = obstacle_color
        self.rock_color = rock_color
        self.heading_color = heading_color

    def transform(
        self,
        ground_coords,
        obstacle_coords,
        rock_coords,
        heading
    ):
        # Initialize canvas:
        canvas = np.zeros(
            (self.H, self.W, 3),
            dtype=np.uint8
        )

        for obj_coords, obj_color in zip(
            (
                ground_coords,
                obstacle_coords
            ),
            (
                self.ground_color,
                self.obstacle_color
            )
        ):
            h = np.int_(self.H - obj_coords[0])
            w = np.int_(self.W/2 - obj_coords[1])
            canvas[h, w, :] = obj_color

        # Add heading:
        delta_x, delta_y = self.H/2 * np.sin(heading), self.H/2*np.cos(heading)
        cv2.line(
            canvas,
            (int(self.W//2 - delta_x), int(self.H - delta_y)),
            (self.W//2, self.H),
            self.heading_color,
            3
        )

        # Draw rock:
        if len(rock_coords[0]) != 0:
            cv2.circle(
                canvas,
                (
                    int(np.mean(rock_coords[0])),
                    int(np.mean(rock_coords[1])),
                ),
                8,
                self.rock_color,
                -1
            )

        return canvas

class WorldMapPainter:
    def __init__(
        self,
        ground_color = (255, 0, 0),
        obstacle_color = (0, 0, 255),
        rock_color = (255, 255, 255)
    ):
        self.ground_color = ground_color
        self.obstacle_color = obstacle_color
        self.rock_color = rock_color

    def transform(
        self,
        world_map,
        ground_coords,
        obstacle_coords,
        rock_coords
    ):
        # Initialize canvas:
        canvas = np.zeros_like(world_map)

        # Draw ground and obstacles:
        for obj_coords, obj_color in zip(
            (
                ground_coords,
                obstacle_coords
            ),
            (
                self.ground_color,
                self.obstacle_color
            )
        ):
            canvas[obj_coords[1], obj_coords[0], :] = obj_color

        # Add overlay:
        painted = cv2.addWeighted(
            world_map, 1.0, canvas, 0.5, 0
        )

        # Draw rock:
        if len(rock_coords[0]) != 0:
            cv2.circle(
                painted,
                (
                    int(np.mean(rock_coords[0])),
                    int(np.mean(rock_coords[1])),
                ),
                5,
                self.rock_color,
                -1
            )

        return painted
