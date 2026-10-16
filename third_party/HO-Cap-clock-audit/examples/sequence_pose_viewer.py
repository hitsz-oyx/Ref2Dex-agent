"""Example of visualizing hand and object poses of one frame in a sequence."""

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"  # GPU-based offscreen rendering

from hocap_toolkit.utils import *
from hocap_toolkit.renderers import SequenceRenderer

PROJ_ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    sequence_folder = PROJ_ROOT / "datasets/subject_1/20231025_165502"
    renderer = SequenceRenderer(sequence_folder, device="cuda")

    frame_id = 70

    # Render the scene and get the rendered images
    renderer.create_scene(frame_id)
    render_colors = renderer.get_render_colors()

    # Display the rendered images
    overlays = [
        cv2.addWeighted(
            renderer.get_rgb_image(frame_id, serial), 0.4, render_color, 0.6, 0
        )
        for serial, render_color in render_colors.items()
    ]

    draw_all_camera_images(overlays, list(render_colors.keys()), show_only=True)
