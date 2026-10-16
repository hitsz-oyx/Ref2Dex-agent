"""Example of visualizing hand and object poses of one frame in a sequence."""

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"  # GPU-based offscreen rendering

from hocap_toolkit.utils import *
from hocap_toolkit.loaders import SequenceLoader
from hocap_toolkit.renderers import OffscreenRenderer

PROJ_ROOT = Path(__file__).parent.parent


if __name__ == "__main__":
    sequence_folder = PROJ_ROOT / "datasets/HOCap/subject_2/20231022_201449"

    data_loader = SequenceLoader(str(sequence_folder), device="cuda")
    rs_serials = data_loader.rs_serials
    rs_height = data_loader.rs_height
    rs_width = data_loader.rs_width
    num_frames = data_loader.num_frames
    mano_sides = data_loader.mano_sides
    obj_meshes = [trimesh.load(p) for p in data_loader.object_textured_mesh_files]

    # Initialize renderer
    renderer = OffscreenRenderer(rs_width, rs_height)

    for frame_id in range(num_frames):
        for serial in rs_serials:
            image_color = data_loader.get_rgb_image(frame_id, serial)
            image_label = data_loader.get_image_label(frame_id, serial)

            if image_label:
                cam_K = image_label["cam_K"]
                obj_poses = image_label["obj_poses"]
                hand_joints_3d = image_label["hand_joints_3d"]
                hand_joints_2d = image_label["hand_joints_2d"]
                segmentation_mask = image_label["seg_mask"]
                obj_class_inds = image_label["obj_class_inds"].astype(int)
                obj_class_names = image_label["obj_class_names"].astype(str)

                # Render object poses
                render_color, render_depth = renderer.get_render_image(
                    obj_meshes, obj_poses, cam_K
                )
                image_pose = draw_image_overlay(image_color, render_color)

                # Draw hand joints
                image_handmarks = image_color.copy()
                for idx, marks in enumerate(hand_joints_2d):
                    side = mano_sides[idx]
                    image_handmarks = draw_hand_landmarks(image_handmarks, marks, side)

                # Draw segmentation visualization
                image_seg = np.zeros_like(image_color)
                for idx in np.unique(segmentation_mask):
                    if idx == 0:  # skip background
                        continue
                    image_seg[segmentation_mask == idx] = HO_CAP_SEG_COLOR[idx].rgb
                image_seg = draw_image_overlay(image_color, image_seg)

                labels_vis = draw_image_grid(
                    [image_pose, image_handmarks, image_seg],
                    ["ObjectPose", "Handmarks", "Segmentation"],
                )

                # Display visualization
                plt.imshow(labels_vis)
                plt.title(f"{serial} - frame_{frame_id:06d}")
                plt.axis("off")
                plt.tight_layout()
                plt.show()
                plt.close()

                exit()
