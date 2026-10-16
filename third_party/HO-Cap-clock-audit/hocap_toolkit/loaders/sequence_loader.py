import numpy as np
import torch
from pathlib import Path
from hocap_toolkit.utils import read_data_from_yaml, read_rgb_image, read_depth_image


class SequenceLoader:
    """
    Class for loading and processing sequence data.

    Supports loading MANO and object layers, along with their poses, intrinsics,
    extrinsics, and metadata required for 3D reconstruction and analysis.
    """

    def __init__(self, sequence_folder: str, device: str = "cuda"):
        """
        Initializes the SequenceLoader object.

        Args:
            sequence_folder (str): The path to the sequence folder.
            device (str): The device to run computations on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self._data_folder = Path(sequence_folder)
        self._calib_folder = self._data_folder.parent.parent / "calibration"
        self._models_folder = self._data_folder.parent.parent / "models"
        self._device = device

        # Crop limits in world frame, [x_min, x_max, y_min, y_max, z_min, z_max]
        self._crop_lim = [-0.60, +0.60, -0.35, +0.35, -0.01, +0.80]

        # Load metadata
        self._load_metadata()

        # Create mapping from 2D coordinates to 3D rays
        self._rays = self._create_3d_rays()

        # Create projection matrices from camera to master/world
        self._M2world = torch.bmm(self._rs_Ks, self._rs_RTs_inv[:, :3, :])

        # Initialize points, colors, and masks
        self._frame_id = -1
        self._points = torch.zeros(
            (len(self._rs_serials), self._rs_height * self._rs_width, 3),
            dtype=torch.float32,
            device=self._device,
        )
        self._colors = torch.zeros(
            (len(self._rs_serials), self._rs_height * self._rs_width, 3),
            dtype=torch.float32,
            device=self._device,
        )
        self._masks = torch.zeros(
            (len(self._rs_serials), self._rs_height * self._rs_width),
            dtype=torch.bool,
            device=self._device,
        )

    def _load_metadata(self):
        data = read_data_from_yaml(self._data_folder / "meta.yaml")

        self._num_frames = data["num_frames"]
        self._object_ids = data["object_ids"]
        self._mano_sides = data["mano_sides"]
        self._task_id = data["task_id"]
        self._subject_id = data["subject_id"]
        # RealSense camera metadata
        self._rs_serials = data["realsense"]["serials"]
        self._rs_width = data["realsense"]["width"]
        self._rs_height = data["realsense"]["height"]
        self._num_cams = len(self._rs_serials)
        # HoloLens metadata
        self._hl_serial = data["hololens"]["serial"]
        self._hl_pv_width = data["hololens"]["pv_width"]
        self._hl_pv_height = data["hololens"]["pv_height"]
        # Object models file paths
        self._object_textured_files = [
            self._models_folder / obj_id / "textured_mesh.obj"
            for obj_id in self._object_ids
        ]
        self._object_cleaned_files = [
            self._models_folder / obj_id / "cleaned_mesh_10000.obj"
            for obj_id in self._object_ids
        ]

        # Load camera intrinsics
        self._load_intrinsics()

        # Load rs camera extrinsics
        self._load_extrinsics(data["extrinsics"])

        # Load MANO shape parameters
        self._mano_beta = self._load_mano_beta()

    def _load_intrinsics(self):
        def read_K_from_yaml(serial, cam_type="color"):
            yaml_file = self._calib_folder / "intrinsics" / f"{serial}.yaml"
            data = read_data_from_yaml(yaml_file)[cam_type]
            K = np.array(
                [
                    [data["fx"], 0.0, data["ppx"]],
                    [0.0, data["fy"], data["ppy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            return K

        rs_Ks = np.stack(
            [read_K_from_yaml(serial) for serial in self._rs_serials], axis=0
        )
        rs_Ks_inv = np.stack([np.linalg.inv(K) for K in rs_Ks], axis=0)

        hl_K = read_K_from_yaml(self._hl_serial)
        hl_K_inv = np.linalg.inv(hl_K)

        # Convert intrinsics to torch tensors
        self._rs_Ks = torch.from_numpy(rs_Ks).to(self._device)
        self._rs_Ks_inv = torch.from_numpy(rs_Ks_inv).to(self._device)
        self._hl_K = torch.from_numpy(hl_K).to(self._device)
        self._hl_K_inv = torch.from_numpy(hl_K_inv).to(self._device)

    def _load_extrinsics(self, file_name):
        def create_mat(values):
            return np.array(
                [values[0:4], values[4:8], values[8:12], [0, 0, 0, 1]], dtype=np.float32
            )

        data = read_data_from_yaml(self._calib_folder / "extrinsics" / f"{file_name}")

        # Read rs_master serial
        self._rs_master = data["rs_master"]

        # Create extrinsics matrices
        extrinsics = data["extrinsics"]
        tag_0 = create_mat(extrinsics["tag_0"])
        tag_0_inv = np.linalg.inv(tag_0)
        tag_1 = create_mat(extrinsics["tag_1"])
        tag_1_inv = np.linalg.inv(tag_1)
        extr2master = np.stack(
            [create_mat(extrinsics[s]) for s in self._rs_serials], axis=0
        )
        extr2master_inv = np.stack([np.linalg.inv(t) for t in extr2master], axis=0)
        extr2world = np.stack([tag_1_inv @ t for t in extr2master], axis=0)
        extr2world_inv = np.stack([np.linalg.inv(t) for t in extr2world], axis=0)

        # Convert extrinsics to torch tensors
        self._tag_0 = torch.from_numpy(tag_0).to(self._device)
        self._tag_0_inv = torch.from_numpy(tag_0_inv).to(self._device)
        self._tag_1 = torch.from_numpy(tag_1).to(self._device)
        self._tag_1_inv = torch.from_numpy(tag_1_inv).to(self._device)
        self._extr2master = torch.from_numpy(extr2master).to(self._device)
        self._extr2master_inv = torch.from_numpy(extr2master_inv).to(self._device)
        self._rs_RTs = torch.from_numpy(extr2world).to(self._device)
        self._rs_RTs_inv = torch.from_numpy(extr2world_inv).to(self._device)

    def _load_mano_beta(self) -> torch.Tensor:
        file_path = self._calib_folder / "mano" / f"{self._subject_id}.yaml"
        data = read_data_from_yaml(file_path)
        return torch.tensor(data["betas"], dtype=torch.float32, device=self._device)

    def _create_3d_rays(self) -> torch.Tensor:
        """Creates 3D rays for deprojecting depth images to 3D space."""

        def create_2d_coords() -> torch.Tensor:
            xv, yv = torch.meshgrid(
                torch.arange(self._rs_width),
                torch.arange(self._rs_height),
                indexing="xy",
            )
            coord_2d = torch.stack(
                (xv, yv, torch.ones_like(xv)), dim=0
            ).float()  # (3, H, W)
            coords_2d = (
                coord_2d.unsqueeze(0)
                .repeat(self._num_cams, 1, 1, 1)
                .view(self._num_cams, 3, -1)
            )  # (N, 3, H*W)
            coords_2d = coords_2d.to(self._device)
            return coords_2d

        coords_2d = create_2d_coords()
        return torch.bmm(self._rs_Ks_inv, coords_2d)  # (N, 3, H*W)

    def _deproject(self, colors, depths) -> tuple:
        """
        Deprojects depth images to 3D points.

        Args:
            colors (np.ndarray): List of color images, [N, H, W, 3], dtype=float32.
            depths (np.ndarray): List of depth images, [N, H, W], dtype=np.float32.

        Returns:
            tuple: Colors, 3D points, and masks.
        """
        # Process color images
        colors = torch.from_numpy(colors.reshape(self._num_cams, -1, 3)).to(
            self._device
        )  # [N, H*W, 3]

        # Process depth images
        depths = torch.from_numpy(depths.reshape(self._num_cams, 1, -1)).to(
            self._device
        )  # [N, 1, H*W]

        # Deproject depth images to 3D points in camera frame
        pts_c = self._rays * depths  # [N, 3, H*W]
        # Transform 3D points from camera frame to world frame
        pts = torch.baddbmm(
            self._rs_RTs[:, :3, 3].unsqueeze(2),
            self._rs_RTs[:, :3, :3],
            pts_c,
        ).permute(
            0, 2, 1
        )  # (N, H*W, 3)

        # Crop 3D points
        mx1 = pts[..., 0] > self._crop_lim[0]
        mx2 = pts[..., 0] < self._crop_lim[1]
        my1 = pts[..., 1] > self._crop_lim[2]
        my2 = pts[..., 1] < self._crop_lim[3]
        mz1 = pts[..., 2] > self._crop_lim[4]
        mz2 = pts[..., 2] < self._crop_lim[5]
        masks = mx1 & mx2 & my1 & my2 & mz1 & mz2

        return colors, pts, masks

    def _update_pcd(self, frame_id: int):
        """Update point cloud data."""
        colors, points, masks = self._deproject(
            np.stack(
                [self.get_rgb_image(frame_id, serial) for serial in self._rs_serials],
                axis=0,
                dtype=np.float32,
            )
            / 255.0,
            np.stack(
                [self.get_depth_image(frame_id, serial) for serial in self._rs_serials],
                axis=0,
                dtype=np.float32,
            ),
        )
        self._points.copy_(points)
        self._colors.copy_(colors)
        self._masks.copy_(masks)

    def get_rgb_image(self, frame_id: int, serial: str) -> np.ndarray:
        """Get RGB image in numpy format, dtype=uint8, [H, W, 3]."""
        image_file = self._data_folder / f"{serial}/color_{frame_id:06d}.jpg"
        return read_rgb_image(image_file)

    def get_depth_image(self, frame_id: int, serial: str) -> np.ndarray:
        """Get depth image in numpy format, dtype=uint16, [H, W]."""
        image_file = self._data_folder / f"{serial}/depth_{frame_id:06d}.png"
        return read_depth_image(image_file, scale=1000.0)

    def get_image_label(self, frame_id: int, serial: str) -> dict:
        """Get image label data."""
        label_file = self._data_folder / f"{serial}/label_{frame_id:06d}.npz"
        if not label_file.exists():
            return {}
        return np.load(label_file)

    def step(self):
        """Step to the next frame."""
        self._frame_id = (self._frame_id + 1) % self._num_frames
        self._update_pcd(self._frame_id)

    def step_by_frame_id(self, frame_id: int):
        """Step to a specific frame."""
        self._frame_id = frame_id % self._num_frames
        self._update_pcd(self._frame_id)

    @property
    def object_ids(self) -> list:
        return self._object_ids

    @property
    def subject_id(self) -> str:
        return self._subject_id

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def rs_width(self) -> int:
        return self._rs_width

    @property
    def rs_height(self) -> int:
        return self._rs_height

    @property
    def rs_serials(self) -> list:
        return self._rs_serials

    @property
    def rs_master(self) -> str:
        return self._rs_master

    @property
    def holo_pv_width(self) -> int:
        return self._hl_pv_width

    @property
    def holo_pv_height(self) -> int:
        return self._hl_pv_height

    @property
    def holo_serial(self) -> list:
        return self._hl_serial

    @property
    def mano_beta(self) -> torch.Tensor:
        return self._mano_beta

    @property
    def mano_sides(self) -> list:
        return self._mano_sides

    @property
    def rs_Ks(self) -> torch.Tensor:
        return self._rs_Ks

    @property
    def rs_Ks_inv(self) -> torch.Tensor:
        return self._rs_Ks_inv

    @property
    def rs_RTs(self) -> torch.Tensor:
        return self._rs_RTs

    @property
    def rs_RTs_inv(self) -> torch.Tensor:
        return self._rs_RTs_inv

    @property
    def tag_0(self) -> torch.Tensor:
        """tag_0 to rs_master transformation matrix"""
        return self._tag_0

    @property
    def tag_0_inv(self) -> torch.Tensor:
        """rs_master to tag_0 transformation matrix"""
        return self._tag_0_inv

    @property
    def tag_1(self) -> torch.Tensor:
        """tag_1 to rs_master transformation matrix"""
        return self._tag_1

    @property
    def tag_1_inv(self) -> torch.Tensor:
        """rs_master to tag_1 transformation matrix"""
        return self._tag_1_inv

    @property
    def M(self) -> torch.Tensor:
        """camera to world transformation matrix"""
        return self._M2world

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @property
    def object_textured_mesh_files(self) -> list:
        return [
            str(self._models_folder / f"{object_id}/textured_mesh.obj")
            for object_id in self._object_ids
        ]

    @property
    def object_cleaned_mesh_files(self) -> list:
        return [
            str(self._models_folder / f"{object_id}/cleaned_mesh_10000.obj")
            for object_id in self._object_ids
        ]

    @property
    def points(self) -> torch.Tensor:
        return self._points

    @property
    def colors(self) -> torch.Tensor:
        return self._colors

    @property
    def masks(self) -> torch.Tensor:
        return self._masks
