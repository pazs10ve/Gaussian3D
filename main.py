import os
import numpy as np
import cv2
import open3d as o3d

# Define a class that takes a 2D image as an input and returns a depth map
import numpy as np
import open3d as o3d

class DepthEstimator:
    def __init__(self, images):
        self.images = images
        self.depth = None

    def estimate_depth(self):
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.Gray32
        )

        for i, image in enumerate(self.images):
            # Ensure that the image has the correct format (uint8)
            color_image = image.astype(np.uint8)
            depth_image = image[:, :, 0].astype(np.float32)

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_image),
                o3d.geometry.Image(depth_image)
            )

            # Adjust your camera parameters accordingly
            intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width=image.shape[1],
                height=image.shape[0],
                fx=500.0,
                fy=500.0,
                cx=image.shape[1] / 2.0,
                cy=image.shape[0] / 2.0,
            )

            # Assuming you have a method to get the camera pose for each image
            pose = self.get_camera_pose(i)

            volume.integrate(rgbd_image, intrinsics, np.linalg.inv(pose))

        # Extract a point cloud from the volume
        pcd = volume.extract_point_cloud()

        return pcd  # or return the depth map if you created one

    def get_camera_pose(self, i):
        # Create a 4x4 identity matrix
        pose = np.eye(4)

        # Assume that the camera is moving along the z-axis
        # The amount of movement depends on the index of the image
        pose[2, 3] = i * 0.05  # Adjust this value as needed

        return pose

# Define a class that takes a folder of 2D images as an input and returns a list of depth maps
class DepthEstimatorMultiView:
    def __init__(self, folder):
        self.folder = folder
        self.images = []
        self.depths = []

    def load_images(self):
        for filename in os.listdir(self.folder):
            image = cv2.imread(os.path.join(self.folder, filename))
            self.images.append(image)

    def estimate_depths(self):
        depth_estimator = DepthEstimator(self.images)  # Pass the list of images to DepthEstimator
        depth = depth_estimator.estimate_depth()
        self.depths.append(depth)
        return self.depths
# Define a class that takes a list of images and a list of depth maps as inputs and returns a 3D point cloud
"""
class PointCloudGenerator:
    def __init__(self, images, depths):
        self.images = images
        self.depths = depths
        self.point_cloud = None

    def generate_point_cloud(self):
        # Generate a simple 3D point cloud using the depth maps and camera intrinsics
        point_cloud_list = []
        print(len(self.images))
        for i in range(0, len(self.images), 3):
            print("i = ", i)
            image = self.images[i]
            depth = self.depths[i]

            h, w = image.shape[:2]
            fx = 500.0  # Replace with your camera intrinsics
            fy = 500.0
            cx = w / 2.0
            cy = h / 2.0

            for y in range(h):
                for x in range(w):
                    # Convert pixel coordinates to 3D coordinates using depth and camera intrinsics
                    Z = depth[y, x]
                    X = (x - cx) * Z / fx
                    Y = (y - cy) * Z / fy

                    point_cloud_list.append([X, Y, Z])

        self.point_cloud = np.array(point_cloud_list)
        return self.point_cloud

"""

class PointCloudGenerator:
    def __init__(self, images, depths):
        self.images = images
        self.depths = depths
        self.point_cloud = None

    def generate_point_cloud(self):
        # Generate a simple 3D point cloud using the depth maps and camera intrinsics
        point_cloud_list = []
        print(len(self.images))
        for i in range(0, len(self.depths)):  # Adjusted the loop condition
            image = self.images[i]
            depth = self.depths[i]

            for point in np.asarray(depth.points):
                point_cloud_list.append(point)

        self.point_cloud = np.array(point_cloud_list)
        return self.point_cloud


# Main code
if __name__ == "__main__":
    folder_path = "images"
    depth_estimator_multi_view = DepthEstimatorMultiView(folder_path)
    depth_estimator_multi_view.load_images()
    depth_maps = depth_estimator_multi_view.estimate_depths()
    point_cloud_generator = PointCloudGenerator(depth_estimator_multi_view.images, depth_maps)
    point_cloud = point_cloud_generator.generate_point_cloud()
    # Visualize the point cloud using open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Set visualization parameters
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 2.0  # Adjust point size if needed
    vis.get_render_option().background_color = np.array([0, 1, 1])

    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])  # Set the point the camera is looking at
    ctr.set_front([1, 0, 0])  # Set the direction of the camera's up vector
    ctr.set_up([0, 0, 1])  # Set the direction of the camera's up vector
    ctr.set_zoom(0.8)  # Adjust zoom level if needed

    # Show the visualization
    vis.run()

