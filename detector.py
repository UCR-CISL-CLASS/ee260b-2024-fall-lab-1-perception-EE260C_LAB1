import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from utils.transform import Transform
import pdb

class Detector:
    def __init__(self):
        # Initialize the detection model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Camera parameters (these should be adjusted based on your specific setup)
        self.camera_height = 1.5  # meters
        self.camera_pitch = -15  # degrees
        self.camera_fov = 90  # degrees
        self.image_width = 800
        self.image_height = 600
        self.sensors_list=[]
    
    def sensors(self):
        self.sensors_list = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 800, 'height': 600, 'fov': 100, 'sensor_tick': 1, 'id': 'RGB'},
            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
            'range': 100, 'rotation_frequency': 20, 'channels': 32, 'upper_fov': 15, 'lower_fov': -30,
            'points_per_second': 500000, 'id': 'LIDAR'}
        ]
        return self.sensors_list
    
    def get_camera_parameters(self):
        for sensor in self.sensors_list:
            if sensor['id'] == 'RGB':
                return {
                    'width': sensor['width'],
                    'height': sensor['height'],
                    'fov': sensor['fov']
                }
        return None  # Return None if RGB camera is not found
    
    def build_projection_matrix(self, w, h, fov):
        '''
        Computing K:
                    [fx  0  cx]
                    [0  fy  cy]
                    [0   0   1]
        '''
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K

    def detect(self, sensor_data):
        det_boxes = []
        det_class = []
        det_score = []

        for sensor_id, (timestamp, data) in sensor_data.items():
            if sensor_id in ['RGB', 'Left', 'Right']:
                boxes, classes, scores = self._detect_on_image(data)
                # Convert 2D boxes to 3D world coordinates
                world_boxes = self._convert_to_world_coordinates(boxes, sensor_id)
                det_boxes.extend(world_boxes)
                det_class.extend(classes)
                det_score.extend(scores)
            elif sensor_id == 'LIDAR':
                lidar_boxes, lidar_classes, lidar_scores = self._detect_on_lidar(data)
                det_boxes.extend(lidar_boxes)
                det_class.extend(lidar_classes)
                det_score.extend(lidar_scores)

        return {
            'det_boxes': np.array(det_boxes),
            'det_class': np.array(det_class),
            'det_score': np.array(det_score)
        }

    def _detect_on_image(self, image):
        # Convert RGBA to RGB
        image_rgb = image[:, :, :3]
        
        # Preprocess the image
        image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(self.device)

        # Perform inference
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Process predictions
        boxes = []
        classes = []
        scores = []
        for i, score in enumerate(predictions[0]['scores']):
            if score > 0.5:
                box = predictions[0]['boxes'][i].cpu().numpy()
                label = predictions[0]['labels'][i].cpu().numpy()
                
                # Map COCO classes to our classes (0: vehicle, 1: pedestrian, 2: cyclist)
                if label in [3, 4, 6, 8]:  # COCO vehicle classes
                    class_id = 0
                elif label == 1:  # COCO person class
                    class_id = 1
                elif label == 2:  # COCO bicycle class
                    class_id = 2
                else:
                    continue  # Skip other classes

                boxes.append(box)
                classes.append(class_id)
                scores.append(score.cpu().numpy())

        return boxes, classes, scores

    def _detect_on_lidar(self, lidar_data):
        # Placeholder implementation
        return [], [], []

    def _convert_to_world_coordinates(self, boxes, camera_id):
        world_boxes = []
        
        # Get camera parameters
        w, h = self.image_width, self.image_height
        fov = self.camera_fov
        
        # Build projection matrix
        K = self.build_projection_matrix(w, h, fov)
        
        # Define camera transform
        if camera_id == 'RGB':
            camera_transform = Transform(0.7, 0.0, 1.60, 0.0, 0.0, 0.0)
        elif camera_id == 'Left':
            camera_transform = Transform(0.7, 0.5, 1.60, 15, 0.0, 0.0)
        elif camera_id == 'Right':
            camera_transform = Transform(0.7, -0.5, 1.60, -15, 0.0, 0.0)
        else:
            raise ValueError(f"Unknown camera_id: {camera_id}")

        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Estimate depth (you might want to improve this)
            depth = self._estimate_depth(x2 - x1, y2 - y1)
            
            # Get 3D points in camera coordinates
            points_3d = np.array([
                [(x1 - K[0, 2]) * depth / K[0, 0], (y1 - K[1, 2]) * depth / K[1, 1], depth],
                [(x2 - K[0, 2]) * depth / K[0, 0], (y1 - K[1, 2]) * depth / K[1, 1], depth],
                [(x2 - K[0, 2]) * depth / K[0, 0], (y2 - K[1, 2]) * depth / K[1, 1], depth],
                [(x1 - K[0, 2]) * depth / K[0, 0], (y2 - K[1, 2]) * depth / K[1, 1], depth]
            ])
            
            # Transform points to world coordinates
            world_points = camera_transform.transform(points_3d)
            
            # Create a 3D bounding box
            min_point = np.min(world_points, axis=0)
            max_point = np.max(world_points, axis=0)
            
            world_box = np.array([
                [min_point[0], min_point[1], min_point[2]],
                [max_point[0], min_point[1], min_point[2]],
                [max_point[0], max_point[1], min_point[2]],
                [min_point[0], max_point[1], min_point[2]],
                [min_point[0], min_point[1], max_point[2]],
                [max_point[0], min_point[1], max_point[2]],
                [max_point[0], max_point[1], max_point[2]],
                [min_point[0], max_point[1], max_point[2]]
            ])
            
            world_boxes.append(world_box)

        return world_boxes


    def _estimate_depth(self, width, height):
        # This is a very simple depth estimation. You might want to use a more sophisticated method.
        # Assuming the average car width is about 2 meters
        estimated_depth = 2 * self.image_width / (width * np.tan(np.radians(self.camera_fov / 2)))
        return estimated_depth

    def _image_to_camera(self, x, y, depth):
        # Convert image coordinates to camera coordinates
        fx = self.image_width / (2 * np.tan(np.radians(self.camera_fov / 2)))
        fy = fx
        cx = self.image_width / 2
        cy = self.image_height / 2

        x_cam = (x - cx) * depth / fx
        y_cam = (y - cy) * depth / fy
        z_cam = depth

        return np.array([x_cam, y_cam, z_cam])

    def _camera_to_world(self, camera_point, camera_id):
        # Assuming the camera is mounted at the front of the car
        camera_transform = Transform(0, 0, self.camera_height, 0, self.camera_pitch, 0)
        
        # If we have multiple cameras, we might need to adjust the transform
        if camera_id == 'Right':
            camera_transform = Transform(0, -0.5, self.camera_height, -15, self.camera_pitch, 0)
        elif camera_id == 'Left':
            camera_transform = Transform(0, 0.5, self.camera_height, 15, self.camera_pitch, 0)

        world_point = camera_transform.transform(camera_point.reshape(1, 3))
        return world_point[0]

    def _create_3d_box(self, center, width, height, depth):
        # This is a simple 3D box creation. You might want to refine this based on your needs.
        half_width = width / 2
        half_height = height / 2
        half_depth = depth / 2

        box_corners = np.array([
            [-half_width, -half_height, -half_depth],
            [half_width, -half_height, -half_depth],
            [half_width, half_height, -half_depth],
            [-half_width, half_height, -half_depth],
            [-half_width, -half_height, half_depth],
            [half_width, -half_height, half_depth],
            [half_width, half_height, half_depth],
            [-half_width, half_height, half_depth]
        ])

        # Transform box corners to world coordinates
        world_corners = box_corners + center

        return world_corners

