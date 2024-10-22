'''
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specifed route

This module uses a perception module to get feedback on object distance as opposed
to ground truth values.
'''

from collections import OrderedDict
import math
import random
import numpy as np
import carla
from eval import box_2_polygon, caluclate_tp_fp, eval_final_results
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
from agents.navigation.behavior_types import Cautious, Aggressive, Normal

from agents.tools.misc import get_speed, positive, is_within_distance, compute_distance
from detector import Detector  # pylint: disable=import-rror
from shapely.geometry import Polygon

from agents.navigation.sensor_interface import SensorInterface, CallBack
import cv2
import numpy as np
import pdb


class PerceptionAgent(BasicAgent):
    '''
    PeceptionAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles.
    It uses a custom perception module to control speed and brakes.
    '''

    def __init__(self, vehicle, world, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 30
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5
        self._vehicle = vehicle
        self._world = world

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

        # Initalize the detector
        self._detector = Detector()

        # Mount sensors on the ego vehicle
        # self._mount_sensors()

        # Evaluate detection results
        self.result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                            0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                            0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    def destroy(self):
        eval_final_results(self.result_stat, global_sort_detections=True)
    
    def _mount_sensors(self):
        sensor_specs = self._detector.sensors()
        for sensor_spec in sensor_specs:
            self._init_sensor(sensor_spec)

    def _init_sensor(self, sensor_spec):
        sensor_type = sensor_spec['type']
        sensor_id = sensor_spec['id']
        sensor_transform = sensor_spec.get('transform', carla.Transform())
        
        if sensor_type == 'sensor.camera.rgb':
            camera_bp = self._world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(sensor_spec['width']))
            camera_bp.set_attribute('image_size_y', str(sensor_spec['height']))
            camera_bp.set_attribute('fov', str(sensor_spec['fov']))
            sensor = self._world.spawn_actor(camera_bp, sensor_transform, attach_to=self._vehicle)
        elif sensor_type == 'sensor.lidar.ray_cast':
            lidar_bp = self._world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('channels', str(sensor_spec['channels']))
            lidar_bp.set_attribute('range', str(sensor_spec['range']))
            lidar_bp.set_attribute('points_per_second', str(sensor_spec['points_per_second']))
            sensor = self._world.spawn_actor(lidar_bp, sensor_transform, attach_to=self._vehicle)
        else:
            raise ValueError(f"Unsupported sensor type: {sensor_type}")
        
        # Register the sensor with SensorInterface
        self.sensor_interface.register_sensor(sensor_id, sensor)
        
        # Set up the callback for the sensor
        sensor.listen(CallBack(sensor_id, sensor, self.sensor_interface))

    def sensors(self):  # pylint: disable=no-self-use
        return self._detector.sensors()
    
    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30)

            # Check for tailgating
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 10]

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def gt_box_vertice_sequence(self, box):
        box = [box[1],
               box[3],
               box[7],
               box[5],
               box[0],
               box[2],
               box[6],
               box[4]]
        return np.array(box)

    def actor_detected(self, actor, detection_results, actor_class):
        """
            actor: carla.Actor
            detection_results: dict
            actor_class: 0 for vehicle, 1 for pedestrian, 2 for cyclist
        """
        # Prepare GT box
        
        gt_box = [[v.x, v.y, v.z] for v in actor.bounding_box.get_world_vertices(actor.get_transform())]
        gt_box = self.gt_box_vertice_sequence(gt_box)
        gt_polygon = box_2_polygon(gt_box)
        # Compare detection boxes with GT boxes
        if ("det_boxes" not in detection_results) or ("det_class" not in detection_results):
            return False
        det_boxes = detection_results["det_boxes"]
        classes = detection_results["det_class"]
        for obj_id in range(len(classes)):
            if classes[obj_id] != actor_class:
                continue
            polygon = box_2_polygon(det_boxes[obj_id])
            
            union = polygon.union(gt_polygon).area
            if union == 0:
                return False
            try: 
                iou = polygon.intersection(gt_polygon).area / union
                if iou > 0.5:
                    return True
            except:
                print(polygon)
        return False

    def gt_actors(self):
        """
        Get all the ground truth actors in the scene
        """
        actor_list = self._world.get_actors()
        vehicles = actor_list.filter("*vehicle*")
        walkers = actor_list.filter("*walker.pedestrian*")
        detection_results = dict()
        detection_results["det_boxes"] = []
        detection_results["det_class"] = []
        detection_results["det_score"] = []
        transform = self._vehicle.get_transform()
        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        for v in vehicles:
            if dist(v.get_location()) > 50:
                continue
            if v.id == self._vehicle.id:
                continue
            bbox = [[v.x, v.y, v.z] for v in v.bounding_box.get_world_vertices(v.get_transform())]
            bbox = self.gt_box_vertice_sequence(bbox)
            detection_results["det_boxes"].append(bbox)
            detection_results["det_class"].append(0)
            detection_results["det_score"].append(1.0)
        for w in walkers:
            if dist(w.get_location()) > 50:
                continue
            bbox = [[w.x, w.y, w.z] for w in w.bounding_box.get_world_vertices(w.get_transform())]
            bbox = self.gt_box_vertice_sequence(bbox)
            detection_results["det_boxes"].append(bbox)
            detection_results["det_class"].append(1)
            detection_results["det_score"].append(1.0)
        detection_results["det_boxes"] = np.array(detection_results["det_boxes"])
        detection_results["det_class"] = np.array(detection_results["det_class"])
        detection_results["det_score"] = np.array(detection_results["det_score"])
        return detection_results
    
    def visualize_bounding_boxes(self, image, gt_boxes, det_boxes, det_scores, camera_params):
        img = image[:, :, :3]  # Remove the alpha channel if present
        # Ensure the image is contiguous in memory
        img = np.ascontiguousarray(img)
        
        # Define colors for GT and detected boxes
        gt_color = (0, 255, 0)  # Green for ground truth
        det_color = (255, 0, 0)  # Red for detections
        
        # Draw GT boxes
        for box in gt_boxes:
            box_2d = self.project_3d_to_2d(box, camera_params)
            pdb.set_trace()
            cv2.polylines(img, [np.array(box_2d[:4], dtype=np.int32)], True, gt_color, 2)
            cv2.polylines(img, [np.array(box_2d[4:], dtype=np.int32)], True, gt_color, 2)
            for i in range(4):
                cv2.line(img, tuple(box_2d[i].astype(int)), tuple(box_2d[i+4].astype(int)), gt_color, 2)

        # Draw detected boxes
        for box, score in zip(det_boxes, det_scores):
            box_2d = self.project_3d_to_2d(box, camera_params)
            cv2.polylines(img, [np.array(box_2d[:4], dtype=np.int32)], True, det_color, 2)
            cv2.polylines(img, [np.array(box_2d[4:], dtype=np.int32)], True, det_color, 2)
            for i in range(4):
                cv2.line(img, tuple(box_2d[i].astype(int)), tuple(box_2d[i+4].astype(int)), det_color, 2)
            
            # Add detection score
            cv2.putText(img, f"{score:.2f}", tuple(box_2d[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, det_color, 2)

        return img

    def project_3d_to_2d(self, box_3d, camera_params):
        # Project 3D points to 2D image plane
        K = self._detector.build_projection_matrix(camera_params['width'], camera_params['height'], camera_params['fov'])
        box_2d = []
        for point in box_3d:
            point_camera = np.array([point[1], -point[2], point[0], 1])  # Convert to camera coordinate system
            # point_img = np.dot(K, point_camera[:3])
            # point_img = point_img[:2] / point_img[2]
            box_2d.append(point_camera)
        return np.array(box_2d)

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        sensor_data = self.get_sensor_data()
        detections = self._detector.detect(sensor_data)
        gt_detections = self.gt_actors()

        print('Perception detection:', detections)
        print('Ground Truth detection:', gt_detections)

        # Visualization
        if 'RGB' in sensor_data:
            image = sensor_data['RGB'][1]

            # pdb.set_trace()
            
            # Get camera parameters from the detector
            camera_params = self._detector.get_camera_parameters()
            if camera_params is None:
                print("Error: RGB camera parameters not found")
                return
            
            image_w = camera_params['width']
            image_h = camera_params['height']
            fov = camera_params['fov']

            # Visualize bounding boxes
            vis_image = self.visualize_bounding_boxes(image, gt_detections['det_boxes'], detections['det_boxes'], detections['det_score'], camera_params)

            # Display the image
            cv2.imshow('Bounding Boxes', vis_image)
            cv2.waitKey(1)

        # Evaluate detection results
        det_boxes = np.array([[[0,0,0],[0,1,0],[1,1,0],[1,0,0],[0,0,1],[0,1,1],[1,1,1],[1,0,1]]])
        det_score = np.array([0])
        if "det_boxes" in detections:
            det_boxes = detections["det_boxes"]
        if "det_score" in detections:
            det_score = detections["det_score"]
            
        caluclate_tp_fp(det_boxes, det_score, gt_detections["det_boxes"], self.result_stat, iou_thresh=0.3)
        caluclate_tp_fp(det_boxes, det_score, gt_detections["det_boxes"], self.result_stat, iou_thresh=0.5)
        caluclate_tp_fp(det_boxes, det_score, gt_detections["det_boxes"], self.result_stat, iou_thresh=0.7)

        self._update_information()

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # 1: Red lights and stops behavior
        if self.traffic_light_manager():
            return self.emergency_stop()

        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            if self.actor_detected(walker, gt_detections, 1):
                # Distance is computed from the center of the two cars,
                # we use bounding boxes to calculate the actual distance
                distance = w_distance - max(
                    walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                        self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

                print('walker distance:', distance)

                # Emergency brake if the car is very close.
                if distance < self._behavior.braking_distance:
                    return self.emergency_stop()

        # 2.2: Car following behaviors
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        if vehicle_state:
            if self.actor_detected(vehicle, gt_detections, 0):
                # Distance is computed from the center of the two cars,
                # we use bounding boxes to calculate the actual distance
                distance = distance - max(
                    vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                        self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
                
                print('vehicle distance:', distance)

                # Emergency brake if the car is very close.
                if distance < self._behavior.braking_distance:
                    return self.emergency_stop()
                else:
                    control = self.car_following_manager(vehicle, distance)

        # 3: Intersection behavior
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - 5])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control
