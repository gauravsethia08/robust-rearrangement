import furniture_bench
import cv2
import gym
import torch
import numpy as np
from transitions import Machine
from transitions.extensions import GraphMachine
from scipy.spatial.transform import Rotation as R


class FurnitureBench:

    def __init__(self):
        self.env = gym.make("FurnitureSimBiManual-v0",
                            furniture="square_table",
                            num_envs = 1,
                            resize_img=False,
                            init_assembled=False,
                            record=True,
                            headless=False,
                            save_camera_input=False,
                            randomness="low",
                            high_random_idx=0,
                            act_rot_repr="quat",
                            action_type="pos",
                            compute_device_id=0,
                            graphics_device_id=0,
                            ee_laser=True)
        
        self.ob = self.env.reset()
        self.done = False
       
    def step(self, action):
        self.ob, _, self.done, _ = self.env.step(action)
        return self.ob, self.done
     
    def reset(self):
        return self.env.reset()
    
    def close(self):
        self.env.close()

    def action_tensor(self, ac, num_envs=1):
        if isinstance(ac, (list, np.ndarray)):
            return torch.tensor(ac).float().to(self.env.device)

        ac = ac.clone()
        if len(ac.shape) == 1:
            ac = ac[None]
        return ac.tile(num_envs, 1).float().to(self.env.device)

    def take_action(self, trans1, euler1, gripper1, trans2, euler2, gripper2, 
                    steps=20, action_name="Action", check_success=True):
        
        quat1 = R.from_euler('xyz', euler1, degrees=True).as_quat()
        quat2 = R.from_euler('xyz', euler2, degrees=True).as_quat()
        
        print("Trans1: ", trans1)
        print("Euler1: ", euler1)
        print("Quat1: ", quat1)
        print("Gripper1: ", gripper1)
        print("Trans2: ", trans2)
        print("Euler2: ", euler2)
        print("Quat2: ", quat2)
        print("Gripper2: ", gripper2)
        
        action = [[[trans1[0], trans1[1], trans1[2], quat1[0], quat1[1], quat1[2], quat1[3], gripper1],
                  [trans2[0], trans2[1], trans2[2], quat2[0], quat2[1], quat2[2], quat2[3], gripper2]]]
        
        print(action)
        print("Action Shape", np.shape(action))
        action = self.action_tensor(action)

        for _ in range(steps):
            print(f"Action Taken - {action_name}") 
            self.step(action)
        
        # Check if the action was successful
        if check_success:
            return self.check_success(trans1, euler1, gripper1) and self.check_success(trans2, euler2, gripper2)
        else:
            return None

    def convert_trans_quat_to_robot_coords(self, pose):
        # Convert trans quat to matrix
        H = np.eye(4)
        H[:3, :3] = R.from_quat(pose[3:]).as_matrix()
        H[:3, 3] = pose[:3]

        # Transform to robot coords
        H_robot = self.env.april_to_robot_mat.cpu().numpy() @ H 

        # Get the translation and quaternion
        trans = H_robot[:3, 3]
        euler = R.from_matrix(H_robot[:3, :3]).as_euler('xyz', degrees=True)

        return trans, euler
    
    def convert_trans_quat_to_robot_coords_2(self, pose):
        # Convert trans quat to matrix
        H = np.eye(4)
        H[:3, :3] = R.from_quat(pose[3:]).as_matrix()
        H[:3, 3] = pose[:3]

        # Transform to robot coords
        H_robot = self.env.april_to_robot_mat_2.cpu().numpy() @ H 

        # Get the translation and quaternion
        trans = H_robot[:3, 3]
        euler = R.from_matrix(H_robot[:3, :3]).as_euler('xyz', degrees=True)

        return trans, euler

    def get_obj_pose_in_sim(self):
        # Get table and leg poses in apriltag frame
        table_pose = self.ob['parts_poses'].squeeze(0).cpu().numpy()[:7]
        leg_poses = self.ob['parts_poses'].squeeze(0).cpu().numpy()[7:14]

        # Get the translation and quaternion of the table
        self.table_trans, self.table_euler = self.convert_trans_quat_to_robot_coords(table_pose)
        self.leg_trans, self.leg_euler = self.convert_trans_quat_to_robot_coords(leg_poses)

        return [self.table_trans, self.table_euler], [self.leg_trans, self.leg_euler]
    
    def get_obj_pose_in_sim_2(self):
        # Get table and leg poses in apriltag frame
        table_pose = self.ob['parts_poses'].squeeze(0).cpu().numpy()[:7]
        leg_poses = self.ob['parts_poses'].squeeze(0).cpu().numpy()[7:14]

        # Get the translation and quaternion of the table
        self.table_trans, self.table_euler = self.convert_trans_quat_to_robot_coords_2(table_pose)
        self.leg_trans, self.leg_euler = self.convert_trans_quat_to_robot_coords_2(leg_poses)

        return [self.table_trans, self.table_euler], [self.leg_trans, self.leg_euler]
  
    def get_robot_ee_pose(self):
        self.ee_trans = self.ob['robot_state']['ee_pos'].squeeze(0).cpu().numpy()
        self.ee_quat = self.ob['robot_state']['ee_quat'].squeeze(0).cpu().numpy()
        self.ee_gripper = self.ob['robot_state']['gripper_width'].squeeze(0).cpu().numpy() # Gripper width is different from the commanded gripper width

        self.ee_euler = R.from_quat(self.ee_quat).as_euler('xyz', degrees=True)

        return self.ee_trans, self.ee_euler, self.ee_gripper
    
    def get_robot_ee_pose_2(self):
        self.ee_trans2 = self.ob['robot_state_2']['ee_pos'].squeeze(0).cpu().numpy()
        self.ee_quat2 = self.ob['robot_state_2']['ee_quat'].squeeze(0).cpu().numpy()
        self.ee_gripper2 = self.ob['robot_state_2']['gripper_width'].squeeze(0).cpu().numpy()

        self.ee_euler2 = R.from_quat(self.ee_quat2).as_euler('xyz', degrees=True)

        return self.ee_trans2, self.ee_euler2, self.ee_gripper2
    
    def check_success(self, pose, euler, gripper_width):
        self.get_robot_ee_pose()
        ee_mat = R.from_euler('xyz', self.ee_euler, degrees=True).as_matrix()
        leg_mat = R.from_euler('xyz', euler, degrees=True).as_matrix()
        r, _ = cv2.Rodrigues(ee_mat@leg_mat.T)
        trans_err = np.linalg.norm(self.ee_trans - pose)
        rot_err = np.linalg.norm(r)
        gripper_err = np.abs(self.ee_gripper - gripper_width)

        print(trans_err, rot_err, gripper_err, self.ee_gripper)

        if trans_err < 0.02 and rot_err < 0.1:
            self.success()
            return True
        else:
            self.failure()
            return False


class FurnitureAssemblyRobot(FurnitureBench):

    states = ['approach_leg',
              'orient_pickup',
              'lower_pickip',
              'pick_up',
              'lift_leg',
              'align_leg',
              'lower_leg',
              'release_leg',
              'pre_align_screw',
              'align_screw',
              'grasp_screw',
              'screw_leg',
              'check_done',
              'approach_leg_2',
              'reach_leg_2',
              'grasp_leg_2',
              'release_leg_2',
              {'name': 'done', 'final': True}]
    

    def __init__(self):
        super().__init__()
        # Define a state machine
        self.machine = GraphMachine(model=self, states=FurnitureAssemblyRobot.states, initial='approach_leg')

        # Add transitions
        # Approach Leg
        self.machine.add_transition('success', 'approach_leg', 'orient_pickup')
        self.machine.add_transition('failure', 'approach_leg', 'approach_leg')

        # Orient Pickup
        self.machine.add_transition('success', 'orient_pickup', 'lower_pickip')
        self.machine.add_transition('failure', 'orient_pickup', 'approach_leg')

        # Lower Pickup
        self.machine.add_transition('success', 'lower_pickip', 'pick_up')
        self.machine.add_transition('failure', 'lower_pickip', 'approach_leg')

        # Pick Up
        self.machine.add_transition('success', 'pick_up', 'lift_leg')
        self.machine.add_transition('failure', 'pick_up', 'approach_leg')

        # Lift Leg
        self.machine.add_transition('success', 'lift_leg', 'align_leg')
        self.machine.add_transition('failure', 'lift_leg', 'approach_leg')

        # Align Leg
        self.machine.add_transition('success', 'align_leg', 'lower_leg')
        self.machine.add_transition('failure', 'align_leg', 'approach_leg')

        # Lower Leg
        self.machine.add_transition('success', 'lower_leg', 'approach_leg_2')
        self.machine.add_transition('failure', 'lower_leg', 'approach_leg')

        # Let 2nd Arm take over
        self.machine.add_transition('success', 'approach_leg_2', 'reach_leg_2')
        self.machine.add_transition('failure', 'approach_leg_2', 'approach_leg_2')

        # Reach Leg 2
        self.machine.add_transition('success', 'reach_leg_2', 'grasp_leg_2')
        self.machine.add_transition('failure', 'reach_leg_2', 'approach_leg_2')

        # Grasp Leg 2
        self.machine.add_transition('success', 'grasp_leg_2', 'release_leg')
        self.machine.add_transition('failure', 'grasp_leg_2', 'approach_leg_2')


        # Release Leg
        self.machine.add_transition('success', 'release_leg', 'pre_align_screw')
        self.machine.add_transition('failure', 'release_leg', 'approach_leg')

        # Pre Align Screw
        self.machine.add_transition('success', 'pre_align_screw', 'align_screw')
        self.machine.add_transition('failure', 'pre_align_screw', 'approach_leg')

        # Align Screw
        self.machine.add_transition('success', 'align_screw', 'grasp_screw')
        self.machine.add_transition('failure', 'align_screw', 'release_leg')

        # Grasp Screw
        # self.machine.add_transition('success', 'grasp_screw', 'screw_leg')
        # self.machine.add_transition('failure', 'grasp_screw', 'release_leg')
        self.machine.add_transition('success', 'grasp_screw', 'release_leg_2')
        self.machine.add_transition('failure', 'grasp_screw', 'release_leg')

        # Release Leg 2
        self.machine.add_transition('success', 'release_leg_2', 'screw_leg')
        self.machine.add_transition('failure', 'release_leg_2', 'approach_leg')

        # Screw Leg
        # self.machine.add_transition('success', 'screw_leg', 'check_done')
        self.machine.add_transition('success', 'screw_leg', 'approach_leg_2')
        self.machine.add_transition('failure', 'screw_leg', 'pre_align_screw')

        # Done
        self.machine.add_transition('success', 'check_done', 'done')
        self.machine.add_transition('failure', 'check_done', 'pre_align_screw')

        self.retry_count = 0

        self.machine.get_graph().draw('furniture_assembly_robot.png', prog='dot')


    # Add action for each state
    def on_enter_approach_leg(self):
        # Use the leg position to approach the leg
        [_, _], [leg_trans, leg_euler] = self.get_obj_pose_in_sim()
        
        trans1 = [leg_trans[0]+0.025, leg_trans[1], leg_trans[2]+0.1]
        euler1 = [180, 0, 0]
        gripper1 = -1
        
        ee_pos, ee_euler, ee_gripper = self.get_robot_ee_pose_2()
        trans2 = ee_pos
        euler2 = ee_euler
        gripper2 = -1 #ee_gripper

        success = self.take_action(trans1=trans1, euler1=euler1, gripper1=gripper1, 
                                   trans2=trans2, euler2=euler2, gripper2=gripper2, steps=20, 
                                   action_name="Approaching Leg") 
        
    def on_exit_approach_leg(self):
        self.retry_count += 1
        if self.retry_count > 3:
            raise Exception("Failed to approach leg")

    def on_enter_orient_pickup(self):
        # Use the leg position to orient the gripper
        [_, _], [leg_trans, leg_euler] = self.get_obj_pose_in_sim()
        
        trans1 = [leg_trans[0]+0.025, leg_trans[1], leg_trans[2]]
        euler1 = [180, 65, 0]
        gripper1 = -1  # Open gripper

        ee_pos, ee_euler, ee_gripper = self.get_robot_ee_pose_2()
        trans2 = ee_pos
        euler2 = ee_euler
        gripper2 = -1 #ee_gripper
        
        success = self.take_action(trans1=trans1, euler1=euler1, gripper1=gripper1, 
                                    trans2=trans2, euler2=euler2, gripper2=gripper2, steps=20, 
                                    action_name="Orienting Pickup")
        
    
    def on_enter_lower_pickip(self):
        # Use the leg position to lowering for pickup
        [_, _], [leg_trans, leg_euler] = self.get_obj_pose_in_sim()
        
        trans1 = [leg_trans[0]+0.025, leg_trans[1], leg_trans[2]]
        euler1 = [180, 65, 0]
        gripper1 = -1  # Open gripper
        
        ee_pos, ee_euler, ee_gripper = self.get_robot_ee_pose_2()
        trans2 = ee_pos
        euler2 = ee_euler
        gripper2 = -1 #ee_gripper
        
        success = self.take_action(trans1=trans1, euler1=euler1, gripper1=gripper1, 
                                    trans2=trans2, euler2=euler2, gripper2=gripper2, steps=20, 
                                    action_name="Lowering Pickup")


    def on_enter_pick_up(self):
        # Use the leg position to pick up the leg
        [_, _], [leg_trans, leg_euler] = self.get_obj_pose_in_sim()
        
        trans1 = [leg_trans[0]+0.025, leg_trans[1], leg_trans[2]]
        euler1 = [180, 65, 0]
        gripper1 = 0.3  # Closed gripper
        
        ee_pos, ee_euler, ee_gripper = self.get_robot_ee_pose_2()
        trans2 = ee_pos
        euler2 = ee_euler
        gripper2 = -1 #ee_gripper
        
        success = self.take_action(trans1=trans1, euler1=euler1, gripper1=gripper1, 
                                    trans2=trans2, euler2=euler2, gripper2=gripper2, steps=20, 
                                    action_name="Pick Up")
 
    
    def on_enter_lift_leg(self):
        # Use the leg position to lift the leg
        [_, _], [leg_trans, leg_euler] = self.get_obj_pose_in_sim()
        
        trans1 = [leg_trans[0], leg_trans[1], 0.25]
        euler1 = [180, 0, 0]
        gripper1 = 0.3  # Closed gripper
        
        ee_pos, ee_euler, ee_gripper = self.get_robot_ee_pose_2()
        trans2 = ee_pos
        euler2 = ee_euler
        gripper2 = -1 #ee_gripper
        
        success = self.take_action(trans1=trans1, euler1=euler1, gripper1=gripper1, 
                                    trans2=trans2, euler2=euler2, gripper2=gripper2, steps=20, 
                                    action_name="Lift Leg")
    

    def on_enter_align_leg(self):
        # Use the leg position to align the leg to the table
        [table_trans, table_euler], [_, _] = self.get_obj_pose_in_sim()

        # Displacement from the corner of the table to the screw hole center
        X_disp = 0.08 - 0.0225 + 0.015 #- 0.0125 #0.0545#0.08 - 0.03
        Y_disp = 0.08 - 0.0205 + 0.015
        table_theta = -90#table_euler[2]

        print("Table Theta: ", table_trans)
        # Get the position of the screw hole center
        # 0.59 + 0.01, -0.0625 
        self.x_pos = table_trans[0] + (X_disp*np.cos(np.deg2rad(table_theta)) - Y_disp*np.sin(np.deg2rad(table_theta)))
        self.y_pos = table_trans[1] + (X_disp*np.sin(np.deg2rad(table_theta)) + Y_disp*np.cos(np.deg2rad(table_theta)))
        
        trans1 = [self.x_pos, self.y_pos, 0.25]
        euler1 = [180, 65-90, 0]
        gripper1 = 0.3

        ee_pos, ee_euler, ee_gripper = self.get_robot_ee_pose_2()
        trans2 = ee_pos
        euler2 = ee_euler
        gripper2 = -1 #ee_gripper
        
        success = self.take_action(trans1=trans1, euler1=euler1, gripper1=gripper1, 
                                    trans2=trans2, euler2=euler2, gripper2=gripper2, steps=20,
                                    action_name="Align Leg")


    def on_enter_lower_leg(self):
        # # Use the leg position to align the leg to the table
        [table_trans, table_euler], [_, _] = self.get_obj_pose_in_sim()

        # # Displacement from the corner of the table to the screw hole center
        X_disp = 0.08 - 0.0225 #- 0.0125
        Y_disp = 0.08 - 0.0205 + 0.015
        table_theta = table_euler[2]

        # Get the position of the screw hole center
        self.x_pos = table_trans[0] + (X_disp*np.cos(np.deg2rad(table_theta)) - Y_disp*np.sin(np.deg2rad(table_theta)))
        self.y_pos = table_trans[1] + (X_disp*np.sin(np.deg2rad(table_theta)) + Y_disp*np.cos(np.deg2rad(table_theta)))

        # Use the leg position to lower the leg       
        trans1 = [self.x_pos, self.y_pos, 0.1]
        euler1 = [180, 65-90, 0]
        gripper1 = 0.3

        ee_pos, ee_euler, ee_gripper = self.get_robot_ee_pose_2()
        trans2 = ee_pos
        euler2 = ee_euler
        gripper2 = -1 #ee_gripper
        
        success = self.take_action(trans1=trans1, euler1=euler1, gripper1=gripper1, 
                                    trans2=trans2, euler2=euler2, gripper2=gripper2, steps=20,
                                    action_name="Lower Leg")

        # TODO : Check if the leg is inserted in screw hole or not
        # If not, raise and go back to align leg

    
    ##########################################################################
    # 2nd Arm Actions
    def on_enter_approach_leg_2(self):
        # Use the leg position to approach the leg
        [_, _], [leg_trans, leg_euler] = self.get_obj_pose_in_sim_2()
        ee_pos, ee_euler, ee_gripper = self.get_robot_ee_pose()
        
        trans1 = ee_pos
        euler1 = ee_euler#[180, 0, 90]
        gripper1 = ee_gripper[0]
        
        trans2 = [leg_trans[0] + 0.01, leg_trans[1] + 0.1, leg_trans[2] - 0.01]
        euler2 = [90, 90, 0]
        gripper2 = -1 #ee_gripper

        success = self.take_action(trans1=trans1, euler1=euler1, gripper1=gripper1, 
                                   trans2=trans2, euler2=euler2, gripper2=gripper2, steps=20, 
                                   action_name="Approaching Leg 2") 
        

    def on_enter_reach_leg_2(self):
        # Use the leg position to approach the leg
        [_, _], [leg_trans, leg_euler] = self.get_obj_pose_in_sim_2()
        ee_pos, ee_euler, ee_gripper = self.get_robot_ee_pose()
        
        trans1 = ee_pos
        euler1 = ee_euler#[180, 0, 90]
        gripper1 = ee_gripper[0]
        
        # trans2, euler2, gripper2 = self.get_robot_ee_pose_2()
        trans2 = [leg_trans[0] + 0.01, leg_trans[1], leg_trans[2] - 0.01]
        euler2 = [90, 90, 0]
        gripper2 = -1 #ee_gripper

        success = self.take_action(trans1=trans1, euler1=euler1, gripper1=gripper1, 
                                   trans2=trans2, euler2=euler2, gripper2=gripper2, steps=20, 
                                   action_name="Reach Leg 2") 
        

    def on_enter_grasp_leg_2(self):
        # Use the leg position to approach the leg
        [_, _], [leg_trans, leg_euler] = self.get_obj_pose_in_sim_2()
        ee_pos, ee_euler, ee_gripper = self.get_robot_ee_pose()
        
        trans1 = ee_pos
        euler1 = ee_euler#[180, 0, 90]
        gripper1 = ee_gripper[0]
        
        # trans2, euler2, gripper2 = self.get_robot_ee_pose_2()
        trans2 = [leg_trans[0] + 0.01, leg_trans[1], leg_trans[2] - 0.01]
        euler2 = [90, 90, 0]
        gripper2 = 1 #ee_gripper

        success = self.take_action(trans1=trans1, euler1=euler1, gripper1=gripper1, 
                                   trans2=trans2, euler2=euler2, gripper2=gripper2, steps=20, 
                                   action_name="Grasp Leg 2") 
        
   
    def on_enter_release_leg_2(self):
        # [_, _], [leg_trans, leg_euler] = self.get_obj_pose_in_sim_2e()
        # Use the leg position to release the leg
        ee_pos, ee_euler, ee_gripper = self.get_robot_ee_pose()
        ee_pos2, ee_euler2, ee_gripper2 = self.get_robot_ee_pose_2()

        trans1 = ee_pos
        euler1 = ee_euler
        gripper1 = ee_gripper[0]

        trans2 = ee_pos2
        trans2[1] += 0.1
        euler2 = ee_euler2
        gripper2 = -1 #ee_gripper2

        success = self.take_action(trans1=trans1, euler1=euler1, gripper1=gripper1,
                                    trans2=trans2, euler2=euler2, gripper2=gripper2, steps=20,
                                    action_name="Release Leg 2", check_success=False)
        
        self.success()
        
        
    ##########################################################################
    def on_enter_release_leg(self):
        [_, _], [leg_trans, leg_euler] = self.get_obj_pose_in_sim()
        # Use the leg position to release the leg
        ee_pos, ee_euler, ee_gripper = self.get_robot_ee_pose()
        trans1 = ee_pos #[self.x_pos, self.y_pos, 0.1]
        euler1 = ee_euler #[180, 65-90, 0]
        gripper1 = -1

        ee_pos, ee_euler, ee_gripper = self.get_robot_ee_pose_2()
        trans2 = ee_pos
        euler2 = ee_euler
        gripper2 = ee_gripper[0]
        
        success = self.take_action(trans1=trans1, euler1=euler1, gripper1=gripper1, 
                                    trans2=trans2, euler2=euler2, gripper2=gripper2, steps=20,
                                     action_name="Release Leg", check_success=False)

        if abs(leg_euler[0] - 90) < 10:
            self.success()
        else:
            self.failure()
    
    
    def on_enter_pre_align_screw(self):
        # Use the leg position to approach the screw
        [_, _], [trans1, leg_euler] = self.get_obj_pose_in_sim()

        trans1[2] += 0.1
        euler1 = [180, 0, 0]
        gripper1 = -1

        trans2, euler2, gripper2 = self.get_robot_ee_pose_2()
        # trans2[1] += 0.01
        # trans2[2] = trans1[2] - 0.11
        print("Trans2: ", trans2)
        success = self.take_action(trans1=trans1, euler1=euler1, gripper1=gripper1, 
                                    trans2=trans2, euler2=euler2, gripper2=gripper2[0], steps=20, 
                                    action_name="Pre Align Screw")
        

    def on_enter_align_screw(self):
        # Use the leg position to align the screw to the leg
        [_, _], [trans1, leg_euler] = self.get_obj_pose_in_sim()

        trans1[2] += 0.02
        euler1 = [180, 0, 0]
        gripper1 = -1

        ee_pos, ee_euler, ee_gripper = self.get_robot_ee_pose_2()
        trans2 = ee_pos
        euler2 = ee_euler
        gripper2 = 1 #ee_gripper
        
        success = self.take_action(trans1=trans1, euler1=euler1, gripper1=gripper1, 
                                    trans2=trans2, euler2=euler2, gripper2=gripper2, steps=20, 
                                    action_name="Align Screw")

    
    def on_enter_grasp_screw(self):
        # Use the leg position to grasp the screw
        [_, _], [trans1, leg_euler] = self.get_obj_pose_in_sim()

        trans1[2] += 0.02
        euler1 = [180, 0, 0]
        gripper1 = 1

        ee_pos, ee_euler, ee_gripper = self.get_robot_ee_pose_2()
        trans2 = ee_pos
        euler2 = ee_euler
        gripper2 = 1 #ee_gripper
        
        success = self.take_action(trans1=trans1, euler1=euler1, gripper1=gripper1, 
                                    trans2=trans2, euler2=euler2, gripper2=gripper2, steps=20,
                                      action_name="Grasp Screw")

    
    def on_enter_screw_leg(self):
        # Use the leg position to screw the leg
        [_, _], [trans1, leg_euler] = self.get_obj_pose_in_sim()

        euler1 = [180, 0, -90]
        gripper1 = 1

        trans1, _, _ = self.get_robot_ee_pose()
        trans1[2] -= 0.01

        ee_pos, ee_euler, ee_gripper = self.get_robot_ee_pose_2()
        trans2 = ee_pos
        euler2 = ee_euler
        gripper2 = -1 #ee_gripper
        
        success = self.take_action(trans1=trans1, euler1=euler1, gripper1=gripper1, 
                                    trans2=trans2, euler2=euler2, gripper2=gripper2, steps=20,
                                    action_name="Screw Leg")


    def on_enter_check_done(self):
        # Check if the assembly is done
        [_, _], [leg_trans, leg_euler] = self.get_obj_pose_in_sim()

        # if abs(leg_euler[0] - 90) < 10:
        #     self.success()
        # else:
        self.failure()





robot = FurnitureAssemblyRobot()
print(robot)
# robot.machine.get_graph().draw('furniture_assembly_robot.png', prog='dot')

print(robot.state)
robot.failure()

print(robot.state)