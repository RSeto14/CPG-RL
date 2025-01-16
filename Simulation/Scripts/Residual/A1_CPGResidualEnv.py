import mujoco
import mujoco.viewer

import numpy as np
import random

import datetime
import pytz
import os
import sys

import imageio

import time

script_dir = os.path.dirname(__file__)    # .../A1_WS/Envs
parent_dir = os.path.dirname(script_dir)  # .../A1_WS
sys.path.append(parent_dir)

script_name = os.path.basename(__file__)[: -len(".py")]

from A1_description.model_xml import FlatXml,BoxXml

from cfg.config_residual import Config

from utils import Trajectory,Joint_q_dq,Inverse_kinematics,PD_control,Position_Velocity,Foot_force,Quaternion_to_rotation_matrix,Rotation_matrix_to_euler,IMU,CPG,Joint_safty,Box,visual_foot_pos,Forward_kinematics

class A1CPGEnv():
    
    def __init__(self,cfg=Config(),capture=False,eval=False):
        
        self.cfg = cfg
        
        self.eval = eval
        self.capture = capture
        
        self.frames = []
        self.video_width = self.cfg.frame_width
        self.video_height = self.cfg.frame_height
        self.framerate = self.cfg.framerate
            
        self.sim_time = 0
        self.max_sim_time = self.cfg.episodeLength_s
        
        self.Kp = np.full(12, self.cfg.Kp[0])
        self.Kd = np.full(12, self.cfg.Kd[0])
        
        self.d_step = self.cfg.d
        self.h = self.cfg.h[0]
        self.gc = self.cfg.gc[0]
        self.gp = self.cfg.gp[0]
        
        self.a = self.cfg.a
        
        self.mu = np.full(4, self.cfg.mu[0])
        self.mu_low = self.cfg.mu[1]
        self.mu_high = self.cfg.mu[2]
        self.omega = np.full(4, 2*np.pi*self.cfg.omega[0])
        self.omega_low = 2*np.pi*self.cfg.omega[1]
        self.omega_high = 2*np.pi*self.cfg.omega[2]
        self.psi = np.full(4, 2*np.pi*self.cfg.psi[0])
        self.psi_low = 2*np.pi*self.cfg.psi[1]
        self.psi_high = 2*np.pi*self.cfg.psi[2]
        
        self.r = np.full(4, self.cfg.mu[0])
        self.r_dot = np.full(4, 0)
        self.r_ddot = np.full(4, 0)
        self.theta = np.array([0,np.pi,np.pi,0])
        self.phi = np.array([0,0,0,0])
        
        self.e_q = np.zeros(12)
        self.e_dq = np.full(12, self.cfg.e_dq[0])
        self.e_dq_last = np.full(12, self.cfg.e_dq[0])
        self.e_dq_min = self.cfg.e_dq[1]
        self.e_dq_max = self.cfg.e_dq[2]
        
        self.e_q_max = np.array(self.cfg.e_q_max)
        self.e_q_min = np.array(self.cfg.e_q_min)
        
        
        self.action_space = Box(dim=12)
        
        self.observation_space = Box(dim=self.cfg.observation_space[1])
        
        self.observation_space_cpg = Box(dim=self.cfg.observation_space_cpg[1])
        
        
        self.dt = self.cfg.dt
        
        self.ctrl_per_mjc = self.cfg.control_dt_per_mujoco_dt
        
        self.nn_per_mjc = self.cfg.NN_dt_per_mujoco_dt
        
        
        
        self.last_push = 0
        self.last_command_change = 0
        
        # reset で再定義する
        
        
        
        self.delay = self.cfg.delay[0]
        
        self.target_q_history = np.zeros((self.delay,12))
        
        self.box_noise = self.cfg.box_noise[0]
        
        
    def reset(self):
        self.sim_time = 0
        self.e_q = np.zeros(12)
        self.e_dq = np.full(12, self.cfg.e_dq[0])
        self.frames = []
        self.renderer = None
        
        if self.capture==True:
            skeleton = False
        else:
            skeleton = True
            
            
        if self.eval:
            additional_trunk_mass = self.cfg.additional_trunk_mass[0]
            limb_mass_scaling_factor = self.cfg.limb_mass_scaling_factor[0]
            friction = self.cfg.friction[0]
            
            self.command = np.array([self.cfg.command_x[0],self.cfg.command_y[0],self.cfg.command_w[0]])
            
            self.gc = self.cfg.gc[0]
            self.gp = self.cfg.gp[0]
            self.h = self.cfg.h[0]
            
            self.delay = self.cfg.delay[0]
            self.target_q_historry = np.zeros((self.delay, 12))
            
        else:
            additional_trunk_mass = random.uniform(self.cfg.additional_trunk_mass[1], self.cfg.additional_trunk_mass[2])
            limb_mass_scaling_factor = random.uniform(self.cfg.limb_mass_scaling_factor[1], self.cfg.limb_mass_scaling_factor[2])
            friction = random.uniform(self.cfg.friction[1], self.cfg.friction[2])
            
            self.command = np.array([random.uniform(self.cfg.command_x[1], self.cfg.command_x[2]),
                                 random.uniform(self.cfg.command_y[1], self.cfg.command_y[2]),
                                 random.uniform(self.cfg.command_w[1], self.cfg.command_w[2])])
            
            self.gc = random.uniform(self.cfg.gc[1], self.cfg.gc[2])
            self.gp = random.uniform(self.cfg.gp[1], self.cfg.gp[2])
            self.h = random.uniform(self.cfg.h[1], self.cfg.h[2])
            
            if self.cfg.Gain_randomization: 
                self.Kp = np.full(12, random.uniform(self.cfg.Kp[1], self.cfg.Kp[2]))
                self.Kd = np.full(12, random.uniform(self.cfg.Kd[1], self.cfg.Kd[2]))
            
            self.delay = random.randint(self.cfg.delay[1], self.cfg.delay[2])
            self.target_q_historry = np.zeros((self.delay, 12))
        
        
        qpos_x = 0.0
        qpos_y = 0.0
        qpos_z = 0.5
        
        if self.cfg.terrain == "flat":
            xml = FlatXml(skeleton=skeleton,
                          additional_trunk_mass=additional_trunk_mass,
                          limb_mass_scaling_factor=limb_mass_scaling_factor,
                          friction=friction)

            
        elif self.cfg.terrain == "box":
            
            if self.eval:
                box_size = self.cfg.box_size[0]
                box_height = self.cfg.box_height[0]
                num_row = int(self.cfg.row_length/self.cfg.box_size[0]) + 1
                num_col = int(self.cfg.col_length/self.cfg.box_size[0]) + 1
                self.box_noise = self.cfg.box_noise[0]
                down = self.cfg.step_down
            else:
                box_size = np.random.uniform(self.cfg.box_size[1],self.cfg.box_size[2])
                box_height = np.random.uniform(self.cfg.box_height[1],self.cfg.box_height[2])
                num_row = int(self.cfg.row_length/self.cfg.box_size[0]) + 1
                num_col = int(self.cfg.col_length/self.cfg.box_size[0]) + 1
                self.box_noise = np.random.uniform(self.cfg.box_noise[1],self.cfg.box_noise[2])
                down = np.random.choice([True,False])
            
            xml = BoxXml(friction=friction,
                           box_size=box_size,
                           height=box_height,
                           num_row=num_row,
                           num_col=num_col,
                           noise=self.box_noise,
                           down=down,
                           skeleton=skeleton,
                           additional_trunk_mass=additional_trunk_mass, limb_mass_scaling_factor=limb_mass_scaling_factor)
            
            qpos_x = - self.cfg.col_length * 0.5 + 1.5
            # qpos_x =  self.cfg.col_length * 0.5 - 1.5
            # qpos_y = 0.0
            
            if down:
                # qpos_z = - box_height * 0.5 * max(num_row,num_col) + 0.5 + (1.0//box_size + 1) * box_height
                qpos_z = 0.5 - (abs(qpos_x)//box_size + 1) * box_height  
            else:
                # qpos_z =  box_height * 0.5 * max(num_row,num_col) + 0.5 - (1.0//box_size + 1) * box_height
                qpos_z = 0.5 + (abs(qpos_x)//box_size + 1) * box_height
        
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model,self.data)
        
        if self.capture == True:
            self.renderer = mujoco.Renderer(self.model,self.video_height,self.video_width)
        
        # self.theta = np.array([random.uniform(0, 2*np.pi),random.uniform(0, 2*np.pi),random.uniform(0, 2*np.pi),random.uniform(0, 2*np.pi),])
        theta_FR = random.uniform(0, 2*np.pi)
        theta_FL = (theta_FR + np.pi) % (2*np.pi)
        theta_RR = (theta_FR + np.pi/2) % (2*np.pi)
        theta_RL = theta_FR 
        self.theta = np.array([theta_FR,theta_FL,theta_RR,theta_RL])
        self.phi = np.array([0,0,0,0])
        
        
        self.data.joint("free").qvel = [0,0,0,0,0,0]
        self.data.joint("free").qpos = [qpos_x,qpos_y,qpos_z,1,0,0,0]
        
        # for reset_step in range(500):
        for reset_step in range(1000):
            target_pos = Trajectory(self.h,self.gc,self.gp,self.d_step,self.r,self.theta,self.phi)
            joint_q, joint_dq = Joint_q_dq(self.data)
            target_q = Inverse_kinematics(target_pos)
            target_q = Joint_safty(target_q, self.cfg.hip_limit, self.cfg.thigh_limit, self.cfg.calf_limit)
            self.target_q_historry = np.tile(target_q, (self.delay, 1))
            torque = PD_control(self.Kp, self.Kd,self.target_q_historry[0], joint_q, joint_dq)
            self.data.ctrl[:] = torque
            self.data.joint("free").qvel = [0,0,0,0,0,0]
            mujoco.mj_step(self.model, self.data)
            
            
            if reset_step > 100:
                qpos_z += -0.001
                self.data.joint("free").qpos = [qpos_x,qpos_y,qpos_z,1,0,0,0]
                self.data.joint("free").qvel = [0,0,0,0,0,0]
                foot_force = Foot_force(self.data)
                mujoco.mj_step(self.model, self.data)
                
                if np.sum(foot_force) > 0:
                    break
        
                
            

        # Observation
        joint_q, joint_dq, foot_force, euler, angular_vel, linear_acc, linear_vel, R = self.Sensor_data()
        observation, observation_cpg = self.Observation(joint_q, joint_dq, foot_force, euler, angular_vel, linear_acc, linear_vel, R)
        
        return observation, observation_cpg
    
    def save_video(self,video_folder=f"{script_dir}/Videos", video_name=None):
        if self.capture == True:
            if len(self.frames) > 0:
                
                timezone = pytz.timezone('Asia/Tokyo')
                current_datetime = datetime.datetime.now(timezone)
                formatted_datetime = current_datetime.strftime("%y%m%d_%H%M%S")
                
                
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)
                
                if video_name == None:
                    video_path = video_folder + f"/{script_name}_{formatted_datetime}.mp4"
                else:
                    video_path = video_folder + f"/{video_name}"
                
                
                imageio.mimsave(video_path, self.frames, fps=self.framerate, macro_block_size=1)
                print(f"Saved video to {video_path}")
    
    def step(self, action_res, action_cpg):
        self.mu = ((self.mu_high-self.mu_low)/2)*action_cpg[0:4] + ((self.mu_high + self.mu_low)/2)
        self.omega = ((self.omega_high - self.omega_low) / 2) * action_cpg[4:8] + ((self.omega_high + self.omega_low) / 2)
        self.psi = ((self.psi_high - self.psi_low) / 2) * action_cpg[8:12] + ((self.psi_high + self.psi_low) / 2)
        
        self.e_dq = ((self.e_dq_max - self.e_dq_min) / 2) * action_res + ((self.e_dq_max + self.e_dq_min) / 2)
        self.e_dq = np.clip(self.e_dq, self.e_dq_last + self.cfg.e_ddq[1] * self.dt * self.nn_per_mjc, self.e_dq_last + self.cfg.e_ddq[2] * self.dt * self.nn_per_mjc)
        self.e_dq_last = self.e_dq
        
    
        # push
        self.last_push += self.dt * self.nn_per_mjc
        if self.last_push > self.cfg.external_push_interval and not self.eval:
            self.last_push = 0
            angle = np.random.uniform(0, 2 * np.pi)
            self.data.joint("free").qvel += np.array([self.cfg.external_push_vel*np.cos(angle),self.cfg.external_push_vel*np.sin(angle),0,0,0,0])
            
        self.last_command_change += self.dt * self.nn_per_mjc
        if self.last_command_change > self.cfg.command_change_interval and not self.eval:
        # if self.last_command_change > self.cfg["env"]["command_change_interval"]:
            self.last_command_change = 0
            self.command = np.array([random.uniform(self.cfg.command_x[1], self.cfg.command_x[2]),
                                 random.uniform(self.cfg.command_y[1], self.cfg.command_y[2]),
                                 random.uniform(self.cfg.command_w[1], self.cfg.command_w[2])])
            
           
        # Simulation
        work, mileage, angle = self.Do_simulation()

        # Observation
        joint_q, joint_dq, foot_force, euler, angular_vel, linear_acc, linear_vel, R = self.Sensor_data()
        observation, observation_cpg = self.Observation(joint_q, joint_dq, foot_force, euler, angular_vel, linear_acc, linear_vel, R)
        
        # Reward
        reward, reward_list = self.reward(linear_vel,work,angular_vel)
        
        # Done
        if np.cos(euler[0])*np.cos(euler[1]) < 0:
            termination = True
        else:
            termination = False
            
        if self.data.sensor("bottom").data[0] or self.data.sensor("bottom2").data[0] > 10:
            termination = True
            
        if self.sim_time >= self.max_sim_time:
            truncation = True
        elif self.cfg.terrain == "box":
            global_pos, _ = Position_Velocity(self.data)
            
            if abs(global_pos[0]) >= self.cfg.col_length/2 - 0.3 or abs(global_pos[1]) >= self.cfg.row_length/2 - 0.3:
                truncation = True
            else:
                truncation = False
        else:
            truncation = False
        
        # Info
        
        info = [work,mileage,angle,reward_list]
        
        
        return observation, observation_cpg, reward, termination, truncation, info
    
    def Do_simulation(self):
        work = 0
        mileage = np.array([0.0,0.0,0.0])
        angle = np.array([0.0,0.0,0.0])
    
        for i in range(self.nn_per_mjc):
            if i % self.ctrl_per_mjc == 0:
                self.r, self.theta, self.phi, self.r_dot = CPG(self.dt, self.a, self.mu, self.omega, self.psi, self.r, self.theta, self.phi, self.r_dot, self.r_ddot)
                target_pos = Trajectory(self.h,self.gc,self.gp,self.d_step,self.r,self.theta,self.phi)
                target_q = Inverse_kinematics(target_pos)
                self.e_q = self.e_q + self.dt * self.ctrl_per_mjc * self.e_dq
                self.e_q = np.clip(self.e_q, self.e_q_min, self.e_q_max)
                target_q += self.e_q
                target_q = Joint_safty(target_q, self.cfg.hip_limit, self.cfg.thigh_limit, self.cfg.calf_limit)
                self.target_q_historry = np.vstack((self.target_q_historry[1:], target_q))
                joint_q, joint_dq = Joint_q_dq(self.data)
                torque = PD_control(self.Kp, self.Kd, self.target_q_historry[0], joint_q, joint_dq)
                self.data.ctrl[:] = torque
                

            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.dt
            work += np.sum(np.abs(torque *joint_dq)) * self.dt
            
            
            quaternion, angular_vel, linear_acc, linear_vel = IMU(self.data)
            
            mileage += self.dt*linear_vel
            
            angle += self.dt*angular_vel
        
            
        if self.capture == True:
            if len(self.frames) < self.sim_time * self.framerate:
                
                self.renderer.update_scene(self.data, camera="track")
                
                visual_foot_pos(self.renderer.scene, target_pos, self.data, rgba=[1,0,0,1])
                res_target_pos = Forward_kinematics(target_q)
                visual_foot_pos(self.renderer.scene, res_target_pos, self.data, rgba=[0,1,0,1])
                pixels = self.renderer.render()
                self.frames.append(pixels)


        return work, mileage, angle
        
    
    def Sensor_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        joint_q, joint_dq = Joint_q_dq(self.data)
        foot_force = Foot_force(self.data)
        quaternion, angular_vel, linear_acc, linear_vel = IMU(self.data)
        R = Quaternion_to_rotation_matrix(quaternion)
        euler = Rotation_matrix_to_euler(R)
        return joint_q, joint_dq, foot_force, euler, angular_vel, linear_acc, linear_vel, R
    
    def Observation(self,joint_q, joint_dq, foot_force, euler, angular_vel, linear_acc, linear_vel, R):
        
        cpg_state = np.hstack((self.r, self.theta%(2*np.pi), self.phi%(2*np.pi), self.r_dot, self.omega, self.psi))
        
        if self.cfg.observation_space[0] == "min": #31
            observation = np.hstack((foot_force, cpg_state, self.command))
            
        elif self.cfg.observation_space[0] == "full": #87
            observation = np.hstack((joint_q, joint_dq, foot_force, euler[0:2], angular_vel, linear_acc, cpg_state, self.e_q, self.e_dq, self.command))
            
        if self.cfg.observation_space_cpg[0] == "min": #31
            observation_cpg = np.hstack((foot_force, cpg_state, self.command))
        elif self.cfg.observation_space_cpg[0] == "full": #63
            observation_cpg = np.hstack((joint_q, joint_dq, foot_force, euler[0:2], angular_vel, linear_acc, cpg_state, self.command))
       
        return observation, observation_cpg
    
    def reward(self,linear_vel,work,angular_vel):
        dt = self.dt*self.nn_per_mjc
        
        def f(x):
            return np.exp(-4*(np.abs(x))**2)
            

        rew_lin_vel_x = f(self.command[0]-linear_vel[0]) * self.cfg.rewardScale_lin_vel_x * dt
        rew_lin_vel_y = f(self.command[1]-linear_vel[1]) * self.cfg.rewardScale_lin_vel_y * dt
        rew_ang_vel_z = f(self.command[2]-angular_vel[2]) * self.cfg.rewardScale_ang_vel_z * dt
        
        rew_lin_vel_z = linear_vel[2]**2 * self.cfg.rewardScale_lin_vel_z * dt
        rew_ang_vel_x = angular_vel[0]**2 * self.cfg.rewardScale_ang_vel_x * dt
        rew_ang_vel_y = angular_vel[1]**2 * self.cfg.rewardScale_ang_vel_y * dt
        
        rew_work = work * self.cfg.rewardScale_work
        
        rew_res = np.sum(np.abs(self.e_q)) * self.cfg.rewardScale_e_q * dt
        rew_res_dot = np.sum(np.abs(self.e_dq)) * self.cfg.rewardScale_e_dq * dt
        
        reward = rew_lin_vel_x + rew_lin_vel_y + rew_ang_vel_z + rew_lin_vel_z + rew_ang_vel_x + rew_ang_vel_y + rew_work + rew_res + rew_res_dot
        reward_list = [rew_lin_vel_x,rew_lin_vel_y,rew_ang_vel_z,rew_lin_vel_z,rew_ang_vel_x,rew_ang_vel_y,rew_work,rew_res,rew_res_dot]
        
        return reward,reward_list
    
    def sample_action(self,type=0):
        if type == 0:
            action_res = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,0,0])
        elif type == 1:
            action_res = np.random.normal(0, 1, self.action_space.shape)  # 平均0、標準偏差1の正規分布に基づく値を生成
            action_res = np.tanh(action_res)
        return action_res 

        
        
if __name__ == "__main__":
    
    env = A1CPGEnv(capture=True,eval=True)

    state, state_cpg = env.reset()
    
    # env.phi = np.array([np.pi/2,np.pi/2,np.pi/2,np.pi/2])
    env.theta = np.array([0.0,np.pi,np.pi,0.0])

    start = time.time()
    done = False
    total_reward = 0
    mileage = np.array([0.0,0.0,0.0])
    angle = np.array([0.0,0.0,0.0])
    print(env.command)
    # print(state.shape)
    print(env.observation_space.shape)

    while not done:
        # action = np.array([1,1,1,1,1,1,1,1,0,0,0,0])
        # action = np.array([-0.5,-0.5,-0.5,-0.5,0.0,0.0,0.0,0.0,0,0,0,0])
        action_cpg = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,0,0])
        action_res = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,0,0])
        # action_res = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
        # action_res = np.array([-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1])
        # action_res = np.array([0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0])*np.sin(2*np.pi*env.sim_time)
        
        # action = np.array([-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0])
        observation, observation_cpg, reward, termination, truncation, info = env.step(action_cpg,action_res)
        # print(observation.shape)
        # steps +=1
        total_reward += reward
        mileage += info[1]
        angle += info[2]
        done = termination or truncation
        if done == True:
            print(f"sim time: {env.sim_time}")
            print("reward",total_reward)
            print("mileage",mileage)
            print("angle",angle)
            env.save_video()
        # time_until_next_step = env.sim_time - (time.time() -start)
        # if time_until_next_step > 0:
        #     time.sleep(time_until_next_step)

    current_time = time.time() - start
    print("Simulation ran for:", current_time, "seconds") 
    # env.reset()