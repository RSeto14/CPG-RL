from typing import List, Union
import json
from dataclasses import dataclass, field
import numpy as np
# dataclassの定義
@dataclass
class Config:
    gpu: int = 0
    seed: int = 123456
    
    # SAC
    policy: str = "Gaussian"
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 0.0001
    alpha: float = 0.2
    automatic_entropy_tuning: bool = True
    batch_size: int = 256
    num_steps: int = 10000000
    hidden_size: List[int] = field(default_factory=lambda: [512, 256, 128]) # 512, 256, 128
    updates_interval: int = 1
    updates_per_step: int = 1
    log_interval: int = 1000
    start_steps: int = 100000
    target_update_interval: int = 1
    replay_size: int = 5000000
    capture_interval: int = 500

    # Environment
    dt: float = 0.001
    delay: List[int] = field(default_factory=lambda: [30,15,35]) # defalut min max
    terrain: str = "flat"

    box_size: List[float] = field(default_factory=lambda: [0.4,0.3,0.5]) # default min max
    box_height: List[float] = field(default_factory=lambda: [0.0,0.0,0.0]) # default min max
    row_length: float = 5 # 縦
    col_length: float = 10 # 横
    box_noise: List[float] = field(default_factory=lambda: [0.00,0.00,0.00]) # default min max
    step_down: bool = True

    # Randomization Terrain
    random_terrain: bool = True
    random_terrain_start: int = 500000
    random_terrain: List[str] = field(default_factory=lambda: ["box"])
    
    curriculum_learn_gc: bool = True
    curriculum_start_end_gc: List[int] = field(default_factory=lambda: [5000000,8000000])
    curriculum_gc_max: List[float] = field(default_factory=lambda: [0.05,0.20]) # min max
    curriculum_gc_min: List[float] = field(default_factory=lambda: [0.03,0.03])
    
    
    curriculum_learn_gain: bool = True
    curriculum_start_end_gain: List[int] = field(default_factory=lambda: [4000000,8000000])
    curriculum_Kp_max: List[float] = field(default_factory=lambda: [100.0,100.0]) # min max
    curriculum_Kp_min: List[float] = field(default_factory=lambda: [80.0,50.0])
    curriculum_Kd_max: List[float] = field(default_factory=lambda: [2.0,3.0]) # min max
    curriculum_Kd_min: List[float] = field(default_factory=lambda: [2.0,1.0])
    
    # Box curriculum
    curriculum_learn_box_noise: bool = True
    curriculum_start_end_box_noise: List[int] = field(default_factory=lambda: [500000,5000000])
    curriculum_box_noise_max: List[float] = field(default_factory=lambda: [0.00,0.06]) # min max
    curriculum_box_noise_min: List[float] = field(default_factory=lambda: [0.00,0.03]) # min max


    # CPG
    cpg_log:str = None
    cpg_net:str = None
    a: int = 150
    d: float = 0.15
    gc: List[float] = field(default_factory=lambda: [0.10,0.03,0.05]) # default min max
    gp: List[float] = field(default_factory=lambda: [0.02,0.0,0.02]) # default min max
    h: List[float] = field(default_factory=lambda: [0.25,0.22,0.32]) # default min max  
    mu: List[float] = field(default_factory=lambda: [1.5,1.0,2.0]) # default min max
    omega: List[float] = field(default_factory=lambda: [0.0,0.0,3.0]) # default min max
    psi: List[float] = field(default_factory=lambda: [0.0,-1.5,1.5]) # default min max
    
    init_r: List[float] = field(default_factory=lambda: [1.0,2.0]) # default min max
    init_theta: List[float] = field(default_factory=lambda: [0.0,2*np.pi]) # default min max
    init_phi: List[float] = field(default_factory=lambda: [-np.pi/12,np.pi/12]) # default min max
    
    # x_offset: List[float] = field(default_factory=lambda: [0.0,0.0,-0.035,-0.035])
    x_offset: List[float] = field(default_factory=lambda: [0.0,0.0,0.0,0.0])


    e_dq: List[float] = field(default_factory=lambda: [0.0,-5.0,5.0]) # default min max
    e_ddq: List[float] = field(default_factory=lambda: [0.0,-50.0,50.0]) # default min max
    e_q_max: List[float] = field(default_factory=lambda: [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,])
    e_q_min: List[float] = field(default_factory=lambda: [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,])
    
    # Train
    observation_space: List[Union[str, int]] = field(default_factory=lambda: ["full", 87])
    observation_space_cpg: List[Union[str, int]] = field(default_factory=lambda: ["full", 63])
    command_x: List[float] = field(default_factory=lambda: [0.4, -0.0, 0.6]) # default min max
    command_y: List[float] = field(default_factory=lambda: [0.0, -0.0, 0.0]) # default min max
    command_w: List[float] = field(default_factory=lambda: [0.0, -0.0, 0.0]) # default min max
    episodeLength_s: int = 20
    command_change_interval: int = 5
    external_push_interval: int = 15
    external_push_vel: float = 0.5
    
    # Control
    Gain_randomization: bool = False
    Kp: List[float] = field(default_factory=lambda: [100.0, 100.0, 100.0]) # default min max
    Kd: List[float] = field(default_factory=lambda: [2.0, 2.0, 2.0]) # default min max
    control_dt_per_mujoco_dt: int = 1
    NN_dt_per_mujoco_dt: int = 10
    hip_limit: List[float] = field(default_factory=lambda: [-0.8,0.8]) # min max
    thigh_limit: List[float] = field(default_factory=lambda: [-0.524,3.927]) # min max
    calf_limit: List[float] = field(default_factory=lambda: [-2.69,-0.9163]) # min max
    
    # Reward
    rewardScale_ang_vel_x: float = -0.05
    rewardScale_ang_vel_y: float = -0.05
    rewardScale_ang_vel_z: float = 0.5
    rewardScale_lin_vel_x: float = 3.0
    rewardScale_lin_vel_y: float = 0.75
    rewardScale_lin_vel_z: float = -2.0
    rewardScale_work: float = -0.001
    
    # rewardScale_ang_vel_x: float = -0.05
    # rewardScale_ang_vel_y: float = -0.05
    # rewardScale_ang_vel_z: float = 0.5
    # rewardScale_lin_vel_x: float = 0.75
    # rewardScale_lin_vel_y: float = 0.75
    # rewardScale_lin_vel_z: float = -2.0
    # rewardScale_work: float = -0.001

    rewardScale_e_q: float = -0.0
    rewardScale_e_dq: float = -0.00
    
    # Randomization
    additional_trunk_mass: List[float] = field(default_factory=lambda: [0.0,0.0,5.0]) # default min max
    limb_mass_scaling_factor: List[float] = field(default_factory=lambda: [1.0,0.8,1.2]) # default min max
    friction: List[float] = field(default_factory=lambda: [1.5,0.5,2.5]) # default min max
    
    # Visualization
    framerate: int = 25
    frame_height: int = 1080
    frame_width: int = 1920
    track_cam: bool = True
    track_id: int = 0
