import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R


def CPG(dt: float, a: float, mu: np.ndarray, omega: np.ndarray, psi: np.ndarray, r: np.ndarray, theta: np.ndarray, phi: np.ndarray, r_dot: np.ndarray, r_ddot: np.ndarray):
    
    r_ddot = a*(a*(mu -r)/4 - r_dot)*dt
    
    r_dot = r_dot + r_ddot*dt
    r = r_dot*dt + r
    theta = omega*dt + theta
    phi = psi*dt + phi
    
    theta = theta%(2*np.pi)
    phi = phi%(2*np.pi)
    
    return r, theta, phi, r_dot, r_ddot




def Trajectory(h: float, gc: float, gp: float, d_step: float, r: np.ndarray, theta: np.ndarray, phi: np.ndarray,x_offset:np.ndarray=np.array([0,0,0,0])) -> np.ndarray:
    g = np.array([gc if np.sin(theta[0]) > 0 else gp,
                    gc if np.sin(theta[1]) > 0 else gp,
                    gc if np.sin(theta[2]) > 0 else gp,
                    gc if np.sin(theta[3]) > 0 else gp,])
    
    x = -d_step*(r - 1)*np.cos(theta)*np.cos(phi)
    y = -d_step*(r - 1)*np.cos(theta)*np.sin(phi)
    yr = -0.0838
    yl =  0.0838
    z = -h + g*np.sin(theta)
    
    trajectory_pos = np.array([[x[0]+x_offset[0],y[0]+yr,z[0]],[x[1]+x_offset[1],y[1]+yl,z[1]],[x[2]+x_offset[2],y[2]+yr,z[2]],[x[3]+x_offset[3],y[3]+yl,z[3]]])
    
    return trajectory_pos


def Joint_q_dq(data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:

    j_names = ["FR_hip_", "FR_thigh_", "FR_calf_", "FL_hip_", "FL_thigh_", "FL_calf_",
            "RR_hip_", "RR_thigh_", "RR_calf_", "RL_hip_", "RL_thigh_", "RL_calf_"]

    joint_q = np.array([data.sensor(j_name + "pos").data[0] for j_name in j_names])
    joint_dq = np.array([data.sensor(j_name + "vel").data[0] for j_name in j_names])

    return joint_q, joint_dq

def IMU(data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    quaternion = np.array(data.sensor("Body_Quat").data)
    angular_vel = np.array(data.sensor("Body_Gyro").data)
    linear_acc = np.array(data.sensor("Body_Acc").data)
    linear_vel = np.array(data.sensor("Body_Vel").data)
    return quaternion, angular_vel, linear_acc, linear_vel

def Foot_force(data: mujoco.MjData) -> np.ndarray:
    foot_force = np.array([1 if data.sensor("FR_foot").data[0] > 0 else 0,
                    1 if data.sensor("FL_foot").data[0] > 0 else 0,
                    1 if data.sensor("RR_foot").data[0] > 0 else 0,
                    1 if data.sensor("RL_foot").data[0] > 0 else 0])
    return foot_force

def Quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    R = np.array([
        [1 - 2 * (q[2]**2 + q[3]**2), 2 * (q[1]*q[2] - q[0]*q[3]), 2 * (q[1]*q[3] + q[0]*q[2])],
        [2 * (q[1]*q[2] + q[0]*q[3]), 1 - 2 * (q[1]**2 + q[3]**2), 2 * (q[2]*q[3] - q[0]*q[1])],
        [2 * (q[1]*q[3] - q[0]*q[2]), 2 * (q[2]*q[3] + q[0]*q[1]), 1 - 2 * (q[1]**2 + q[2]**2)]
    ])
    return R

def Rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    # ヨー角を計算
    yaw = np.arctan2(R[1, 0], R[0, 0])

    # ピッチ角を計算
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))

    # ロール角を計算
    roll = np.arctan2(R[2, 1], R[2, 2])

    return np.array([roll, pitch, yaw])
    



def Forward_kinematics(joint_q: np.ndarray) -> np.ndarray:
    L2 = 0.2
    L3 = 0.2
    
    # フットポジションを計算するためにjoint_qをreshape
    joint_q_reshape = joint_q.reshape(4, 3)

    L1 = np.array([0.0838, -0.0838, 0.0838, -0.0838])

    th1 = joint_q_reshape[:, 0]
    th2 = joint_q_reshape[:, 1]
    th3 = joint_q_reshape[:, 2]
    
    sin_th1 = np.sin(th1)
    cos_th1 = np.cos(th1)
    sin_th2 = np.sin(th2)
    cos_th2 = np.cos(th2)
    sin_th23 = np.sin(th2 + th3)
    cos_th23 = np.cos(th2 + th3)

    x = -L3 * sin_th23 - L2 * sin_th2
    y = L3 * sin_th1 * cos_th23 + L2 * sin_th1 * cos_th2 - L1 * cos_th1
    z = -L3 * cos_th1 * cos_th23 - L2 * cos_th1 * cos_th2 - L1 * sin_th1

    FootPos = np.column_stack((x, y, z))
    
    return FootPos

def Inverse_kinematics(target_positions: np.ndarray) -> np.ndarray:
    L1 = 0.0838
    L2 = 0.2
    L3 = 0.2
    
    joint_angles = []
    
    for i, target_pos in enumerate(target_positions):
        if i % 2 == 0:  # Right legs
            th_f_yz = np.arctan2(-target_pos[2], -target_pos[1])
            th1 = th_f_yz - np.arccos(L1/np.sqrt(target_pos[1]**2 + target_pos[2]**2))
        else:  # Left legs
            th_f_yz = np.arctan2(target_pos[2], target_pos[1])
            th1 = th_f_yz + np.arccos(L1/np.sqrt(target_pos[1]**2 + target_pos[2]**2))
        
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(th1), -np.sin(th1)],
            [0, np.sin(th1), np.cos(th1)]
        ])
        
        R_x_inv = R_x.T
        rotated_target_pos = np.dot(R_x_inv, target_pos)
        
        phi = np.arccos((L2**2 + L3**2 - rotated_target_pos[0]**2 - rotated_target_pos[2]**2)/(2*L2*L3))
        
        th_f_xz = np.arctan2(-rotated_target_pos[0], -rotated_target_pos[2])
        
        th2 = th_f_xz + (np.pi - phi)/2
        th3 = -np.pi + phi
        
        joint_angles.append([th1, th2, th3])
    
    return np.array(joint_angles).flatten()



def Position_Velocity(data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    position = np.array(data.sensor("Global_Body_Pos").data)
    global_linear_vel = np.array(data.sensor("Global_Body_Vel").data)
    return position, global_linear_vel


def PD_control(Kp: np.ndarray, Kd: np.ndarray,target_q: np.ndarray, joint_q: np.ndarray, joint_dq: np.ndarray) -> np.ndarray:
    torques = Kp * (target_q -joint_q)  - Kd * joint_dq
    return torques

def Joint_safty(joint_q: np.ndarray, hip_limit: np.ndarray, thigh_limit: np.ndarray, calf_limit: np.ndarray) -> bool:
    limit_min = np.array([hip_limit[0],thigh_limit[0],calf_limit[0],hip_limit[0],thigh_limit[0],calf_limit[0],hip_limit[0],thigh_limit[0],calf_limit[0],hip_limit[0],thigh_limit[0],calf_limit[0]])
    limit_max = np.array([hip_limit[1],thigh_limit[1],calf_limit[1],hip_limit[1],thigh_limit[1],calf_limit[1],hip_limit[1],thigh_limit[1],calf_limit[1],hip_limit[1],thigh_limit[1],calf_limit[1]])
    
    clipped_joint_q = np.clip(joint_q, limit_min, limit_max)
    clipped = not np.array_equal(joint_q, clipped_joint_q)
    # if clipped:
    #     print("joint_q is clipped")
    
    return clipped_joint_q

def add_visual_sphere(scene, point: np.ndarray, radius: float, rgba: np.ndarray):
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_connector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[radius, 0, 0],
                        pos=point,
                        mat=np.eye(3).flatten(),
                        rgba=rgba
                        )
def rotate_point(point, quaternion):
    # Convert the quaternion to a rotation object
    quaternion = np.array([quaternion[1],quaternion[2],quaternion[3],quaternion[0]], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    rotation = R.from_quat(quaternion)
    
    # Apply the rotation to the point
    rotated_point = rotation.apply(point)
    
    return rotated_point

def visual_foot_pos(scene,target_pos: np.ndarray, data: mujoco.MjData, rgba: np.ndarray):
    FR_foot_pos = target_pos[0] + np.array([0.183,-0.047,0.0])
    FL_foot_pos = target_pos[1] + np.array([0.183,0.047,0.0])
    RR_foot_pos = target_pos[2] + np.array([-0.183,-0.047,0.0])
    RL_foot_pos = target_pos[3] + np.array([-0.183,0.047,0.0])
    point1 = rotate_point(FR_foot_pos, data.body("trunk").xquat)
    point2 = rotate_point(FL_foot_pos, data.body("trunk").xquat)
    point3 = rotate_point(RR_foot_pos, data.body("trunk").xquat)
    point4 = rotate_point(RL_foot_pos, data.body("trunk").xquat)
    point1 = point1 + data.body("trunk").xpos
    point2 = point2 + data.body("trunk").xpos
    point3 = point3 + data.body("trunk").xpos
    point4 = point4 + data.body("trunk").xpos
    add_visual_sphere(scene, point=point1, radius=0.03, rgba=rgba)
    add_visual_sphere(scene, point=point2, radius=0.03, rgba=rgba)
    add_visual_sphere(scene, point=point3, radius=0.03, rgba=rgba)
    add_visual_sphere(scene, point=point4, radius=0.03, rgba=rgba)


class Box:
    def __init__(self, dim, low=None, high=None):
        self.low = low
        self.high = high
        self.shape = dim
