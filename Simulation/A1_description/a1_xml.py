import os
import sys

script_dir = os.path.dirname(__file__)


def a1_xml(additional_trunk_mass=0.0, limb_mass_scaling_factor=1.0, skeleton=True):
    
    trunk_mass = 4.713 + additional_trunk_mass
    thigh_mass = 1.013 * limb_mass_scaling_factor
    calf_mass = 0.226 * limb_mass_scaling_factor

    a1_xml =f"""

    <!-- <mujoco model="a1"> -->
        <compiler angle="radian" meshdir="{script_dir}/assets" texturedir="{script_dir}/assets" autolimits="true"/>

        <option cone="elliptic" impratio="100"/>
        <default>
            <motor ctrlrange="-33.5 33.5" ctrllimited="true"/>
            <default class="a1">
            <geom friction="0.6" margin="0.001"/>
            <joint axis="0 1 0" damping="2" armature="0.01" frictionloss="0.2"/>
            <position kp="100" forcerange="-33.5 33.5"/>
            <default class="abduction">
                <joint axis="1 0 0" damping="1" range="-0.802851 0.802851"/>
                <position ctrlrange="-0.802851 0.802851"/>
            </default>
            <default class="hip">
                <joint range="-1.0472 4.18879"/>
                <position ctrlrange="-1.0472 4.18879"/>
            </default>
            <default class="knee">
                <joint range="-2.69653 -0.916298"/>
                <position ctrlrange="-2.69653 -0.916298"/>
            </default>
            <default class="visual">
                <geom type="mesh" contype="0" conaffinity="0" group="2" material="dark"/>
            </default>
            <default class="collision">
                <geom group="3" type="capsule"/>
                <default class="hip_left">
                <geom size="0.04 0.04" quat="1 1 0 0" type="cylinder" pos="0 0.055 0"/>
                </default>
                <default class="hip_right">
                <geom size="0.04 0.04" quat="1 1 0 0" type="cylinder" pos="0 -0.055 0"/>
                </default>
                <default class="thigh1">
                <geom size="0.015" fromto="-0.02 0 0 -0.02 0 -0.16"/>
                </default>
                <default class="thigh2">
                <geom size="0.015" fromto="0 0 0 -0.02 0 -0.1"/>
                </default>
                <default class="thigh3">
                <geom size="0.015" fromto="-0.02 0 -0.16 0 0 -0.2"/>
                </default>
                <default class="calf1">
                <geom size="0.01" fromto="0 0 0 0.02 0 -0.13"/>
                </default>
                <default class="calf2">
                <geom size="0.01" fromto="0.02 0 -0.13 0 0 -0.2"/>
                </default>
                <default class="foot">
                <geom type="sphere" size="0.02" pos="0 0 -0.2" priority="1" solimp="0.015 1 0.031" condim="6"
                    friction="0.8 0.02 0.01"/>
                </default>
            </default>
            </default>
        </default>

        <asset>
            <!-- <material name="dark" specular="0" shininess="0.25" rgba="0.2 0.2 0.2 1"/> -->
            <material name="dark" specular="0.99" shininess="0.25" rgba="0.15 0.15 0.15 1"/>
            <texture type="2d" name="trunk_A1" file="trunk_A1.png"/>
            <!-- <material name="carbonfibre" texture="trunk_A1" specular="0" shininess="0.25"/> -->
            <material name="carbonfibre" texture="trunk_A1" specular="0" shininess="0.25" rgba="0.2 0.2 0.2 1" />

            <mesh class="a1" file="calf.obj"/>
            <mesh class="a1" file="hip.obj"/>
            <mesh class="a1" file="thigh.obj"/>
            <mesh class="a1" file="thigh_mirror.obj"/>
            <mesh class="a1" file="trunk.obj"/>
        </asset>

        <worldbody>
            <light name="spotlight1" mode="targetbodycom" target="trunk" pos="5 -5 5" diffuse="1.2 1.2 1.2"/>
            <!--<light name="spotlight2" mode="targetbodycom" target="trunk" pos="3 -5 10" diffuse="0.8 0.8 0.8"/>-->
        
            <body name="trunk" pos="0 0 0.43" childclass="a1">
            <!--<camera name="track" mode="track" pos="0 -1.0 0.5" xyaxes="1 0 0 0 1 1.5"/>-->
            <camera name="track" mode="track" pos="0.736 -0.882 0.2" xyaxes="0.832 0.555 0.000 -0.193 0.290 0.937"/>


            <!-- <camera name="track" mode="track" pos="1.125 -0.6495190528383289 0.45000000000000023" xyaxes="0.6495190528383289 1.125 -0.0 -0.8437500000000002 0.4871392896287468 1.6875"/> -->
            <!-- pos="0.5+x y 0.3+z" xyaxes="-y x 0 xz zy x^2+y^2" -->
            <freejoint name="free"/>
            <inertial mass="{trunk_mass}" pos="0 0.0041 -0.0005"
                fullinertia="0.0158533 0.0377999 0.0456542 -3.66e-05 -6.11e-05 -2.75e-05"/>
            <geom class="visual" mesh="trunk" material="dark"/>
            <geom class="collision" size="0.125 0.04 0.057" type="box"/>
            <site name="bottom" pos="0 0 -0.057" size="0.125 0.04 0.005" type="box" rgba="0.0 1.0 0.0 0.0"/>
            <geom class="collision" quat="1 0 1 0" pos="0 -0.04 0" size="0.058 0.125" type="cylinder"/>
            <geom class="collision" quat="1 0 1 0" pos="0 +0.04 0" size="0.058 0.125" type="cylinder"/>
            <geom class="collision" pos="0.25 0 0" size="0.005 0.06 0.05" type="box"/>
            <geom class="collision" pos="0.25 0.06 -0.01" size="0.009 0.035"/>
            <geom class="collision" pos="0.25 -0.06 -0.01" size="0.009 0.035"/>
            <geom class="collision" pos="0.25 0 -0.05" size="0.005 0.06" quat="1 1 0 0"/>
            <site name="bottom2" pos="0.25 0 -0.05" size="0.01 0.06" quat="1 1 0 0" type="capsule" rgba="0.0 1.0 0.0 0.5"/>
            <geom class="collision" pos="0.255 0 0.0355" size="0.021 0.052" quat="1 1 0 0"/>
            <site name="imu" pos="0 0 0"/>

            <body name="FR_hip" pos="0.183 -0.047 0">
                <inertial mass="0.696" pos="-0.003311 -0.000635 3.1e-05" quat="0.507528 0.506268 0.491507 0.494499"
                diaginertia="0.000807752 0.00055293 0.000468983"/>
                <joint class="abduction" name="FR_hip_joint"/>
                <geom class="visual" mesh="hip" quat="0 1 0 0"/>
                <geom class="hip_right"/>
                <body name="FR_thigh" pos="0 -0.08505 0">
                <inertial mass="{thigh_mass}" pos="-0.003237 0.022327 -0.027326" quat="0.999125 -0.00256393 -0.0409531 -0.00806091"
                    diaginertia="0.00555739 0.00513936 0.00133944"/>
                <joint class="hip" name="FR_thigh_joint"/>
                <geom class="visual" mesh="thigh_mirror"/>
                <geom class="thigh1"/>
                <geom class="thigh2"/>
                <geom class="thigh3"/>
                <body name="FR_calf" pos="0 0 -0.2">
                    <inertial mass="{calf_mass}" pos="0.00472659 0 -0.131975" quat="0.706886 0.017653 0.017653 0.706886"
                    diaginertia="0.00340344 0.00339393 3.54834e-05"/>
                    <joint class="knee" name="FR_calf_joint"/>
                    <geom class="visual" mesh="calf"/>
                    <geom class="calf1"/>
                    <geom class="calf2"/>
                    <geom class="foot"/>
                    <site name="FR_foot" type="sphere" rgba="0 0 0 0" pos="0 0 -0.2" size="0.02"/>
                </body>
                </body>
            </body>
            <body name="FL_hip" pos="0.183 0.047 0">
                <inertial mass="0.696" pos="-0.003311 0.000635 3.1e-05" quat="0.494499 0.491507 0.506268 0.507528"
                diaginertia="0.000807752 0.00055293 0.000468983"/>
                <joint class="abduction" name="FL_hip_joint"/>
                <geom class="visual" mesh="hip"/>
                <geom class="hip_left"/>
                <geom class="collision" size="0.04 0.04" pos="0 0.055 0" quat="1 1 0 0" type="cylinder"/>
                <body name="FL_thigh" pos="0 0.08505 0">
                <inertial mass="{thigh_mass}" pos="-0.003237 -0.022327 -0.027326" quat="0.999125 0.00256393 -0.0409531 0.00806091"
                    diaginertia="0.00555739 0.00513936 0.00133944"/>
                <joint class="hip" name="FL_thigh_joint"/>
                <geom class="visual" mesh="thigh"/>
                <geom class="thigh1"/>
                <geom class="thigh2"/>
                <geom class="thigh3"/>
                <body name="FL_calf" pos="0 0 -0.2">
                    <inertial mass="{calf_mass}" pos="0.00472659 0 -0.131975" quat="0.706886 0.017653 0.017653 0.706886"
                    diaginertia="0.00340344 0.00339393 3.54834e-05"/>
                    <joint class="knee" name="FL_calf_joint"/>
                    <geom class="visual" mesh="calf"/>
                    <geom class="calf1"/>
                    <geom class="calf2"/>
                    <geom class="foot"/>
                    <site name="FL_foot" type="sphere" rgba="0 0 0 0" pos="0 0 -0.2" size="0.02"/>
                </body>
                </body>
            </body>
            <body name="RR_hip" pos="-0.183 -0.047 0">
                <inertial mass="0.696" pos="0.003311 -0.000635 3.1e-05" quat="0.491507 0.494499 0.507528 0.506268"
                diaginertia="0.000807752 0.00055293 0.000468983"/>
                <joint class="abduction" name="RR_hip_joint"/>
                <geom class="visual" quat="0 0 0 -1" mesh="hip"/>
                <geom class="hip_right"/>
                <body name="RR_thigh" pos="0 -0.08505 0">
                <inertial mass="{thigh_mass}" pos="-0.003237 0.022327 -0.027326" quat="0.999125 -0.00256393 -0.0409531 -0.00806091"
                    diaginertia="0.00555739 0.00513936 0.00133944"/>
                <joint class="hip" name="RR_thigh_joint"/>
                <geom class="visual" mesh="thigh_mirror"/>
                <geom class="thigh1"/>
                <geom class="thigh2"/>
                <geom class="thigh3"/>
                <body name="RR_calf" pos="0 0 -0.2">
                    <inertial mass="{calf_mass}" pos="0.00472659 0 -0.131975" quat="0.706886 0.017653 0.017653 0.706886"
                    diaginertia="0.00340344 0.00339393 3.54834e-05"/>
                    <joint class="knee" name="RR_calf_joint"/>
                    <geom class="visual" mesh="calf"/>
                    <geom class="calf1"/>
                    <geom class="calf2"/>
                    <geom class="foot"/>
                    <site name="RR_foot" type="sphere" rgba="0 0 0 0" pos="0 0 -0.2" size="0.02"/>
                </body>
                </body>
            </body>
            <body name="RL_hip" pos="-0.183 0.047 0">
                <inertial mass="0.696" pos="0.003311 0.000635 3.1e-05" quat="0.506268 0.507528 0.494499 0.491507"
                diaginertia="0.000807752 0.00055293 0.000468983"/>
                <joint class="abduction" name="RL_hip_joint"/>
                <geom class="visual" quat="0 0 1 0" mesh="hip"/>
                <geom class="hip_left"/>
                <body name="RL_thigh" pos="0 0.08505 0">
                <inertial mass="{thigh_mass}" pos="-0.003237 -0.022327 -0.027326" quat="0.999125 0.00256393 -0.0409531 0.00806091"
                    diaginertia="0.00555739 0.00513936 0.00133944"/>
                <joint class="hip" name="RL_thigh_joint"/>
                <geom class="visual" mesh="thigh"/>
                <geom class="thigh1"/>
                <geom class="thigh2"/>
                <geom class="thigh3"/>
                <body name="RL_calf" pos="0 0 -0.2">
                    <inertial mass="{calf_mass}" pos="0.00472659 0 -0.131975" quat="0.706886 0.017653 0.017653 0.706886"
                    diaginertia="0.00340344 0.00339393 3.54834e-05"/>
                    <joint class="knee" name="RL_calf_joint"/>
                    <geom class="visual" mesh="calf"/>
                    <geom class="calf1"/>
                    <geom class="calf2"/>
                    <geom class="foot"/>
                    <site name="RL_foot" type="sphere" rgba="0 0 0 0" pos="0 0 -0.2" size="0.02"/>
                </body>
                </body>
            </body>
            </body>
        </worldbody>

        <actuator>
            <!-- <position class="abduction" name="FR_hip" joint="FR_hip_joint"/>
            <position class="hip" name="FR_thigh" joint="FR_thigh_joint"/>
            <position class="knee" name="FR_calf" joint="FR_calf_joint"/>
            <position class="abduction" name="FL_hip" joint="FL_hip_joint"/>
            <position class="hip" name="FL_thigh" joint="FL_thigh_joint"/>
            <position class="knee" name="FL_calf" joint="FL_calf_joint"/>
            <position class="abduction" name="RR_hip" joint="RR_hip_joint"/>
            <position class="hip" name="RR_thigh" joint="RR_thigh_joint"/>
            <position class="knee" name="RR_calf" joint="RR_calf_joint"/>
            <position class="abduction" name="RL_hip" joint="RL_hip_joint"/>
            <position class="hip" name="RL_thigh" joint="RL_thigh_joint"/>
            <position class="knee" name="RL_calf" joint="RL_calf_joint"/> -->

            <motor name="FR_hip"       gear="1" joint="FR_hip_joint"/>
            <motor name="FR_thigh"       gear="1" joint="FR_thigh_joint"/>
            <motor name="FR_calf"       gear="1" joint="FR_calf_joint"/>
            <motor name="FL_hip"     gear="1" joint="FL_hip_joint"/>
            <motor name="FL_thigh"     gear="1" joint="FL_thigh_joint"/>
            <motor name="FL_calf"     gear="1" joint="FL_calf_joint"/>
            <motor name="RR_hip"      gear="1" joint="RR_hip_joint"/>
            <motor name="RR_thigh"   gear="1" joint="RR_thigh_joint"/>
            <motor name="RR_calf"   gear="1" joint="RR_calf_joint" />
            <motor name="RL_hip"      gear="1" joint="RL_hip_joint"/>
            <motor name="RL_thigh"      gear="1" joint="RL_thigh_joint"/>
            <motor name="RL_calf"      gear="1" joint="RL_calf_joint"/>

        </actuator>

        <sensor>

            <jointpos name="FR_hip_pos"     joint="FR_hip_joint"/>
            <jointpos name="FR_thigh_pos"   joint="FR_thigh_joint"/>
            <jointpos name="FR_calf_pos"    joint="FR_calf_joint"/>
            <jointpos name="FL_hip_pos"     joint="FL_hip_joint"/>
            <jointpos name="FL_thigh_pos"   joint="FL_thigh_joint"/>
            <jointpos name="FL_calf_pos"    joint="FL_calf_joint"/>
            <jointpos name="RR_hip_pos"     joint="RR_hip_joint"/>
            <jointpos name="RR_thigh_pos"   joint="RR_thigh_joint"/>
            <jointpos name="RR_calf_pos"    joint="RR_calf_joint" />
            <jointpos name="RL_hip_pos"     joint="RL_hip_joint"/>
            <jointpos name="RL_thigh_pos"   joint="RL_thigh_joint"/>
            <jointpos name="RL_calf_pos"    joint="RL_calf_joint"/>

            <jointvel name="FR_hip_vel"     joint="FR_hip_joint"/>
            <jointvel name="FR_thigh_vel"   joint="FR_thigh_joint"/>
            <jointvel name="FR_calf_vel"    joint="FR_calf_joint"/>
            <jointvel name="FL_hip_vel"     joint="FL_hip_joint"/>
            <jointvel name="FL_thigh_vel"   joint="FL_thigh_joint"/>
            <jointvel name="FL_calf_vel"    joint="FL_calf_joint"/>
            <jointvel name="RR_hip_vel"     joint="RR_hip_joint"/>
            <jointvel name="RR_thigh_vel"   joint="RR_thigh_joint"/>
            <jointvel name="RR_calf_vel"    joint="RR_calf_joint" />
            <jointvel name="RL_hip_vel"     joint="RL_hip_joint"/>
            <jointvel name="RL_thigh_vel"   joint="RL_thigh_joint"/>
            <jointvel name="RL_calf_vel"    joint="RL_calf_joint"/>

            <framequat name="Body_Quat" objtype="site" objname="imu"/>
            <gyro name="Body_Gyro" site="imu"/>
            <accelerometer name="Body_Acc" site="imu"/>
            <velocimeter name="Body_Vel" site="imu"/>

            <touch name="FR_foot" site="FR_foot"/>
            <touch name="FL_foot" site="FL_foot"/>
            <touch name="RR_foot" site="RR_foot"/>
            <touch name="RL_foot" site="RL_foot"/>
            
            <touch name="bottom" site="bottom"/>
            <touch name="bottom2" site="bottom2"/>
            
            <framepos name="Global_Body_Pos" objtype="site" objname="imu"/>
            <framelinvel name="Global_Body_Vel" objtype="site" objname="imu"/>
            
            </sensor>

        <!-- <keyframe>
            <key name="home" qpos="0 0 0.27 1 0 0 0 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8"
            ctrl="0 0 0 0 0 0 0 0 0 0 0 0"/>
        </keyframe> -->
    <!--</mujoco>-->

    """
    a1_skeleton_xml = f"""
    
    <!--<mujoco model="a1">-->
        <compiler angle="radian" meshdir="{script_dir}/assets" texturedir="{script_dir}/assets" autolimits="true"/>

        <option cone="elliptic" impratio="100"/>
        <default>
            <motor ctrlrange="-33.5 33.5" ctrllimited="true"/>
            <default class="a1">
            <geom friction="0.6" margin="0.001"/>
            <joint axis="0 1 0" damping="2" armature="0.01" frictionloss="0.2"/>
            <position kp="100" forcerange="-33.5 33.5"/>
            <default class="abduction">
                <joint axis="1 0 0" damping="1" range="-0.802851 0.802851"/>
                <position ctrlrange="-0.802851 0.802851"/>
            </default>
            <default class="hip">
                <joint range="-1.0472 4.18879"/>
                <position ctrlrange="-1.0472 4.18879"/>
            </default>
            <default class="knee">
                <joint range="-2.69653 -0.916298"/>
                <position ctrlrange="-2.69653 -0.916298"/>
            </default>
            <!-- <default class="visual">
                <geom type="mesh" contype="0" conaffinity="0" group="2" material="dark"/>
            </default> -->
            <default class="collision">
                <geom group="0" type="capsule" rgba= "0.5 0.5 0.5 0.5"/>
                <!-- <geom group="3" type="capsule"/> -->
                <default class="hip_left">
                <geom size="0.04 0.04" quat="1 1 0 0" type="cylinder" pos="0 0.055 0"/>
                </default>
                <default class="hip_right">
                <geom size="0.04 0.04" quat="1 1 0 0" type="cylinder" pos="0 -0.055 0"/>
                </default>
                <default class="thigh1">
                <geom size="0.015" fromto="-0.02 0 0 -0.02 0 -0.16"/>
                </default>
                <default class="thigh2">
                <geom size="0.015" fromto="0 0 0 -0.02 0 -0.1"/>
                </default>
                <default class="thigh3">
                <geom size="0.015" fromto="-0.02 0 -0.16 0 0 -0.2"/>
                </default>
                <default class="calf1">
                <geom size="0.01" fromto="0 0 0 0.02 0 -0.13"/>
                </default>
                <default class="calf2">
                <geom size="0.01" fromto="0.02 0 -0.13 0 0 -0.2"/>
                </default>
                <default class="foot">
                <geom type="sphere" size="0.02" pos="0 0 -0.2" priority="1" solimp="0.015 1 0.031" condim="6"
                    friction="0.8 0.02 0.01"/>
                </default>
            </default>
            </default>
        </default>

        <asset>
            <!-- <material name="dark" specular="0" shininess="0.25" rgba="0.2 0.2 0.2 1"/> -->
            <!-- <material name="dark" specular="0" shininess="0.25" rgba="0.05 0.05 0.05 1"/> -->
            <!-- <texture type="2d" name="trunk_A1" file="trunk_A1.png"/> -->
            <!-- <material name="carbonfibre" texture="trunk_A1" specular="0" shininess="0.25"/> -->
            <!-- <material name="carbonfibre" texture="trunk_A1" specular="0" shininess="0.25" rgba="0.2 0.2 0.2 1" /> -->

            <!-- <mesh class="a1" file="calf.obj"/>
            <mesh class="a1" file="hip.obj"/>
            <mesh class="a1" file="thigh.obj"/>
            <mesh class="a1" file="thigh_mirror.obj"/>
            <mesh class="a1" file="trunk.obj"/> -->
        </asset>

        <worldbody>
            <!-- <light name="spotlight1" mode="targetbodycom" target="trunk" pos="-5 5 10" diffuse="0.8 0.8 0.8"/>
            <light name="spotlight2" mode="targetbodycom" target="trunk" pos="3 -5 10" diffuse="0.8 0.8 0.8"/> -->
        
            <body name="trunk" pos="0 0 0.43" childclass="a1">
            <!-- <camera name="track" mode="track" pos="1.5 0 1.3" xyaxes="0 1 0 -1 0 1"/> -->
            <camera name="track" mode="track" pos="1.125 -0.6495190528383289 0.45000000000000023" xyaxes="0.6495190528383289 1.125 -0.0 -0.8437500000000002 0.4871392896287468 1.6875"/>
            <!-- pos="0.5+x y 0.3+z" xyaxes="-y x 0 xz zy x^2+y^2" -->
            <freejoint name="free"/>
            <inertial mass="{trunk_mass}" pos="0 0.0041 -0.0005"
                fullinertia="0.0158533 0.0377999 0.0456542 -3.66e-05 -6.11e-05 -2.75e-05"/>
            <!-- <geom class="visual" mesh="trunk" material="carbonfibre"/> -->
            <geom class="collision" size="0.125 0.04 0.057" type="box"/>
            <site name="bottom" pos="0 0 -0.057" size="0.125 0.075 0.01" type="box" rgba="0.0 1.0 0.0 0.5"/>
            <geom class="collision" quat="1 0 1 0" pos="0 -0.04 0" size="0.058 0.125" type="cylinder"/>
            <geom class="collision" quat="1 0 1 0" pos="0 +0.04 0" size="0.058 0.125" type="cylinder"/>
            <geom class="collision" pos="0.25 0 0" size="0.005 0.06 0.05" type="box"/>
            <geom class="collision" pos="0.25 0.06 -0.01" size="0.009 0.035"/>
            <geom class="collision" pos="0.25 -0.06 -0.01" size="0.009 0.035"/>
            <geom class="collision" pos="0.25 0 -0.05" size="0.005 0.06" quat="1 1 0 0"/>
            <site name="bottom2" pos="0.25 0 -0.05" size="0.01 0.06" quat="1 1 0 0" type="capsule" rgba="0.0 1.0 0.0 0.5"/>
            <geom class="collision" pos="0.255 0 0.0355" size="0.021 0.052" quat="1 1 0 0" />
            <site name="imu" pos="0 0 0"/>

            <body name="FR_hip" pos="0.183 -0.047 0">
                <inertial mass="0.696" pos="-0.003311 -0.000635 3.1e-05" quat="0.507528 0.506268 0.491507 0.494499"
                diaginertia="0.000807752 0.00055293 0.000468983"/>
                <joint class="abduction" name="FR_hip_joint"/>
                <!-- <geom class="visual" mesh="hip" quat="0 1 0 0"/> -->
                <geom class="hip_right"/>
                <body name="FR_thigh" pos="0 -0.08505 0">
                <inertial mass="{thigh_mass}" pos="-0.003237 0.022327 -0.027326" quat="0.999125 -0.00256393 -0.0409531 -0.00806091"
                    diaginertia="0.00555739 0.00513936 0.00133944"/>
                <joint class="hip" name="FR_thigh_joint"/>
                <!-- <geom class="visual" mesh="thigh_mirror"/> -->
                <geom class="thigh1"/>
                <geom class="thigh2"/>
                <geom class="thigh3"/>
                <site name="FR_knee" fromto="-0.02 0 -0.16 0 0 -0.2" size="0.02" type="capsule" rgba="0.0 1.0 0.0 0.5"/>
                <body name="FR_calf" pos="0 0 -0.2">
                    <inertial mass="{calf_mass}" pos="0.00472659 0 -0.131975" quat="0.706886 0.017653 0.017653 0.706886"
                    diaginertia="0.00340344 0.00339393 3.54834e-05"/>
                    <joint class="knee" name="FR_calf_joint"/>
                    <!-- <geom class="visual" mesh="calf"/> -->
                    <geom class="calf1"/>
                    <geom class="calf2"/>
                    <geom class="foot"/>
                    <site name="FR_foot" type="sphere" rgba="0 0 0 0" pos="0 0 -0.2" size="0.02"/>
                </body>
                </body>
            </body>
            <body name="FL_hip" pos="0.183 0.047 0">
                <inertial mass="0.696" pos="-0.003311 0.000635 3.1e-05" quat="0.494499 0.491507 0.506268 0.507528"
                diaginertia="0.000807752 0.00055293 0.000468983"/>
                <joint class="abduction" name="FL_hip_joint"/>
                <!-- <geom class="visual" mesh="hip"/> -->
                <geom class="hip_left"/>
                <geom class="collision" size="0.04 0.04" pos="0 0.055 0" quat="1 1 0 0" type="cylinder"/>
                <body name="FL_thigh" pos="0 0.08505 0">
                <inertial mass="{thigh_mass}" pos="-0.003237 -0.022327 -0.027326" quat="0.999125 0.00256393 -0.0409531 0.00806091"
                    diaginertia="0.00555739 0.00513936 0.00133944"/>
                <joint class="hip" name="FL_thigh_joint"/>
                <!-- <geom class="visual" mesh="thigh"/> -->
                <geom class="thigh1"/>
                <geom class="thigh2"/>
                <geom class="thigh3"/>
                <site name="FL_knee" fromto="-0.02 0 -0.16 0 0 -0.2" size="0.02" type="capsule" rgba="0.0 1.0 0.0 0.5"/>
                <body name="FL_calf" pos="0 0 -0.2">
                    <inertial mass="{calf_mass}" pos="0.00472659 0 -0.131975" quat="0.706886 0.017653 0.017653 0.706886"
                    diaginertia="0.00340344 0.00339393 3.54834e-05"/>
                    <joint class="knee" name="FL_calf_joint"/>
                    <!-- <geom class="visual" mesh="calf"/> -->
                    <geom class="calf1"/>
                    <geom class="calf2"/>
                    <geom class="foot"/>
                    <site name="FL_foot" type="sphere" rgba="0 0 0 0" pos="0 0 -0.2" size="0.02"/>
                </body>
                </body>
            </body>
            <body name="RR_hip" pos="-0.183 -0.047 0">
                <inertial mass="0.696" pos="0.003311 -0.000635 3.1e-05" quat="0.491507 0.494499 0.507528 0.506268"
                diaginertia="0.000807752 0.00055293 0.000468983"/>
                <joint class="abduction" name="RR_hip_joint"/>
                <!-- <geom class="visual" quat="0 0 0 -1" mesh="hip"/> -->
                <geom class="hip_right"/>
                <body name="RR_thigh" pos="0 -0.08505 0">
                <inertial mass="{thigh_mass}" pos="-0.003237 0.022327 -0.027326" quat="0.999125 -0.00256393 -0.0409531 -0.00806091"
                    diaginertia="0.00555739 0.00513936 0.00133944"/>
                <joint class="hip" name="RR_thigh_joint"/>
                <!-- <geom class="visual" mesh="thigh_mirror"/> -->
                <geom class="thigh1"/>
                <geom class="thigh2"/>
                <geom class="thigh3"/>
                <site name="RR_knee" fromto="-0.02 0 -0.16 0 0 -0.2" size="0.02" type="capsule" rgba="0.0 1.0 0.0 0.5"/>
                <body name="RR_calf" pos="0 0 -0.2">
                    <inertial mass="{calf_mass}" pos="0.00472659 0 -0.131975" quat="0.706886 0.017653 0.017653 0.706886"
                    diaginertia="0.00340344 0.00339393 3.54834e-05"/>
                    <joint class="knee" name="RR_calf_joint"/>
                    <!-- <geom class="visual" mesh="calf"/> -->
                    <geom class="calf1"/>
                    <geom class="calf2"/>
                    <geom class="foot"/>
                    <site name="RR_foot" type="sphere" rgba="0 0 0 0" pos="0 0 -0.2" size="0.02"/>
                </body>
                </body>
            </body>
            <body name="RL_hip" pos="-0.183 0.047 0">
                <inertial mass="0.696" pos="0.003311 0.000635 3.1e-05" quat="0.506268 0.507528 0.494499 0.491507"
                diaginertia="0.000807752 0.00055293 0.000468983"/>
                <joint class="abduction" name="RL_hip_joint"/>
                <!-- <geom class="visual" quat="0 0 1 0" mesh="hip"/> -->
                <geom class="hip_left"/>
                <body name="RL_thigh" pos="0 0.08505 0">
                <inertial mass="{thigh_mass}" pos="-0.003237 -0.022327 -0.027326" quat="0.999125 0.00256393 -0.0409531 0.00806091"
                    diaginertia="0.00555739 0.00513936 0.00133944"/>
                <joint class="hip" name="RL_thigh_joint"/>
                <!-- <geom class="visual" mesh="thigh"/> -->
                <geom class="thigh1"/>
                <geom class="thigh2"/>
                <geom class="thigh3"/>
                <site name="RL_knee" fromto="-0.02 0 -0.16 0 0 -0.2" size="0.02" type="capsule" rgba="0.0 1.0 0.0 0.5"/>
                <body name="RL_calf" pos="0 0 -0.2">
                    <inertial mass="{calf_mass}" pos="0.00472659 0 -0.131975" quat="0.706886 0.017653 0.017653 0.706886"
                    diaginertia="0.00340344 0.00339393 3.54834e-05"/>
                    <joint class="knee" name="RL_calf_joint"/>
                    <!-- <geom class="visual" mesh="calf"/> -->
                    <geom class="calf1"/>
                    <geom class="calf2"/>
                    <geom class="foot"/>
                    <site name="RL_foot" type="sphere" rgba="0 0 0 0" pos="0 0 -0.2" size="0.02"/>
                </body>
                </body>
            </body>
            </body>
        </worldbody>

        <actuator>
            <!-- <position class="abduction" name="FR_hip" joint="FR_hip_joint"/>
            <position class="hip" name="FR_thigh" joint="FR_thigh_joint"/>
            <position class="knee" name="FR_calf" joint="FR_calf_joint"/>
            <position class="abduction" name="FL_hip" joint="FL_hip_joint"/>
            <position class="hip" name="FL_thigh" joint="FL_thigh_joint"/>
            <position class="knee" name="FL_calf" joint="FL_calf_joint"/>
            <position class="abduction" name="RR_hip" joint="RR_hip_joint"/>
            <position class="hip" name="RR_thigh" joint="RR_thigh_joint"/>
            <position class="knee" name="RR_calf" joint="RR_calf_joint"/>
            <position class="abduction" name="RL_hip" joint="RL_hip_joint"/>
            <position class="hip" name="RL_thigh" joint="RL_thigh_joint"/>
            <position class="knee" name="RL_calf" joint="RL_calf_joint"/> -->

            <motor name="FR_hip"       gear="1" joint="FR_hip_joint"/>
            <motor name="FR_thigh"       gear="1" joint="FR_thigh_joint"/>
            <motor name="FR_calf"       gear="1" joint="FR_calf_joint"/>
            <motor name="FL_hip"     gear="1" joint="FL_hip_joint"/>
            <motor name="FL_thigh"     gear="1" joint="FL_thigh_joint"/>
            <motor name="FL_calf"     gear="1" joint="FL_calf_joint"/>
            <motor name="RR_hip"      gear="1" joint="RR_hip_joint"/>
            <motor name="RR_thigh"   gear="1" joint="RR_thigh_joint"/>
            <motor name="RR_calf"   gear="1" joint="RR_calf_joint" />
            <motor name="RL_hip"      gear="1" joint="RL_hip_joint"/>
            <motor name="RL_thigh"      gear="1" joint="RL_thigh_joint"/>
            <motor name="RL_calf"      gear="1" joint="RL_calf_joint"/>

        </actuator>

        <sensor>

            <jointpos name="FR_hip_pos"     joint="FR_hip_joint"/>
            <jointpos name="FR_thigh_pos"   joint="FR_thigh_joint"/>
            <jointpos name="FR_calf_pos"    joint="FR_calf_joint"/>
            <jointpos name="FL_hip_pos"     joint="FL_hip_joint"/>
            <jointpos name="FL_thigh_pos"   joint="FL_thigh_joint"/>
            <jointpos name="FL_calf_pos"    joint="FL_calf_joint"/>
            <jointpos name="RR_hip_pos"     joint="RR_hip_joint"/>
            <jointpos name="RR_thigh_pos"   joint="RR_thigh_joint"/>
            <jointpos name="RR_calf_pos"    joint="RR_calf_joint" />
            <jointpos name="RL_hip_pos"     joint="RL_hip_joint"/>
            <jointpos name="RL_thigh_pos"   joint="RL_thigh_joint"/>
            <jointpos name="RL_calf_pos"    joint="RL_calf_joint"/>

            <jointvel name="FR_hip_vel"     joint="FR_hip_joint"/>
            <jointvel name="FR_thigh_vel"   joint="FR_thigh_joint"/>
            <jointvel name="FR_calf_vel"    joint="FR_calf_joint"/>
            <jointvel name="FL_hip_vel"     joint="FL_hip_joint"/>
            <jointvel name="FL_thigh_vel"   joint="FL_thigh_joint"/>
            <jointvel name="FL_calf_vel"    joint="FL_calf_joint"/>
            <jointvel name="RR_hip_vel"     joint="RR_hip_joint"/>
            <jointvel name="RR_thigh_vel"   joint="RR_thigh_joint"/>
            <jointvel name="RR_calf_vel"    joint="RR_calf_joint" />
            <jointvel name="RL_hip_vel"     joint="RL_hip_joint"/>
            <jointvel name="RL_thigh_vel"   joint="RL_thigh_joint"/>
            <jointvel name="RL_calf_vel"    joint="RL_calf_joint"/>

            <framequat name="Body_Quat" objtype="site" objname="imu"/>
            <gyro name="Body_Gyro" site="imu"/>
            <accelerometer name="Body_Acc" site="imu"/>
            <velocimeter name="Body_Vel" site="imu"/>

            <touch name="FR_foot" site="FR_foot"/>
            <touch name="FL_foot" site="FL_foot"/>
            <touch name="RR_foot" site="RR_foot"/>
            <touch name="RL_foot" site="RL_foot"/>
            
            <touch name="bottom" site="bottom"/>
            <touch name="bottom2" site="bottom2"/>
            <touch name="FR_knee" site="FR_knee"/>
            <touch name="FL_knee" site="FL_knee"/>
            <touch name="RR_knee" site="RR_knee"/>
            <touch name="RL_knee" site="RL_knee"/>

            
            <framepos name="Global_Body_Pos" objtype="site" objname="imu"/>
            <framelinvel name="Global_Body_Vel" objtype="site" objname="imu"/>
            
            </sensor>

        <!-- <keyframe>
            <key name="home" qpos="0 0 0.27 1 0 0 0 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8"
            ctrl="0 0 0 0 0 0 0 0 0 0 0 0"/>
        </keyframe> -->
    <!--</mujoco>-->
    
    """
    
    if skeleton:
        return a1_skeleton_xml
    else:
        return a1_xml

