import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

script_dir = os.path.dirname(__file__)    # .../A1_WS/Envs
parent_dir = os.path.dirname(script_dir)  # .../A1_WS
sys.path.append(parent_dir)
from .a1_xml import a1_xml
# from a1_xml import a1_xml


def FlatXml(additional_trunk_mass=0.0, limb_mass_scaling_factor=1.0,skeleton=True,friction=0.8):
    a1_model = a1_xml(additional_trunk_mass=additional_trunk_mass, limb_mass_scaling_factor=limb_mass_scaling_factor,skeleton=skeleton)
    
    xml = f"""
    <mujoco model="a1 Flat Ground">
        <option gravity='0 0 -9.806' iterations='50' solver='Newton' timestep='0.001'/> <!--追記-->

        {a1_model}

        <statistic center="0 0 0.1" extent="0.8"/>

        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
        </visual>

        <asset>
            <!-- <texture type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/> -->
            <!--<texture type="2d" name="groundplane" builtin="checker" rgb1="0.59 0.6 0.66" rgb2="0.49 0.5 0.56"  width="300" height="300"/>-->
            <texture type="2d" name="groundplane" builtin="checker" rgb1="0.9804 0.9882 0.9882" rgb2="0.9804 0.9882 0.9882"  width="500" height="500" mark="edge" markrgb="0.1 0.1 0.1" />
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.0"/>
        </asset>

        <worldbody>
            <geom name="floor" size="100 100 0.05" type="plane" material="groundplane" friction="{friction}"/>
        </worldbody>

    </mujoco>
    """
    return xml

def BoxXml(friction=0.8,box_size=0.3,height=0.1,num_row=5,num_col=5,noise=0.01,down=True,additional_trunk_mass=0.0, limb_mass_scaling_factor=1.0,skeleton=True,
           max_noise=0.06,cmap=mcolors.LinearSegmentedColormap.from_list("custom",["#d0d4d6","#74878f"]),vmin=0.0,vmax=1.0):
    # cmap=plt.get_cmap("viridis")
    
    a1_model = a1_xml(additional_trunk_mass=additional_trunk_mass, limb_mass_scaling_factor=limb_mass_scaling_factor,skeleton=skeleton)
    
    max_height = max((num_col-1)/2, (num_row-1)/2)
    
    boxes_xml =""
    for i in range(num_col):
        for j in range(num_row):

            pos_x = -num_col/2 * box_size + box_size/2 + box_size * i
            pos_y = -num_row/2 * box_size + box_size/2 + box_size * j
            
            if down:
                h = height*(max_height - max(abs(i - (num_col-1)/2), abs(j - (num_row-1)/2))) + 0.01 + noise + noise * np.random.uniform(-1, 1)
                pos_z = h/2 - max_height*height - 0.01 -  noise
            else:
                h = height*max(abs(i - (num_col-1)/2), abs(j - (num_row-1)/2)) + 0.01 + noise + noise * np.random.uniform(-1, 1)
                pos_z = h/2 -  0.01 -  noise
            # rgba = f"{0.5 * i/num_col + 0.1} {0.5 * j/num_row + 0.1} 0.0 1"
            _rgba = value_to_rgba((h-0.01)/(max_height*height+2*max_noise),cmap=cmap,vmin=vmin,vmax=vmax)
            rgba = f"{_rgba[0]} {_rgba[1]} {_rgba[2]} {_rgba[3]}"
            boxes_xml += f"""
            <body name="box_{i}_{j}_parent" pos="0 0 0.0">
                <body name="box_{i}_{j}_body" pos="{pos_x} {pos_y} {pos_z}">
                    <geom name="box_{i}_{j}" type="box" size="{box_size/2-0.001} {box_size/2-0.001} {h/2}" rgba="{rgba}" friction="{friction}"   solref="0.01 1" solimp="0.9 0.95 0.001" />
                </body>
            </body>
            """
      
    
    
    xml = f"""
    <mujoco model="a1 Flat Ground">
        <option gravity='0 0 -9.806' iterations='50' solver='Newton' timestep='0.001'/> <!--追記-->

        {a1_model}

        <statistic center="0 0 0.1" extent="0.8"/>

        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
        </visual>

        <asset>
            <!-- <texture type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/> -->
            <texture type="2d" name="groundplane" builtin="checker" rgb1="0.59 0.6 0.66" rgb2="0.49 0.5 0.56"  width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.0"/>
        </asset>

        <worldbody>

            {boxes_xml}
        </worldbody>

    </mujoco>
    """
    return xml

def value_to_rgba(value,cmap=mcolors.LinearSegmentedColormap.from_list("custom",["#fafcfc","#74878f"]),vmin=0.0,vmax=1.0):
    # 値が0から1の範囲内にあることを保証
    value = max(0, min(1, value))
    
    
    rgba = cmap(value*(1 - vmin - (1-vmax)) + vmin)
    
    # rgba = cmap(norm(value))
    # RGBAは0～1の範囲で返されるので、255スケールに変換する場合
    # rgba_255 = [int(255 * c) for c in rgba]
    
    return rgba