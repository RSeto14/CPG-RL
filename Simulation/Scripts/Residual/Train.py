# buffer torch実装
import argparse
import os
import datetime
import pytz
import sys
import numpy as np
import itertools
import torch
import csv

import random

import glob

script_dir = os.path.dirname(__file__)    
parent_dir1 = os.path.dirname(script_dir)
parent_dir2 = os.path.dirname(parent_dir1)
sys.path.append(parent_dir2)

# script_name = os.path.basename(__file__)[: -len(".py")]


from SoftActorCritic.SAC import SAC,SAC_Eval
from SoftActorCritic.ReplayMemory import ReplayMemory
from A1_CPGResidualEnv import A1CPGEnv
from cfg.config_residual import Config
from cfg.config_cpgrl import Config as CPG_Config
from cfg.Save_Load_cfg import dataclass_to_json,json_to_dataclass




def Parse_args():
    parser = argparse.ArgumentParser(description='SAC train')
    
    parser.add_argument("--gpu", type=int, default=0, help="run on CUDA (default: 2)")
    parser.add_argument("--seed", type=int, default=123456, help="seed")
    parser.add_argument("--cap", type=bool, default=False,help="capture video")

    parser.add_argument("--cpg_log", type=str, default="/Users/hayashibe-lab/CPG-RL/Real/NN/241021_173827",help="cpg log folder") # Forward


    parser.add_argument("--cpg_net", type=str, default=None,help="cpg network")
    
    parser.add_argument("--title", type=str, default=None,help="comment1")
    parser.add_argument("--com", type=str, default="rewards e_q -0.001",help="comment2")

    args = parser.parse_args()
    
    return args



def main(args):
    
    # Networks
    if args.cpg_net is None:
        network_files = glob.glob(f"{args.cpg_log}/Networks/episode_*.pt")
        if network_files:
            latest_network = max(network_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            cpg_net = latest_network
        else:
            raise FileNotFoundError("No network files found.")
    elif args.net == 0:
            cpg_net = f"{args.cpg_log}/Networks/best.pt"
    else:
        cpg_net = f"{args.cpg_log}/Networks/episode_{args.cpg_net}.pt"
    
    data = json_to_dataclass(file_path=f"{args.cpg_log}/config.json")
    
    
    # cpg_cfg = CPG_Config(**data)
    cpg_cfg = CPG_Config()
    
    
    
    cfg = Config(seed=args.seed, gpu=args.gpu, cpg_log=args.cpg_log, cpg_net=cpg_net, a=cpg_cfg.a, d=cpg_cfg.d, gp=cpg_cfg.gp, h=cpg_cfg.h, mu=cpg_cfg.mu, omega=cpg_cfg.omega, psi=cpg_cfg.psi, observation_space_cpg=cpg_cfg.observation_space)
    
    # if cpg_cfg.gc[1] > cfg.gc[1]:
    #     cfg.gc[1] = cpg_cfg.gc[1]
    # if cpg_cfg.gc[2] < cfg.gc[2]:
    #     cfg.gc[2] = cpg_cfg.gc[2]
    if cpg_cfg.gp[1] > cfg.gp[1]:
        cfg.gp[1] = cpg_cfg.gp[1]
    if cpg_cfg.gp[2] < cfg.gp[2]:
        cfg.gp[2] = cpg_cfg.gp[2]
    if cpg_cfg.h[1] > cfg.h[1]:
        cfg.h[1] = cpg_cfg.h[1]
    if cpg_cfg.h[2] < cfg.h[2]:
        cfg.h[2] = cpg_cfg.h[2]
        
    
    # make log dir
    timezone = pytz.timezone('Asia/Tokyo')
    start_datetime = datetime.datetime.now(timezone)    
    start_formatted = start_datetime.strftime("%y%m%d_%H%M%S")
    train_log_dir = f"{script_dir}/Log/{start_formatted}"
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
        os.makedirs(f"{train_log_dir}/CSVs")
        os.makedirs(f"{train_log_dir}/Networks")
        
    

    # train log
    with open(f"{train_log_dir}/log.txt", 'w') as file:
        start = start_datetime.strftime("%y/%m/%d %H:%M:%S")
        #PID
        pid = os.getpid()
        file.write(f'Comment: {args.com}\n')
        file.write(f'Process ID: {pid}\n')
        file.write(f'Start: {start}\n')
    
    if args.title is not None:
        file_title = args.title
        
    else:
        file_title = ""
        if cfg.command_x[2] > 0:
            file_title += "Forward"
        if cfg.command_x[1] < 0:
            file_title += "Backward"
        if cfg.command_y[2] > 0:
            file_title += "Left"
        if cfg.command_y[1] < 0:
            file_title += "Right"
        if cfg.command_w[2] > 0:
            file_title += "TurnLeft"
        if cfg.command_w[1] < 0:
            file_title += "TurnRight"
        
    with open(f"{train_log_dir}/#{file_title}", 'w') as file:
        pass
    
    with open(f"{train_log_dir}/CSVs/rewards.csv", 'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerows([["Episode","Episode_Steps","Total Steps","Rewards","Rew_lin_vel_x","Rew_lin_vel_y","Rew_ang_vel_z","Rew_lin_vel_z","Rew_ang_vel_x","Rew_ang_vel_y","Rew_work","Rew_res","Rew_res_dot","Ave_Rewards","Mileage_x","Mileage_y","Angle_w","Work","Time","Terrain","Box_noise","Termination","Truncation"]])
        writer.writerows([[0,]])
        
    with open(f"{train_log_dir}/CSVs/log.csv", 'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerows([["Global Steps","Critic 1 loss", "Critic 2 loss", "Policy loss", "Entropy loss", "alpha"]])
        writer.writerows([[0,]])
    
    dataclass_to_json(cfg,file_path=f"{train_log_dir}/config.json")
    
    
    # Seed
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Device
    visible_device = cfg.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))
    cfg.gpu = 0
    cpg_cfg.gpu = 0
    
    # Environment
    env = A1CPGEnv(cfg=cfg,capture=args.cap,eval=False)

    # Agent
    agent = SAC(env.observation_space.shape, env.action_space.shape, cfg)
    
    cpg_agent = SAC_Eval(cfg.observation_space_cpg[1], 12, cpg_cfg)
    cpg_agent.load_checkpoint(cpg_net)

    # Memory
    memory = ReplayMemory(cfg.replay_size, env.observation_space.shape, env.action_space.shape, seed, gpu=cfg.gpu)
        
    # Training Loop
    total_numsteps = 0
    updates = 0
    max_episode_rewards = -10000
    best_episode = 0
    
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        episode_mileage = np.array([0.0,0.0,0.0])
        episode_angle = np.array([0.0,0.0,0.0])
        episode_work = 0
        episode_reward_list = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        done = False
        
        if cfg.capture_interval > 0 and i_episode%cfg.capture_interval == 0:
            env.capture = True
        else:
            env.capture = False
            
        
        ###### Curriculum Learning #######################################################################################################################
        
        if cfg.curriculum_learn_gc:
            if cfg.curriculum_start_end_gc[0] <= total_numsteps <= cfg.curriculum_start_end_gc[1]:
                env.cfg.gc[2] = cfg.curriculum_gc_max[0] + (cfg.curriculum_gc_max[1] - cfg.curriculum_gc_max[0]) * (total_numsteps - cfg.curriculum_start_end_gc[0]) / (cfg.curriculum_start_end_gc[1] - cfg.curriculum_start_end_gc[0])
                env.cfg.gc[1] = cfg.curriculum_gc_min[0] + (cfg.curriculum_gc_min[1] - cfg.curriculum_gc_min[0]) * (total_numsteps - cfg.curriculum_start_end_gc[0]) / (cfg.curriculum_start_end_gc[1] - cfg.curriculum_start_end_gc[0])
            elif total_numsteps > cfg.curriculum_start_end_gc[1]:
                env.cfg.gc[2] = cfg.curriculum_gc_max[1]
                env.cfg.gc[1] = cfg.curriculum_gc_min[1]
            else:
                env.cfg.gc[2] = cfg.curriculum_gc_max[0]
                env.cfg.gc[1] = cfg.curriculum_gc_min[0]
                
        if cfg.random_terrain:
            if total_numsteps >= cfg.random_terrain_start:
                env.cfg.terrain = random.choice(cfg.random_terrain)
            else:
                env.cfg.terrain = cfg.terrain
                
        if cfg.curriculum_learn_box_noise:
            if cfg.curriculum_start_end_box_noise[0] <= total_numsteps <= cfg.curriculum_start_end_box_noise[1]:
                env.cfg.box_noise[2] = cfg.curriculum_box_noise_max[0] + (cfg.curriculum_box_noise_max[1] - cfg.curriculum_box_noise_max[0]) * (total_numsteps - cfg.curriculum_start_end_box_noise[0]) / (cfg.curriculum_start_end_box_noise[1] - cfg.curriculum_start_end_box_noise[0])
                env.cfg.box_noise[1] = cfg.curriculum_box_noise_min[0] + (cfg.curriculum_box_noise_min[1] - cfg.curriculum_box_noise_min[0]) * (total_numsteps - cfg.curriculum_start_end_box_noise[0]) / (cfg.curriculum_start_end_box_noise[1] - cfg.curriculum_start_end_box_noise[0])
            elif total_numsteps > cfg.curriculum_start_end_box_noise[1]:
                env.cfg.box_noise[2] = cfg.curriculum_box_noise_max[1]
                env.cfg.box_noise[1] = cfg.curriculum_box_noise_min[1]
            else:
                env.cfg.box_noise[2] = cfg.curriculum_box_noise_max[0]
                env.cfg.box_noise[1] = cfg.curriculum_box_noise_min[0]    
                
        #########################################################################################################################################################
        
        
        state, state_cpg = env.reset()
        

        while not done:
            if cfg.start_steps > total_numsteps:
                # if episode_steps == 0:
                #     action = np.random.uniform(-1, 1, env.action_space.shape)
                # action += np.random.uniform(-0.005, 0.005, env.action_space.shape)
                # action = np.clip(action, -1, 1) # Sample random action
                
                # action = np.random.normal(0, 1, env.action_space.shape)  # 平均0、標準偏差1の正規分布に基づく値を生成
                # action = np.tanh(action)
                
                if i_episode%10 == 0:
                    action_type = 0
                else:
                    action_type = 1
                
                action = env.sample_action(type=action_type)
                
            else:
                action = agent.select_action(state)  # Sample action from policy

                if len(memory) > cfg.batch_size:
                    # Number of updates per step in environment
                    if total_numsteps%cfg.updates_interval==0:
                        for _ in range(cfg.updates_per_step):
                            # Update parameters of all the networks
                            
                            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, cfg.batch_size, updates)
                            updates += 1
                    if total_numsteps%cfg.log_interval==0:
                        with open(f"{train_log_dir}/CSVs/log.csv", 'a',newline='') as file:
                            writer = csv.writer(file)
                            writer.writerows([[total_numsteps, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha]])
            
            action_cpg = cpg_agent.select_action(state_cpg)
            
            next_state, next_state_cpg, reward, termination, truncation, info = env.step(action, action_cpg) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            episode_work += info[0]
            episode_mileage += info[1]
            episode_angle += info[2]
            episode_reward_list += info[3]
            mask = int(not termination)
            
            
            if not np.isnan(reward):
                memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state
            state_cpg = next_state_cpg
            done = termination or truncation
            
        if env.capture == True:
            env.save_video(video_folder=f"{train_log_dir}/Videos", video_name=f"episode_{i_episode}.mp4")
            
        if total_numsteps >= cfg.start_steps and episode_reward > max_episode_rewards:
            agent.save_checkpoint(ckpt_dir=f"{train_log_dir}/Networks", file_name=f"best.pt" )
            max_episode_rewards = episode_reward
            best_episode = i_episode
        if total_numsteps >= cfg.start_steps and i_episode%100 == 0:
            agent.save_checkpoint(ckpt_dir=f"{train_log_dir}/Networks", file_name=f"episode_{i_episode}.pt" )
            
        # print("Episode: {},  episode steps: {}, total numsteps: {}, reward: {}".format(i_episode,  episode_steps, total_numsteps, round(episode_reward, 2)))
        with open(train_log_dir + "/CSVs/rewards.csv", 'a',newline='') as file:
            writer = csv.writer(file)
            # writer.writerows([[i_episode, episode_steps, total_numsteps, round(episode_reward, 2)]])
            writer.writerows([[i_episode, episode_steps, total_numsteps, round(episode_reward, 4), round(episode_reward_list[0], 4), round(episode_reward_list[1], 4), round(episode_reward_list[2], 4),round(episode_reward_list[3], 4),round(episode_reward_list[4], 4),round(episode_reward_list[5], 4),round(episode_reward_list[6], 4),round(episode_reward_list[7], 4),round(episode_reward_list[8], 4),round(episode_reward/episode_steps, 4),round(episode_mileage[0], 4), round(episode_mileage[1], 4), round(episode_angle[2], 4),round(episode_work, 4),datetime.datetime.now(timezone)-start_datetime,env.cfg.terrain,env.box_noise,termination,truncation],])
                
        if total_numsteps >= cfg.num_steps:
            break
    
    memory.save_buffer(save_dir=train_log_dir + "/Buffer")
    
    with open(f"{train_log_dir}/log.txt", 'a') as file:
        finish_datetime = datetime.datetime.now(timezone)
        finish = finish_datetime.strftime("%y/%m/%d %H:%M:%S")
        file.write(f'Finished: {finish}\n')
        file.write(f'It takes {finish_datetime - start_datetime}\n')      
        file.write(f'Best Episode: {best_episode}\n')  

if __name__ == '__main__':
    args = Parse_args()
    main(args)