import argparse

import os

import datetime
import pytz
import sys
import numpy as np
import itertools
import torch
import csv
import glob

import random



script_dir = os.path.dirname(__file__)    
parent_dir1 = os.path.dirname(script_dir)  
parent_dir2 = os.path.dirname(parent_dir1)
sys.path.append(parent_dir2)

script_name = os.path.basename(__file__)[: -len(".py")]

from SoftActorCritic.SAC import SAC_Eval
from A1_CPGEnv import A1CPGEnv
from Simulation.cfg.config_cpgrl import Config
from cfg.Save_Load_cfg import json_to_dataclass,dataclass_to_json

def Parse_args():
    parser = argparse.ArgumentParser(description='SAC eval')

    parser.add_argument("--train_log", type=str, default="~/Log/241021_173827",help="train log dir name")

    
    parser.add_argument("--gpu", type=int, default=0, help="run on CUDA")
    parser.add_argument("--seed", type=int, default=123456, help="seed")
    

    parser.add_argument("--cap", type=bool, default=True,help="capture video")
    parser.add_argument("--view", type=bool, default=False,help="Render")
    
    parser.add_argument("--net", type=int, default=None,help="Networks(episode)")
    
    parser.add_argument("--n_ep", type=int, default=1, help="num episodes")
    parser.add_argument("--mil", type=int, default=10,help="mileage[m]")
    parser.add_argument("--epl", type=int, default=20,help="episode_length[s]")
    
    parser.add_argument("--ter", type=str, default="flat",help="Terrain")
    parser.add_argument("--dkbk", type=float, default=0.15, help="dekoboko")
    
    parser.add_argument("--cx", type=float, default=0.4, help="command")
    parser.add_argument("--cy", type=float, default=0.0, help="command")
    parser.add_argument("--cw", type=float, default=0.0, help="command")
    
    parser.add_argument("--alog", type=bool, default=True,help="action log")
    parser.add_argument("--olog", type=bool, default=True,help="observation log")
    parser.add_argument("--com", type=str, default="obs_min",help="comment")
    
    args = parser.parse_args()
    
    return args



def main(args):
    
    # Networks
    if args.net is None:
        network_files = glob.glob(f"{args.train_log}/Networks/episode_*.pt")
        if network_files:
            latest_network = max(network_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            networks = latest_network
        else:
            raise FileNotFoundError("No network files found.")
    elif args.net == 0:
            networks = f"{args.train_log}/Networks/best.pt"
    else:
        networks = f"{args.train_log}/Networks/episode_{args.net}.pt"
    
    # make log dir
    timezone = pytz.timezone('Asia/Tokyo')
    start_datetime = datetime.datetime.now(timezone)    
    start_formatted = start_datetime.strftime("%y%m%d_%H%M%S")
    test_log_dir = f"{args.train_log}/Test/{start_formatted}"
    if not os.path.exists(test_log_dir):
        os.makedirs(test_log_dir)
        os.makedirs(f"{test_log_dir}/CSVs")


    # train log
    with open(f"{test_log_dir}/log.txt", 'w') as file:
        file.write(f'Networks: {networks}\n')
        start = start_datetime.strftime("%y/%m/%d %H:%M:%S")
        #PID
        pid = os.getpid()
        file.write(f'Comment: {args.com}\n')
        file.write(f'Process ID: {pid}\n')
        file.write(f'Start: {start}\n')
    
    with open(f"{test_log_dir}/CSVs/rewards.csv", 'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerows([["Episode","Episode_Steps","Rewards","Ave_Rewards","Mileage_x","Mileage_y","Angle_w","Work","Time"]])
        writer.writerows([[0,]])
    

    # Set config
    cfg_data = json_to_dataclass(file_path=f"{args.train_log}/config.json")
    cfg = Config(**cfg_data)
    cfg.seed = args.seed
    cfg.gpu = args.gpu
    cfg.terrain = args.ter
    cfg.command_x[0] = args.cx
    cfg.command_y[0] = args.cy
    cfg.command_w[0] = args.cw
    cfg.episodeLength_s = args.epl
    
    dataclass_to_json(cfg,file_path=f"{test_log_dir}/config.json")
    
    
    
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
    
    # Environment
    env = A1CPGEnv(cfg,capture=args.cap,eval=True)

    # Agent
    agent = SAC_Eval(env.observation_space.shape, env.action_space.shape, cfg)

    agent.load_checkpoint(ckpt_path=networks,evaluate=True)

    commands = []

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        commands.append([env.command[0],env.command[1],env.command[2]])
        episode_work = 0
        episode_mileage = np.array([0.0,0.0,0.0])
        episode_angle = np.array([0.0,0.0,0.0])
        
        print(f"Episode:{i_episode}")
        
        if args.alog:
            with open(f"{test_log_dir}/CSVs/action_{i_episode}.csv", 'w',newline='') as file:
                writer = csv.writer(file)
                writer.writerows([["step",
                                "mu_FR","mu_FL","mu_RR","mu_RL",
                                "omega_FR","omega_FL","omega_RR","omega_RL",
                                "psi_FR","psi_FL","psi_RR","psi_RL",
                                "FR_x","FR_y","FR_z",
                                "FL_x","FL_y","FL_z",
                                "RR_x","RR_y","RR_z",
                                "RL_x","RL_y","RL_z",]])
                
        if args.olog:
            with open(f"{test_log_dir}/CSVs/obs_{i_episode}.csv", 'w',newline='') as file:
                writer = csv.writer(file)
                if cfg.observation_space[0] == "min":
                    writer.writerows([["step",
                                    "foot_force_FR","foot_force_FL","foot_force_RR","foot_force_RL",
                                    "r_FR","r_FL","r_RR","r_RL",
                                    "theta_FR","theta_FL","theta_RR","theta_RL",
                                    "phi_FR","phi_FL","phi_RR","phi_RL",
                                    "r_dot_FR","r_dot_FL","r_dot_RR","r_dot_RL",
                                    "omega_FR","omega_FL","omega_RR","omega_RL",
                                    "psi_FR","psi_FL","psi_RR","psi_RL",
                                    "command_x","command_y","command_omega_z",]])

                else:    
                    writer.writerows([["step",
                                    "q_0","q_1","q_2","q_3","q_4","q_5","q_6","q_7","q_8","q_9","q_10","q_11",
                                    "dq_0","dq_1","dq_2","dq_3","dq_4","dq_5","dq_6","dq_7","dq_8","dq_9","dq_10","dq_11",
                                    "foot_force_FR","foot_force_FL","foot_force_RR","foot_force_RL",
                                    "euler_roll","euler_pitch",
                                    "angle_vel_roll","angle_vel_pitch","angle_vel_yaw",
                                    "linear_acc_x","linear_acc_y","linear_acc_z",
                                    "r_FR","r_FL","r_RR","r_RL",
                                    "theta_FR","theta_FL","theta_RR","theta_RL",
                                    "phi_FR","phi_FL","phi_RR","phi_RL",
                                    "r_dot_FR","r_dot_FL","r_dot_RR","r_dot_RL",
                                    "omega_FR","omega_FL","omega_RR","omega_RL",
                                    "psi_FR","psi_FL","psi_RR","psi_RL",
                                    "command_x","command_y","command_omega_z",]])

        while not done:
            
            # print(f"gain_factor:{env.gain_factor}")
            
            action = agent.select_action(state,evaluate=True)  # Sample action from policy
            
            if args.alog:
                with open(f"{test_log_dir}/CSVs/action_{i_episode}.csv", 'a',newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows([[episode_steps]+action.tolist()])
            
            if args.olog:
                with open(f"{test_log_dir}/CSVs/obs_{i_episode}.csv", 'a',newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([episode_steps] + state.tolist())
        
            next_state, reward, termination, truncation, info = env.step(action) # Step
            episode_steps += 1
            episode_reward += reward
            episode_work += info[0]
            episode_mileage += info[1]
            episode_angle += info[2]

            state = next_state
            
            # truncation = truncation or episode_mileage[0] >= args.mil
            truncation = truncation or episode_mileage[0] >= args.mil or episode_mileage[1] >= args.mil
            
            done = termination or truncation
            
        if env.capture:
            env.save_video(f"{test_log_dir}/Videos",f"video_{i_episode}.mp4")

        with open(f"{test_log_dir}/CSVs/rewards.csv", 'a',newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[i_episode, episode_steps, round(episode_reward, 4), round(episode_reward/episode_steps, 4),round(episode_mileage[0], 4), round(episode_mileage[1], 4), round(episode_angle[2], 4),round(episode_work, 4),datetime.datetime.now(timezone)-start_datetime]])
                
        if i_episode >= args.n_ep:
            break
    
    with open(f"{test_log_dir}/log.txt", 'a') as file:
        finish_datetime = datetime.datetime.now(timezone)
        finish = finish_datetime.strftime("%y/%m/%d %H:%M:%S")
        file.write(f'Commands: {commands}\n')
        file.write(f'Finished: {finish}\n')
        file.write(f'It takes {finish_datetime - start_datetime}\n')

if __name__ == '__main__':
    args = Parse_args()
    main(args)