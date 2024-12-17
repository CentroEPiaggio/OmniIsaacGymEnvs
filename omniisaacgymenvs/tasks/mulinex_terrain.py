

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.mulinex import Mulinex
from omniisaacgymenvs.robots.articulations.views.mulinex_view import MulinexView
from omniisaacgymenvs.tasks.utils.anymal_terrain_generator import *
from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *
from omniisaacgymenvs.tasks.utils.data_plotter import DataPlotter
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.prims import GeometryPrim
from omni.isaac.core.materials import PhysicsMaterial
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from time import monotonic, sleep

import omni.replicator.isaac as dr
import omni.replicator.core as rep
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import math
import sys

from pxr import UsdPhysics, UsdLux

from omniisaacgymenvs.tasks.utils.plot_utils import plot_results, append_mean_and_std, add_data
from tensorboardX import SummaryWriter


class MulinexTerrainTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.height_samples = None
        self.custom_origins = False
        self.init_done = False

        self.quadrants = ['LF', 'LH', 'RF', 'RH']
        
        
        self.dp_show = self._cfg.get('dp_show', False)
        self.dp_sdp = self._cfg.get('dp_sdp')
        self.with_dp = self.dp_show or (self.dp_sdp is not None)
        self.dp_fls = self._cfg.get('dp_fls')
        self.dp_env_id = self._cfg.get('dp_env_id', 0)

        
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.height_meas_scale = self._task_cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale = self._task_cfg["env"]["control"]["actionScale"]

        
        self.rew_scales = {}
        self.rew_scales["termination"] = self._task_cfg["env"]["learn"]["terminalReward"] 
        self.rew_scales["lin_vel_xy"] = self._task_cfg["env"]["learn"]["linearVelocityXYRewardScale"] 
        self.rew_scales["lin_vel_z"] = self._task_cfg["env"]["learn"]["linearVelocityZRewardScale"] 
        self.rew_scales["ang_vel_z"] = self._task_cfg["env"]["learn"]["angularVelocityZRewardScale"] 
        self.rew_scales["ang_vel_z_penalty"] = self._task_cfg["env"]["learn"]["angularVelocityZPenaltyScale"] 
        self.rew_scales["ang_vel_xy"] = self._task_cfg["env"]["learn"]["angularVelocityXYRewardScale"] 
        self.rew_scales["orient"] = self._task_cfg["env"]["learn"]["orientationRewardScale"] 
        self.rew_scales["torque"] = self._task_cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["base_height"] = self._task_cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["hip"] = self._task_cfg["env"]["learn"]["hipRewardScale"]
        self.rew_scales["fallen_over"] = self._task_cfg["env"]["learn"]["fallenOverRewardScale"]
        self.rew_scales["feet_pos_z"] = self._task_cfg["env"]["learn"]["feetPosZRewardScale"]
        self.rew_scales["stand_still"] = self._task_cfg["env"]["learn"]["standStillRewardScale"]
        self.rew_scales["action_accel"] = self._task_cfg["env"]["learn"]["action_accel"]
        self.rew_scales["action_jerk"] = self._task_cfg["env"]["learn"]["action_jerk"]
        self.rew_scales["joint_jerk"] = self._task_cfg["env"]["learn"]["joint_jerk"]
        self.rew_scales["falcata"] = self._task_cfg["env"]["learn"]["falcataRewardScale"]
        self.rew_scales["actionjerk_max"] = self._task_cfg["env"]["learn"]["actionjerk_maxRewardScale"]
        self.rew_scales["jointjerk_max"] = self._task_cfg["env"]["learn"]["jointjerk_maxRewardScale"]
        self.rew_scales["actionaccel_max"] = self._task_cfg["env"]["learn"]["actionaccel_maxRewardScale"]
        self.rew_scales["jointaccel_max"] = self._task_cfg["env"]["learn"]["jointaccel_maxRewardScale"]
        self.rew_scales["air_time"] = self._task_cfg["env"]["learn"]["airtimeRewardScale"]
        self.rew_scales["vz+"] = self._task_cfg["env"]["learn"]["vzRewardScale+"]
        self.rew_scales["base_height_+"] = self._task_cfg["env"]["learn"]["base_height_+"]
        self.rew_scales["AngularVelocity+"] = self._task_cfg["env"]["learn"]["AngularVelocityRewardScale+"]
        self.rew_scales["actRate+"] = self._task_cfg["env"]["learn"]["actRateRewScale+"]
        self.rew_scales["jointAcc+"] = self._task_cfg["env"]["learn"]["jointAccRewScale+"]
        self.rew_scales["actionAccel+"] = self._task_cfg["env"]["learn"]["actionAccelRewScale+"]
        self.rew_scales["jointJerk+"] = self._task_cfg["env"]["learn"]["jointJerkRewardScale+"]
        self.rew_scales["actionJerk+"] = self._task_cfg["env"]["learn"]["actionJerkRewScale+"]
        self.rew_scales["orient+"] = self._task_cfg["env"]["learn"]["orientationsRewardScale+"]
        self.rew_scales["falcata+"] = self._task_cfg["env"]["learn"]["falcataScale+"]
        self.rew_scales["feetZ+"] = self._task_cfg["env"]["learn"]["feetZScale+"]
        self.rew_scales["actionjerk_max+"] = self._task_cfg["env"]["learn"]["actionjerk_maxRewardScale+"]
        self.rew_scales["jointjerk_max+"] = self._task_cfg["env"]["learn"]["jointjerk_maxRewardScale+"]
        self.rew_scales["actionaccel_max+"] = self._task_cfg["env"]["learn"]["actionaccel_maxRewardScale+"]
        self.rew_scales["jointaccel_max+"] = self._task_cfg["env"]["learn"]["jointaccel_maxRewardScale+"]
        self.rew_scales["air_time+"] = self._task_cfg["env"]["learn"]["airtimeRewardScale+"]
        self.rew_scales["action_rateGN"] = self._task_cfg["env"]["learn"]["actionRateLocalMalus"]

        self.feet_pos_z_val = self._task_cfg["env"]["learn"]["feetPosZ"]
        self.des_base_height = self._task_cfg["env"]["learn"]["des_base_height"]
        self.falcata_des = self._task_cfg["env"]["learn"]["falcata_des"]
        self.air_time_des = self._task_cfg["env"]["learn"]["air_time_des"]

        
        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        self.command_yaw_rate_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw_rate"]

        
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        
        self.sim_dt = self._task_cfg["sim"]["dt"]
        self.decimation = self._task_cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self._task_cfg["sim"]["dt"]                          
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"] 
        self.max_episode_length = int(self.max_episode_length_s/ self.dt + 0.5)
        self.push_interval = int(self._task_cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.randomize_terrain_interval = int(self._task_cfg["env"]["terrain"]["terrain_rand_int_s"] / self.dt + 0.5)   
        self.change_command_interval = int(self._task_cfg["env"]["learn"]["command_change_int_s"] / self.dt + 0.5)    
        self.mix_com_percentage = self._task_cfg["env"]["learn"]["mixed_command_percentage"]

        self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]
        self.base_threshold = 0.2
        self.knee_threshold = 0.1

        
        
        self.use_dataset_percentage = self._task_cfg["env"]["learn"]["use_dataset_percentage"]              
        
        self.failure_dataset = {
                            'dof_pos': {
                                'yaw_180': [2.094, -2.094, -2.094, 2.094, -1.0472, 1.0472, 1.0472, -1.0472],

                            },
                            
                            
                            
                            
                            'base_quat': {
                                'yaw_180': [-4.3711e-8, 0.0, 0.0, 1.0],

                            },
                            'base_pos': {
                                'yaw_180': [0.0, 0.0, 0.35],

                            },
                            'base_velocities': {
                                'yaw_180': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

                            }
        }
        
        lista_dof_pos = list(self.failure_dataset['dof_pos'].values())    
        self.tensore_dof_pos = torch.tensor(lista_dof_pos, dtype=torch.float) 
        
        
        lista_base_quat = list(self.failure_dataset['base_quat'].values())
        self.tensore_base_quat = torch.tensor(lista_base_quat, dtype=torch.float)
        lista_base_pos = list(self.failure_dataset['base_pos'].values())
        self.tensore_base_pos = torch.tensor(lista_base_pos, dtype=torch.float)
        lista_base_velocities = list(self.failure_dataset['base_velocities'].values())
        self.tensore_base_velocities = torch.tensor(lista_base_velocities, dtype=torch.float)
        
        self.tests = list(self.failure_dataset['dof_pos'].keys())
        
        
        
        self.var_lin_vel = self._task_cfg["env"]["learn"]["var_lin_vel"]
        self.fBc_base = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_base"]
        self.fBc_vz = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_vz"]
        self.fBc_rollrate = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_rollrate"]
        self.fBc_pitchrate = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_pitchrate"]
        self.fBc_kneeactionRate = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_kneeactionRate"]
        self.fBc_hipactionRate = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_hipactionRate"]
        self.fBcactRateGNscale = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_actionRateScale"]
        self.fBc_kneeAccel = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_kneeAccel"]
        self.fBc_hipAccel = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_hipAccel"]
        self.fBc_kneeactionAccel = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_kneeactionAccel"]
        self.fBc_hipactionAccel = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_hipactionAccel"]
        self.fBc_kneej = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_kneej"]
        self.fBc_hipj = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_hipj"]
        self.fBc_kneeactionJ = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_kneeactionJ"]
        self.fBc_hipactionJ = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_hipactionJ"]
        self.fBc_orient = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_orient"]
        self.fBc_falcata = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_falcata"]
        self.fBc_feetZ = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_feet_posZ"]
        self.fBc_actionjerk_max = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_actionjerk_max"]
        self.fBc_jointjerk_max = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_jointjerk_max"]
        self.fBc_actionaccel_max = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_actionaccel_max"]
        self.fBc_jointaccel_max = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_jointaccel_max"]
        self.fBc_airtime = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_airtime"]
        self.fBc_angz = self._task_cfg["env"]["learn"]["fuzzyBlendCoef_angz"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._num_observations = self._task_cfg["env"]["numObservations"]
        self._num_states = self._task_cfg["env"]["numStates"]
        self._num_actions = self._task_cfg["env"]["numActions"]

        self._task_cfg["sim"]["default_physics_material"]["static_friction"] = self._task_cfg["env"]["terrain"]["staticFriction"]
        self._task_cfg["sim"]["default_physics_material"]["dynamic_friction"] = self._task_cfg["env"]["terrain"]["dynamicFriction"]
        self._task_cfg["sim"]["default_physics_material"]["restitution"] = self._task_cfg["env"]["terrain"]["restitution"]
   
        self._task_cfg["sim"]["add_ground_plane"] = False
        self._env_spacing = 0.0                                                          

        self.cnt = 0                                                                     
        self.t = 0
        self.t0 = None
        self.action1 = True                                                              
        self.randomize_terrain_properties = self._task_cfg["env"]["terrain"]["randomize_terrain"]  
        self.enable_command_changes = self._task_cfg["env"]["learn"]["enable_command_changes"]    

        RLTask.__init__(self, name, env)
        
        if self._dr_randomizer.randomize:
            self.randomize_actions = True
            self.dr = dr

        self.Kp_hip, self.Kp_knee = self._task_cfg['env']['control']['Kp']['hip'], self._task_cfg['env']['control']['Kp']['knee']
        self.Kd_hip, self.Kd_knee = self._task_cfg['env']['control']['Kd']['hip'], self._task_cfg['env']['control']['Kd']['knee']
        self.Kp = torch.tensor([self.Kp_hip] * 4 + [self.Kp_knee] * 4, dtype=torch.float, device=self.device)
        self.Kd = torch.tensor([self.Kd_hip] * 4 + [self.Kd_knee] * 4, dtype=torch.float, device=self.device)
        self.max_torque = self._task_cfg['env']['control']['maxTorque']
        self.use_implicit_PD = self._task_cfg['env']['control']['useImplicitPD']

        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.has_fallen = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.simulation_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        
        self.up_axis_idx = 2
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self._task_cfg)
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) 
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], device=self.device, requires_grad=False,)
        self.gravity_vec = torch.tensor(get_axis_params(-1., self.up_axis_idx), dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = torch.tensor([1., 0., 0.], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, 4, dtype=torch.float, device=self.device, requires_grad=False)
    
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
    
        self.torques_ind = torch.zeros(self.num_envs, self.num_actions, 2, dtype=torch.float, device=self.device, requires_grad=False)

        self.height_points = self.init_height_points()
        self.height_points_upsample = torch.zeros((self.num_envs, 861), dtype=torch.float, device=self.device, requires_grad=False)
        print("Height point initialized")
        self.measured_heights = None
        
        self.default_dof_pos = torch.zeros((self.num_envs, 8), dtype=torch.float, device=self.device, requires_grad=False)
        
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"lin_vel_xy": torch_zeros(), "lin_vel_z": torch_zeros(), "ang_vel_z": torch_zeros(), "ang_vel_xy": torch_zeros(),
                             "orient": torch_zeros(), "torques": torch_zeros(), "joint_acc": torch_zeros(), "base_height": torch_zeros(),
                             "air_time": torch_zeros(), "collision": torch_zeros(), "stumble": torch_zeros(), "action_rate": torch_zeros(), "hip": torch_zeros(),
                             "hip_joint_errors": torch_zeros(), "knee_joint_errors": torch_zeros(),
                             "torque": torch_zeros(), "torque_max": torch_zeros(),
                             "LF_foot_air_time": torch_zeros(), "LH_foot_air_time": torch_zeros(), "RF_foot_air_time": torch_zeros(), "RH_foot_air_time": torch_zeros(),
                             "LF_foot_pos_z": torch_zeros(), "LH_foot_pos_z": torch_zeros(), "RF_foot_pos_z": torch_zeros(), "RH_foot_pos_z": torch_zeros(),
                             "action_accel": torch_zeros(), "action_jerk": torch_zeros(), "joint_jerk": torch_zeros(), "falcata": torch_zeros(), "feet_pos_z": torch_zeros(),
                             "action_jerk_max": torch_zeros(), "joint_jerk_max": torch_zeros(), "action_accel_max": torch_zeros(), "joint_accel_max": torch_zeros(), 
                             "mean_falcata": torch_zeros(), "max_falcata": torch_zeros(), "feet_air_time": torch_zeros(), "mean_air_time": torch_zeros(), "max_air_time": torch_zeros()}
        
        self.writer = SummaryWriter(f"/isaac-sim/OmniIsaacGymEnvs/omniisaacgymenvs/runs/{self._cfg['train']['params']['config']['name']}/summaries")
        self.ep_count = 0
        self.step_counter_terrain = 0    
        self.step_counter_command = 0    






















 


    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self._task_cfg["env"]["learn"]["addNoise"]
        noise_level = self._task_cfg["env"]["learn"]["noiseLevel"]
        noise_vec[:3] = self._task_cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        noise_vec[3:6] = self._task_cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        noise_vec[6:9] = self._task_cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[9:12] = 0. 
        noise_vec[12:20] = self._task_cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        noise_vec[20:28] = self._task_cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
        noise_vec[28:259] = self._task_cfg["env"]["learn"]["heightMeasurementNoise"] * noise_level * self.height_meas_scale  
        noise_vec[259:] = 0. 
        return noise_vec
    
    def init_height_points(self):
        
        y = 0.1 * torch.tensor([-5, -4, -3,-2,-1, 0, 1, 2, 3, 4, 5], device=self.device, requires_grad=False)                                       
        x = 0.1 * torch.tensor([-10, -9,-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], device=self.device, requires_grad=False) 
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _create_trimesh(self):
        self.terrain = Terrain(self._task_cfg["env"]["terrain"], num_robots=self.num_envs)
        vertices = self.terrain.vertices
        triangles = self.terrain.triangles
        position = torch.tensor([-self.terrain.border_size , -self.terrain.border_size , 0.0])
        add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)  
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        self.get_terrain()
        self.get_mulinex()
        super().set_up_scene(scene)
        self._mulinexs = MulinexView(prim_paths_expr="/World/envs/.*/mulinex", name="mulinex_view", track_contact_forces=True)
        scene.add(self._mulinexs)
        scene.add(self._mulinexs._bases)
        scene.add(self._mulinexs._knees)
        scene.add(self._mulinexs._feet)
        if self.randomize_terrain_properties:                                            
            self.terrain_geom = GeometryPrim('/World/terrain', 'terrain_geometry_prim')  
            
            self.random_static_friction = torch_rand_float(self._task_cfg["env"]["terrain"]["lower_staticFriction"], self._task_cfg["env"]["terrain"]["upper_staticFriction"], (1,1), device=self.device).item()
            self.random_dynamic_friction = torch_rand_float(self._task_cfg["env"]["terrain"]["lower_dynamicFriction"], self._task_cfg["env"]["terrain"]["upper_dynamicFriction"], (1,1), device=self.device).item()
            self.random_restitution = torch_rand_float(self._task_cfg["env"]["terrain"]["lower_restitution"], self._task_cfg["env"]["terrain"]["upper_restitution"], (1,1), device=self.device).item()
            
            self.terrain_material = PhysicsMaterial('/World/terrain/physics_material', 'terrain_physics_material', self.random_static_friction, self.random_dynamic_friction, self.random_restitution)
            self.terrain_geom.apply_physics_material(self.terrain_material)
            self.terrain_material.set_dynamic_friction(self.random_dynamic_friction)
            self.terrain_material.set_static_friction(self.random_static_friction)
            self.terrain_material.set_restitution(self.random_restitution)              
        self._dr_randomizer.apply_on_startup_domain_randomization(self)  
        print("setup scene done")

    def get_terrain(self):
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        if not self.curriculum: self._task_cfg["env"]["terrain"]["maxInitMapLevel"] = self._task_cfg["env"]["terrain"]["numLevels"] - 1
        self.terrain_levels = torch.randint(0, self._task_cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.randint(0, self._task_cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)
        self._create_trimesh()
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
        print("Get terrain done")

    def get_mulinex(self):
        self.base_init_state = torch.tensor(self.base_init_state, dtype=torch.float, device=self.device, requires_grad=False)
        mulinex_translation = torch.tensor([0.0, 0.0, 0.35])                    

        mulinex_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
        mulinex = Mulinex(prim_path=self.default_zero_env_path + "/mulinex", 
                          name="mulinex",
                          translation=mulinex_translation, 
                          orientation=mulinex_orientation)
        self._sim_config.apply_articulation_settings("mulinex", get_prim_at_path(mulinex.prim_path), self._sim_config.parse_actor_config("mulinex"))
        mulinex.set_mulinex_properties(self._stage, mulinex.prim)
        mulinex.prepare_contacts(self._stage, mulinex.prim)
        if self.use_implicit_PD:
            self.set_drives()

        self.dof_names = mulinex.dof_names
        for i in range(self.num_actions):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

    def set_drives(self):
        for quadrant in self.quadrants:
            set_drive(f'{self.default_zero_env_path}/mulinex/base_link/{quadrant}_HFE', 'angular', 'position', 0, self.Kp_hip, self.Kd_hip, self.max_torque)
            set_drive(f'{self.default_zero_env_path}/mulinex/{quadrant}_UPPER_LEG/{quadrant}_KFE', 'angular', 'position', 0, self.Kp_knee, self.Kd_knee, self.max_torque)


    def post_reset(self):
        for i in range(self.num_envs):
            self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
        self.num_dof = self._mulinexs.num_dof
        self.dof_pos_cmd = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof, 3), dtype=torch.float, device=self.device)
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.projected_gravity = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        
        
        self.rotation_ = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.knee_pos = torch.zeros((self.num_envs*4, 3), dtype=torch.float, device=self.device)
        self.knee_quat = torch.zeros((self.num_envs*4, 4), dtype=torch.float, device=self.device)

        self.feet_pos_z = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_pos_x = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_pos = torch.zeros((self.num_envs, 4, 3), dtype=torch.float, device=self.device, requires_grad=False)
    
        self.spost_norm = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_contact = torch.ones((self.num_envs, 4), dtype=torch.bool, device=self.device, requires_grad=False)
        self.prev_feet_contact = torch.ones((self.num_envs, 4), dtype=torch.bool, device=self.device, requires_grad=False)
        self.step_completetd = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_steps = torch.zeros((self.num_envs, 4), dtype=torch.long, device=self.device, requires_grad=False)

        if self.with_dp:
            self.dp = DataPlotter(dof_names=self.dof_names, filter_length_s=self.dp_fls)

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
        self.init_done = True
        
        if self._dr_randomizer.randomize:    
            self._dr_randomizer.set_up_domain_randomization(self)    

    def add_plot_data(self):
        self.dp.add_data(t=self.simulation_time[self.dp_env_id].clone(),
                         joint_pos_cmd=self.dof_pos_cmd[self.dp_env_id].clone(),
                         joint_pos=self.dof_pos[self.dp_env_id].clone(),
                         joint_vel=self.dof_vel[self.dp_env_id,:,-1].clone(),
                         joint_torques=self.torques[self.dp_env_id,].clone(),
                         base_lin_vel_cmd=[self.commands[self.dp_env_id, 0].clone(), 0., 0.],
                         base_ang_vel_cmd=[0., 0., self.commands[self.dp_env_id, 1].clone()],
                         base_lin_vel=self.base_lin_vel[self.dp_env_id].clone(),
                         base_ang_vel=self.base_ang_vel[self.dp_env_id].clone(),
                         base_pg=self.projected_gravity[self.dp_env_id].clone(),
                         base_quat=self.base_quat[self.dp_env_id].clone())

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)

        if self.with_dp and self.init_done and (self.dp_env_id in env_ids):
            self.dp.plot(show=self.dp_show, save_dir_path=self.dp_sdp)
            del self.dp
            self.dp = DataPlotter(dof_names=self.dof_names, filter_length_s=self.dp_fls)

        if self.use_dataset_percentage != 0.0:     
            
            
            def assign_values(env_ids, use_dataset_percentage, tests):

                """input: use_dataset_percentage è la percentuale di robot che riceveranno valori dal dataset;
                tests è una lista che contiene le key corrispondenti ai nomi dei vari test presenti nel dataset. \n
                ===> La funzione assign_values sceglie casualmente un sottoinsieme di env_ids che useranno i valori dal
                dataset ed assegna i valori ai robot in base a questa selezione.
                Per i robot che useranno il dataset, viene scelto casualmente un test da tests per ogni env_ids"""
                
                env_ids_ = env_ids.tolist()     
                num_envs_using_dataset = int(len(env_ids_) * self.use_dataset_percentage)  
                env_ids_using_dataset = random.sample(env_ids_, num_envs_using_dataset)    

                for env_id in env_ids_:                    
                    if env_id in env_ids_using_dataset:    
                        
                        test_key = random.choice(tests)            
                        test_key_idx = self.tests.index(test_key)  
                        self.update_terrain_level(env_ids_)
                        
                        self.dof_pos[env_id] = self.tensore_dof_pos[test_key_idx]
                        
                        self.base_quat[env_id] = self.tensore_base_quat[test_key_idx]
                        self.base_pos[env_id] = self.tensore_base_pos[test_key_idx]
                        self.base_velocities[env_id] = self.tensore_base_velocities[test_key_idx]

                    else:
                        
                        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids_), self.num_dof), device=self.device)
                        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids_), self.num_dof), device=self.device)
                        self.dof_pos[env_ids_] = self.default_dof_pos[env_ids_] * positions_offset
                        self.dof_vel[env_ids_, :, -1] = velocities 
                        self.update_terrain_level(env_ids_)
                        self.base_pos[env_ids_] = self.base_init_state[0:3]
                        self.base_pos[env_ids_, 0:3] += self.env_origins[env_ids_]
                        self.base_pos[env_ids_, 0:2] += torch_rand_float(-0.5, 0.5, (len(env_ids_), 2), device=self.device)
                        self.base_quat[env_ids_] = self.base_init_state[3:7]
                        self.base_velocities[env_ids_] = self.base_init_state[7:]
                        

    
    
            
            assign_values(env_ids=env_ids, use_dataset_percentage = self.use_dataset_percentage, tests = self.tests)
            
        else:
            
            positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
            velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

            self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
            self.dof_vel[env_ids, :, -1] = velocities 

            self.update_terrain_level(env_ids)
            self.base_pos[env_ids] = self.base_init_state[0:3]
            self.base_pos[env_ids, 0:3] += self.env_origins[env_ids]
            self.base_pos[env_ids, 0:2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
            self.base_quat[env_ids] = self.base_init_state[3:7]
            self.base_velocities[env_ids] = self.base_init_state[7:]
            

        
        
        
        def degrees_to_radians(degrees):    
            return degrees * math.pi / 180
        
        def quaternion_to_euler_angle(q):   
            w, x, y, z = q
            ysqr = y * y
            
            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + ysqr)
            roll_x = math.atan2(t0, t1)
            
            t2 = +2.0 * (w * y - z * x)
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch_y = math.asin(t2)
            
            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (ysqr + z * z)
            yaw_z = math.atan2(t3, t4)

            return roll_x, pitch_y, yaw_z 

        def euler_angle_to_quaternion(roll, pitch, yaw):    
            qx = torch.sin(roll/2) * torch.cos(pitch/2) * torch.cos(yaw/2) - torch.cos(roll/2) * torch.sin(pitch/2) * torch.sin(yaw/2)
            qy = torch.cos(roll/2) * torch.sin(pitch/2) * torch.cos(yaw/2) + torch.sin(roll/2) * torch.cos(pitch/2) * torch.sin(yaw/2)
            qz = torch.cos(roll/2) * torch.cos(pitch/2) * torch.sin(yaw/2) - torch.sin(roll/2) * torch.sin(pitch/2) * torch.cos(yaw/2)
            qw = torch.cos(roll/2) * torch.cos(pitch/2) * torch.cos(yaw/2) + torch.sin(roll/2) * torch.sin(pitch/2) * torch.sin(yaw/2)
            return torch.tensor([qw, qx, qy, qz])
        
        def torch_rand_quaternion(roll_min, roll_max, pitch_min, pitch_max, yaw_min, yaw_max, env_ids, device):
            num_env_ids = env_ids.shape[0]                    
            shape = (num_env_ids, 3)                          
            random_values = torch.rand(shape, device=device)  
            
            random_values[:, 0] = random_values[:, 0] * (degrees_to_radians(roll_max - roll_min)) + degrees_to_radians(roll_min)
            random_values[:, 1] = random_values[:, 1] * (degrees_to_radians(pitch_max - pitch_min)) + degrees_to_radians(pitch_min)
            random_values[:, 2] = random_values[:, 2] * (degrees_to_radians(yaw_max - yaw_min)) + degrees_to_radians(yaw_min)
            
            random_quaternions = torch.zeros(num_env_ids, 4)
            for i in range(num_env_ids):    
                random_quaternions[i] = euler_angle_to_quaternion(random_values[i, 0], random_values[i, 1], random_values[i, 2])

            return random_quaternions

        
        roll_min = -5     
        roll_max = 5      
        pitch_min = -5    
        pitch_max = 5     
        yaw_min = -10     
        yaw_max = 10      
        
        quat_randomization = torch_rand_quaternion(roll_min, roll_max, pitch_min, pitch_max, yaw_min, yaw_max, env_ids, device=self.device)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        quat_randomization = quat_randomization.to(self.base_quat.device)    
        self.base_quat[env_ids] += quat_randomization                        
        


        self._mulinexs.set_world_poses(positions=self.base_pos[env_ids].clone(), 
                                      orientations=self.base_quat[env_ids].clone(),
                                      indices=indices)
        self._mulinexs.set_velocities(velocities=self.base_velocities[env_ids].clone(),
                                          indices=indices)
        self._mulinexs.set_joint_positions(positions=self.dof_pos[env_ids].clone(), 
                                          indices=indices)
        self._mulinexs.set_joint_velocities(velocities=self.dof_vel[env_ids, :, -1].clone(), 
                                          indices=indices)

        self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()  
        self.commands[env_ids, 1] = 0. 
        
        if self.mix_com_percentage != 0.0:
            env_ids_mc = env_ids.tolist()                                                  
            num_envs_using_yaw_rate = int(len(env_ids_mc) * self.mix_com_percentage)       
            env_ids_using_yaw_rate = random.sample(env_ids_mc, num_envs_using_yaw_rate)    
            for env_id in env_ids_mc:                                                      
                if env_id in env_ids_using_yaw_rate:                                       
                    self.commands[env_id, 2] = torch_rand_float(self.command_yaw_rate_range[0], self.command_yaw_rate_range[1], (1, 1), device=self.device).squeeze()  
                else:
                    
                    self.commands[env_id, 3] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (1, 1), device=self.device).squeeze()  
        else:
            

            self.commands[env_ids, 3] =  torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()  
        
        self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.15).unsqueeze(1) 

       
       
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        
        self.extras["episode"] = {}   
        for key in self.episode_sums.keys():
            if key not in ["hip_joint_errors", "knee_joint_errors", "torque", "torque_max", "mean_falcata", "max_falcata", "mean_air_time", "max_air_time"]:
                self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
                self.episode_sums[key][env_ids] = 0.
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        self.extras["episode"]["hip_joint_errors"] = torch.mean(self.episode_sums["hip_joint_errors"][env_ids] / (self.progress_buf[env_ids] * self.decimation))  
        self.extras["episode"]["knee_joint_errors"] = torch.mean(self.episode_sums["knee_joint_errors"][env_ids] / (self.progress_buf[env_ids] * self.decimation))
        self.extras["episode"]["torque_mean"] = torch.mean(self.episode_sums["torque"][env_ids] / (self.progress_buf[env_ids] * self.decimation))
        self.extras["episode"]["torque_standard_deviation"] = torch.std(self.episode_sums["torque"][env_ids] / (self.progress_buf[env_ids] * self.decimation))
        self.extras["episode"]["torque_max"] = torch.mean(self.episode_sums["torque_max"][env_ids])
        
        self.extras["episode"]["LF_foot_pos_z"] = torch.mean(self.episode_sums['LF_foot_pos_z'][env_ids] / (self.feet_steps[env_ids, 0] + sys.float_info.epsilon))
        self.extras["episode"]["LH_foot_pos_z"] = torch.mean(self.episode_sums['LH_foot_pos_z'][env_ids] / (self.feet_steps[env_ids, 1] + sys.float_info.epsilon))
        self.extras["episode"]["RF_foot_pos_z"] = torch.mean(self.episode_sums['RF_foot_pos_z'][env_ids] / (self.feet_steps[env_ids, 2] + sys.float_info.epsilon))
        self.extras["episode"]["RH_foot_pos_z"] = torch.mean(self.episode_sums['RH_foot_pos_z'][env_ids] / (self.feet_steps[env_ids, 3] + sys.float_info.epsilon))
        
        self.episode_sums['LF_foot_pos_z'][env_ids] = 0.
        self.episode_sums['LH_foot_pos_z'][env_ids] = 0.
        self.episode_sums['RF_foot_pos_z'][env_ids] = 0.
        self.episode_sums['RH_foot_pos_z'][env_ids] = 0.
      
        self.extras["episode"]["fallen"] = torch.mean(self.has_fallen[env_ids].float())
        
        self.episode_sums["hip_joint_errors"][env_ids] = 0.
        self.episode_sums["knee_joint_errors"][env_ids] = 0.
        self.episode_sums["torque"][env_ids] = 0.
        self.episode_sums["torque_max"][env_ids] = 0.

        self.extras["episode"]["mean_falcata"] = torch.mean(self.episode_sums["mean_falcata"][env_ids] / (self.progress_buf[env_ids] * self.decimation))
        self.extras["episode"]["max_falcata"] = torch.mean(self.episode_sums["max_falcata"][env_ids] / (self.progress_buf[env_ids] * self.decimation))
        self.episode_sums["mean_falcata"][env_ids] = 0.
        self.episode_sums["max_falcata"][env_ids] = 0.

        self.extras["episode"]["mean_air_time"] = torch.mean(self.episode_sums["mean_air_time"][env_ids] / (self.progress_buf[env_ids] * self.decimation))
        self.extras["episode"]["max_air_time"] = torch.mean(self.episode_sums["max_air_time"][env_ids] / (self.progress_buf[env_ids] * self.decimation))
        self.episode_sums["mean_air_time"][env_ids] = 0.
        self.episode_sums["max_air_time"][env_ids] = 0.

        
        self.dof_vel[env_ids, :, :-1] = 0.  
        self.actions[env_ids] = 0.

        self.feet_contact[env_ids] = True
        self.step_completetd[env_ids] = False
        self.feet_steps[env_ids] = 0.
        feet_ids = []


        self.feet_pos[env_ids] = self._mulinexs._feet.get_world_poses(clone=False)[0].view(self.num_envs, 4, 3)[env_ids] - self.base_pos[env_ids, None] * torch.ones((len(env_ids), 4, 3), device = self.device)
        self.feet_pos_z[env_ids] = self.feet_pos[env_ids,:,2]
        self.feet_pos_x[env_ids] = self.feet_pos[env_ids,:,0]
    
    
    
        
        self.progress_buf[env_ids] = 0
        self.simulation_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        
        
            
                
                
                
                
                
                

    def update_terrain_level(self, env_ids):   
        if not self.init_done or not self.curriculum:
            
            return
        root_pos, _ = self._mulinexs.get_world_poses(clone=False)
        distance = torch.norm(root_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(self.commands[env_ids, :2])*self.max_episode_length_s*0.25)
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def refresh_dof_state_tensors(self):
        self.dof_pos = self._mulinexs.get_joint_positions(clone=False)
        self.dof_vel.roll(-1, dims = -1)
        self.dof_vel[:, :, -1] = self._mulinexs.get_joint_velocities(clone=False)
    
    def refresh_body_state_tensors(self):
        self.base_pos, self.base_quat = self._mulinexs.get_world_poses(clone=False)
        
        self.base_velocities = self._mulinexs.get_velocities(clone=False)
        
        
        self.knee_pos, self.knee_quat = self._mulinexs._knees.get_world_poses(clone=False)
        self.feet_pos = self._mulinexs._feet.get_world_poses(clone=False)[0].view(self.num_envs, 4, 3) - self.base_pos[:, None] * torch.ones((self.num_envs, 4, 3), device = self.device) 
        for i in range(4):
            self.feet_pos[:,i] = quat_rotate_inverse(self.base_quat, self.feet_pos[:,i])   

    def pre_physics_step(self, actions):   
        if self._cfg['test']:     
            self.t1 = monotonic()
            if self.t0 is not None:
                time_res = 0.1 - (self.t1 - self.t0)     
                if time_res > 0:
                    sleep(time_res)
                
                
                
                
                
            self.t0 = monotonic()
            self.cnt += 1  

        if not self._env._world.is_playing():
            return

        self.actions.roll(-1, dims = -1)   
        self.actions[:, :, -1] = actions.clone().to(self.device)   
        
        if self.use_implicit_PD:
            self.dof_pos_cmd = self.action_scale*self.actions[:, :, -1] + self.default_dof_pos
            self._mulinexs.set_joint_position_targets(self.action_scale*self.actions[:, :, -1] + self.default_dof_pos)
            
        for i in range(self.decimation):
            if self._env._world.is_playing():
                
                
                
                
                
                
                
                
                
                if not self.use_implicit_PD:
                    self.dof_pos_cmd = self.action_scale*self.actions[:, :, -1] + self.default_dof_pos
                    torques = torch.clip(self.Kp*(self.action_scale*self.actions[:, :, -1] + self.default_dof_pos - self.dof_pos) - self.Kd*self.dof_vel[:, :, -1], -5., 5.)  
                    self._mulinexs.set_joint_efforts(torques) 
                    self.torques = torques
                    self.episode_sums["torque"] += torch.mean(self.torques.abs(), dim = 1)
                    self.episode_sums["torque_max"] = torch.max(self.episode_sums["torque_max"], self.torques.abs().max(dim = 1).values)

                self.jointerrors = self.action_scale*self.actions[:, :, -1] + self.default_dof_pos - self.dof_pos
                self.episode_sums["hip_joint_errors"] += torch.mean(self.jointerrors[:,:4], dim = 1)  
                self.episode_sums["knee_joint_errors"] += torch.mean(self.jointerrors[:,4:], dim = 1)
                
                self.torques_ind.roll(-1, dims=-1)
                self.torques_ind = self.Kp*(self.action_scale*self.actions[:, :, -1] + self.default_dof_pos - self.dof_pos) - self.Kd*self.dof_vel[:, :, -1]
        
                self.simulation_time += self.sim_dt
                if self.with_dp:
                    self.add_plot_data()

                SimulationContext.step(self._env._world, render=False)
                self.refresh_dof_state_tensors()

        
        self.feet_pos_z[self.step_completetd] = 0.
        self.feet_pos_x[self.step_completetd] = 0.
    
    




        

    def post_physics_step(self):               
        self.progress_buf[:] += 1              

        if self._env._world.is_playing():      
            
            self.refresh_dof_state_tensors()   
            self.refresh_body_state_tensors()  
            
            
            self.feet_air_time[self.feet_contact] = 0.
            self.prev_feet_contact[:] = self.feet_contact                                                                                                    
            self.feet_contact[:] = torch.norm(self._mulinexs._feet.get_net_contact_forces(clone=False, dt=self.dt).view(self.num_envs, 4, 3), dim=-1) > 0.   
            self.feet_pos_x[self.prev_feet_contact & (~ self.feet_contact)] = self.feet_pos[:,:,0][self.prev_feet_contact & (~ self.feet_contact)]           

            self.step_completetd[:] = (~self.prev_feet_contact) & self.feet_contact                                                                          
            self.feet_steps[self.step_completetd] += 1                                                                                                       
            self.feet_pos_z[:] = self.feet_pos_z.max(self.feet_pos[:, :, 2])                                                                                 
            self.spost_norm[self.step_completetd] = torch.abs(self.feet_pos[:,:,0][self.step_completetd] - self.feet_pos_x[self.step_completetd])            
            self.feet_air_time[~self.feet_contact] += self.dt

            
            self.episode_sums["mean_falcata"] += torch.mean(self.spost_norm, dim=1)         
            self.episode_sums["max_falcata"] += torch.max(self.spost_norm, dim=1).values    

            
            self.episode_sums["mean_air_time"] += torch.mean(self.feet_air_time*self.step_completetd, dim=1)         
            self.episode_sums["max_air_time"] += torch.max(self.feet_air_time*self.step_completetd, dim=1).values
            
            
            self.common_step_counter += 1                            
            if self.common_step_counter % self.push_interval == 0:   
                self.push_robots()                                   
            
            if self.randomize_terrain_properties:                                     
                self.step_counter_terrain += 1                                        
                if self.step_counter_terrain % self.randomize_terrain_interval == 0:  
                    
                    on_int_rand_static_friction = torch_rand_float(self._task_cfg["env"]["terrain"]["on_int_lower_staticFriction"], self._task_cfg["env"]["terrain"]["on_int_upper_staticFriction"], (1,1), device=self.device).item()
                    on_int_rand_dynamic_friction = torch_rand_float(self._task_cfg["env"]["terrain"]["on_int_lower_dynamicFriction"], self._task_cfg["env"]["terrain"]["on_int_upper_dynamicFriction"], (1,1), device=self.device).item()
                    on_int_rand_restitution = torch_rand_float(self._task_cfg["env"]["terrain"]["on_int_lower_restitution"], self._task_cfg["env"]["terrain"]["on_int_upper_restitution"], (1,1), device=self.device).item()
                    
                    self.random_static_friction = on_int_rand_static_friction
                    self.random_dynamic_friction = on_int_rand_dynamic_friction
                    
                    if on_int_rand_restitution > 1:
                        self.random_restitution = 1
                    else:
                        self.random_restitution = on_int_rand_restitution
                    
                    self.terrain_material.set_dynamic_friction(self.random_dynamic_friction)
                    self.terrain_material.set_static_friction(self.random_static_friction)
                    self.terrain_material.set_restitution(self.random_restitution)
            
            if self.enable_command_changes:                                         
                self.step_counter_command += 1                                      
                if self.step_counter_command % self.change_command_interval == 0:   
                    
                    self.commands[:, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (self.num_envs, 1), device=self.device).squeeze()       
                    

            
            self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 0:3])     
            self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 3:6])     
            self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)            
            forward = quat_apply(self.base_quat, self.forward_vec)                                    
            heading = torch.atan2(forward[:, 1], forward[:, 0])                                       
            if self.mix_com_percentage != 0.0:
                zero_yaw_rate_mask = (self.commands[:, 2] == 0.)                                            
                yaw_rate_from_heading =  torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.) 
                self.commands[zero_yaw_rate_mask, 2] = yaw_rate_from_heading[zero_yaw_rate_mask]            
            else:                                                                                           
                self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)    

            self.check_termination()      
            
            self.calculate_metrics()      

            
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()  
            if len(env_ids) > 0:                                        
                self.reset_idx(env_ids)                                 
                if self._dr_randomizer.randomize:                       
                    self.dr.physics_view.step_randomization(env_ids)    
                    
            
            if self._dr_randomizer.randomize:  
                self.dr.physics_view.step_randomization()

            self.get_observations()                                                              
            if self.add_noise:                                                                   
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec   

            
            

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras        

    def push_robots(self):                                                                                  
        self.base_velocities[:, :1] = torch_rand_float(-0.5, 0.5, (self.num_envs, 1), device=self.device)   
        self._mulinexs.set_velocities(self.base_velocities)                                                 
    
    def check_termination(self):                                                                                                                                  
        self.timeout_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf))   
        knee_contact = torch.norm(self._mulinexs._knees.get_net_contact_forces(clone=False).view(self._num_envs, 4, 3), dim=-1) > 0.                              
        self.has_fallen = (torch.norm(self._mulinexs._bases.get_net_contact_forces(clone=False), dim=1) > 0.) | (torch.sum(knee_contact, dim=-1) > 0.)            
        self.reset_buf = self.has_fallen.clone()                                                                                                                  
        self.reset_buf = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)                                                    

    def calculate_metrics(self):              
        
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)    
        
        
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])                        
        
        rew_lin_vel_xy = torch.exp(-lin_vel_error/(2*self.var_lin_vel)) * self.rew_scales["lin_vel_xy"]    
            
        if self.rew_scales["ang_vel_z_penalty"] == 0.:                     
            varangz = 1                                              
        else:
            varangz = -1/(self.fBc_angz*self.rew_scales["ang_vel_z_penalty"])  
        rew_ang_vel_z = ang_vel_error * self.rew_scales["ang_vel_z_penalty"] + torch.exp(-ang_vel_error/(2*varangz)) * self.rew_scales["ang_vel_z"]                      

        
        if self.rew_scales["lin_vel_z"] == 0.:                     
            varvz = 1                                              
        else:
            varvz = -1/(self.fBc_vz*self.rew_scales["lin_vel_z"])  

        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"] + self.rew_scales["vz+"]*torch.exp(-torch.square(self.base_lin_vel[:, 2]) / (2*varvz))   
        


        
        
        if self.rew_scales["ang_vel_xy"] == 0.:
            var_rollrate = 1   
            var_pitchrate = 1
        else:
            var_rollrate = -1/(self.fBc_rollrate * self.rew_scales["ang_vel_xy"] )
            var_pitchrate = -1/(self.fBc_pitchrate * self.rew_scales["ang_vel_xy"] )
        
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"] + self.rew_scales["AngularVelocity+"] * torch.exp(-torch.square(self.base_ang_vel[:, 0])/(2*var_rollrate) ) * torch.exp(-torch.square(self.base_ang_vel[:, 1])/(2*var_pitchrate) )

        
        if self.rew_scales["orient"] == 0.:
            var_orient = 1   
        else:
            var_orient = -1/(self.fBc_orient * self.rew_scales["orient"])

        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"] + self.rew_scales["orient+"] * torch.exp( -torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)/(2*var_orient) )

        

        mean_heightmap = torch.mean(self.get_heights(), dim=-1)       
        
        base_z_error = torch.square((self.base_pos[:, 2] - mean_heightmap) - self.des_base_height)
        if self.rew_scales["base_height"] == 0.:
            var_base_rew = 1   
        else:
            var_base_rew = -1/(self.fBc_base*self.rew_scales["base_height"])
        

        rew_base_height = torch.exp(-base_z_error/(2*var_base_rew)) * self.rew_scales["base_height_+"] + (torch.square(self.base_pos[:, 2] - self.des_base_height) * self.rew_scales["base_height"])

        
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

        
        if self.rew_scales["joint_acc"] == 0.:
            var_kneeacc = 1   
        else:
            var_kneeacc = -1/(self.fBc_kneeAccel * self.rew_scales["joint_acc"])  
            
        
        rew_joint_acc = torch.sum(torch.square(self.dof_vel.diff(1, dim = -1)[:, :, -1]), dim=1) * self.rew_scales["joint_acc"] + self.rew_scales["jointAcc+"] * torch.exp( -torch.sum(torch.square(self.dof_vel.diff(1, dim = -1)[:, :, -1]), dim=1)/(2*var_kneeacc) )

        
        if self.rew_scales["joint_jerk"] == 0.:
            var_kneejerk = 1   
        else:
            var_kneejerk = -1/(self.fBc_kneej * self.rew_scales["joint_jerk"])  
        
        rew_joint_jerk = torch.sum(torch.square(self.dof_vel.diff(2, dim = -1)[:, :, -1]), dim=1) * self.rew_scales["joint_jerk"] + self.rew_scales["jointJerk+"] * torch.exp( -torch.sum(torch.square(self.dof_vel.diff(2, dim = -1)[:, :, -1]), dim=1)/(2*var_kneejerk) )

        
        rew_fallen_over = self.has_fallen * self.rew_scales["fallen_over"]

        
        if self.rew_scales["action_rate"] == 0.:
            var_actionrate = 1   
            varActRateGN = 1     
        else: 
            var_actionrate = -1/(self.fBc_kneeactionRate * self.rew_scales["action_rate"])   
            if self.rew_scales["actRate+"] == 0.0: 
                fBc_actRateGN_ = 1
                varActRateGN = 1
            else:
                fBc_actRateGN_ = -self.fBcactRateGNscale * (self.rew_scales["action_rateGN"] / self.rew_scales["actRate+"]) * self.fBc_kneeactionRate
                varActRateGN = -1/(fBc_actRateGN_ * self.rew_scales["action_rate"])

        
        rew_action_rate = torch.sum(torch.square(self.actions.diff(1, dim = -1)[:, :, -1]), dim=1) * self.rew_scales["action_rate"] + self.rew_scales["actRate+"] * torch.exp( -torch.sum(torch.square(self.actions.diff(1, dim = -1)[:, :, -1]), dim=1)/(2*var_actionrate) ) + self.rew_scales["action_rateGN"] * torch.exp( -torch.sum(torch.square(self.actions.diff(1, dim = -1)[:, :, -1]), dim=1)/(2*varActRateGN) )

        
        if self.rew_scales["action_accel"] == 0.:
            var_actionaccel = 1   
        else:
            var_actionaccel = -1/(self.fBc_kneeactionAccel * self.rew_scales["action_accel"])   
        
        rew_action_accel = torch.sum(torch.square(self.actions.diff(2, dim = -1)[:, :, -1]), dim=1) * self.rew_scales["action_accel"] + self.rew_scales["actionAccel+"] * torch.exp( -torch.sum(torch.square(self.actions.diff(2, dim = -1)[:, :, -1]), dim=1)/(2*var_actionaccel) )

        
        if self.rew_scales["action_jerk"] == 0.:
            var_actionjerk = 1   
        else:
            var_actionjerk = -1/(self.fBc_kneeactionJ * self.rew_scales["action_jerk"])   
        
        rew_action_jerk = torch.sum(torch.square(self.actions.diff(3, dim = -1)[:, :, -1]), dim=1) * self.rew_scales["action_jerk"] + self.rew_scales["actionJerk+"] * torch.exp( -torch.sum(torch.square(self.actions.diff(3, dim = -1)[:, :, -1]), dim=1)/(2*var_actionjerk) )
        
        
        rew_hip = torch.sum(torch.abs(self.dof_pos[:, 0:4] - self.default_dof_pos[:, 0:4]), dim=1)* self.rew_scales["hip"]

        
        vx_non_nulla_mask = self.commands[:,0] != 0  
        err_falcata = torch.square(self.spost_norm - self.falcata_des * torch.ones_like(self.spost_norm))  
        if self.rew_scales["falcata"] == 0:
            var_falcata = 1
        else:
            var_falcata = -1/(self.fBc_falcata * self.rew_scales["falcata"])
    
        reward_falcata_penality = err_falcata * self.step_completetd * self.rew_scales["falcata"]    
        reward_falcata_bonus = torch.exp(- err_falcata/(2*var_falcata)) * self.step_completetd * self.rew_scales["falcata+"]   
        mask_leastOneStep = self.step_completetd.sum(dim=1) != 0            
        reward_falcata = torch.zeros(self._num_envs, device=self.device)    
        reward_falcata[mask_leastOneStep] = torch.sum(reward_falcata_penality[mask_leastOneStep] + reward_falcata_bonus[mask_leastOneStep], dim=1) / (self.step_completetd[mask_leastOneStep].sum(dim=1))  
        rew_falcata = reward_falcata * vx_non_nulla_mask  
        
        















        
        if self.rew_scales["feet_pos_z"] == 0:
            var_feetZ = 1
        else:
            var_feetZ = -1/(self.fBc_feetZ * self.rew_scales["feet_pos_z"])

        err_feet_pos_z = torch.square(self.feet_pos_z + self.des_base_height - self.feet_pos_z_val)
        reward_feet_pos_z_penality = err_feet_pos_z * self.step_completetd * self.rew_scales["feet_pos_z"]  
        reward_feet_pos_z_bonus = torch.exp( -err_feet_pos_z/(2*var_feetZ))* self.step_completetd * self.rew_scales["feetZ+"]  
        reward_feet_pos_z = torch.sum(reward_feet_pos_z_penality + reward_feet_pos_z_bonus, dim=1)
        rew_feet_pos_z = reward_feet_pos_z * vx_non_nulla_mask + torch.ones_like(reward_feet_pos_z)  * (~vx_non_nulla_mask) * self.rew_scales["feetZ+"]

        
        if self.rew_scales["actionjerk_max"] == 0:
            var_actionjerk_max = 1
        else:
            var_actionjerk_max = -1/(self.fBc_actionjerk_max * self.rew_scales["actionjerk_max"])
        if self.rew_scales["jointjerk_max"] == 0:
            var_jointjerk_max = 1
        else:
            var_jointjerk_max = -1/(self.fBc_jointjerk_max * self.rew_scales["jointjerk_max"])

        max_actionjerk, _ = torch.max(torch.abs(self.actions.diff(3, dim = -1)[:, :, -1]), dim=1)
        max_jointjerk, _ = torch.max(torch.abs(self.dof_vel.diff(2, dim = -1)[:, :, -1]), dim=1)
        rew_actionjerk_max =  max_actionjerk * self.rew_scales["actionjerk_max"] + self.rew_scales["actionjerk_max+"] * torch.exp(-torch.square(max_actionjerk)/(2*var_actionjerk_max))
        rew_jointjerk_max =  max_jointjerk * self.rew_scales["jointjerk_max"] + self.rew_scales["jointjerk_max+"] * torch.exp(-torch.square(max_jointjerk)/(2*var_jointjerk_max))

        
        if self.rew_scales["actionaccel_max"] == 0:
            var_actionaccel_max = 1
        else:
            var_actionaccel_max = -1/(self.fBc_actionaccel_max * self.rew_scales["actionaccel_max"])
        if self.rew_scales["jointaccel_max"] == 0:
            var_jointaccel_max = 1
        else:
            var_jointaccel_max = -1/(self.fBc_jointaccel_max * self.rew_scales["jointaccel_max"])

        max_actionaccel, _ = torch.max(torch.abs(self.actions.diff(2, dim = -1)[:, :, -1]), dim=1)
        max_jointaccel, _ = torch.max(torch.abs(self.dof_vel.diff(1, dim = -1)[:, :, -1]), dim=1)
        rew_actionaccel_max =  max_actionaccel * self.rew_scales["actionaccel_max"] + self.rew_scales["actionaccel_max+"] * torch.exp(-torch.square(max_actionaccel)/(2*var_actionaccel_max))
        rew_jointaccel_max =  max_jointaccel * self.rew_scales["jointaccel_max"] + self.rew_scales["jointaccel_max+"] * torch.exp(-torch.square(max_jointaccel)/(2*var_jointaccel_max))

        
        if self.rew_scales["air_time"] == 0:
            var_airtime = 1
        else:
            var_airtime = -1/(self.fBc_airtime * self.rew_scales["air_time"])
        
        err_airtime = torch.square(self.feet_air_time - self.air_time_des * torch.ones_like(self.feet_air_time))  
        airtime_penalty = err_airtime * self.step_completetd * self.rew_scales["air_time"]
        airtime_bonus = torch.exp(- err_airtime/(2*var_airtime)) *self.step_completetd * self.rew_scales["air_time+"]
        reward_airtime = torch.zeros(self._num_envs, device = self.device)  
        reward_airtime[mask_leastOneStep] = torch.sum(airtime_penalty[mask_leastOneStep] + airtime_bonus[mask_leastOneStep], dim=1) / (self.step_completetd[mask_leastOneStep].sum(dim=1))
        rew_airtime = reward_airtime * vx_non_nulla_mask

        
        
        
        
        self.rew_buf = rew_lin_vel_xy + rew_ang_vel_z + rew_lin_vel_z + rew_ang_vel_xy + rew_orient + rew_base_height + \
                    rew_torque + rew_joint_acc + rew_action_rate + rew_hip + rew_fallen_over + rew_feet_pos_z  + rew_action_accel +rew_action_jerk + rew_joint_jerk + rew_falcata + rew_actionjerk_max + rew_jointjerk_max + rew_actionaccel_max + rew_jointaccel_max + rew_airtime
        self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)

        
        self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

        if self._cfg['test'] and False:  

            if self.cnt < 700:
                append_mean_and_std((lin_vel_error / 2) ** 0.5, self.lin_vel_err_mean, self.lin_vel_err_std)
                append_mean_and_std(ang_vel_error ** 0.5, self.ang_vel_err_mean, self.ang_vel_err_std)
                append_mean_and_std((((rew_joint_acc / self.rew_scales["joint_acc"]) / 12) ** 0.5), self.joint_acc_mean, self.joint_acc_std)
                append_mean_and_std(((rew_action_rate / self.rew_scales["action_rate"]) / 12) ** 0.5, self.action_rate_mean, self.action_rate_std)
                append_mean_and_std(((rew_torque / self.rew_scales["torque"]) / 12) ** 0.5, self.torque_mean, self.torque_std)
                append_mean_and_std(torch.square(self.base_pos[:, 2] - self.get_heights().mean(dim=1)) ** 0.5, self.base_height_mean, self.base_height_std)
                
                add_data((lin_vel_error / 2) ** 0.5, self.lin_vel_err_data)
                add_data(ang_vel_error ** 0.5, self.ang_vel_err_data)
                add_data((((rew_joint_acc / self.rew_scales["joint_acc"]) / 12) ** 0.5), self.joint_acc_data)
                add_data(((rew_action_rate / self.rew_scales["action_rate"]) / 12) ** 0.5, self.action_rate_data)
                add_data(((rew_torque / self.rew_scales["torque"]) / 12) ** 0.5, self.torque_data)
                add_data(torch.square(self.base_pos[:, 2] - self.get_heights().mean(dim=1)) ** 0.5, self.base_height_data)
                
                print('Step count:', self.cnt)
                self.cnt += 1

            if self.cnt == 700:  
                self.cnt += 1
                path = 'Anymal_Rough_Terrains'
                
                
                
                
                
                
                print(f'\n\n\n'
                      f'Task: {path} ({self.num_envs})\n'
                      f'linear velocity error (x, y) [m/s]: {np.asarray(self.lin_vel_err_mean).mean()} ± {np.asarray(self.lin_vel_err_std).mean()}\n'
                      f'angular velocity error [rad/s]: {np.asarray(self.ang_vel_err_mean).mean()} ± {np.asarray(self.ang_vel_err_std).mean()}\n'
                      f'joint acceleration [rad/s]: {np.asarray(self.joint_acc_mean).mean()} ± {np.asarray(self.joint_acc_std).mean()}\n'
                      f'action rate [rad/s]: {np.asarray(self.action_rate_mean).mean()} ± {np.asarray(self.action_rate_std).mean()}\n'
                      f'joint torques [N*m]: {np.asarray(self.torque_mean).mean()} ± {np.asarray(self.torque_std).mean()}\n'
                      f'robot base height [m]: {np.asarray(self.base_height_mean).mean()} ± {np.asarray(self.base_height_std).mean()}'
                      f'\n\n\n')
                lin_vel_err_data_tensor = torch.tensor(self.lin_vel_err_data, device=self._device)
                ang_vel_err_data_tensor = torch.tensor(self.ang_vel_err_data, device=self._device)
                joint_acc_data_tensor = torch.tensor(self.joint_acc_data, device=self._device)
                action_rate_data_tensor = torch.tensor(self.action_rate_data, device=self._device)
                torque_data_tensor = torch.tensor(self.torque_data, device=self._device)
                base_height_data_tensor = torch.tensor(self.base_height_data, device=self._device)
                print(f'\n\n\n'
                      f'Task: {path} ({self.num_envs})\n'
                      f'linear velocity error (x, y) [m/s]: {lin_vel_err_data_tensor.mean()} ± {lin_vel_err_data_tensor.std()}\n'
                      f'angular velocity error [rad/s]: {ang_vel_err_data_tensor.mean()} ± {ang_vel_err_data_tensor.std()}\n'
                      f'joint acceleration [rad/s]: {joint_acc_data_tensor.mean()} ± {joint_acc_data_tensor.std()}\n'
                      f'action rate [rad/s]: {action_rate_data_tensor.mean()} ± {action_rate_data_tensor.std()}\n'
                      f'joint torques [N*m]: {torque_data_tensor.mean()} ± {torque_data_tensor.std()}\n'
                      f'robot base height [m]: {base_height_data_tensor.mean()} ± {base_height_data_tensor.std()}'
                      f'\n\n\n')
                exit(0)

        
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["orient"] += rew_orient
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["base_height"] += rew_base_height
        self.episode_sums["hip"] += rew_hip
        
        
        
        
        self.episode_sums["feet_pos_z"] += rew_feet_pos_z
        self.episode_sums["action_accel"] += rew_action_accel
        self.episode_sums["action_jerk"] += rew_action_jerk
        self.episode_sums["joint_jerk"] += rew_joint_jerk
        self.episode_sums["falcata"] += rew_falcata
        self.episode_sums["action_jerk_max"] += rew_actionjerk_max
        self.episode_sums["joint_jerk_max"] += rew_jointjerk_max
        self.episode_sums["action_accel_max"] += rew_actionaccel_max
        self.episode_sums["joint_accel_max"] += rew_jointaccel_max
        self.episode_sums["feet_air_time"] += rew_airtime
        
    

    def get_observations(self):   
         self.obs_buf[:] = torch.cat(( self.projected_gravity,
                                    self.base_quat,        
                                    self.base_ang_vel * self.ang_vel_scale,
                                    self.commands[:, [0,2]] * self.commands_scale[[0,2]],
                                    self.dof_pos * self.dof_pos_scale,
                                    self.dof_vel[:, :, -1] * self.dof_vel_scale,
                                    self.actions[:, :, -1]
                                    ),dim=-1)
                           
    
    def get_states(self):   
        self.measured_heights = self.get_heights()
        heights = torch.clip(self.base_pos[:, 2].unsqueeze(1) - 0.25 - self.measured_heights, -1, 1.) * self.height_meas_scale
        self.states_buf[:] = torch.cat((  self.base_lin_vel * self.lin_vel_scale,
                                    self.base_ang_vel  * self.ang_vel_scale,
                                    self.projected_gravity,
                                    
                                    
                                    
                                    self.commands[:, [0,2]] * self.commands_scale[[0,2]],
                                    self.dof_pos * self.dof_pos_scale,
                                    self.dof_vel[:, :, -1] * self.dof_vel_scale,
                                    heights,
                                    self.actions[:, :, -1]
                                    ),dim=-1)
        return self.states_buf

    def get_ground_heights_below_knees(self):     
        points = self.knee_pos.reshape(self.num_envs, 4, 3)
        points += self.terrain.border_size
        points = (points/self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py+1]
        heights = torch.min(heights1, heights2)
        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale
    
    def get_ground_heights_below_base(self):   
        points = self.base_pos.clone().reshape(self.num_envs, 1, 3)
        points += self.terrain.border_size
        points = (points/self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py+1]
        heights = torch.min(heights1, heights2)
        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale
                                    
    def get_heights(self, env_ids=None):
        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.base_pos[env_ids, 0:3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.base_pos[:, 0:3]).unsqueeze(1)
 
        points += self.terrain.border_size
        points = (points/self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)


        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)


        heights1 = self.height_samples[px, py]

        heights2 = self.height_samples[px+1, py+1]
        heights = torch.min(heights1, heights2)



        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale



@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles


def get_axis_params(value, axis_idx, x_value=0., dtype=np.float, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))
