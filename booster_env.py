import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

class BoosterLookEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Carregando o modelo do Enviroment
        self.model = mujoco.MjModel.from_xml_path("scene.xml")
        self.data = mujoco.MjData(self.model)

        # Definição das ações: (todos os 23 motores em controle)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(23,), dtype=np.float32)

        # Observações: Orientação (4) + Posição da bola (3) + qpos (todas as juntas) + qvel
        obs_shape = 4 + 3 + self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def _get_obs(self):
        # Coleta dados dos sensores e estados físicos
        quat = self.data.sensor("orientation").data  # 4 valores
        ball_pos = self.data.body("target_ball").xpos # 3 valores
        
        return np.concatenate([
            quat, 
            ball_pos, 
            self.data.qpos.flat, 
            self.data.qvel.flat
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Primeiro limpamos todos os dados e aplicamos o Keyframe (Pose Home)
        mujoco.mj_resetData(self.model, self.data)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        
        # Spawn aleatório da bola em um círculo ao redor do robô
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(1.2, 2.0)
        ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_ball")
        
        new_ball_pos = np.array([
            radius * np.cos(angle), 
            radius * np.sin(angle), 
            0.1
        ])
        
        # Descobre onde começa o qpos da bola
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_ball")
        qpos_adr = self.model.jnt_qposadr[joint_id]

        # Atualiza posição
        self.data.qpos[qpos_adr:qpos_adr+3] = new_ball_pos

        # Mantém orientação neutra
        self.data.qpos[qpos_adr+3:qpos_adr+7] = np.array([1, 0, 0, 0])

        mujoco.mj_forward(self.model, self.data)
        

        return self._get_obs(), {}

    def step(self, action):
        # Aplica as ações em todos os atuadores
        self.data.ctrl[:] = action
        
        # Simula 5 passos de física para cada passo de IA (aumenta estabilidade)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        # Cálculo da recompensa:
        trunk_pos = self.data.body("Trunk").xpos
        ball_pos = self.data.body("target_ball").xpos
        
        # 1. Recompensa de Orientação (Giro)
        # Queremos que o vetor "frente" do robô alinhe com o vetor "robô->bola"
        vec_to_ball = ball_pos[:2] - trunk_pos[:2]
        vec_to_ball /= (np.linalg.norm(vec_to_ball) + 1e-6)
        
        # Recompensa baseada no produto escalar (quanto mais alinhado, mais próximo de 1)
        reward_align = np.dot([1, 0], vec_to_ball)

        # 2. Penalidade de Queda
        height = trunk_pos[2]
        terminated = False
        reward_fall = 0
        
        if height < 0.45: # Se o tronco cair abaixo de 45cm
            reward_fall = -10.0
            terminated = True
        
        reward = reward_align + reward_fall

        return self._get_obs(), reward, terminated, False, {}