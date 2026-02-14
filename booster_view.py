import mujoco.viewer
from stable_baselines3 import PPO
from booster_env import BoosterLookEnv
import time

env = BoosterLookEnv()
model = PPO.load("booster_look_at_ball_model")

obs, _ = env.reset()

# Abre o visualizador do MuJoCo
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        # A IA decide o que fazer com base na política aprendida
        action, _states = model.predict(obs, deterministic=True)
        
        # O ambiente executa a ação (estado->ação->recompensa)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Sincroniza a imagem
        viewer.sync()
        time.sleep(0.01) # Slow motion

        if terminated or truncated:
            obs, _ = env.reset()