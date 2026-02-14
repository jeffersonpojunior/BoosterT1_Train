import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from booster_env import BoosterLookEnv

def train():
    # 1. Instancia o enviroment personalizado que foi criado em booster_env.py
    env = BoosterLookEnv()

    # 2. Configura o salvamento automático (Checkpoints)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./logs/",
        name_prefix="booster_pivot_model"
    )

    # 3. Inicializa o PPO
    # Para lidar com a complexidade de 23 juntas
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        batch_size=128,
        n_steps=2048,
        device="auto" # Usa GPU se disponível
    )

    print("Iniciando treinamento... Pressione Ctrl+C para parar e salvar manualmente.")
    
    try:
        # Treinando por 1.500.000 vezes
        model.learn(total_timesteps=1500000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("Treinamento interrompido pelo usuário. Salvando estado atual...")
    
    model.save("booster_look_at_ball_model")

if __name__ == "__main__":
    train()