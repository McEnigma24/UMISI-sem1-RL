import gym
import crafter
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os

# 1. Konfiguracja i powtarzalność
SEED = 42
LOG_DIR = "./crafter_logs/"
os.makedirs(LOG_DIR, exist_ok=True)

# 2. Utworzenie środowiska Crafter
# Crafter domyślnie zwraca obrazy 64x64 RGB
env = gym.make('CrafterReward-v1')
env = Monitor(env, LOG_DIR) # Monitor pozwala zrzucać logi nagród do pliku
env.seed(SEED)

# Utworzenie osobnego środowiska do ewaluacji (żeby nie zaburzać epizodów treningowych)
eval_env = gym.make('CrafterReward-v1')
eval_env = Monitor(eval_env, os.path.join(LOG_DIR, "eval"))
eval_env.seed(SEED)

# 3. Callback do ewaluacji
# Sprawdza model co 10 000 kroków i zapisuje ten, który osiągnął najlepszy wynik.
eval_callback = EvalCallback(eval_env,
                             best_model_save_path=LOG_DIR,
                             log_path=LOG_DIR,
                             eval_freq=10000,
                             deterministic=False,
                             render=False)

# 4. Utworzenie modelu PPO
# Używamy "CnnPolicy", ponieważ środowisko zwraca siatkę pikseli (obraz), a nie wektory.
model = PPO("CnnPolicy",
            env,
            verbose=1,
            seed=SEED,
            tensorboard_log=LOG_DIR)

print("Rozpoczynam trening agenta...")

# 5. Trening agenta
# 100 000 to wartość startowa. WCrafterze może być potrzebne więcej (np. 1M) żeby zobaczyć zaawansowane zachowania.
model.learn(total_timesteps=100000, callback=eval_callback)

print("Trening zakończony. Zapisuję ostateczny model...")
model.save(os.path.join(LOG_DIR, "ppo_crafter_final"))