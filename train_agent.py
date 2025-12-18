# AA: Stable-Baselines3 (often abbreviated as SB3) is a popular Python library that provides reliable, well-tested implementations of reinforcement learning algorithms. 
# It's essentially a toolkit that makes it much easier to apply RL algorithms to your problems without having to code everything from scratch.stable_baselines3 is model free error

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO          
from stable_baselines3.common.vec_env import DummyVecEnv

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv

def main():
    df = load_and_preprocess_data("data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv")
    
    # create env
    env = ForexTradingEnv(df=df,
                          window_size=30,
                          sl_options=[30, 60, 80],  # example SL distances in pips
                          tp_options=[30, 60, 80])  # example TP distances in pips
    
    # Wrap in a DummyVecEnv (required by stable-baselines for parallelization)
    vec_env = DummyVecEnv([lambda: env])
    
    # Define RL model (PPO) 
    # PPO (Proximal Policy Optimization) is one of the most popular and successful reinforcement learning algorithms, developed by OpenAI in 2017. 
    # It's a policy gradient method that directly learns a policy (a strategy for choosing actions) by optimizing it through experience.
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_log/"
    )
    
    # Train the model
    model.learn(total_timesteps=10100)    # AA: I can change the time steps
    model.save("model_eurusd")            # AA: I can save the model
    print("Model saved successfully!")    
    
    # Evaluate or test the model
    obs = vec_env.reset()
    done = False
    equity_curve = []
    
    while True:
        action, _states = model.predict(obs, deterministic=True)        # AA: Observations are not stochastic
        obs, rewards, done, info = vec_env.step(action)                 # AA: The action is buying or selling
        
        # Collect equity from the unwrapped environment
        # Because we have a DummyVecEnv, we can access env_method to get the attribute
        current_equity = vec_env.get_attr("equity")[0]
        equity_curve.append(current_equity)
        
        if done[0]:
            break
    
    # Plot the final equity curve
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label='Equity')
    plt.title("Equity Curve during Evaluation")
    plt.xlabel("Time Steps")
    plt.ylabel("Equity")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
