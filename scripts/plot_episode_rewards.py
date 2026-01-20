
JSON_PATH = "./gpu_models/A_500ep_20260120_101500.pt_rewards.json"  # <-- 在此替换为你的文件路径
import json
import matplotlib.pyplot as plt
with open(JSON_PATH, 'r') as f:
    data = json.load(f)
episode_rewards = data['episode_rewards']
episodes = list(range(1, len(episode_rewards) + 1))

plt.figure(figsize=(10, 5))
plt.plot(episodes, episode_rewards, marker='o', markersize=3, linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Episode Rewards over Time')
plt.grid(True)
plt.tight_layout()
plt.show()
