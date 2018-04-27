import gym

env = gym.make('SpaceInvaders-v0')
env.reset()
for _ in range(100000):
    env.render()
    env.step(env.action_space.sample())

env.close()
