o, _ = env.reset()
o = np.array(o)
state = torch.tensor([o], dtype=torch.float).to(agent.device)
state = torch.transpose(state, 1, 3)
# print(state.shape)
probs = agent.actor(state)
print(probs)


o, _ = env.reset()