"""
distance_to_w, f_val, rewards = data["distance_to_w"], data["f_val"], data["rewards"]

plt.figure()
plt.ylabel('$w$ - $w_{opt}$')
plt.xlabel('Episode #')
plt.plot(range(len(distance_to_w)), distance_to_w)
plt.title('Distance from optimal solution during training')
plt.savefig(os.path.join(fpath, 'episode-convergence.png'))

plt.figure()
plt.ylabel('f(w)')
plt.xlabel('Episode #')
plt.plot(range(len(f_val)), f_val)
plt.title('Function value at last iteration during training')
plt.savefig(os.path.join(fpath, 'episode-f-val.png'))

plt.figure()
plt.ylabel('rewards')
plt.xlabel('Episode #')
plt.plot(range(len(rewards)), rewards)
plt.title('Rewards during training')
plt.savefig(os.path.join(fpath, 'rewards.png'))

for episode, action in actions.items():
    plt.figure()
    plt.hist(action, bins=100)
    plt.title(f'Actions during episode: {episode}')
    plt.savefig(os.path.join(fpath, 'actions-episode-{}.png'.format(episode)))

"""