params = {}
params["render_pygame"] = False
params["episode_capacity"] = 10 # Number of episodes per epoch (used for SGD)
params["learning_rate"] = 0.001
params["actor_hidden_units"] = 64
params["critic_hidden_units"] = 128 # Also the number of hidden units for the meta-gradient assignment distribution in VAST
params["episodes_per_epoch"] = params["episode_capacity"]
params["evaluations_per_epoch"] = 10
params["output_folder"] = "data/gaussian_model"