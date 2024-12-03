# %% [markdown]
# # Demo for Reproduction Purposes

# %% [markdown]
# ### Install Prerequisites

# %%
#%pip install vizdoom
#%pip install sample-factory

# %% [markdown]
# ### Function Setup

# %%
import functools

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec


# Registers all the ViZDoom environments
def register_vizdoom_envs():
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)

def register_vizdoom_models():
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)


def register_vizdoom_components():
    register_vizdoom_envs()
    register_vizdoom_models()

# parse the command line args and create a config
def parse_vizdoom_cfg(argv=None, evaluation=False):
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)

    add_doom_env_args(parser)
    doom_override_defaults(parser)

    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg

# %% [markdown]
# ### Train Model

# %%
## Start the training, this should take around  minutes
register_vizdoom_components()

# other scenarios include "doom_basic", "doom_two_colors_easy", "doom_dm", "doom_dwango5", "doom_my_way_home", "doom_deadly_corridor", "doom_defend_the_center", "doom_defend_the_line"
env = "doom_deathmatch_bots"
cfg = parse_vizdoom_cfg(argv=[f"--env={env}",
                              "--algo=APPO",
                              "--seed=0",
                              "--num_workers=16",
                              "--num_envs_per_worker=24",
                              "--batch_size=2048",
                              "--recurrence=32",
                              "--gamma=0.995",
                              "--res_w=128",
                              "--res_h=72",
                              "--train_for_env_steps=10000000",
                              "--train_for_seconds=36000",
                              "--use_rnn=True",
                              "--rnn_type=lstm",
                              "--nonlinearity=relu",
                              "--env_frameskip=2"
                             ])

status = run_rl(cfg)

# %% [markdown]
# ### Test Model

# %%
from sample_factory.enjoy import enjoy
register_vizdoom_components()
env = "doom_deathmatch_bots"
cfg = parse_vizdoom_cfg(argv=[f"--env={env}", "--num_workers=1", "--save_video", "--no_render", "--max_num_episodes=10"], evaluation=True)
status = enjoy(cfg)

# %% [markdown]
# ### Visualize Testing

# %%
from base64 import b64encode
from IPython.display import HTML

mp4 = open('train_dir/default_experiment/replay.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=640 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)

# %% [markdown]
# ### Save Model to HuggingFace

# %%
from sample_factory.enjoy import enjoy

hf_username = "Mojitrk" # insert your HuggingFace username here

cfg = parse_vizdoom_cfg(argv=[f"--env={env}",
                              "--num_workers=1",
                              "--save_video",
                              "--no_render",
                              "--max_num_episodes=10",
                              "--max_num_frames=100000",
                              "--push_to_hub",
                              f"--hf_repository={hf_username}/doom_deathmatch_bots"], evaluation=True)
status = enjoy(cfg)


