from argparse import Namespace
import datetime
import os
from tensorboardX import SummaryWriter
import config


def splash_screen(params: Namespace):
    print(r"""
    Stochastic GYM Trainer ; Commit before running                                                           
    """)
    run_name = config.RUN_NAME + params.version # "_" + params.commit[:8] + "_" + params.version
    print(f" Version: {params.version} ".center(88, "="))
    print(f" Run Name: {run_name} ".center(88, "="))

    train_summary_writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR_ROOT, run_name))

    nl = "\n"
    train_summary_writer.add_text("run_name", f"""
### {run_name}
#### {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
---
## params
{nl.join([f"- `{k}`: `{v}`" for k, v in params.__dict__.items()])}
        """, 0)

    train_summary_writer.flush()

    return train_summary_writer


def get_file_descriptor(params: Namespace, episode: int) -> str:
    """
    Creates filename for a training run.

    :param params: Namespace of training run
    :param episode: episode number at moment of saving
    :return: str
    """
    return f"{config.MODELS_DIR}/{params.version}_{config.RUN_NAME}_{episode}.pth" #  '_{params.commit[:8]}'