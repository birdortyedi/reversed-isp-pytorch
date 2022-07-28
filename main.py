import fire

from engine import Trainer, Tester


def run(
    data_path: str,
    save_dir: str,
    output_dir: str,
    ckpt_path: str = None,
    project_name: str = "reversed-isp",
    model_name: str = "ifrnet-v4",
    dataset_name: str = "s7",
    n_channels_D: int = 32,
    n_channels_G: int = 32,
    num_critics_D: int = 3,
    grad_penalty: float = 10.0,
    batch_size: int = 8,
    img_size: int = 504,
    lr: float = 0.0004,
    betas: tuple = (0.5, 0.999),
    num_gpu : int = 2,
    shuffle: bool = True,
    resume: bool = False,
    start_step: int = 0,
    total_step: int = 64000,
    log_interval: int = 100, 
    visualize_interval: int = 100,
    save_interval: int = 10000,
    eval_interval: int = 1000,
    num_experiment: int = 0,
    test: bool = False
):
    params = {
        "data_path": data_path,
        "ckpt_path": ckpt_path,
        "save_dir": save_dir,
        "output_dir": output_dir,
        "project_name": project_name,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "n_channels_D": n_channels_D,
        "n_channels_G": n_channels_G,
        "num_critics_D": num_critics_D,
        "grad_penalty": grad_penalty,
        "batch_size": batch_size,
        "img_size": img_size,
        "lr": lr,
        "betas": betas,
        "num_gpu": num_gpu,
        "shuffle": shuffle,
        "resume": resume,
        "start_step": start_step,
        "total_step": total_step,
        "log_interval": log_interval,
        "visualize_interval": visualize_interval,
        "save_interval": save_interval,
        "eval_interval": eval_interval,
        "num_experiment": num_experiment,
    }
    
    engine = Trainer(params) if not test else Tester(params)
    engine.run()
            

if __name__ == '__main__':
    fire.Fire(run)