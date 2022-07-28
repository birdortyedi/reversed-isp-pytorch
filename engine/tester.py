import os
import cv2
import torch
import numpy as np
import kornia
import time
import glog as log
from torch.utils import data
from torchvision import transforms as T

from datasets.loader import ReversedISPDataset
from modeling.build import build_model
from metrics.ssim import SSIM
from utils.raw_utils import *


class Tester:
    def __init__(self, dict_param) -> None:
        self._init_params(dict_param)
        self.dataset_name = self.data_path.split("/")[-2]
        self.model_name = f"{self.model_name}_{self.dataset_name}_{self.total_step}step_{self.batch_size}bs_{self.lr}lr_{self.num_gpu}gpu_{self.num_experiment}run"
        
        self.dataset = ReversedISPDataset(root=self.data_path, test=True)
        self.image_loader = data.DataLoader(dataset=self.dataset, batch_size=1, shuffle=False)
        
        self.to_pil = T.ToPILImage()

        self.net = build_model(n_channels=self.n_channels_G, out_channels=4)
        
        self.SSIM = SSIM()
        
        self.output_dir = os.path.join(self.output_dir, self.model_name)
        os.makedirs(self.output_dir, exist_ok=True)
        log.info("Checkpoints loading from ckpt file...")
        self.load_checkpoint_from_ckpt(self.ckpt_path)
        self.net.cuda()
        
        
    def _init_params(self, dict_param):
        self.data_path = ""
        self.ckpt_path = None
        self.save_dir = ""
        self.output_dir = ""
        self.project_name = "reversed-isp"
        self.model_name = "ifrnet"
        self.dataset_name = "s7"
        self.n_channels_D = 0
        self.n_channels_G = 0
        self.num_critics_D = 0
        self.grad_penalty = 0.0
        self.batch_size = 0
        self.img_size = 0
        self.lr = 0.0
        self.betas = (0.0, 0.0)
        self.num_gpu = 0
        self.shuffle = False
        self.resume = False
        self.start_step = 0
        self.total_step = 0
        self.num_step = 0
        self.log_interval = 0
        self.visualize_interval = 0
        self.save_interval = 0
        self.eval_interval = 0
        self.num_experiment = 0
        for key, value in dict_param.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def run(self):
        self.net.eval()
        
        # psnr_lst, ssim_lst = list(), list()
        runtime = list()
        for i, (imgs, imgs_name) in enumerate(self.image_loader):
            if i % self.visualize_interval == 0:
                log.info(f" [Step: {i}/{len(self.image_loader)} ({round(i / len(self.image_loader) * 100., 2)}%)] Evaluating...")
            imgs = imgs.float().cuda()

            with torch.no_grad():
                tick = time.time()
                output = self.net(imgs)
                runtime.append(time.time() - tick)
            output = torch.clamp(output, max=1., min=0.).detach().cpu().permute(0, 2, 3, 1).numpy()
            assert output.shape[-1] == 4
            for (out, name) in zip(output, imgs_name):
                raw = (out * 1024).astype(np.uint16)
                np.save(os.path.join(self.output_dir, name.split("/")[-1].split(".")[0] + ".npy"), raw)
                out = postprocess_raw(demosaic(out)) * 255.0
                out = out.astype(np.uint8)
                cv2.imwrite(os.path.join(self.output_dir, "{}.png".format(i)), out, [cv2.IMWRITE_PNG_COMPRESSION, 5])
        log.info(f"Average run-time: {np.mean(runtime) * 1000:.2f}ms")
        #     ssim = self.SSIM(255. * y_imgs, 255. * output).item()
        #     ssim_lst.append(ssim)

        #     psnr = -kornia.losses.psnr_loss(output, y_imgs, max_val=1.).item()  # -self.PSNR(y_imgs, output).item()
        #     psnr_lst.append(psnr)
            
        # results = {"Dataset": self.dataset_name, "PSNR": np.round(np.mean(psnr_lst), 4), "SSIM": np.round(np.mean(ssim_lst), 4)}
        # log.info(results)
        
    def load_checkpoint_from_ckpt(self, ckpt_path):
        checkpoints = torch.load(ckpt_path)
        self.net.load_state_dict(checkpoints["model"])
    
    