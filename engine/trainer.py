from email.policy import strict
import os
import kornia
import numpy as np
import torch
import wandb
import glog as log
import glob

from torch import nn
from torch.utils import data
from torchvision import transforms

from datasets.loader import ReversedISPDataset
from modeling.build import build_discriminators, build_model
from losses.adversarial import compute_gradient_penalty
from losses.angular import AngularError
from metrics.ssim import SSIM
from utils.raw_utils import *


class Trainer:
    def __init__(self, dict_param):
        self._init_params(dict_param)
        self.dataset_name = self.data_path.split("/")[-2]
        assert self.dataset_name in ["s7", "p20"]
        self.model_name = f"{self.model_name}_{self.dataset_name}_{self.total_step}step_{self.batch_size}bs_{self.lr}lr_{self.num_gpu}gpu_{self.num_experiment}run"
    
        self.wandb_log_dir = os.path.join("./logs/", self.model_name)
        self.wandb = wandb
        self.wandb.init(project=self.project_name, resume=self.resume, notes=self.wandb_log_dir, config=dict_param, entity="vvgl-ozu")
        
        self.best_psnr = 0.0

        self.dataset = ReversedISPDataset(root=self.data_path)
        self.image_loader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        self.test_dataset = ReversedISPDataset(root=self.data_path)
        self.test_image_loader = data.DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False)

        self.to_pil = transforms.ToPILImage()

        self.net = build_model(n_channels=self.n_channels_G, out_channels=4, model_type=self.dataset_name)
        # self.discriminator, self.patch_discriminator = build_discriminators(self.n_channels_D, in_channels=4)
        self.discriminator = build_discriminators(256)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr / 4, betas=self.betas)  # ttur 
        # self.optimizer_discriminator = torch.optim.Adam(list(self.discriminator.parameters()) + list(self.patch_discriminator.parameters()), lr=self.lr, betas=self.betas)
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)
        
        self.SSIM = SSIM()
        
        self.output_dir = os.path.join(self.output_dir, self.model_name)
        os.makedirs(self.output_dir, exist_ok=True)
        if self.start_step != 0 and self.resume and self.ckpt_path is not None:
            log.info("Checkpoints loading...")
            self.load_latest_checkpoint()
        elif self.ckpt_path is not None:
            log.info("Checkpoints loading from ckpt file...")
            self.load_checkpoint_from_ckpt(self.ckpt_path)

        self.check_and_use_multi_gpu()
        self._init_criterion()

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
        self.lambdas = {
            "RECON": 0.1,
            "SSIM": 0.1,
            "MS_SSIM": 1.0,
            "TV": 0.001,
            "SEMANTIC": 1e-1,
            "TEXTURE": 2e-1,
            "ADVERSARIAL": 1e-3,
        }  # TODO: add lambda to dict_param
        for key, value in dict_param.items():
            if hasattr(self, key):
                setattr(self, key, value)      
    
    def _init_criterion(self):
        # self.reconstruction_loss = torch.nn.L1Loss().cuda()
        # self.ssim_loss = kornia.losses.SSIMLoss(window_size=11).cuda()
        self.ms_ssim_loss = kornia.losses.MS_SSIMLoss(alpha=0.025, sigmas=[0.5, 1.0, 2.0, 4.0]).cuda()
        # self.reconstruction_loss = torch.nn.HuberLoss(delta=0.6)
        self.tv_loss = kornia.losses.TotalVariation().cuda()
        
        
    def run(self):
        while self.num_step < self.total_step:
            self.num_step += 1
            
            imgs, y_imgs = next(iter(self.image_loader))
            imgs = imgs.float().cuda()
            y_imgs = y_imgs.float().cuda()

            for _ in range(self.num_critics_D):
                d_loss = self.train_D(imgs, y_imgs)
            
            g_loss, output = self.train_G(imgs, y_imgs)

            if self.num_step % self.log_interval == 0:
                info = f"[Step: {self.num_step}/{self.total_step} ({round(100 * self.num_step / self.total_step, 2)}%)] "
                info += f"D Loss: {round(d_loss, 3)}"
                info += f"\tG Loss: {round(g_loss, 3)}"
                log.info(info)

            if self.num_step % self.visualize_interval == 0:
                idx = 0
                reconst_raw = torch.from_numpy(postprocess_raw(demosaic(torch.clamp(output, min=0., max=1.)[idx].detach().cpu().permute(1, 2, 0).numpy()))).permute(2, 0, 1)
                gt_raw = torch.from_numpy(postprocess_raw(demosaic(y_imgs[idx].detach().cpu().permute(1, 2, 0).numpy()))).permute(2, 0, 1)
                self.wandb.log({"examples": [
                    self.wandb.Image(self.to_pil(gt_raw), caption="raw_image"),
                    self.wandb.Image(self.to_pil(imgs[idx].cpu()), caption="rgb_image"),
                    self.wandb.Image(self.to_pil(reconst_raw), caption="output")
                ]}, commit=False)

            if self.num_step % self.save_interval == 0 and self.num_step != 0:
                self.do_checkpoint(self.num_step)

            if self.num_step % self.eval_interval == 0 and self.num_step != 0:
                self.evaluate()
            self.wandb.log({})
        self.do_checkpoint(self.num_step)

    def evaluate(self):
        self.net.eval()
        log.info(f"[Step: {self.num_step}/{self.total_step} ({round(100 * self.num_step / self.total_step, 2)}%)] Evaluating...")
        psnr_lst, ssim_lst = list(), list()
        for imgs, y_imgs in self.test_image_loader:
            imgs = imgs.float().cuda()
            y_imgs = y_imgs.float().cuda()

            with torch.no_grad():
                output = self.net(imgs)
                output = torch.clamp(output, max=1., min=0.)

            ssim = self.SSIM(255. * y_imgs, 255. * output).item()
            ssim_lst.append(ssim)

            psnr = -kornia.losses.psnr_loss(output, y_imgs, max_val=1.).item()  # -self.PSNR(y_imgs, output).item()
            psnr_lst.append(psnr)
            
        results = {"Dataset": self.dataset_name, "PSNR": np.round(np.mean(psnr_lst), 4), "SSIM": np.round(np.mean(ssim_lst), 4)}
        log.info(results)
        self.wandb.log({
            "test/psnr": results["PSNR"],
            "test/ssim": results["SSIM"],
        }, commit=False)
        if np.mean(psnr_lst) > self.best_psnr:
            self.best_psnr = np.mean(psnr_lst)
            log.info("Best PSNR: {}".format(self.best_psnr))
            self.do_checkpoint("best")
        self.net.train()

    def train_D(self, x, y):
        self.optimizer_discriminator.zero_grad()
        output = self.net(x)
        output_r = kornia.geometry.resize(output, 256)
        y_r = kornia.geometry.resize(y, 256)
        
        real_global_validity = self.discriminator(y_r).mean()
        fake_global_validity = self.discriminator(output_r.detach()).mean()
        gp_global = compute_gradient_penalty(self.discriminator, output_r.data, y_r.data)

        # real_patch_validity = self.patch_discriminator(y).mean()
        # fake_patch_validity = self.patch_discriminator(output.detach()).mean()
        # gp_fake = compute_gradient_penalty(self.patch_discriminator, output.data, y.data)

        real_validity = real_global_validity # + real_patch_validity
        fake_validity = fake_global_validity # + fake_patch_validity
        gp = gp_global # + gp_fake

        d_loss = -real_validity + fake_validity + self.grad_penalty * gp
        d_loss.backward()
        # nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5.0)
        self.optimizer_discriminator.step()

        self.wandb.log({
            # "real_global_validity": -real_global_validity.item(),
            # "fake_global_validity": fake_global_validity.item(),
            # "real_patch_validity": -real_patch_validity.item(),
            # "fake_patch_validity": fake_patch_validity.item(),
            # "gp_global": gp_global.item(),
            # "gp_fake": gp_fake.item(),
            "real_validity": -real_validity.item(),
            "fake_validity": fake_validity.item(),
            "gp": gp.item()
        }, commit=False)
        return d_loss.item()

    def train_G(self, x, y):
        self.optimizer.zero_grad()

        output = self.net(x)
        # recon_loss = self.reconstruction_loss(output, y)
        # ssim_loss = self.ssim_loss(output, y)
        ms_ssim_loss = self.ms_ssim_loss(output, y)
        tv_loss = self.tv_loss(output).mean()
        adv_global_loss = -self.discriminator(kornia.geometry.resize(output, 256)).mean()
        # adv_patch_loss = -self.patch_discriminator(output).mean()

        adv_loss = (adv_global_loss)  # + adv_patch_loss)

        g_loss = self.lambdas["MS_SSIM"] * ms_ssim_loss + \
            self.lambdas["TV"] * tv_loss + \
            self.lambdas["ADVERSARIAL"] * adv_loss
                # self.lambdas["RECON"] * recon_loss + \
                # self.lambdas["SSIM"] * ssim_loss + \
                
        g_loss.backward()
        self.optimizer.step()

        self.wandb.log({
            # "recon_loss": recon_loss.item(),
            # "ssim_loss": ssim_loss.item(),
            "ms_ssim_loss": ms_ssim_loss.item(),
            "tv_loss": tv_loss.item(),
            # "adv_global_loss": adv_global_loss.item(),
            # "adv_patch_loss": adv_patch_loss.item(),
            "adv_loss": adv_loss.item()
        }, commit=False)
        return g_loss.item(), output.detach()
    
    def check_and_use_multi_gpu(self):
        if torch.cuda.device_count() > 1 and self.num_gpu > 1:
            log.info("Using {} GPUs...".format(torch.cuda.device_count()))
            self.net = torch.nn.DataParallel(self.net).cuda()
            self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
            # self.patch_discriminator = torch.nn.DataParallel(self.patch_discriminator).cuda()
        else:
            log.info("GPU ID: {}".format(torch.cuda.current_device()))
            self.net = self.net.cuda()
            self.discriminator = self.discriminator.cuda()
            # self.patch_discriminator = self.patch_discriminator.cuda()

    def do_checkpoint(self, num_step):
        os.makedirs(self.save_dir, exist_ok=True)

        checkpoint = {
            'num_step': num_step if isinstance(num_step, int) else self.num_step,
            'model': self.net.module.state_dict() if isinstance(self.net, torch.nn.DataParallel) else self.net.state_dict(),
            'D': self.discriminator.module.state_dict() if isinstance(self.discriminator, torch.nn.DataParallel) else self.discriminator.state_dict(),
            # 'patch_D': self.patch_discriminator.module.state_dict() if isinstance(self.patch_discriminator, torch.nn.DataParallel) else self.patch_discriminator.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'optimizer_D': self.optimizer_discriminator.state_dict()
        }
        torch.save(checkpoint, "./{}/checkpoint-{}.pth".format(self.save_dir, num_step))

    def load_latest_checkpoint(self):
        ckpt_names  = glob.glob('{}/*.pth'.format(self.save_dir))
        ckpt_names = sorted(ckpt_names)
        ckpt_path = None

        if len(ckpt_names) == 0:
            log.info('No checkpoints found in directory {}'.format(self.save_dir))
        else:
            latest_eps  = max([int(k.split("/")[-1].split(".")[0].split("_")[-1]) for k in ckpt_names])
            log.info('Checkpoint found with step at {}'.format(latest_eps))
            ckpt_path = ckpt_names[0].replace('0', str(latest_eps))

        if ckpt_path is not None:
            self.load_checkpoint_from_ckpt(ckpt_path)

    def load_checkpoint_from_ckpt(self, ckpt_path):
        checkpoints = torch.load(ckpt_path)
        self.num_step = checkpoints["num_step"]
        self.net.load_state_dict(checkpoints["model"], strict=False)
        self.discriminator.load_state_dict(checkpoints["D"])
        # self.patch_discriminator.load_state_dict(checkpoints["patch_D"])

        self.optimizer.load_state_dict(checkpoints["optimizer"])        
        self.optimizer_discriminator.load_state_dict(checkpoints["optimizer_D"])
        self.optimizers_to_cuda()

    def optimizers_to_cuda(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        for state in self.optimizer_discriminator.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

