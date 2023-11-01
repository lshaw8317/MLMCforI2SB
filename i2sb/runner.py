# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics

import distributed_util as dist_util
from evaluation import build_resnet50

from . import util
from .network import Image256Net
from .diffusion import Diffusion

from ipdb import set_trace as debug


def mom2norm(sqsums):
    #sqsums should have shape L,C,H,W
    s=sqsums.shape
    if len(s)!=4:
        raise Exception('shape of sqsums likely not LHCW')
    return torch.sum(torch.flatten(sqsums, start_dim=1, end_dim=-1),dim=-1)/np.prod(s[1:])

def imagenorm(img):
    s=img.shape
    if len(s)==1: #fix for when img is single dimensional (batch_size,) -> (batch_size,1)
        img=img[:,None]
    n=torch.linalg.norm(torch.flatten(img, start_dim=1, end_dim=-1),dim=-1) #flattens non-batch dims and calculates norm
    n/=np.sqrt(np.prod(s[1:]))
    return n

def build_optimizer_sched(opt, net, log):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

class Runner(object):
    def __init__(self, opt, log,mlmcoptions=None, save_opt=True):
        super(Runner,self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)
    
        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")
        
        if mlmcoptions:
            #MLMC params
            self.M=mlmcoptions.M
            self.Lmax=mlmcoptions.Lmax
            self.Lmin=mlmcoptions.Lmin
            self.mlmc_batch_size=mlmcoptions.batch_size
            self.accsplit=mlmcoptions.accsplit
            self.N0=mlmcoptions.N0
            self.eval_dir=mlmcoptions.eval_dir
        
            if mlmcoptions.payoff=='mean':
                self.payoff=lambda x: torch.clip((x+1)/2,0.,1.) #[-1,1]->[0,1]
            elif mlmcoptions.payoff=='secondmoment':
                self.payoff=lambda x: torch.clip((x+1)/2,0.,1.)**2 #[-1,1]->[0,1]
            else:
                print('mlmcoptions payoff arg not recognised, Setting to mean payoff by default')
                self.payoff=lambda x: torch.clip((x+1)/2,0.,1.) #[-1,1]->[0,1]
                
            self.net = torch.nn.DataParallel(self.net)

        self.net.to(opt.device)
        self.ema.to(opt.device)

        self.log = log

    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def sample_batch(self, opt, loader, corrupt_method):
        if opt.corrupt == "mixture":
            clean_img, corrupt_img, y = next(loader)
            mask = None
        elif "inpaint" in opt.corrupt:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img, mask = corrupt_method(clean_img.to(opt.device))
        else:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img = corrupt_method(clean_img.to(opt.device))
            mask = None

        # os.makedirs(".debug", exist_ok=True)
        # tu.save_image((clean_img+1)/2, ".debug/clean.png", nrow=4)
        # tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png", nrow=4)
        # debug()

        y  = y.detach().to(opt.device)
        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)
        if mask is not None:
            mask = mask.detach().to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
        cond = x1.detach() if opt.cond_x1 else None

        if opt.add_x1_noise: # only for decolor
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape

        return x0, x1, mask, y, cond

    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        self.accuracy = torchmetrics.Accuracy().to(opt.device)
        self.resnet = build_resnet50().to(opt.device)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method)

                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (x0.shape[0],))

                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(step, x0, xt)

                pred = net(xt, step, cond=cond)
                assert xt.shape == label.shape == pred.shape

                if mask is not None:
                    pred = mask * pred
                    label = mask * label

                loss = F.mse_loss(pred, label)
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 5000 == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            if it == 500 or it % 3000 == 0: # 0, 0.5k, 3k, 6k 9k
                net.eval()
                self.evaluation(opt, it, val_loader, corrupt_method)
                net.train()
        self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                out = self.net(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0
    
    def mlmclooper(self, Nl, l ,M, opt, corrupt_img, mask=None, cond=None, clip_denoise=False, log_count=0, verbose=True):
        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = M**l
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        assert cond==None
        # # create log steps
        # log_count = min(len(steps)-1, log_count)
        # log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        # assert log_steps[0] == 0
        # self.log.info(f"[MLMC loop Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")
        
        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                out = self.net(xt, step, cond=cond)
                std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
                pred_x0 = xt - std_fwd * out
                if clip_denoise: pred_x0.clamp_(-1., 1.)
                return pred_x0
              
            
            num_sampling_rounds = Nl // self.mlmc_batch_size + 1
            numrem=Nl % self.mlmc_batch_size
            for r in range(num_sampling_rounds):
                bs=numrem if r==num_sampling_rounds-1 else self.mlmc_batch_size
                if bs==0:
                    break
                with torch.no_grad():
                    Xf, Xc = self.diffusion.mlmcsample(
                        steps, M, pred_x0_fn, corrupt_img, bs, mask=mask, ot_ode=opt.ot_ode, 
                        log_steps=None, verbose=verbose)
                fine_payoff=self.payoff(Xf)
                coarse_payoff=self.payoff(Xc)
                if r==0:
                    sums=torch.zeros((3,*fine_payoff.shape[1:])) #skip batch_size
                    sqsums=torch.zeros((4,*fine_payoff.shape[1:]))
                sumXf=torch.sum(fine_payoff,axis=0).to('cpu') #sum over batch size
                sumXf2=torch.sum(fine_payoff**2,axis=0).to('cpu')
                if l==self.Lmin:
                    sqsums+=torch.stack([sumXf2,sumXf2,torch.zeros_like(sumXf2),torch.zeros_like(sumXf2)])
                    sums+=torch.stack([sumXf,sumXf,torch.zeros_like(sumXf)])
                elif l<self.Lmin:
                    raise ValueError("l must be at least Lmin")
                else:
                    dX_l=fine_payoff-coarse_payoff #Image difference
                    sumdX_l=torch.sum(dX_l,axis=0).to('cpu') #sum over batch size
                    sumdX_l2=torch.sum(dX_l**2,axis=0).to('cpu')
                    sumXc=torch.sum(coarse_payoff,axis=0).to('cpu')
                    sumXc2=torch.sum(coarse_payoff**2,axis=0).to('cpu')
                    sumXcXf=torch.sum(coarse_payoff*fine_payoff,axis=0).to('cpu')
                    sums+=torch.stack([sumdX_l,sumXf,sumXc])
                    sqsums+=torch.stack([sumdX_l2,sumXf2,sumXc2,sumXcXf])

            # Directory to save samples. Just to save an example sample for debugging
            if l>self.Lmin:
                this_sample_dir = os.path.join(self.eval_dir, f"level_{l}")
                if not os.path.exists(this_sample_dir):
                    os.mkdir(this_sample_dir)
                samples_f=np.clip(Xf.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
                samples_c=np.clip(Xc.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
                with open(os.path.join(this_sample_dir, "samples_f.npz"), "wb") as fout:
                    np.savez_compressed(fout, samplesf=samples_f)
                with open(os.path.join(this_sample_dir, "samples_c.npz"), "wb") as fout:
                    np.savez_compressed(fout, samplesc=samples_c)
                                    
        return sums,sqsums 
    
    
    def Giles_plot(self,acc,opt, corrupt_img, mask, cond):
        torch.cuda.empty_cache() 
        #Set mlmc params
        M=self.M
        N0=self.N0
        Lmax=self.Lmax
        eval_dir = self.eval_dir
        Nsamples=1000
        Lmin=self.Lmin
        
        # Directory to save means and norms                                                                                               
        this_sample_dir = os.path.join(eval_dir, f"VarMean_M_{M}_Nsamples_{Nsamples}")
        tpayoffshape=self.payoff(torch.randn(*corrupt_img.shape)).shape[1:]
        sums=torch.zeros((1,3,*tpayoffshape))
        sqsums=torch.zeros((1,4,*tpayoffshape))

        if not os.path.exists(this_sample_dir):
            #Variance and mean samples
            sums=torch.zeros((Lmax+1,*sums.shape[1:]))
            sqsums=torch.zeros((Lmax+1,*sqsums.shape[1:]))
            os.mkdir(this_sample_dir)
            print(f'Proceeding to calculate variance and means with {Nsamples} estimator samples')
            for i,l in enumerate(range(Lmax+1)):
                print(f'l={l}')
                sums[i],sqsums[i] = self.mlmclooper(Nsamples, l ,M, opt, corrupt_img, mask, cond, clip_denoise=opt.clip_denoise, log_count=0, verbose=False)
            
            
            # Write samples to disk or Google Cloud Storage
            with open(os.path.join(this_sample_dir, "averages.pt"), "wb") as fout:
                torch.save(sums/Nsamples,fout)
            with open(os.path.join(this_sample_dir, "sqaverages.pt"), "wb") as fout:
                torch.save(sqsums/Nsamples,fout)
            with open(os.path.join(this_sample_dir, "avgcost.pt"), "wb") as fout:
                torch.save(torch.cat((torch.tensor([1]),(1+1./M)*M**torch.arange(1,Lmax+1))),fout)
            
            means_dp=imagenorm(sums[:,0])/Nsamples
            V_dp=mom2norm(sqsums[:,0])/Nsamples-means_dp**2  
            means_p=imagenorm(sums[:,1])/Nsamples
            V_p=mom2norm(sqsums[:,1])/Nsamples-means_p**2  
            #Estimate orders of weak (alpha from means) and strong (beta from variance) convergence using LR
            cutoff=np.argmax(V_dp<(np.sqrt(M)-1.)**2*V_p[-1]/(1+M))-1 #index of optimal lmin 
            means_p=means_p[cutoff:]
            V_p=V_p[cutoff:]
            means_dp=means_dp[cutoff:]
            V_dp=V_dp[cutoff:]
            
            X=np.ones((Lmax-cutoff+1,2))
            X[:,0]=np.arange(1,Lmax+1)
            a = np.linalg.lstsq(X,np.log(means_dp[1:]),rcond=None)[0]
            alpha = -a[0]/np.log(M)
            Y0=np.exp(a[1])
            b = np.linalg.lstsq(X,np.log(V_dp[1:]),rcond=None)[0]
            beta = -b[0]/np.log(M) 

            print(f'Estimated alpha={alpha}\n Estimated beta={beta}\n')
            with open(os.path.join(this_sample_dir, "info_text.txt"),'w') as f:
                f.write(f'MLMC params: Nsamples={Nsamples}, M={M}, accsplit={self.accsplit}.\n')
                f.write(f'Estimated alpha={alpha}. Estimated beta={beta}. Estimated Lmin={cutoff}.')
            with open(os.path.join(this_sample_dir, "alphabetagamma.pt"), "wb") as fout:
                torch.save(torch.tensor([alpha,beta,cutoff]),fout)
                
        with open(os.path.join(this_sample_dir, "alphabetagamma.pt"),'rb') as f:
            temp=torch.load(f)
            alpha=temp[0].item()
            beta=temp[1].item()
            Lmin=int(temp[-1])
            print(alpha,beta,Lmin)
        assert Lmin == self.Lmin
        
        #Do the calculations and simulations for num levels and complexity plot
        for i in range(len(acc)):
            e=acc[i]
            print(f'Performing mlmc for accuracy={e}')
            sums,sqsums,N=self.mlmc(e,Lmin,alpha_0=alpha,beta_0=beta,opt=opt, corrupt_img=corrupt_img, mask=mask, cond=cond) #sums=[dX,Xf,Xc], sqsums=[||dX||^2,||Xf||^2,||Xc||^2]
            L=len(N)-1+Lmin
            means_p=imagenorm(sums[:,1])/N #Norm of mean of fine discretisations
            V_p=torch.clip(mom2norm(sqsums[:,1])/N-means_p**2,min=0)

            #cost
            cost_mlmc=torch.sum(N*(M**np.arange(Lmin,L+1)+np.hstack((0,M**np.arange(Lmin,L))))) #cost is number of NFE
            cost_mc=V_p[-1]*(self.M**L)/(e*self.accsplit)**2
            
            
            # Directory to save means, norms and N
            dividerN=N.clone() #add axes to N to broadcast correctly on division
            for i in range(len(sums.shape[1:])):
                dividerN.unsqueeze_(-1)
            this_sample_dir = os.path.join(eval_dir, f"M_{M}_accuracy_{e}")
            
            if not os.path.exists(this_sample_dir):
                os.mkdir(this_sample_dir)        
            with open(os.path.join(this_sample_dir, "averages.pt"), "wb") as fout:
                torch.save(sums/dividerN,fout)
            with open(os.path.join(this_sample_dir, "sqaverages.pt"), "wb") as fout:
                torch.save(sqsums/dividerN,fout) #sums has shape (L,4,C,H,W) if img (L,4,2048) if activations
            with open(os.path.join(this_sample_dir, "N.pt"), "wb") as fout:
                torch.save(N,fout)
            with open(os.path.join(this_sample_dir, "costs.npz"), "wb") as fout:
               np.savez_compressed(fout,costmlmc=np.array(cost_mlmc),costmc=np.array(cost_mc))

            meanimg=torch.sum(sums[:,0]/dividerN[:,0,...],axis=0)#cut off one dummy axis
            meanimg=torch.clip(meanimg,0.,1.).permute(1,2,0).cpu().numpy
            # Write samples to disk or Google Cloud Storage
            with open(os.path.join(this_sample_dir, "meanpayoff.npz"), "wb") as fout:
                np.savez_compressed(fout, meanpayoff=meanimg)

        return None

    def mlmc(self,accuracy,Lmin,alpha_0,beta_0, opt, corrupt_img, mask, cond, clip_denoise=False, log_count=0, verbose=False):
        accsplit=self.accsplit
        #Orders of convergence
        alpha=max(0,alpha_0)
        beta=max(0,beta_0)
        M=self.M
        N0=self.N0
        Lmax=self.Lmax
        L=Lmin+1

        mylen=L+1-Lmin
        V=torch.zeros(mylen) #Initialise variance vector of each levels' variance
        N=torch.zeros(mylen) #Initialise num. samples vector of each levels' num. samples
        dN=N0*torch.ones(mylen) #Initialise additional samples for this iteration vector for each level
        sqrt_cost=torch.sqrt(M**torch.arange(Lmin,L+1.)+torch.hstack((torch.tensor([0.]),M**torch.arange(Lmin,1.*L))))
        it0_ind=False
        while (torch.sum(dN)>0): #Loop until no additional samples asked for
            mylen=L+1-Lmin
            for i,l in enumerate(torch.arange(Lmin,L+1)):
                num=dN[i]
                if num>0: #If asked for additional samples...
                    tempsums,tempsqsums=self.mlmclooper(int(num), l ,M, opt, corrupt_img, mask, cond, clip_denoise, log_count, verbose) #Call function which gives sums
                    if not it0_ind:
                        sums=torch.zeros((mylen,*tempsums.shape)) #Initialise sums array of unnormed [dX,Xf,Xc], each column is a level
                        sqsums=torch.zeros((mylen,*tempsqsums.shape)) #Initialise sqsums array of normed [dX^2,Xf^2,Xc^2,XcXf], each column is a level
                        it0_ind=True
                    sqsums[i,...]+=tempsqsums
                    sums[i,...]+=tempsums
                    
            N+=dN #Increment samples taken counter for each level
            Yl=imagenorm(sums[:,0])/N
            V=torch.clip(mom2norm(sqsums[:,0])/N-(Yl)**2,min=0) #Calculate variance based on updated samples
            
            ##Fix to deal with zero variance or mean by linear extrapolation
            Yl[2:]=torch.maximum(Yl[2:],.5*Yl[1:-1]*M**(-alpha))
            V[2:]=torch.maximum(V[2:],.5*V[1:-1]*M**(-beta))
            
            #Estimate order of weak convergence using LR
            #Yl=(M^alpha-1)khl^alpha=(M^alpha-1)k(TM^-l)^alpha=((M^alpha-1)kT^alpha)M^(-l*alpha)
            #=>log(Yl)=log(k(M^alpha-1)T^alpha)-alpha*l*log(M)
            X=torch.ones((mylen-1,2))
            X[:,0]=torch.arange(1,mylen)
            a = torch.linalg.lstsq(X,torch.log(Yl[1:]))[0]
            alpha_ = max(-a[0]/np.log(M),0.)
            b = torch.linalg.lstsq(X,torch.log(V[1:]))[0]
            beta_= -b[0]/np.log(M)
            if alpha_0==-1:
                alpha=alpha_
            if beta_0==-1:
                beta=beta_
                
            sqrt_V=torch.sqrt(V)
            Nl_new=torch.ceil(((accsplit*accuracy)**-2)*torch.sum(sqrt_V*sqrt_cost)*(sqrt_V/sqrt_cost)) #Estimate optimal number of samples/level
            dN=torch.clip(Nl_new-N,min=0) #Number of additional samples
            print(f'Asking for {dN} new samples for l=[{Lmin,L}]')
            if torch.sum(dN > 0.01*N).item() == 0: #Almost converged
                if max(Yl[-2]/(M**alpha),Yl[-1])>(M**alpha-1)*accuracy*np.sqrt(1-accsplit**2):
                    L+=1
                    print(f'Increased L to {L}')
                    if (L>Lmax):
                        print('Asked for an L greater than maximum allowed Lmax. Ending MLMC algorithm.')
                        break
                    #Add extra entries for the new level and estimate sums with N0 samples 
                    V=torch.cat((V,V[-1]*M**(-beta)*torch.ones(1)), dim=0)
                    sqrt_V=torch.sqrt(V)
                    newcost=torch.sqrt(torch.tensor([M**L+M**((L-1.))]))
                    sqrt_cost=torch.cat((sqrt_cost,newcost),dim=0)
                    Nl_new=torch.ceil(((accsplit*accuracy)**-2)*torch.sum(sqrt_V*sqrt_cost)*(sqrt_V/sqrt_cost)) #Estimate optimal number of sample
                    N=torch.cat((N,torch.tensor([0])),dim=0)
                    dN=torch.clip(Nl_new-N,min=0) #Number of additional samples
                    print(f'With new L, estimate of {dN} new samples for l=[{Lmin,L}]')
                    sums=torch.cat((sums,torch.zeros((1,*sums[0].shape))),dim=0)
                    sqsums=torch.cat((sqsums,torch.zeros((1,*sqsums[0].shape))),dim=0)
                    
        print(f'Estimated alpha = {alpha_}')
        print(f'Estimated beta = {beta_}')
        return sums,sqsums,N

    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, corrupt_method):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        img_clean, img_corrupt, mask, y, cond = self.sample_batch(opt, val_loader, corrupt_method)

        x1 = img_corrupt.to(opt.device)

        xs, pred_x0s = self.ddpm_sampling(
            opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
        )

        log.info("Collecting tensors ...")
        img_clean   = all_cat_cpu(opt, log, img_clean)
        img_corrupt = all_cat_cpu(opt, log, img_corrupt)
        y           = all_cat_cpu(opt, log, y)
        xs          = all_cat_cpu(opt, log, xs)
        pred_x0s    = all_cat_cpu(opt, log, pred_x0s)

        batch, len_t, *xdim = xs.shape
        assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
        assert xs.shape == pred_x0s.shape
        assert y.shape == (batch,)
        log.info(f"Generated recon trajectories: size={xs.shape}")

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]

        def log_accuracy(tag, img):
            pred = self.resnet(img.to(opt.device)) # input range [-1,1]
            accu = self.accuracy(pred, y.to(opt.device))
            self.writer.add_scalar(it, tag, accu)

        log.info("Logging images ...")
        img_recon = xs[:, 0, ...]
        log_image("image/clean",   img_clean)
        log_image("image/corrupt", img_corrupt)
        log_image("image/recon",   img_recon)
        log_image("debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
        log_image("debug/recon_traj",      xs.reshape(-1, *xdim),      nrow=len_t)

        log.info("Logging accuracies ...")
        log_accuracy("accuracy/clean",   img_clean)
        log_accuracy("accuracy/corrupt", img_corrupt)
        log_accuracy("accuracy/recon",   img_recon)

        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
