import os
import torch
import math
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import wandb
import loggers
logging.getLogger("numba").setLevel(logging.WARNING)
import commons
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
)
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from text.symbols import symbols
from download_utils import load_pretrain_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

torch.backends.cudnn.benchmark = True
global_step = 0
count = 0

def run():
    wand_project_melo = 'melotts-optimization'
    wandb.init(project=wand_project_melo)
    hps = utils.get_hparams()
    
    torch.manual_seed(hps.train.seed)
    
    global global_step
    
    loggers.log_train.info("TRAIN.PY")
    loggers.log_train.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_size=hps.train.batch_size,
        persistent_workers=True,
        prefetch_factor=4,
    )
    


    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(
        eval_dataset,
        num_workers=4,
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).cuda()
    
    wandb.watch(net_g, log="all")
    
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda()
    
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
        weight_decay= hps.train.lr_decay
        
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
        weight_decay= hps.train.lr_decay
        
    )
    
    net_dur_disc = None
    optim_dur_disc = None
    if hasattr(hps.model, 'use_duration_discriminator') and hps.model.use_duration_discriminator:
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda()
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
            weight_decay= hps.train.lr_decay
        )

    pretrain_G, pretrain_D, pretrain_dur = load_pretrain_model()
    hps.pretrain_G = hps.pretrain_G or pretrain_G
    hps.pretrain_D = hps.pretrain_D or pretrain_D
    hps.pretrain_dur = hps.pretrain_dur or pretrain_dur

    if hps.pretrain_G:
        utils.load_checkpoint(hps.pretrain_G, net_g, None, skip_optimizer=True)
    if hps.pretrain_D:
        utils.load_checkpoint(hps.pretrain_D, net_d, None, skip_optimizer=True)
    if net_dur_disc is not None and hps.pretrain_dur:
        utils.load_checkpoint(hps.pretrain_dur, net_dur_disc, None, skip_optimizer=True)
    
    try:
        _, optim_g, g_resume_lr, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
            net_g,
            optim_g,
            skip_optimizer=hps.train.skip_optimizer if "skip_optimizer" in hps.train else True,
        )
        _, optim_d, d_resume_lr, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
            net_d,
            optim_d,
            skip_optimizer=hps.train.skip_optimizer if "skip_optimizer" in hps.train else True,
        )
        if net_dur_disc is not None:
            _, optim_dur_disc, dur_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                net_dur_disc,
                optim_dur_disc,
                skip_optimizer=hps.train.skip_optimizer if "skip_optimizer" in hps.train else True,
            )

        epoch_str = max(epoch_str, 1)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        print('except')
        epoch_str = 1
        global_step = 0
        
    def exponential_adsr(attack, decay, sustain, release, attack_start, attack_max, n):
        total = attack + decay + release
        attack_range = attack_max - attack_start

        if n < 0:
            return attack_start
        elif n >= total:
            return attack_start

        if attack > 0:
            if n < attack:
                return attack_start + attack_range * (1 - math.exp(-2.5 * n / attack)) / (1 - math.exp(-2.5))
        else:
            n = max(n, 1)  # Avoid division by zero

        if decay > 0:
            if n < attack + decay:
                return sustain + (attack_max - sustain) * math.exp(-2.5 * (n - attack) / decay)
        else:
            if n == attack:
                return sustain

        if release > 0:
            release_start = sustain
            release_range = release_start - attack_start
            if n < total:
                return attack_start + release_range * math.exp(-2.5 * (n - attack - decay) / release)
        else:
            if n < total:
                return sustain

        return attack_start

    def lr_lambda(epoch):
        return exponential_adsr(
        attack=hps.train.warmup_epochs, # train_warmup_epochs
        decay=hps.train.epochs-hps.train.warmup_epochs, # el resto
        sustain=0,
        release=0,
        attack_start = 0.1,
        attack_max = 0.9,
        n=epoch)
    
    


 
   

    
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(optim_g, lr_lambda)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optim_d, lr_lambda)
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.LambdaLR(optim_dur_disc, lr_lambda)
    
    scaler = GradScaler(enabled=hps.train.fp16_run)

    print([epoch_str,hps.train.epochs + 1])
    for epoch in range(epoch_str, hps.train.epochs + 1):
        print('patata'+ str(epoch))
        train_and_evaluate(
            epoch,
            hps,
            [net_g, net_d, net_dur_disc],
            [optim_g, optim_d, optim_dur_disc],
            [scheduler_g, scheduler_d, scheduler_dur_disc],
            scaler,
            [train_loader, eval_loader],
            loggers,
            [writer, writer_eval],
        )
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()

def train_and_evaluate(
    epoch, hps, nets, optims, schedulers, scaler, loaders, loggers, writers
):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    train_loader, eval_loader = loaders
    writer, writer_eval = writers
    """
    print(type(loggers))
    print(type(loggers.log))
    print(type(loggers.log_train))
    """
    global global_step
    global count
    best_loss_sum = float('inf')  
    best_model = None
    
    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
        
    HALT = False
        
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, tone, language, bert, ja_bert) in enumerate(tqdm(train_loader)):
        x, x_lengths = x.cuda(non_blocking=True), x_lengths.cuda(non_blocking=True)
        spec, spec_lengths = spec.cuda(non_blocking=True), spec_lengths.cuda(non_blocking=True)
        y, y_lengths = y.cuda(non_blocking=True), y_lengths.cuda(non_blocking=True)
        speakers = speakers.cuda(non_blocking=True)
        tone = tone.cuda(non_blocking=True)
        language = language.cuda(non_blocking=True)
        bert = bert.cuda(non_blocking=True)
        ja_bert = ja_bert.cuda(non_blocking=True)
        
                   
        
        if HALT==True:
            HALT=False
            break
            
        if (x.cpu().sum()==0):
            print('error');
                               
         
        with autocast(enabled=hps.train.fp16_run):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (hidden_x, logw, logw_) = net_g(x, x_lengths, spec, spec_lengths, speakers, tone, language, bert, ja_bert)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            
            y_spec = commons.slice_segments(
                spec, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            
            y_hat_spec = spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
            )
            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )

            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
            
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw.detach(), logw_.detach()
                )
                with autocast(enabled=False):
                    loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d, _ = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)
            with autocast(enabled=False):
                
                loggers.log_loss.info("EPOCH")
                loggers.log_loss.info(epoch)
                
                loss_dur = torch.sum(l_length.float())
                
                log_dur1 = [w1.detach() - w2.detach() for w1, w2 in zip(logw, logw_)]


                loss_duraciones = F.l1_loss(logw.detach(), logw_.detach())


                
                loggers.log_loss.info("loss_dur")
                loggers.log_loss.info(loss_dur.cpu().detach().numpy())
                """
                print(y_mel.shape)
                print(y_hat_mel.shape)
                """
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loggers.log_loss.info("loss_mel")
                loggers.log_loss.info(loss_mel.cpu().detach().numpy())
                
                loss_spec = F.l1_loss(y_spec, y_hat_spec)
                loggers.log_loss.info("loss_spec")
                loggers.log_loss.info(loss_spec.cpu().detach().numpy())
                
                # Keep the latent-space loss for the KL-driven stage transition.
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                
                loggers.log_loss.info("z_mask")
                loggers.log_loss.info(z_mask.cpu().detach().numpy())
               
                loggers.log_loss.info(z_mask.shape)
                
                loggers.log_loss.info("x_mask")
                loggers.log_loss.info(x_mask.cpu().detach().numpy())
               
                loggers.log_loss.info(x_mask.shape)
                    
                
                loss_fm = feature_loss(fmap_r, fmap_g)
                
                loggers.log_loss.info("loss_fm")
                loggers.log_loss.info(loss_fm.cpu().detach().numpy())
                
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                
                loggers.log_loss.info("loss_gen")
                loggers.log_loss.info(loss_gen.cpu().detach().numpy())
                for i, loss in enumerate(losses_gen):
                    loggers.log_loss.info(f"losses_gen[{i}]")
                    loggers.log_loss.info(loss.cpu().detach().numpy())
                    
                
                
                def controller(losses_g, selector=[1,1,1,1,1], weights=[1,1,1,1,1]):
                    """
                    losses_g: lista de tensores o listas de tensores (pérdidas)
                    selector: lista de 0/1 indicando qué pérdidas incluir (default: todos)
                    weights: lista de pesos para cada pérdida (default: 1 para todas)
                    """

                    if selector is None:
                        selector = [1] * len(losses_g)
                    if weights is None:
                        weights = [1] * len(losses_g)

                    assert len(selector) == len(losses_g), "selector y losses_g deben tener igual longitud"
                    assert len(weights) == len(losses_g), "weights y losses_g deben tener igual longitud"

                    loss_gen_all = torch.tensor(0.0, device=losses_g[0].device)  # o torch.tensor(0.0) si quieres tensor inicial
                    for i in range(len(selector)):
                        if selector[i] == 1:
                            loss = losses_g[i]

                            if isinstance(loss, list):
                                loss = sum(loss)

                            loss_gen_all += loss * weights[i]
                      
                    return loss_gen_all
                losses_g = [loss_gen, loss_fm, loss_mel, loss_dur, loss_kl, loss_spec]
                
                    
                
                    
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g, _ = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl, loss_spec]
            loggers.log_train.info(
                "Train Epoch: {} [{:.0f}%]".format(
                    epoch, 100.0 * batch_idx / len(train_loader)
                )
            )
            loggers.log_train.info([x.item() for x in losses] + [global_step, lr])

            scalar_dict = {
                "loss/g/total": loss_gen_all,
                "loss/d/total": loss_disc_all,
                "learning_rate": lr,
                "grad_norm_d": grad_norm_d,
                "grad_norm_g": grad_norm_g,
            }
            scalar_dict.update(
                {
                    "loss/g/fm": loss_fm,
                    "loss/g/mel": loss_mel,
                    "loss/g/dur": loss_dur,
                    "loss/g/kl": loss_kl,
                    "loss/g/spec": loss_spec,
                }
            )
            scalar_dict.update(
                {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
            )
            scalar_dict.update(
                {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
            )
            scalar_dict.update(
                {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
            )

            image_dict = {
                "slice/mel_org": utils.plot_spectrogram_to_numpy(
                    y_mel[0].data.cpu().numpy()
                ),
                "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                    y_hat_mel[0].data.cpu().numpy()
                ),
                "all/mel": utils.plot_spectrogram_to_numpy(
                    mel[0].data.cpu().numpy()
                ),
                "all/attn": utils.plot_alignment_to_numpy(
                    attn[0, 0].data.cpu().numpy()
                ),
            }
            utils.summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                scalars=scalar_dict,
            )
            wandb.log(scalar_dict)

        if global_step % hps.train.eval_interval == 0:
            evaluate(hps, net_g, eval_loader, writer_eval)
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
            )
            if net_dur_disc is not None:
                utils.save_checkpoint(
                    net_dur_disc,
                    optim_dur_disc,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                )
            keep_ckpts = getattr(hps.train, "keep_ckpts", 0)
            if keep_ckpts > 0:
                utils.clean_checkpoints(
                    path_to_models=hps.model_dir,
                    n_ckpts_to_keep=keep_ckpts,
                    sort_by_time=True,
                )
        total_loss = loss_gen_all + loss_disc_all
        if total_loss < best_loss_sum:
            best_loss_sum = total_loss
            best_model = net_g  
            torch.save(best_model.state_dict(), "best_model.pth")
        wandb.save("best_model.pth")
        global_step += 1

    loggers.log_train.info("====> Epoch: {}".format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    print("Evaluating ...")
    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, tone, language, bert, ja_bert) in enumerate(eval_loader):
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            bert = bert.cuda()
            ja_bert = ja_bert.cuda()
            tone = tone.cuda()
            language = language.cuda()
            for use_sdp in [True, False]:
                y_hat, attn, mask, *_ = generator.infer(
                    x,
                    x_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    ja_bert,
                    y=spec,
                    max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                image_dict.update(
                    {
                        f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].cpu().numpy()
                        )
                    }
                )
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{use_sdp}": y_hat[
                            0, :, : y_hat_lengths[0]
                        ]
                    }
                )
                image_dict.update(
                    {
                        f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            mel[0].cpu().numpy()
                        )
                    }
                )
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()
    print('Evaluate done')


if __name__ == "__main__":
    loggers.init()
    error_history_mel = []
    momentum_history_mel = []
    filtered_history_mel = []
    error_history_dur = []
    momentum_history_dur = []
    filtered_history_dur = []
    run()
