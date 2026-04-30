from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models import  Wav2Lip_student, Wav2Lip as Wav2Lip
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob

import os, random, cv2, argparse
import json
# from hparams import hparams, get_image_list
from hparams import get_image_list, hparams_debug_string

from models import pytorch_ssim
from models.vgg_feature import VGGFeature
torch.set_float32_matmul_precision('high') # for speed


parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model without the visual quality discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', required=True, type=str)
parser.add_argument('--teacher_checkpoint_path', help='Load the pre-trained Teacher model', required=True, type=str) # need teacher model

parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)

parser.add_argument('--hparams_config', help='Which hparams config to use (e.g., hparams_96, hparams_256)', default='hparams', type=str)


args = parser.parse_args()

hparams = getattr(__import__('hparams'), args.hparams_config)
audio.set_hparams(hparams) # need to pass hparams to audio.py!


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue
            
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            #CHANGE - use mel spectrograms instead  # for speed
            # NEED preproccessed mel files for this to work
            try:
                #wavpath = join(vidname, "audio.wav")
                #wav = audio.load_wav(wavpath, hparams.sample_rate)
                #orig_mel = audio.melspectrogram(wav).T
                
                melpath = join(vidname, 'mel.npy')
                orig_mel = np.load(melpath).T
                
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            
            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2]//2:] = 0.

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            return x, indiv_mels, mel, y

def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False

recon_loss = nn.L1Loss()
def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

def get_CD_loss(student_kd_feats_, teach_kd_feats_): # intermediate channel distillation loss
    CD_loss = []
    for s, t in zip(student_kd_feats_, teach_kd_feats_):
        w_s = s.mean(dim=(2, 3), keepdim=False) # attention weight: average over dim 2 and 3 (H×W)
        w_t = t.mean(dim=(2, 3), keepdim=False)
        loss = torch.mean(torch.pow(w_t - w_s, 2))
        CD_loss.append(loss)
    return sum(CD_loss)

def get_TV_loss(student_output): # Total Variation (TV) on output of student - measures how diffrent pixels are from the one next to them in an image
    # sum of absolute pixel differences between pixes next to eachother
    # student g.size(): torch.Size([16, 3, 5, 96, 96]) << input to this function (B, C, T, H, W)
    diff_i = torch.sum(torch.abs(student_output[:, :, :, :, 1:] - student_output[:, :, :, :, :-1]))
    diff_j = torch.sum(torch.abs(student_output[:, :, :, 1:, :] - student_output[:, :, :, :-1, :]))
    return diff_i + diff_j

def gram(x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w*h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

def train(device, student_model, teacher_model, vgg, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
    
    train_l1loss_curve = []
    train_sync_loss_curve = []
    train_running_l1loss_curve = []
    train_running_sync_loss_curve = []
    train_loss_curve = []
    
    train_CD_loss = []
    train_TV_loss = []
    train_ssim_loss = []
    train_style_loss = []
    train_feature_loss = []
    
    train_loss_time_steps = [] # need to track what steps I record train_loss
    
    val_sync_loss_curve = []
    val_recon_losses, val_loss_liz, val_CD_loss_liz, val_TV_loss_liz, val_ssim_loss_liz, val_style_loss_liz, val_feature_loss_liz = [], [], [], [], [], [], []
    val_loss_time_steps = [] # need to track what steps I record val_loss
    
 
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_l1_loss = 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, indiv_mels, mel, gt) in prog_bar:
            student_model.train()
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device) # ground truth target frames

            g, student_kd_feats = student_model(indiv_mels, x) # generated student frames
            g_teach, teach_kd_feats = teacher_model(indiv_mels, x) # generated teacher frames
            
            #   generated frames                                                   (B, C, T, H, W)
            # print("student g.size():", g.size()) # student g.size(): torch.Size([16, 3, 5, 96, 96]) 
            # print("teach g_teach.size():", g_teach.size()) # teach g_teach.size(): torch.Size([16, 3, 5, 96, 96])
            
            
            
#             for s,t in zip(student_kd_feats, teach_kd_feats): # check if same size KD layers! 
#                 print()
#                 print("student_kd_feats.size():", s.size())
#                 print("teach_kd_feats.size():", t.size())
                
# #                 student_kd_feats.size(): torch.Size([80, 1024, 1, 1])
# #                 teach_kd_feats.size(): torch.Size([80, 1024, 1, 1])

# #                 student_kd_feats.size(): torch.Size([80, 1024, 3, 3])
# #                 teach_kd_feats.size(): torch.Size([80, 1024, 3, 3])

# #                 student_kd_feats.size(): torch.Size([80, 768, 6, 6])
# #                 teach_kd_feats.size(): torch.Size([80, 768, 6, 6])

# #                 student_kd_feats.size(): torch.Size([80, 512, 12, 12])
# #                 teach_kd_feats.size(): torch.Size([80, 512, 12, 12])

# #                 student_kd_feats.size(): torch.Size([80, 320, 24, 24])
# #                 teach_kd_feats.size(): torch.Size([80, 320, 24, 24])

# #                 student_kd_feats.size(): torch.Size([80, 160, 48, 48])
# #                 teach_kd_feats.size(): torch.Size([80, 160, 48, 48])

# #                 student_kd_feats.size(): torch.Size([80, 80, 96, 96])
# #                 teach_kd_feats.size(): torch.Size([80, 80, 96, 96])

# #                 student_kd_feats.size(): torch.Size([80, 3, 96, 96])
# #                 teach_kd_feats.size(): torch.Size([80, 3, 96, 96])
            

            #----------------------------------------------------
            # Get ALL the losses: 
            #----------------------------------------------------
            # sync_loss
            #---------------------
            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g)
            else:
                sync_loss = 0.

            #---------------------
            # recon_loss or L1 loss
            #---------------------
            if hparams.L1_gt_wt != 0:
                l1loss = recon_loss(g, gt)
                L1_wt = hparams.L1_gt_wt
                train_l1loss_curve.append(l1loss.detach().cpu().item())
            elif hparams.L1_teach_wt != 0:
                l1loss = recon_loss(g, g_teach)
                L1_wt = hparams.L1_teach_wt
                train_l1loss_curve.append(l1loss.detach().cpu().item())
            else:
                l1loss = 0
                L1_wt = 0
                train_l1loss_curve.append(None)
            
            #---------------------
            # CD_loss - intermediate channel distillation loss
            #---------------------  
            if hparams.cd_wt != 0:
                CD_loss = get_CD_loss(student_kd_feats, teach_kd_feats)
                # print()
                # print("CD_loss:", CD_loss) # CD_loss: tensor(2.0059, device='cuda:0', grad_fn=<AddBackward0>)
                train_CD_loss.append(CD_loss.detach().cpu().item())
                CD_loss_report = CD_loss.detach().cpu().item()
            else:
                CD_loss = 0.
                CD_loss_report = None
                train_CD_loss.append(None)
            
            #---------------------
            # Total Variation loss - measures how diffrent pixels are from the one next to them in an image
            #---------------------
            if hparams.tv_wt != 0:
                TV_loss = get_TV_loss(g)
                # print("TV_loss:",TV_loss) # 'TV_loss': tensor(345843.0938, device='cuda:0', grad_fn=<AddBackward0>)
                train_TV_loss.append(TV_loss.detach().cpu().item())
                # print("train_TV_loss:",train_TV_loss)
                TV_loss_report = TV_loss.detach().cpu().item()
    
            else:
                TV_loss = 0.
                TV_loss_report = None
                train_TV_loss.append(None)
                
            
            #---------------------
            # reshape g and g_teach for SSIM, style, and feature Loss
            #--------------------- 
            #       generated frames                                               (B, C, T, H, W)
            # print("student g.size():", g.size()) # student g.size(): torch.Size([16, 3, 5, 96, 96]) 
            # print("teach g_teach.size():", g_teach.size()) # teach g_teach.size(): torch.Size([16, 3, 5, 96, 96])
            if hparams.ssim_wt != 0 or hparams.feature_wt != 0 or hparams.style_wt != 0:
                # losses expects (N, C, H, W)
                B, C, T, H, W = g.shape
                g_4d = g.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
                gtech_4d = g_teach.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
            
                
            #---------------------
            # Structural Similarity (SSIM) Loss
            #---------------------   
            if hparams.ssim_wt != 0:
                ssim_loss = pytorch_ssim.SSIM()
                SSIM_student_loss = (1 - ssim_loss(g_4d, gtech_4d))
                train_ssim_loss.append(SSIM_student_loss.detach().cpu().item())
                # print("SSIM_loss:", SSIM_student_loss.detach().cpu().item())
                SSIM_student_loss_report = SSIM_student_loss.detach().cpu().item()
            else:
                SSIM_student_loss = 0
                SSIM_student_loss_report = None
                train_ssim_loss.append(None)
                
            
            #-------------------------------
            # VGG for feature and style loss
            #------------------------------- 
            if hparams.feature_wt != 0 or hparams.style_wt != 0:
                Tfeatures = vgg(gtech_4d)
                Sfeatures = vgg(g_4d)
                
                Tgram = [gram(fmap) for fmap in Tfeatures]
                Sgram = [gram(fmap) for fmap in Sfeatures]
                
                
                #--------------------
                # style loss
                #--------------------
                if hparams.style_wt != 0:
                    style_loss = 0
                    for i in range(len(Tgram)):
                        style_loss += F.l1_loss(Sgram[i], Tgram[i])
                    
                    train_style_loss.append(style_loss.detach().cpu().item())
                    style_loss_report = style_loss.detach().cpu().item()
                else:
                    style_loss = 0
                    style_loss_report = None
                    train_style_loss.append(None)
                    
                #--------------------
                # feature loss
                #--------------------
                if hparams.feature_wt != 0:
                    Srecon, Trecon = Sfeatures[1], Tfeatures[1]
                    feature_loss = F.l1_loss(Srecon, Trecon)
                    
                    train_feature_loss.append(feature_loss.detach().cpu().item())
                    feature_loss_report = feature_loss.detach().cpu().item()
                else:
                    feature_loss = 0
                    feature_loss_report = None
                    train_feature_loss.append(None)
            
            
            else:
                style_loss = 0
                style_loss_report = None
                train_style_loss.append(None)
                
                feature_loss = 0
                feature_loss_report = None
                train_feature_loss.append(None)
                
                    
                    
            #---------------------------------------------------------------
            # Bring all losses together! 
            #---------------------------------------------------------------
            # loss = hparams.syncnet_wt * sync_loss + (1 - hparams.syncnet_wt) * l1loss                 # OG loss eq
            loss = hparams.syncnet_wt * sync_loss + L1_wt*(1 - hparams.syncnet_wt) * l1loss + hparams.cd_wt*CD_loss + hparams.tv_wt*TV_loss + hparams.ssim_wt*SSIM_student_loss + hparams.style_wt*style_loss + hparams.feature_wt*feature_loss
            loss.backward()
            optimizer.step()
            

            if global_step % checkpoint_interval == 0:
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

            global_step += 1
            cur_session_steps = global_step - resumed_step
            

            if hparams.L1_gt_wt != 0 or hparams.L1_teach_wt != 0:
                running_l1_loss += l1loss.item()
            else:
                running_l1_loss = 0
            
            if hparams.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.
                
                
            #----------------------------------------------------
            # save training loss data!!!!!
            #----------------------------------------------------
            
            if sync_loss == 0.0:
                train_sync_loss_curve.append(sync_loss)
            else:
                train_sync_loss_curve.append(sync_loss.detach().cpu().item())
                
            train_loss_curve.append(loss.detach().cpu().item())
            
            train_loss_time_steps.append(global_step)
                
            train_running_l1loss_curve.append(running_l1_loss)
            train_running_sync_loss_curve.append(running_sync_loss)

#             if global_step == 1 or global_step % checkpoint_interval == 0:
#                 save_checkpoint(
#                     student_model, optimizer, global_step, checkpoint_dir, global_epoch)


            if global_step == 1 or global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    averaged_sync_loss, averaged_recon_loss, average_loss, average_CD_loss, average_TV_loss, average_SSIM_loss, average_style_loss, average_feature_loss = eval_model(test_data_loader, global_step, device, student_model, teacher_model, vgg, checkpoint_dir)
                    
                    val_sync_loss_curve.append(averaged_sync_loss)
                    val_loss_time_steps.append(global_step)
                    
                    val_recon_losses.append(averaged_recon_loss)
                    val_loss_liz.append(average_loss)
                    val_CD_loss_liz.append(average_CD_loss)
                    val_TV_loss_liz.append(average_TV_loss)
                    val_ssim_loss_liz.append(average_SSIM_loss)
                    val_style_loss_liz.append(average_style_loss)
                    val_feature_loss_liz.append(average_feature_loss)
                    

                    if averaged_sync_loss < .75:
                        hparams.set_hparam('syncnet_wt', 0.01) # without image GAN a lesser weight is sufficient
                        
            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    student_model, optimizer, global_step, checkpoint_dir, global_epoch)
                
                # print("train_l1loss:", train_l1loss_curve)
                # print("train_sync_loss:", train_sync_loss_curve)
                # print("train_loss:", train_loss_curve)
                # print("val_loss:", val_sync_loss_curve)
                # print()

                plt.figure()
                plt.plot(train_loss_time_steps, train_loss_curve, marker='+', label='Train Loss')
                plt.plot(train_loss_time_steps, train_sync_loss_curve, marker='o', label='Train sync_loss')
                plt.plot(val_loss_time_steps, val_sync_loss_curve, marker='s', label='Validation Loss')
                plt.title('Loss vs. Steps \n steps per epoch: %s' % len(train_data_loader))
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(checkpoint_dir+"/checkpoint_loss_graph_step{:09d}.png".format(global_step)) # SAVE THAT PLOT!!! 
                # plt.show()
                
                train_loss_data_dict = { # save last checkpoint window of data
                    'train_loss_time_steps': train_loss_time_steps[-checkpoint_interval:],
                    'train_loss_curve': train_loss_curve[-checkpoint_interval:],
                    'train_sync_loss_curve': train_sync_loss_curve[-checkpoint_interval:],
                    'train_l1loss_curve': train_l1loss_curve[-checkpoint_interval:],
                    'train_running_l1loss_curve': train_running_l1loss_curve[-checkpoint_interval:],
                    'train_running_sync_loss_curve': train_running_sync_loss_curve[-checkpoint_interval:],
                    'train_CD_loss': train_CD_loss[-checkpoint_interval:],
                    'train_TV_loss':train_TV_loss[-checkpoint_interval:],
                    'train_ssim_loss':train_ssim_loss[-checkpoint_interval:],
                    'train_style_loss':train_style_loss[-checkpoint_interval:],
                    'train_feature_loss':train_feature_loss[-checkpoint_interval:]
                }
                
                num_o_eval_pts = checkpoint_interval // hparams.eval_interval
                
                val_loss_data_dict = {
                    'val_loss_time_steps': val_loss_time_steps[-num_o_eval_pts:],
                    'val_sync_loss_curve': val_sync_loss_curve[-num_o_eval_pts:],
                    'val_recon_losses':val_recon_losses[-num_o_eval_pts:],
                    'val_loss_liz':val_loss_liz[-num_o_eval_pts:],
                    'val_CD_loss_liz':val_CD_loss_liz[-num_o_eval_pts:],
                    'val_TV_loss_liz':val_TV_loss_liz[-num_o_eval_pts:],
                    'val_ssim_loss_liz':val_ssim_loss_liz[-num_o_eval_pts:],
                    'val_style_loss_liz':val_style_loss_liz[-num_o_eval_pts:],
                    'val_feature_loss_liz':val_feature_loss_liz[-num_o_eval_pts:]
                }
                
                # print(loss_data_dict)
                
                train_loss_df = pd.DataFrame(train_loss_data_dict)
                val_loss_df = pd.DataFrame(val_loss_data_dict)
                

                # # Save the data to a JSON file
                # with open(checkpoint_dir+"/checkpoint_loss_graph_step{:09d}.json".format(global_step), 'w') as f:
                #     json.dump(loss_data_dict, f, indent=4)
                    
                    
                if not os.path.exists(checkpoint_dir+"/train_loss_data.csv"):
                    train_loss_df.to_csv(checkpoint_dir+"/train_loss_data.csv", index=False)
                else:
                    train_loss_df.to_csv(checkpoint_dir+"/train_loss_data.csv", mode='a', header=False, index=False) # mode='a' means append data!!! YAY!!
                    
                if not os.path.exists(checkpoint_dir+"/val_loss_data.csv"):
                    val_loss_df.to_csv(checkpoint_dir+"/val_loss_data.csv", index=False)
                else:
                    val_loss_df.to_csv(checkpoint_dir+"/val_loss_data.csv", mode='a', header=False, index=False) # mode='a' means append data!!! YAY!!
                
                    
                
#                 # Empty all loss liz. This way we only append new data! 
#                 train_l1loss_curve = []
#                 train_sync_loss_curve = []
#                 train_running_l1loss_curve = []
#                 train_running_sync_loss_curve = []
#                 train_loss_curve = []

#                 train_CD_loss = []
#                 train_TV_loss = []
#                 train_ssim_loss = []
#                 train_style_loss = []
#                 train_feature_loss = []

#                 train_loss_time_steps = [] # need to track what steps I record train_loss

#                 val_sync_loss_curve = []
#                 val_recon_losses, val_loss_liz, val_CD_loss_liz, val_TV_loss_liz, val_ssim_loss_liz, val_style_loss_liz, val_feature_loss_liz = [], [], [], [], [], [], []
#                 val_loss_time_steps = [] # need to track what steps I record val_loss



            prog_bar.set_description('train: L1: {}, Sync Loss: {}, CD Loss {}, TV Loss {}, SSIM Loss {}, Style Loss {}, Feature Loss {}'.format(running_l1_loss / (step + 1),
                                                                                                                                                 running_sync_loss / (step + 1),
                                                                                                                                                 CD_loss_report,
                                                                                                                                                 TV_loss_report,
                                                                                                                                                 SSIM_student_loss_report,
                                                                                                                                                 style_loss_report,
                                                                                                                                                 feature_loss_report
                                                                                                                ))

        global_epoch += 1
        

def eval_model(test_data_loader, global_step, device, student_model, teacher_model, vgg, checkpoint_dir):
    eval_steps = 700
    print('Evaluating for {} steps'.format(eval_steps))
    sync_losses, recon_losses, loss_liz, CD_loss_liz, TV_loss_liz, ssim_loss_liz, style_loss_liz, feature_loss_liz = [], [], [], [], [], [], [], []
    step = 0
    while 1:
        for x, indiv_mels, mel, gt in test_data_loader:
            step += 1
            student_model.eval()

            # Move data to CUDA device
            x = x.to(device)
            gt = gt.to(device)
            indiv_mels = indiv_mels.to(device)
            mel = mel.to(device)

            # g = student_model(indiv_mels, x) # generated student frames
            # g_teach = teacher_model(indiv_mels, x) # generated teacher frames
            g, student_kd_feats = student_model(indiv_mels, x) # generated student frames
            g_teach, teach_kd_feats = teacher_model(indiv_mels, x) # generated teacher frames

#             sync_loss = get_sync_loss(mel, g)
#             # l1loss = recon_loss(g, gt)
#             l1loss = recon_loss(g, g_teach)

#             sync_losses.append(sync_loss.item())
#             recon_losses.append(l1loss.item())
            
            
            #----------------------------------------------------
            # Get ALL the losses: 
            #----------------------------------------------------
            # sync_loss
            #---------------------
            sync_loss = get_sync_loss(mel, g)
            sync_losses.append(sync_loss.item())
            

            #---------------------
            # recon_loss or L1 loss
            #---------------------
            if hparams.L1_gt_wt != 0:
                l1loss = recon_loss(g, gt)
                L1_wt = hparams.L1_gt_wt
                recon_losses.append(l1loss.item())
            else:
                l1loss = recon_loss(g, g_teach)
                L1_wt = hparams.L1_teach_wt
                recon_losses.append(l1loss.item())

            
            #---------------------
            # CD_loss - intermediate channel distillation loss
            #---------------------  
            if hparams.cd_wt != 0:
                CD_loss = get_CD_loss(student_kd_feats, teach_kd_feats)
                # print()
                # print("CD_loss:", CD_loss) # CD_loss: tensor(2.0059, device='cuda:0', grad_fn=<AddBackward0>)
                CD_loss_liz.append(CD_loss.detach().cpu().item())
            else:
                CD_loss = 0.
            
            #---------------------
            # Total Variation loss - measures how diffrent pixels are from the one next to them in an image
            #---------------------
            if hparams.tv_wt != 0:
                TV_loss = get_TV_loss(g)
                # print("TV_loss:",TV_loss) # 'TV_loss': tensor(345843.0938, device='cuda:0', grad_fn=<AddBackward0>)
                TV_loss_liz.append(TV_loss.detach().cpu().item())
                # print("train_TV_loss:",train_TV_loss)
    
            else:
                TV_loss = 0.
                
                
            #---------------------
            # reshape g and g_teach for SSIM, style, and feature Loss
            #--------------------- 
            #       generated frames                                               (B, C, T, H, W)
            # print("student g.size():", g.size()) # student g.size(): torch.Size([16, 3, 5, 96, 96]) 
            # print("teach g_teach.size():", g_teach.size()) # teach g_teach.size(): torch.Size([16, 3, 5, 96, 96])
            if hparams.ssim_wt != 0 or hparams.feature_wt != 0 or hparams.style_wt != 0:
                # losses expects (N, C, H, W)
                B, C, T, H, W = g.shape
                g_4d = g.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
                gtech_4d = g_teach.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
                
                
            #---------------------
            # Structural Similarity (SSIM) Loss
            #---------------------     
            if hparams.ssim_wt != 0:
                ssim_loss = pytorch_ssim.SSIM()
                SSIM_student_loss = (1 - ssim_loss(g_4d, gtech_4d))
                ssim_loss_liz.append(SSIM_student_loss.detach().cpu().item())
                # print("SSIM_loss:", SSIM_student_loss.detach().cpu().item())
            else:
                SSIM_student_loss = 0
                
                
            #-------------------------------
            # VGG for feature and style loss
            #------------------------------- 
            if hparams.feature_wt != 0 or hparams.style_wt != 0:
                Tfeatures = vgg(gtech_4d)
                Sfeatures = vgg(g_4d)
                
                Tgram = [gram(fmap) for fmap in Tfeatures]
                Sgram = [gram(fmap) for fmap in Sfeatures]
                
                
                #--------------------
                # style loss
                #--------------------
                if hparams.style_wt != 0:
                    style_loss = 0
                    for i in range(len(Tgram)):
                        style_loss += F.l1_loss(Sgram[i], Tgram[i])
                    
                    style_loss_liz.append(style_loss.detach().cpu().item())
                    style_loss_report = style_loss.detach().cpu().item()
                else:
                    style_loss = 0
                    
                #--------------------
                # feature loss
                #--------------------
                if hparams.feature_wt != 0:
                    Srecon, Trecon = Sfeatures[1], Tfeatures[1]
                    feature_loss = F.l1_loss(Srecon, Trecon)
                    
                    feature_loss_liz.append(feature_loss.detach().cpu().item())
                    feature_loss_report = feature_loss.detach().cpu().item()
                else:
                    feature_loss = 0
            
            else:
                style_loss = 0
                feature_loss = 0

            #---------------------
            # Bring all losses together! 
            #---------------------
            # loss = hparams.syncnet_wt * sync_loss + (1 - hparams.syncnet_wt) * l1loss                 # OG loss eq
            loss = hparams.syncnet_wt * sync_loss + L1_wt*(1 - hparams.syncnet_wt) * l1loss + hparams.cd_wt*CD_loss + hparams.tv_wt*TV_loss + hparams.ssim_wt*SSIM_student_loss
            loss_liz.append(loss.detach().cpu().item())
            
            

            if step > eval_steps: 
                averaged_sync_loss = sum(sync_losses) / len(sync_losses)
                
                if hparams.L1_gt_wt!= 0 or hparams.L1_teach_wt!= 0:
                    averaged_recon_loss = sum(recon_losses) / len(recon_losses)
                else:
                    averaged_recon_loss = None
                
                average_loss = sum(loss_liz) / len(loss_liz)
                
                if len(CD_loss_liz) != 0:
                    average_CD_loss = sum(CD_loss_liz) / len(CD_loss_liz)
                else:
                    average_CD_loss = None
                
                if len(TV_loss_liz) != 0:
                    average_TV_loss = sum(TV_loss_liz) / len(TV_loss_liz)
                else:
                    average_TV_loss = None
                
                if len(ssim_loss_liz) != 0:
                    average_SSIM_loss = sum(ssim_loss_liz) / len(ssim_loss_liz)
                else:
                    average_SSIM_loss = None
                
                if len(style_loss_liz) != 0:
                    average_style_loss = sum(style_loss_liz) / len(style_loss_liz)
                else:
                    average_style_loss = None
                
                if len(feature_loss_liz) != 0:
                    average_feature_loss = sum(feature_loss_liz) / len(feature_loss_liz)
                else:
                    average_feature_loss = None

                # print('eval_model:')
                # print('test:  L1: {}, Sync loss: {}'.format(averaged_recon_loss, averaged_sync_loss))
                # print()
                print('Eval:  L1: {}, Sync loss: {}, CD Loss {}, TV Loss {}, SSIM Loss {}, Style Loss {}, Feature Loss {}'.format(averaged_recon_loss, 
                                                                                                                                  averaged_sync_loss, 
                                                                                                                                  average_CD_loss, 
                                                                                                                                  average_TV_loss, 
                                                                                                                                  average_SSIM_loss,
                                                                                                                                  average_style_loss,
                                                                                                                                  average_feature_loss))

                return averaged_sync_loss, averaged_recon_loss, average_loss, average_CD_loss, average_TV_loss, average_SSIM_loss, average_style_loss, average_feature_loss

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
#        checkpoint = torch.load(checkpoint_path,
#                                map_location=lambda storage, loc: storage)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    return checkpoint

#def _load(checkpoint_path):
#    if use_cuda:
#        map_loc = torch.device('cuda')
#    else:
#        map_loc = torch.device('cpu')
#    try:
#        # For TorchScript models (.pt, zip archive), use torch.jit.load
#        return torch.jit.load(checkpoint_path, map_location=map_loc)
#    except Exception:
#        # For regular PyTorch checkpoints, fallback
#        return torch.load(checkpoint_path, map_location=map_loc)

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir

    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers)
    
    steps_per_epoch = int(np.ceil(len(train_dataset) / hparams.batch_size))
    print("len(train_dataset):", len(train_dataset))
    print("Steps per epoch:", steps_per_epoch)
    
    # print out hparams
    print(hparams_debug_string(hparams))

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    teacher_model = Wav2Lip().to(device)
    student_model = Wav2Lip_student().to(device)
    vgg = VGGFeature().to(device)
    
    
    print('total Teacher trainable params {}'.format(sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)))
    print('total Student trainable params {}'.format(sum(p.numel() for p in student_model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in student_model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate)

    # load student model from a check point
    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, student_model, optimizer, reset_optimizer=False)
        
    # load syncnet model 
    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)
    
    # load pre‑trained teacher model 
    load_checkpoint(args.teacher_checkpoint_path, teacher_model, None, reset_optimizer=True, overwrite_global_states=False)
    
    # compile for speed - compile AFTER loading models
    # may want to comment this out when developing your code. It takes time up front to complie the code. 
    teacher_model = torch.compile(teacher_model)
    student_model = torch.compile(student_model)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        
    #save hparams file
    with open(checkpoint_dir+'/hparams.json', 'w') as f:
        json.dump(hparams.data, f, indent=4)

    # Train!
    train(device, student_model, teacher_model, vgg, train_data_loader, test_data_loader, optimizer,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs)
