# %load_ext autoreload
# %autoreload 2
import config

import os
import joblib
import sys
import pickle

from tqdm import tqdm

from braindecode.datautil.iterators import get_balanced_batches
from eeggan.examples.conv_lin.augmented_model import Generator, Discriminator
from eeggan.util import weight_filler
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import wandb

matplotlib.use('TKAgg')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

jobid = 0
input_length = 768

n_critic = 1
n_batch = 2648 * 8
n_z = 16
lr = 0.05
n_blocks = 6
rampup = 100.
block_epochs = [150, 100, 200, 200, 400, 800]
# block_epochs = list(np.array(block_epochs)//100)
# block_epochs = [1,1,1,1,1,1]
subj_ind = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
task_ind = 0  # subj_ind

# Params used in GAN before mapping network extension
# subj_ind = 9
# n_critic = 5
# n_batch = 2048 * 2
# n_z = 16
# lr = 0.01
# n_blocks = 6
# rampup = 2000.
# # block_epochs = [500,1000,1000,1000,4000,4000]
# block_epochs = [500,300,300,500,500,500]
# subj_ind = int(os.getenv('SLURM_ARRAY_TASK_ID','0'))
# task_ind = 0#subj_ind
# #subj_ind = 9

subj_names = ['BhNoMoSc1',
              'FaMaMoSc1',
              'FrThMoSc1',
              'GuJoMoSc01',
              'KaUsMoSc1',
              'LaKaMoSc1',
              'LuFiMoSc3',
              'MaJaMoSc1',
              'MaKiMoSC01',
              'MaVoMoSc1',
              'PiWiMoSc1',
              'RoBeMoSc03',
              'RoScMoSc1',
              'StHeMoSc01']

np.random.seed(task_ind)
torch.manual_seed(task_ind)
torch.cuda.manual_seed_all(task_ind)
random.seed(task_ind)
rng = np.random.RandomState(task_ind)

datapath = config.compiled_data_path
if not os.path.exists(datapath):
    from eeggan.dataset.dataset import EEGDataClass

    dc = EEGDataClass(config.dataset_path)

    train = np.vstack([e[0] for e in dc.events])
    target = np.ones(train.shape[0]).astype(int)
    data_tuple = (train, target)
    pickle.dump(data_tuple, open(datapath, 'wb'))

train, target = pickle.load(open(datapath, 'rb'))

train = train[:, None, :, None].astype(np.float32)

train = train - train.mean()
train = train / train.std()

train_quantile = np.percentile(np.abs(train), 98)
train = train[(np.abs(train) < train_quantile)[:,0,:,0].all(axis = 1),:,:,:]
train = train/(train_quantile + 1e-8)

target_onehot = np.zeros((target.shape[0], 2))
target_onehot[:, target] = 1

modelpath = config.model_path

modelname = 'Progressive%s'
if not os.path.exists(modelpath):
    os.makedirs(modelpath)

generator = Generator(1, n_z)
discriminator = Discriminator(1)

generator.train_init(alpha=lr, betas=(0., 0.99))
discriminator.train_init(alpha=lr, betas=(0., 0.99), eps_center=0.001,
                         one_sided_penalty=True, distance_weighting=True)
generator = generator.apply(weight_filler)
discriminator = discriminator.apply(weight_filler)

i_block_tmp = 0
i_epoch_tmp = 0
generator.model.cur_block = i_block_tmp
discriminator.model.cur_block = n_blocks - 1 - i_block_tmp
fade_alpha = 1.
generator.model.alpha = fade_alpha
discriminator.model.alpha = fade_alpha
print("Size of the training set:",train.shape)
# plt.plot(train[1000,0,:,0])
# plt.show()

generator = generator.cuda()
discriminator = discriminator.cuda()
generator.train()
discriminator.train()

losses_d = []
losses_g = []
i_epoch = 0
z_vars_im = rng.normal(0, 1, size=(n_batch, n_z)).astype(np.float32)

# wandb.init(project="EEG_GAN", entity="hubertp")
# wandb.watch(generator, log_freq=5)

for i_block in range(i_block_tmp, n_blocks):
    print("-----------------")
    c = 0

    train_tmp = discriminator.model.downsample_to_block(
        Variable(torch.from_numpy(train).cuda(), volatile=True),
        discriminator.model.cur_block
    ).data.cpu()

    for i_epoch in tqdm(range(i_epoch_tmp, block_epochs[i_block])):
        i_epoch_tmp = 0

        if fade_alpha < 1:
            fade_alpha += 1. / rampup
            generator.model.alpha = fade_alpha
            discriminator.model.alpha = fade_alpha

        batches = get_balanced_batches(train.shape[0], rng, True, batch_size=n_batch)
        iters = max(int(len(batches) / n_critic), 1)

        for it in range(iters):
            #critic training
            for i_critic in range(n_critic):
                try:
                    train_batches = train_tmp[batches[it * n_critic + i_critic]]
                except IndexError:
                    continue
                batch_real = Variable(train_batches, requires_grad=True).cuda()

                z_vars = rng.normal(0, 1, size=(len(batches[it * n_critic + i_critic]), n_z)).astype(np.float32)
                z_vars = Variable(torch.from_numpy(z_vars), volatile=True).cuda()

                output = generator(z_vars)
                batch_fake = Variable(output.data, requires_grad=True).cuda()

                loss_d = discriminator.train_batch(batch_real, batch_fake)

                # assert np.all(np.isfinite(loss_d))


            z_vars = rng.normal(0, 1, size=(n_batch, n_z)).astype(np.float32)
            z_vars = Variable(torch.from_numpy(z_vars), requires_grad=True).cuda()
            loss_g = generator.train_batch(z_vars, discriminator)

        losses_d.append(loss_d)
        losses_g.append(loss_g)

        # if wandb:
        #     wandb.log(
        #         {
        #             "Learning_rate": lr,
        #             "Loss_F": loss_d[0],
        #             "Loss_R": loss_d[1],
        #             "Penalty": loss_d[2],
        #             "Generator Loss": loss_g
        #         }
        #     )

        if i_epoch % 10 == 0:
            generator.eval()
            discriminator.eval()

            print('Epoch: %d   Loss_F: %.3f   Loss_R: %.3f   Penalty: %.4f   Loss_G: %.3f' % (
            i_epoch, loss_d[0], loss_d[1], loss_d[2], loss_g))
            joblib.dump((i_epoch, losses_d, losses_g), os.path.join(modelpath, modelname % jobid + '_.data'),
                        compress=True)
            joblib.dump((i_epoch, losses_d, losses_g),
                        os.path.join(modelpath, modelname % jobid + '_%d.data' % i_epoch), compress=True)
            # joblib.dump((n_epochs,n_z,n_critic,batch_size,lr),os.path.join(modelpath,modelname%jobid+'_%d.params'%i_epoch),compress=True)

            freqs_tmp = np.fft.rfftfreq(train_tmp.numpy().shape[2], d=1 / (250. / np.power(2, n_blocks - 1 - i_block)))

            train_fft = np.fft.rfft(train_tmp.numpy(), axis=2)
            train_amps = np.abs(train_fft).mean(axis=3).mean(axis=0).squeeze()

            z_vars = Variable(torch.from_numpy(z_vars_im), volatile=True).cuda()
            batch_fake = generator(z_vars)

            fake_fft = np.fft.rfft(batch_fake.data.cpu().numpy(), axis=2)
            fake_amps = np.abs(fake_fft).mean(axis=3).mean(axis=0).squeeze()

            plt.figure()
            plt.plot(freqs_tmp, np.log(fake_amps), label='Fake')
            plt.plot(freqs_tmp, np.log(train_amps), label='Real')
            plt.title(f'Frequency Spektrum, block {i_block}')
            plt.xlabel('Hz')
            plt.legend()
            plt.savefig(os.path.join(modelpath, modelname % jobid + '_fft_%d_%d.png' % (i_block, i_epoch)))
            plt.close()

            batch_fake = batch_fake.data.cpu().numpy()
            plt.figure(figsize=(10, 10))
            plt.title(f'Fake samples, block {i_block}')
            for i in range(10):
                plt.subplot(10, 1, i + 1)
                plt.plot(batch_fake[i].squeeze())
                plt.xticks((), ())
                plt.yticks((), ())
            plt.subplots_adjust(hspace=0)
            plt.savefig(os.path.join(modelpath, modelname % jobid + '_fakes_%d_%d.png' % (i_block, i_epoch)))
            plt.close()

            discriminator.save_model(os.path.join(modelpath, modelname % jobid + '.disc'))
            generator.save_model(os.path.join(modelpath, modelname % jobid + '.gen'))

            # plt.figure(figsize=(10,15))
            # plt.subplot(3,2,1)
            # plt.plot(np.asarray(losses_d)[:,0],label='Loss Real')
            # plt.plot(np.asarray(losses_d)[:,1],label='Loss Fake')
            # plt.title('Losses Discriminator')
            # plt.legend()
            # plt.subplot(3,2,2)
            # plt.plot(np.asarray(losses_d)[:,0]+np.asarray(losses_d)[:,1]+np.asarray(losses_d)[:,2],label='Loss')
            # plt.title('Loss Discriminator')
            # plt.legend()
            # plt.subplot(3,2,3)
            # plt.plot(np.asarray(losses_d)[:,2],label='Penalty Loss')
            # plt.title('Penalty')
            # plt.legend()
            # plt.subplot(3,2,4)
            # plt.plot(-np.asarray(losses_d)[:,0]-np.asarray(losses_d)[:,1],label='Wasserstein Distance')
            # plt.title('Wasserstein Distance')
            # plt.legend()
            # plt.subplot(3,2,5)
            # plt.plot(np.asarray(losses_g),label='Loss Generator')
            # plt.title('Loss Generator')
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig(os.path.join(modelpath,modelname%jobid+'_losses.png'))
            # plt.close()

            generator.train()
            discriminator.train()
        lr /= 1.05
        lr = max(lr, 0.001)

    fade_alpha = 0.
    generator.model.cur_block += 1
    discriminator.model.cur_block -= 1
    lr = 0.01

    n_critic+=1
    if i_block in [0,1,2]:
        n_batch //= 3
    if i_block in [3]:
        n_batch = 20
    print(n_batch)
