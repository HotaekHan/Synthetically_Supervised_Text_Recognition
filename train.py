# python
import os
import argparse
import random
import numpy as np
import shutil
import cv2

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

# user-defined
from models.crnn import load_model
import dataset
import utils
from models.Synth_gan import ImageDiscriminator
from models.Synth_gan import FeatureDiscriminator
from models.Synth_gan import ImageGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path of config file')
opt = parser.parse_args()

config = utils.get_config(opt.config)
start_epoch = 0

# make output folder
if not os.path.exists(config['model']['exp_path']):
    os.mkdir(config['model']['exp_path'])

shutil.copy(opt.config, os.path.join(config['model']['exp_path'], 'config.yaml'))

# set random seed
random.seed(config['hyperparameters']['random_seed'])
np.random.seed(config['hyperparameters']['random_seed'])
torch.manual_seed(config['hyperparameters']['random_seed'])
torch.cuda.manual_seed(config['hyperparameters']['random_seed'])

if torch.cuda.is_available() and not config['cuda']['using_cuda']:
    print("WARNING: You have a CUDA device, so you should probably run with using cuda")

cuda_str = 'cuda:' + str(config['cuda']['gpu_id'])
device = torch.device(cuda_str if config['cuda']['using_cuda'] else "cpu")

# make dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

encoder = utils.LabelEncoder(labels=config['hyperparameters']['labels'],
                             is_ignore_case=config['hyperparameters']['is_ignore_case'])

img_width = config['hyperparameters']['imgWidth']
img_height = config['hyperparameters']['imgHeight']

train_dataset = dataset.listDataset(root_path=config['data']['train'], transform=transform, encoder=encoder,
                                    img_size=(img_width, img_height))
valid_dataset = dataset.listDataset(root_path=config['data']['valid'], transform=transform, encoder=encoder,
                                    img_size=(img_width, img_height))

assert train_dataset
assert valid_dataset

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config['hyperparameters']['batch_size'],
    shuffle=True, num_workers=0,
    collate_fn=train_dataset.collate_fn)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=config['hyperparameters']['batch_size'],
    shuffle=False, num_workers=0,
    collate_fn=valid_dataset.collate_fn)

# EOS
num_classes = len(encoder.labels) + 1
max_length_of_sequence = config['hyperparameters']['max_length']

# load model
crnn = load_model(num_classes=num_classes)
crnn = crnn.to(device)

generator_g = ImageGenerator()
generator_g = generator_g.to(device)
discriminator_ga = ImageDiscriminator()
discriminator_ga = discriminator_ga.to(device)
discriminator_fa = FeatureDiscriminator()
discriminator_fa = discriminator_fa.to(device)

# tensorboard
summary_writer = SummaryWriter(os.path.join(config['model']['exp_path'], 'log'))

# print networks
num_parameters = 0.
for param in crnn.parameters():
    sizes = param.size()

    num_layer_param = 1.
    for size in sizes:
        num_layer_param *= size
    num_parameters += num_layer_param

print(crnn)
print("num. of parameters : " + str(num_parameters))

seq_criterion = nn.CTCLoss(blank=0, reduction='sum')
recon_criterion = nn.L1Loss(reduction='sum')
feat_criterion = nn.MSELoss(reduction='sum')
adversarial_criterion = nn.BCELoss(reduction='sum')

# set optimizer for sequence(Encoder and Decoder)
if config['hyperparameters']['optimizer'] == 'Adam':
    optimizer_ED = optim.Adam(crnn.parameters(), lr=float(config['hyperparameters']['lr']), betas=(0.5, 0.999))
    # generator optimizer
    optimizer_G = optim.Adam(generator_g.parameters(), lr=float(config['hyperparameters']['lr']), betas=(0.5, 0.999))
    # discriminator optimizer
    optimizer_D_ga = optim.Adam(discriminator_ga.parameters(), lr=float(config['hyperparameters']['lr']), betas=(0.5, 0.999))
    optimizer_D_fa = optim.Adam(discriminator_fa.parameters(), lr=float(config['hyperparameters']['lr']), betas=(0.5, 0.999))
elif config['hyperparameters']['optimizer'] == 'Adadelta':
    optimizer_ED = optim.Adadelta(crnn.parameters(), lr=float(config['hyperparameters']['lr']))
    # generator optimizer
    optimizer_G = optim.Adadelta(generator_g.parameters(), lr=float(config['hyperparameters']['lr']))
    # discriminator optimizer
    optimizer_D_ga = optim.Adadelta(discriminator_ga.parameters(), lr=float(config['hyperparameters']['lr']))
    optimizer_D_fa = optim.Adadelta(discriminator_fa.parameters(), lr=float(config['hyperparameters']['lr']))
else:
    raise AssertionError("not supported optimizer")

lambda_seq = config['hyperparameters']['lambda_seq']
lambda_feat_match = config['hyperparameters']['lambda_feat_match']
lambda_recon = config['hyperparameters']['lambda_recon']
lambda_ga = config['hyperparameters']['lambda_ga']
lambda_fa = config['hyperparameters']['lambda_fa']

best_valid_loss = float('inf')

global_iter_train = 0
global_iter_valid = 0

# set lr scheduler
if config['hyperparameters']['lr_patience'] > 0:
    scheduler_for_lr = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_ED, mode='min', factor=0.1,
                                                      patience=config['hyperparameters']['lr_patience'], verbose=True)
else:
    scheduler_for_lr = None

if config['hyperparameters']['lr_multistep'] != 'None':
    milestones = config['hyperparameters']['lr_multistep']
    milestones = milestones.split(', ')
    for iter_milestone in range(len(milestones)):
        milestones[iter_milestone] = int(milestones[iter_milestone])
    scheduler_for_lr = lr_scheduler.MultiStepLR(optimizer=optimizer_ED, milestones=milestones, gamma=0.1)
else:
    scheduler_for_lr = None

# set pre-trained
if config['model']['model_path'] != 'None':
    print('loading pretrained model from %s' % config['model']['model_path'])
    ckpt = torch.load(config['model']['model_path'], map_location=device)
    crnn.load_state_dict(ckpt['crnn'])
    generator_g.load_state_dict(ckpt['generator'])
    discriminator_fa.load_state_dict(ckpt['discriminator_fa'])
    discriminator_ga.load_state_dict(ckpt['discriminator_ga'])
    start_epoch = ckpt['epoch'] + 1
    if config['model']['is_finetune'] is False:
        best_valid_loss = ckpt['loss']
        global_iter_train = ckpt['global_train_iter']
        global_iter_valid = ckpt['global_valid_iter']
    else:
        start_epoch = 0

print("optimizer : " + str(optimizer_ED))
if scheduler_for_lr is None:
    print("lr_scheduler : None")
elif config['hyperparameters']['lr_patience'] > 0:
    print("lr_scheduler : [patience: " + str(scheduler_for_lr.patience) +
          ", gamma: " + str(scheduler_for_lr.factor) +"]")
elif config['hyperparameters']['lr_multistep'] != 'None':
    tmp_str = "lr_scheduler : [milestones: "
    for milestone in scheduler_for_lr.milestones:
        tmp_str = tmp_str + str(milestone) + ', '
    tmp_str += ' gamma: '
    tmp_str += str(scheduler_for_lr.gamma)
    tmp_str += ']'
    print(tmp_str)
print("Size of batch : " + str(train_loader.batch_size))
print("transform : " + str(transform))
print("num. train data : " + str(len(train_dataset)))
print("num. valid data : " + str(len(valid_dataset)))
print("num_classes : " + str(num_classes))
print("classes : " + str(encoder.dict))

utils.print_config(config)
input("Press any key to continue..")


def train(epoch):
    crnn.train()
    generator_g.train()
    discriminator_ga.train()
    discriminator_fa.train()

    train_loss = 0
    global global_iter_train

    batch_size = train_loader.batch_size

    with torch.set_grad_enabled(True):
        for batch_idx, (inputs, targets, synths, lengths, imgpaths) in enumerate(train_loader):
            inputs = inputs.to(device)
            # targets = targets.to(device)
            synths = synths.to(device)

            # init optimizers
            optimizer_ED.zero_grad()
            optimizer_G.zero_grad()

            # Sequence loss
            preds = crnn(inputs)

            preds_steps = torch.tensor([preds.size(0)] * preds.size(1), dtype=torch.int32)
            seq_loss = seq_criterion(preds, targets, preds_steps, lengths)
            seq_loss = lambda_seq * seq_loss

            # variables
            encoded_synth = crnn.encoder(synths)
            encoded_real = crnn.encoder(inputs)
            generated_synth = generator_g(encoded_real)

            if batch_idx == 0:
                np_synth = generated_synth.detach().cpu().numpy()
                np_input = inputs.detach().cpu().numpy()
                for iter_batch in range(np_synth.shape[0]):
                    npp = np_synth[iter_batch, :, :, :]
                    npp = (npp + 1) * (255. / 2.)
                    npp = np.transpose(npp, (1, 2, 0))
                    npp = npp.astype(np.uint8)
                    file_name = 'tmp_img_train/' + str(iter_batch) + '.png'
                    cv2.imwrite(file_name, npp)

                    npp = np_input[iter_batch, :, :, :]
                    npp = (npp + 1) * (255. / 2.)
                    npp = np.transpose(npp, (1, 2, 0))
                    npp = npp.astype(np.uint8)
                    file_name = 'tmp_img_train/' + str(iter_batch) + '.jpg'
                    cv2.imwrite(file_name, npp)

            # Feature match loss
            feat_match_loss = feat_criterion(encoded_synth, encoded_real)
            feat_match_loss = lambda_feat_match * feat_match_loss

            # Reconstruction loss

            recon_loss = recon_criterion(generated_synth, synths)
            recon_loss = recon_loss / batch_size
            recon_loss = lambda_recon * recon_loss

            # adversarial learning
            valid_label = torch.full((inputs.shape[0],), 1, device=device)
            fake_label = torch.full((inputs.shape[0],), 0, device=device)

            # update discriminator(feature)
            real_feat_dis_loss = adversarial_criterion(discriminator_fa(encoded_real.detach()), valid_label)
            fake_feat_dis_loss = adversarial_criterion(discriminator_fa(encoded_synth.detach()), fake_label)
            feat_dis_loss = (real_feat_dis_loss + fake_feat_dis_loss) / 2
            feat_dis_loss = lambda_fa * feat_dis_loss

            # update encoder(feature)
            real_feat_enc_loss = adversarial_criterion(discriminator_fa(encoded_real), fake_label)
            fake_feat_enc_loss = adversarial_criterion(discriminator_fa(encoded_synth), valid_label)
            feat_enc_loss = (real_feat_enc_loss + fake_feat_enc_loss) / 2
            feat_enc_loss = lambda_fa * feat_enc_loss
            feat_enc_loss = feat_enc_loss / batch_size
            feat_enc_loss.backward(retain_graph=True)
            optimizer_D_fa.zero_grad()

            real_pair = torch.cat([inputs, synths], 1)
            fake_pair = torch.cat([inputs, generated_synth], 1)

            # update discriminator(image)
            real_img_dis_loss = adversarial_criterion(discriminator_ga(real_pair.detach()), valid_label)
            fake_img_dis_loss = adversarial_criterion(discriminator_ga(fake_pair.detach()), fake_label)
            img_dis_loss = (real_img_dis_loss + fake_img_dis_loss) / 2
            img_dis_loss = lambda_ga * img_dis_loss

            # update encoder&generator(image)
            real_img_enc_loss = adversarial_criterion(discriminator_ga(real_pair), fake_label)
            fake_img_enc_loss = adversarial_criterion(discriminator_ga(fake_pair), valid_label)
            img_enc_loss = (real_img_enc_loss + fake_img_enc_loss) / 2
            img_enc_loss = lambda_ga * img_enc_loss
            img_enc_loss = img_enc_loss / batch_size
            img_enc_loss.backward(retain_graph=True)
            optimizer_D_ga.zero_grad()

            total_loss = seq_loss + feat_match_loss + recon_loss + feat_dis_loss + img_dis_loss
            total_loss = total_loss / batch_size
            total_loss.backward()

            train_loss += total_loss.item()
            avg_train_loss = train_loss / (batch_idx + 1)

            # update all losses
            optimizer_ED.step()
            optimizer_G.step()
            optimizer_D_ga.step()
            optimizer_D_fa.step()

            log_str ='[Train] epoch: %3d | iter: %4d | train_loss: %.4f | seq: %.4f | ' \
                     'recon: %.4f | feat_match: %.4f | ' \
                     'dis(feat): %.4f | enc(feat): %.4f | dis(img): %.4f | enc,gen(img): %.4f | avg_loss: %.4f' \
                     % (epoch, batch_idx, total_loss.item(), seq_loss.item(),
                        recon_loss.item(), feat_match_loss.item(), feat_dis_loss.item(), feat_enc_loss.item(),
                        img_dis_loss.item(), img_enc_loss.item(), avg_train_loss)
            print(log_str)

            # if batch_idx % config['hyperparameters']['interval_store_grad'] == 0:
            #     for name, param in generator_g.named_parameters():
            #         summary_writer.add_histogram(name, param.grad.detach().cpu().data.numpy(), global_iter_train)
            #     for name, param in crnn.named_parameters():
            #         summary_writer.add_histogram(name, param.grad.detach().cpu().data.numpy(), global_iter_train)

            summary_writer.add_scalar('train_loss/seq_loss', seq_loss.item(), global_iter_train)
            summary_writer.add_scalar('train_loss/feat_match_loss', feat_match_loss.item(), global_iter_train)
            summary_writer.add_scalar('train_loss/recon_loss', recon_loss.item(), global_iter_train)
            summary_writer.add_scalar('train_loss/feat_dis_loss', feat_dis_loss.item(), global_iter_train)
            summary_writer.add_scalar('train_loss/feat_enc_loss', feat_enc_loss.item(), global_iter_train)
            summary_writer.add_scalar('train_loss/img_dis_loss', img_dis_loss.item(), global_iter_train)
            summary_writer.add_scalar('train_loss/img_enc_loss', img_enc_loss.item(), global_iter_train)
            summary_writer.add_scalar('train_loss/total_loss', total_loss.item(), global_iter_train)
            global_iter_train += 1

            if config['hyperparameters']['lr_multistep'] != 'None':
                scheduler_for_lr.step()


def valid(epoch):
    crnn.eval()
    generator_g.eval()
    discriminator_ga.eval()
    discriminator_fa.eval()

    avg_valid_loss = 0
    valid_loss = 0
    global best_valid_loss
    is_saved = False
    global global_iter_valid
    global global_iter_train

    batch_size = valid_loader.batch_size

    with torch.set_grad_enabled(False):
        for batch_idx, (inputs, targets, synths, lengths, imgpaths) in enumerate(valid_loader):
            inputs = inputs.to(device)
            # targets = targets.to(device)
            synths = synths.to(device)

            # Sequence loss
            preds = crnn(inputs)

            preds_steps = torch.tensor([preds.size(0)] * preds.size(1), dtype=torch.int32)
            seq_loss = seq_criterion(preds, targets, preds_steps, lengths)
            seq_loss = lambda_seq * seq_loss

            # variables
            encoded_synth = crnn.encoder(synths)
            encoded_real = crnn.encoder(inputs)
            generated_synth = generator_g(encoded_real)

            if batch_idx == 0:
                np_synth = generated_synth.detach().cpu().numpy()
                np_input = inputs.detach().cpu().numpy()
                for iter_batch in range(np_synth.shape[0]):
                    npp = np_synth[iter_batch, :, :, :]
                    npp = (npp + 1) * (255. / 2.)
                    npp = np.transpose(npp, (1, 2, 0))
                    npp = npp.astype(np.uint8)
                    file_name = 'tmp_img_val/' + str(iter_batch) + '.png'
                    cv2.imwrite(file_name, npp)

                    npp = np_input[iter_batch, :, :, :]
                    npp = (npp + 1) * (255. / 2.)
                    npp = np.transpose(npp, (1, 2, 0))
                    npp = npp.astype(np.uint8)
                    file_name = 'tmp_img_val/' + str(iter_batch) + '.jpg'
                    cv2.imwrite(file_name, npp)

            # Feature match loss
            feat_match_loss = feat_criterion(encoded_synth, encoded_real)
            feat_match_loss = lambda_feat_match * feat_match_loss

            # Reconstruction loss

            recon_loss = recon_criterion(generated_synth, synths)
            recon_loss = recon_loss / batch_size
            recon_loss = lambda_recon * recon_loss

            # adversarial learning
            valid_label = torch.full((inputs.shape[0],), 1, device=device)
            fake_label = torch.full((inputs.shape[0],), 0, device=device)

            # update discriminator(feature)
            real_feat_dis_loss = adversarial_criterion(discriminator_fa(encoded_real.detach()), valid_label)
            fake_feat_dis_loss = adversarial_criterion(discriminator_fa(encoded_synth.detach()), fake_label)
            feat_dis_loss = (real_feat_dis_loss + fake_feat_dis_loss) / 2
            feat_dis_loss = lambda_fa * feat_dis_loss

            # update encoder(feature)
            real_feat_enc_loss = adversarial_criterion(discriminator_fa(encoded_real), fake_label)
            fake_feat_enc_loss = adversarial_criterion(discriminator_fa(encoded_synth), valid_label)
            feat_enc_loss = (real_feat_enc_loss + fake_feat_enc_loss) / 2
            feat_enc_loss = lambda_fa * feat_enc_loss

            # update discriminator(image)
            real_pair = torch.cat([inputs, synths], 1)
            fake_pair = torch.cat([inputs, generated_synth], 1)

            real_img_dis_loss = adversarial_criterion(discriminator_ga(real_pair.detach()), valid_label)
            fake_img_dis_loss = adversarial_criterion(discriminator_ga(fake_pair.detach()), fake_label)
            img_dis_loss = (real_img_dis_loss + fake_img_dis_loss) / 2
            img_dis_loss = lambda_ga * img_dis_loss

            # update encoder&generator(image)
            real_img_enc_loss = adversarial_criterion(discriminator_ga(real_pair), fake_label)
            fake_img_enc_loss = adversarial_criterion(discriminator_ga(fake_pair), valid_label)
            img_enc_loss = (real_img_enc_loss + fake_img_enc_loss) / 2
            img_enc_loss = lambda_ga * img_enc_loss

            total_loss = seq_loss + feat_match_loss + recon_loss + \
                         feat_dis_loss + feat_enc_loss + img_dis_loss + img_enc_loss
            total_loss = total_loss / batch_size

            valid_loss += total_loss.item()
            avg_valid_loss = valid_loss / (batch_idx + 1)

            log_str ='[Valid] epoch: %3d | iter: %4d | valid_loss: %.4f | seq: %.4f | ' \
                     'recon: %.4f | feat_match: %.4f | ' \
                     'dis(feat): %.4f | enc(feat): %.4f | dis(img): %.4f | enc,gen(img): %.4f | avg_loss: %.4f' \
                     % (epoch, batch_idx, total_loss.item(), seq_loss.item(),
                        recon_loss.item(), feat_match_loss.item(), feat_dis_loss.item(), feat_enc_loss.item(),
                        img_dis_loss.item(), img_enc_loss.item(), avg_valid_loss)
            print(log_str)

            summary_writer.add_scalar('valid_loss/seq_loss', seq_loss.item(), global_iter_valid)
            summary_writer.add_scalar('valid_loss/feat_match_loss', feat_match_loss.item(), global_iter_valid)
            summary_writer.add_scalar('valid_loss/recon_loss', recon_loss.item(), global_iter_valid)
            summary_writer.add_scalar('valid_loss/feat_dis_loss', feat_dis_loss.item(), global_iter_valid)
            summary_writer.add_scalar('valid_loss/feat_enc_loss', feat_enc_loss.item(), global_iter_valid)
            summary_writer.add_scalar('valid_loss/img_dis_loss', img_dis_loss.item(), global_iter_valid)
            summary_writer.add_scalar('valid_loss/img_enc_loss', img_enc_loss.item(), global_iter_valid)
            summary_writer.add_scalar('valid_loss/total_loss', total_loss.item(), global_iter_valid)
            global_iter_valid += 1

    print('[Valid] avg. valid loss: ' + str(avg_valid_loss))

    # lr scheduler
    if config['hyperparameters']['lr_patience'] > 0:
        scheduler_for_lr.step(avg_valid_loss)

    # check whether better model or not
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        is_saved = True

    if is_saved is True:
        print('Saving..')
        state = {
            'crnn': crnn.state_dict(),
            'generator': generator_g.state_dict(),
            'discriminator_fa': discriminator_fa.state_dict(),
            'discriminator_ga': discriminator_ga.state_dict(),
            'loss': best_valid_loss,
            'epoch': epoch,
            'lr': config['hyperparameters']['lr'],
            'batch': config['hyperparameters']['batch_size'],
            'labels': config['hyperparameters']['labels'],
            'global_train_iter': global_iter_train,
            'global_valid_iter': global_iter_valid
        }
        torch.save(state, config['model']['exp_path'] + '/ckpt-' + str(epoch) + '.pth')


for iter_epoch in range(start_epoch, config['hyperparameters']['epoch'], 1):
    train(iter_epoch)
    valid(iter_epoch)
summary_writer.close()

print("best valid loss : " + str(best_valid_loss))
