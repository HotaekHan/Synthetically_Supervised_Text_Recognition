import argparse
import os
import sys
import cv2
import numpy as np

# pytorch
import torch
import torchvision.transforms as transforms
import torch.utils.data

# user-defined
from models.crnn import load_model
import dataset
import utils
from models.Synth_gan import ImageGenerator


def get_performace(dataset_name, data_set, correct_dir, incorrect_dir):
    confusion_mat = dict()

    acc = 0.0
    mean_edit_distance = 0

    # make outputs
    correct_out_path = os.path.join(correct_dir, dataset_name)
    incorrect_out_path = os.path.join(incorrect_dir, dataset_name)
    if not os.path.exists(correct_out_path):
        os.mkdir(correct_out_path)
    if not os.path.exists(incorrect_out_path):
        os.mkdir(incorrect_out_path)

    correct_results_out = open(os.path.join(correct_out_path, dataset_name + '.results'), 'w')
    incorrect_results_out = open(os.path.join(incorrect_out_path, dataset_name + '.results'), 'w')

    file_idx = 0
    num_processed_data = 0

    data_loader = torch.utils.data.DataLoader(
        data_set, batch_size=128,
        shuffle=False, num_workers=0,
        collate_fn=data_set.collate_fn)

    with torch.set_grad_enabled(False):
        for batch_idx, (inputs, targets, synths, lengths, imgpaths) in enumerate(data_loader):
            num_processed_data += inputs.size(0)
            sys.stdout.write('\r' + str(dataset_name) + ': ' + str(num_processed_data) + '/' + str(len(data_set)))

            device_inputs = inputs.to(device)
            preds = crnn(device_inputs)
            preds_steps = torch.tensor([preds.size(0)] * preds.size(1), dtype=torch.int32)

            values, indices = preds.max(2)
            indices = indices.transpose(1, 0).contiguous().view(-1)

            blank_targets = torch.empty(targets.size(0) * 2, dtype=torch.int32)
            for idx in range(targets.size(0)):
                blank_targets[2 * idx] = targets[idx]
                blank_targets[2 * idx + 1] = 0

            for idx in range(lengths.size(0)):
                step = lengths[idx].item()
                lengths[idx] = step * 2
            target_texts = encoder.decode(blank_targets, lengths)
            pred_texts = encoder.decode(indices, preds_steps)

            pred_synths = generator_g(crnn.encoder(device_inputs))
            pred_synths = pred_synths.cpu()

            for idx in range(len(target_texts)):
                # classification
                lower_pred_text = pred_texts[idx].lower()
                lower_target_text = target_texts[idx].lower()

                synth = pred_synths[idx]
                synth = synth.numpy()
                synth = synth * 255
                synth = np.transpose(synth, (1, 2, 0))
                synth = synth.astype(np.uint8)

                if lower_pred_text == lower_target_text:
                    acc += 1.0

                    # Each dataset
                    str_idx = utils.idx_to_str(file_idx)
                    tmp_str = os.path.join(correct_out_path, str_idx + '.png')
                    tmp_img = inputs[idx].numpy()
                    tmp_img = tmp_img * 255
                    tmp_img = np.transpose(tmp_img, (1, 2, 0))
                    cv2.imwrite(tmp_str, tmp_img)

                    tmp_str = os.path.join(correct_out_path, str_idx + '.jpg')
                    cv2.imwrite(tmp_str, synth)

                    correct_results_out.write(str_idx + '\t' + imgpaths[idx] + '\t'
                                      + target_texts[idx] + '\t' + pred_texts[idx] + '\n')
                else:
                    edit_distance = utils.levenshtein(lower_pred_text, lower_target_text)
                    mean_edit_distance += edit_distance

                    # Each dataset
                    str_idx = utils.idx_to_str(file_idx)
                    tmp_str = os.path.join(incorrect_out_path, str_idx + '.png')
                    tmp_img = inputs[idx].numpy()
                    tmp_img = tmp_img * 255
                    tmp_img = np.transpose(tmp_img, (1, 2, 0))
                    cv2.imwrite(tmp_str, tmp_img)

                    tmp_str = os.path.join(incorrect_out_path, str_idx + '.jpg')
                    cv2.imwrite(tmp_str, synth)

                    incorrect_results_out.write(str_idx + '\t' + imgpaths[idx] + '\t'
                                                + target_texts[idx] + '\t' + pred_texts[idx] + '\n')

                file_idx += 1

            confusion_mat = utils.get_confusion_matrix(preds=pred_texts, targets=target_texts,
                                                       confusion_dict=confusion_mat)

    correct_results_out.close()
    incorrect_results_out.close()
    acc /= float(num_processed_data)
    mean_edit_distance /= float(num_processed_data)
    print("")
    print("num. of data: " + str(num_processed_data))

    return acc, mean_edit_distance, confusion_mat


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path of config file')
opt = parser.parse_args()

# get config
config = utils.get_config(opt.config)

if torch.cuda.is_available() and not config['cuda']['using_cuda']:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# set device
cuda_str = 'cuda:' + str(config['cuda']['gpu_id'])
device = torch.device(cuda_str if config['cuda']['using_cuda'] else "cpu")

# get best model
best_model_path = utils.get_best_model(config['model']['exp_path'])

if best_model_path is None:
    raise FileNotFoundError('Not found ckpt file')

# set output path
ckpt_filename = best_model_path.split("/")[-1]
result_dir = best_model_path.replace(ckpt_filename, 'results')
correct_dir = os.path.join(result_dir, 'correct')
incorrect_dir = os.path.join(result_dir, 'incorrect')

if not os.path.exists(result_dir):
    os.mkdir(result_dir)
    os.mkdir(correct_dir)
    os.mkdir(incorrect_dir)

# load best checkpoint
print('loading pretrained model from %s' % best_model_path)
ckpt = torch.load(best_model_path, map_location=cuda_str)
labels = ckpt['labels']

# EOS, SOS
num_classes = len(labels) + 1
max_length_of_sequence = config['hyperparameters']['max_length']

# load model
crnn = load_model(num_classes=num_classes)

num_parameters = 0.
for param in crnn.parameters():
    sizes = param.size()

    num_layer_param = 1.
    for size in sizes:
        num_layer_param *= size
    num_parameters += num_layer_param
crnn.load_state_dict(ckpt['crnn'])
crnn = crnn.to(device)
crnn.eval()
print(crnn)
print("num. of parameters : " + str(num_parameters))

generator_g = ImageGenerator()
generator_g.load_state_dict(ckpt['generator'])
generator_g = generator_g.to(device)
generator_g.eval()

# set the datasets
transform = transforms.Compose([
    transforms.ToTensor()
])
encoder = utils.LabelEncoder(labels=labels,
                             is_ignore_case=config['hyperparameters']['is_ignore_case'])

img_width = config['hyperparameters']['imgWidth']
img_height = config['hyperparameters']['imgHeight']

# filter the dataset would be tested.
config_data = config['data']
dataset_path = dict()
for dataset_name in config_data:
    path = config_data[dataset_name]
    if path != 'None':
        split_path = path.split(' ')
        if len(split_path) == 1:
            dataset_path[dataset_name] = path

# get performance for each filtered dataset
dataset_acc = dict()
for dataset_name in dataset_path:
    path = dataset_path[dataset_name]

    dataset_for_test = dataset.listDataset(root_path=path, transform=transform, encoder=encoder,
                                        img_size=(img_width, img_height))

    assert dataset_for_test

    print("num. " + str(dataset_name) + " data : " + str(len(dataset_for_test)))

    acc, mean_ed, conf_mat = \
        get_performace(dataset_name, dataset_for_test, correct_dir, incorrect_dir)
    print('confusion mat for ' + str(dataset_name))
    utils.print_confusion_matrix(conf_mat)
    conf_mat_path = os.path.join(result_dir, str(dataset_name) +'_conf_mat.txt')
    utils.write_confusion_matrix(conf_mat, conf_mat_path)
    print(str(dataset_name) + '. acc : ' + str(acc))
    print(str(dataset_name) + '. MED : ' + str(mean_ed))

    dataset_acc[dataset_name] = [acc, mean_ed]

fout = open(os.path.join(result_dir, 'results.txt'), 'w')
for iter_acc in dataset_acc:
    print("Acc. of " + str(iter_acc) + ": %.5f" % (dataset_acc[iter_acc][0]))
    print("MED. of " + str(iter_acc) + ": %.5f" % (dataset_acc[iter_acc][1]))

    fout.write("Acc. of " + str(iter_acc) + ": %.5f\n" % (dataset_acc[iter_acc][0]))
    fout.write("MED. of " + str(iter_acc) + ": %.5f\n" % (dataset_acc[iter_acc][1]))
fout.close()
