import os
import cv2
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/hdd/Datasets/0_Text/1_Public_Data/SynthVGG/SynthVGG/mnt')
parser.add_argument('--labels', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
opt = parser.parse_args()

def _make_dataset(root_path, target_labels, set_name):
    delim = '\t'

    print('Make dataset: ' + str(set_name))
    target_dict = dict()
    for ch in target_labels:
        target_dict[ch] = 0
    print("num. of classes: " + str(len(target_dict)))
    print(target_labels)

    not_contain_target = dict()

    # trainset
    file_path = 'ramdisk/max/90kDICT32px/annotation_' + str(set_name) + '.txt'
    fp = open(os.path.join(root_path, file_path), 'r')
    lines = fp.readlines()
    num_lines = len(lines)
    fp.close()

    out_name = 'VGG90_' + str(set_name) + '.txt'
    out_fp = open(out_name, 'w')
    for idx, line in enumerate(lines):
        sys.stdout.write('\r' + str(idx) + ' / ' + str(num_lines))
        imgpath = line.strip().split(' ')[0]
        label = imgpath.split('/')[-1].split('_')[1]
        label = label.lower()
        imgpath = imgpath[1:]
        imgpath = 'ramdisk/max/90kDICT32px%s' % imgpath
        imgpath = os.path.join(root_path, imgpath)
        if not os.path.exists(imgpath):
            print("There are no image : " + str(imgpath))
            continue

        img = cv2.imread(imgpath)
        if img is None:
            print('Corrupted image : ' + str(imgpath))
            continue

        length_label = len(label)
        cnt_target = 0
        for ch in label:
            if ch not in target_dict:
                if ch in not_contain_target:
                    not_contain_target[ch] += 1
                else:
                    not_contain_target[ch] = 1
            else:
                target_dict[ch] += 1
                cnt_target += 1

        if cnt_target < int(length_label / 2):
            continue

        output = delim.join([imgpath, label])
        out_fp.write(output + '\n')
    out_fp.close()

    print("target class")
    sorted_list = sorted(target_dict.items(), key=lambda x: x[1], reverse=True)
    for char in sorted_list:
        out_str = char[0] + ': ' + str(char[1])
        print(out_str)

    print("not contain target class")
    sorted_list = sorted(not_contain_target.items(), key=lambda x: x[1], reverse=True)
    for sp_char in sorted_list:
        out_str = sp_char[0] + ': ' + str(sp_char[1])
        print(out_str)

target_labels = opt.labels
root_path = opt.root_path
_make_dataset(root_path=root_path, target_labels=target_labels, set_name='train')
_make_dataset(root_path=root_path, target_labels=target_labels, set_name='val')
_make_dataset(root_path=root_path, target_labels=target_labels, set_name='test')