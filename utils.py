#!/usr/bin/python
# encoding: utf-8

import torch
import yaml
import os
import cv2
import numpy as np
from freetype import *

class LabelEncoder(object):
    def __init__(self, labels, is_ignore_case=True):
        self.is_ignore_case = is_ignore_case
        if self.is_ignore_case:
            labels = labels.lower()
        self.labels = labels  # for `-1` index

        self.dict = {}
        for i, char in enumerate(self.labels):
            # NOTE: 0 is reserved for 'blank' required by ctc
            self.dict[char] = i + 1

    def encode(self, texts):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        num_texts = len(texts)

        for text_idx in range(num_texts):
            string = texts[text_idx]
            tmp_string = list()
            for char in string:
                if char.lower() in self.dict:
                    tmp_string.append(char)
            new_string = ''.join(tmp_string)
            texts[text_idx] = new_string

        length = [len(s) for s in texts]
        text_oneline = ''.join(texts)

        text_encoded = list()

        for char in text_oneline:
            if self.is_ignore_case is True:
                text_encoded.append(self.dict[char.lower()])
            else:
                text_encoded.append(self.dict[char])

        return (torch.tensor(text_encoded, dtype=torch.long), torch.tensor(length, dtype=torch.long))

    def decode(self, labels, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if labels.numel() != length.sum():
            print('texts with length ' + str(labels.numel()) +' does not match declared length ' + str(length.sum()))

        texts = list()
        idx = 0

        for len in length:
            text_labels = labels[idx:idx+len]
            idx += len

            if raw is True:
                raw_text = list()
                for label in text_labels:
                    label = label - 1
                    if label < 0:
                        raw_text.append('-')
                    else:
                        raw_text.append(self.labels[label])
                raw_text = ''.join(raw_text)
                texts.append(raw_text)
            else:
                text = list()
                for iter in range(len):
                    if text_labels[iter] != 0 and (not (iter > 0 and text_labels[iter - 1] == text_labels[iter])):
                        text.append(self.labels[text_labels[iter] - 1])
                text = ''.join(text)
                texts.append(text)

        return texts

class SynthImageMaker(object):
    def __init__(self):
        self.font_face = Face('./Brevia-Black.otf')
        self.font_face.set_char_size(64*64)
        self.slot = self.font_face.glyph

    def do_make_synthData(self, text):
        # First pass to compute bbox
        width, height, baseline = 0, 0, 0
        previous = 0
        for i, c in enumerate(text):
            self.font_face.load_char(c)
            bitmap = self.slot.bitmap
            height = max(height,
                         bitmap.rows + max(0, -(self.slot.bitmap_top - bitmap.rows)))
            baseline = max(baseline, max(0, -(self.slot.bitmap_top - bitmap.rows)))
            kerning = self.font_face.get_kerning(previous, c)
            if self.slot.bitmap_left < 0:
                width += (self.slot.advance.x >> 6) + (kerning.x >> 6) - self.slot.bitmap_left
            else:
                width += (self.slot.advance.x >> 6) + (kerning.x >> 6)
            previous = c

        synth_text = np.zeros((height, width), dtype=np.uint8)

        # Second pass for actual rendering
        x, y = 0, 0
        previous = 0
        for c in text:
            self.font_face.load_char(c)
            bitmap = self.slot.bitmap
            top = self.slot.bitmap_top
            left = self.slot.bitmap_left
            w, h = bitmap.width, bitmap.rows
            y = height - baseline - top
            kerning = self.font_face.get_kerning(previous, c)
            x += (kerning.x >> 6)

            start_y = y
            end_y = y + h
            start_x = x
            end_x = x + w

            if start_y < 0:
                end_y = end_y - start_y
                start_y = 0
            # if start_x < 0:
            #     start_x = 0
            #     end_x = end_x - start_x
            #
            # if end_y > height:
            #     end_y = height
            # if end_x > width:
            #     end_x = width

            synth_text[start_y:end_y, start_x:end_x] += np.array(bitmap.buffer, dtype=np.uint8).reshape(h, w)
            x += (self.slot.advance.x >> 6)
            previous = c

        synth_text = 255 - synth_text

        return synth_text

# get num of corrected samples(char)
def check_corrected_each_label(preds, targets, label_dict):
    for target_idx in range(len(targets)):
        target = targets[target_idx]
        pred = preds[target_idx]

        for label_idx in range(len(target)):
            target_label = target[label_idx]

            if not target_label in label_dict:
                label_dict[target_label] = [0.0, 0.0]

            try:
                pred_label = pred[label_idx]
            except IndexError:
                label_dict[target_label][0] += 1.0
                continue

            if target_label == pred_label:
                label_dict[target_label][0] += 1.0
                label_dict[target_label][1] += 1.0
            else:
                label_dict[target_label][0] += 1.0

    return label_dict

def get_confusion_matrix(preds, targets, confusion_dict):
    for target_idx in range(len(targets)):
        target = targets[target_idx]
        pred = preds[target_idx]

        for label_idx in range(len(target)):
            target_label = target[label_idx]

            if not target_label in confusion_dict:
                confusion_dict[target_label] = dict()

            try:
                pred_label = pred[label_idx]
            except IndexError:
                continue

            if not pred_label in confusion_dict[target_label]:
                confusion_dict[target_label][pred_label] = 1.0
            else:
                confusion_dict[target_label][pred_label] += 1.0

    return confusion_dict

def print_confusion_matrix(confusion_dict):
    print('===========================================')
    for confusion_item in confusion_dict:
        print('[target: ' + str(confusion_item) + ']')
        print(confusion_dict[confusion_item])

        if len(confusion_dict[confusion_item]) <= 0:
            continue

        all_samples = 0
        target_samples = 0
        for idx in confusion_dict[confusion_item]:
            all_samples += confusion_dict[confusion_item][idx]
            if idx == confusion_item:
                target_samples = confusion_dict[confusion_item][idx]

        acc = target_samples / all_samples
        print('Acc : ' + str(acc))
    print('===========================================')

def write_confusion_matrix(confusion_dict, out_path):
    f_out = open(out_path, 'w')
    simple_f_out = open(out_path.replace('.txt', '_simple.txt'), 'w')
    for confusion_item in confusion_dict:
        f_out.write('[target: ' + str(confusion_item) + ']\n')
        f_out.write(str(confusion_dict[confusion_item]) + '\n')

        if len(confusion_dict[confusion_item]) <= 0:
            continue

        all_samples = 0
        target_samples = 0
        for idx in confusion_dict[confusion_item]:
            all_samples += confusion_dict[confusion_item][idx]
            if idx == confusion_item:
                target_samples = confusion_dict[confusion_item][idx]

        acc = target_samples / all_samples
        f_out.write('Acc : ' + str(acc) + '\n')

        simple_f_out.write(str(confusion_item) + '\t' + str(acc) + '\n')
    f_out.close()
    simple_f_out.close()

def get_config(conf):
    with open(conf, 'r') as stream:
        return yaml.load(stream, Loader=yaml.Loader)

def print_config(conf):
    print(yaml.dump(conf, default_flow_style=False, default_style=''))

def get_best_model(dir_path):
    ckpt_file = dict()
    minimum_loss = float('inf')
    minimum_file = ''

    for (path, dirs, files) in os.walk(dir_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]

            if ext == '.pth':
                load_pth = torch.load(os.path.join(path, filename), map_location='cpu')
                valid_loss = load_pth['loss']

                ckpt_idx = filename
                ckpt_idx = int(ckpt_idx.split("-")[-1].split(".")[0])

                ckpt_file[ckpt_idx] = valid_loss

                if valid_loss < minimum_loss:
                    minimum_loss = valid_loss
                    minimum_file = filename

    for idx in ckpt_file:
        print("ckpt-" + str(idx) + " " + str(ckpt_file[idx]))

    if minimum_file == '':
        return None

    return os.path.join(dir_path, minimum_file)

def idx_to_str(index):
    if not isinstance(index, int):
        raise ValueError('Only Int object able to be input')

    if index < 10:
        output_str = '00000' + str(index)
    elif index < 100:
        output_str = '0000' + str(index)
    elif index < 1000:
        output_str = '000' + str(index)
    elif index < 10000:
        output_str = '00' + str(index)
    elif index < 100000:
        output_str = '0' + str(index)
    else:
        output_str = str(index)

    return output_str

# This function compute the receptive field as bounding box coordinate.
# The coordinate system of this function is from 0.
def get_receptive_field(x_coord, y_coord, hyperparams_dict):
    init_x = x_coord
    init_y = y_coord

    # find minimum coordinate first.
    for name, module in hyperparams_dict.items():
        padding = module.padding
        kernel = module.kernel_size
        stride = module.stride

        # only consider stride(jump) and padding(move)
        # e.g, how many pixels skip along with axis and how many pixels move along with axis.
        if isinstance(padding, int) or isinstance(stride, int) or isinstance(kernel, int):
            xmin = x_coord * stride - padding
            ymin = y_coord * stride - padding
        else:
            xmin = x_coord * stride[1] - padding[1]
            ymin = y_coord * stride[0] - padding[0]

        # iteratively on minimum coordinate
        x_coord = xmin
        y_coord = ymin

    # store final output
    xmin_out = x_coord
    ymin_out = y_coord

    # then, find maximum coordinate.
    x_coord = init_x
    y_coord = init_y

    # find maximum coordinate.
    for name, module in hyperparams_dict.items():
        padding = module.padding
        kernel = module.kernel_size
        stride = module.stride

        # only consider stride(jump) and padding(move)
        # e.g, how many pixels skip along with axis and how many pixels move along with axis.
        # max coordinate only is affected by kernel size.
        if isinstance(padding, int) or isinstance(stride, int) or isinstance(kernel, int):
            xmin = x_coord * stride - padding
            xmax = xmin + kernel - 1
            ymin = y_coord * stride - padding
            ymax = ymin + kernel - 1
        else:
            xmin = x_coord * stride[1] - padding[1]
            xmax = xmin + kernel[1] - 1
            ymin = y_coord * stride[0] - padding[0]
            ymax = ymin + kernel[0] - 1

        # iteratively on maximum coordinate
        x_coord = xmax
        y_coord = ymax

    # final output
    xmax_out = x_coord
    ymax_out = y_coord

    return xmin_out, ymin_out, xmax_out, ymax_out

# This function make random color(BGR)
def random_color():
    # make random number from uniform distribution(0-1)
    rand_number = np.random.rand(3)
    rand_number = rand_number * 255

    return rand_number

def draw_crossline(img, center_x, center_y, color, length_of_line=2):
    input_rows = img.shape[0]
    input_cols = img.shape[1]

    state_x = ((center_x-length_of_line) >= 0 and (center_x+length_of_line) < input_cols)
    state_y = ((center_y-length_of_line) >= 0 and (center_y+length_of_line) < input_rows)

    if state_x and state_y:
        pt1 = (center_x, center_y-length_of_line)
        pt2 = (center_x, center_y+length_of_line)
        cv2.line(img, pt1, pt2, color=color)

        pt1 = (center_x-length_of_line, center_y)
        pt2 = (center_x+length_of_line, center_y)
        cv2.line(img, pt1, pt2, color=color)

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]