# -*- coding: utf-8 -*-
"""
Calculating CCA (Canonical Correlation Analysis) over the produced and ground truth gestures

@author: Taras Kucherenko
"""

import argparse
import glob
import os
import numpy as np

from cca import calculate_CCA


def shorten(arr_one, arr_two):
    """
    Make sure that two arrays have the same length
    Args:
        arr_one:  array one
        arr_two:  array two

    Returns:
        shortened versions of both arrays

    """
    min_len = min(arr_one.shape[0], arr_two.shape[0])
    arr_one = arr_one[:min_len]
    arr_two = arr_two[:min_len]

    return arr_one, arr_two


def save_result(lines, out_dir):
    """Write computed measure to CSV

      Args:
          lines:        list of strings to be written
          out_dir:      output directory
          measure:      used measure
    """

    # Make output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    outname = os.path.join(out_dir, 'cca.csv')

    with open(outname, 'w') as out_file:
        out_file.writelines(lines)


def evaluate_folder(cond_name):
    """
    Calculate numerical measure for the coordinates in the given folder
    Args:
        cond_name:   name of the condition / folder to evaluate

    Returns:
        nothing, prints out the metrics results

    """

    cond_dir = "data/" + cond_name + "/"

    generated_files = sorted(glob.glob(os.path.join(cond_dir, '*.npy')))

    gt_files = sorted(glob.glob(os.path.join("data/GT", '*.npy')))

    predicted_out_lines = [','.join(['file']) + '\n']

    predicted_errors = []
    for predicted_file, gt_file in zip(generated_files, gt_files):

        # read and flatten the predicted values
        predicted_coords = np.load(predicted_file)
        predicted_coords = np.reshape(predicted_coords, (predicted_coords.shape[0], -1))

        # read and flatten the ground truth values
        original_coords = np.load(gt_file)
        original_coords = np.reshape(original_coords, (original_coords.shape[0], -1))

        # make sure sequences have the same length
        predicted_coords, original_coords = shorten(predicted_coords, original_coords)

        # calculate CCA value
        current_cca_value = calculate_CCA(original_coords, predicted_coords)

        predicted_errors.append(current_cca_value)

        basename = os.path.basename(predicted_file)
        predicted_line = basename

        predicted_line += ',' + str(current_cca_value) + '\n'

        predicted_out_lines.append(predicted_line)

    predicted_average_line = 'Average'
    error_avgs = np.mean(predicted_errors, axis=0)
    error_stds = np.std(predicted_errors, axis=0)

    predicted_average_line += ',' + str(error_avgs)

    predicted_out_lines.append(predicted_average_line)

    predicted_out_dir = os.path.join("result", cond_name)

    save_result(predicted_out_lines, predicted_out_dir)

    print('{:s}: {:.2f} +- {:.2F}'.format(cond_name, np.mean(predicted_errors), error_stds))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Calculate prediction errors')
    parser.add_argument('--coords_dir', '-c', default='data',
                        help='Predicted gesture directory')
    args = parser.parse_args()

    print('CCA:')

    for cond_name in os.listdir(args.coords_dir):
        if cond_name == "GT":
            continue
        evaluate_folder(cond_name)

    print('More detailed result was writen to the files in the "result" folder ')
    print('')