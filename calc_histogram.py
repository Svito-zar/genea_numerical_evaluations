# -*- coding: utf-8 -*-
"""
Calculating statistics over the produced and ground truth gestures

@author: kaneko.naoshi
"""

import argparse
import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np



def compute_velocity(data, dim=3):
    """Compute velocity between adjacent frames

      Args:
          data:         array containing joint positions of gesture
          dim:          gesture dimensionality

      Returns:
          vel_norms:    velocities of each joint between each adjacent frame
    """

    # Flatten the array of 3d coords
    coords = data.reshape(data.shape[0], -1)

    # First derivative of position is velocity
    vels = np.diff(coords, n=1, axis=0)

    num_vels = vels.shape[0]
    num_joints = vels.shape[1] // dim

    vel_norms = np.zeros((num_vels, num_joints))

    for i in range(num_vels):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            vel_norms[i, j] = np.linalg.norm(vels[i, x1:x2])

    return vel_norms * 20


def compute_acceleration(data, dim=3):
    """Compute acceleration between adjacent frames

      Args:
          data:         array containing joint positions of gesture
          dim:          gesture dimensionality

      Returns:
          acc_norms:    accelerations of each joint between each adjacent frame
    """

    # Second derivative of position is acceleration
    accs = np.diff(data, n=2, axis=0)

    num_accs = accs.shape[0]
    num_joints = accs.shape[1] // dim

    acc_norms = np.zeros((num_accs, num_joints))

    for i in range(num_accs):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            acc_norms[i, j] = np.linalg.norm(accs[i, x1:x2])

    return acc_norms * 20 * 20


def save_result(lines, out_dir, width, measure):
    """Write computed histogram to CSV

      Args:
          lines:        list of strings to be written
          out_dir:      output directory
          width:        bin width of the histogram
          measure:      used measure for histogram calculation
    """

    # Make output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    hist_type = measure[:3]  # 'vel' or 'acc'
    filename = 'hmd_{}_{}.csv'.format(hist_type, width)
    outname = os.path.join(out_dir, filename)

    with open(outname, 'w') as out_file:
        out_file.writelines(lines)


def make_histogram(cond_name, measure, visualize = True, width = 1):
    """
    Calculate frequencies histogram for a given measure
    Args:
        cond_name:   name of the folder to consider
        measure:     measure to be used
        visualize:   flag if we want to visualize the result
        width:       width of the histogram bins

    Returns:
        Saved in the "results" folder

    """

    predicted_dir = "data/" + cond_name + "/"  # os.path.join(args.predicted, args.gesture)

    measures = {
        'velocity': compute_velocity,
        'acceleration': compute_acceleration,
    }

    # Check if error measure was correct
    if measure not in measures:
        raise ValueError('Unknown measure: \'{}\'. Choose from {}'
                         ''.format(measure, list(measures.keys())))

    predicted_files = sorted(glob.glob(os.path.join(predicted_dir, '*.npy')))

    predicted_out_lines = [','.join([''] + ['Total']) + '\n']

    predicted_distances = []
    for predicted_file in predicted_files:
        # read the values and remove hips which are fixed
        predicted = np.load(predicted_file)[:, 2:]

        predicted_distance = measures[measure](
            predicted)  # [:, selected_joints]

        predicted_distances.append(predicted_distance)

    predicted_distances = np.concatenate(predicted_distances)

    # Compute histogram for each joint
    bins = np.arange(0, 59 + width, width)
    num_joints = predicted_distances.shape[1]
    predicted_hists = []
    for i in range(num_joints):
        predicted_hist, _ = np.histogram(predicted_distances[:, i], bins=bins)

        predicted_hists.append(predicted_hist)

    # Sum over all joints
    predicted_total = np.sum(predicted_hists, axis=0)

    # Append total number of bin counts to the last
    predicted_hists = np.stack(predicted_hists + [predicted_total], axis=1)

    num_bins = bins.size - 1
    for i in range(num_bins):
        predicted_line = str(bins[i])
        for j in range(num_joints + 1):
            predicted_line += ',' + str(predicted_hists[i, j])
        predicted_line += '\n'
        predicted_out_lines.append(predicted_line)

    predicted_out_dir  = "result/" + cond_name + "/"  # os.path.join(args.predicted, args.gesture)


    if visualize:
        sum = np.sum(predicted_total)
        actual_freq = predicted_total / sum
        plt.plot(bins[:-1], actual_freq, label=cond_name)
        plt.legend()
        plt.xlabel('Velocity (cm/s)')
        plt.ylabel('Frequency')
        plt.title('Frequencies of Moving Distance ({})'.format(measure))
        plt.tight_layout()
        plt.show()

    save_result(predicted_out_lines, predicted_out_dir, width, measure)

    print('HMD ({}):'.format(measure))
    print('bins: {}'.format(bins))
    print('predicted: {}'.format(predicted_total))

def main():

    parser = argparse.ArgumentParser(
        description='Calculate histograms of moving distances')
    parser.add_argument('--coords_dir', '-p', default='data',
                        help='Predicted gestures directory')
    parser.add_argument('--width', '-w', type=float, default=1,
                        help='Bin width of the histogram')
    parser.add_argument('--measure', '-m', default='velocity',
                        help='Measure to calculate (velocity or acceleration)')
    parser.add_argument('--select', '-s', nargs='+',
                        help='Joint subset to compute (if omitted, use all)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualize histograms')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    for cond_name in os.listdir(args.coords_dir):
        print("\nConsider", cond_name)
        make_histogram(cond_name, args.measure, args.visualize, args.width)

    print('\nMore detailed result was writen to the files in the "results" folder ')
    print('')

if __name__ == '__main__':
    main()
