# GENEA Numerical Evaluations
Scripts for numerical evaluations for the GENEA Gesture Generation Challenge:

https://genea-workshop.github.io/2020/#gesture-generation-challenge

This directory provides the scripts for quantitative evaluation of our gesture generation framework. We currently support the following measures:
- Average Jerk (AJ)
- Histogram of Moving Distance (HMD) for velocity
- Hellinger distance between histograms



## Run

`calc_errors.py`, `calc_histogram.py`, and `hellinger_distance.py` support different quantitative measures, described below.


### Average jerk

Average Jerk (AJ) represent the characteristics of gesture motion.

To calculate AJ, you can use `calc_jerk.py`.

```sh
# Compute AJ
python calc_jerk.py -g your_prediction_dir
```

Note: `calc_jerk.py` computes AJ for both original and predicted gestures. The AJ of the original gestures will be stored in `result/original` by default. The AJ of the predicted gestures will be stored in `result/your_prediction_dir`.

### HMD

Histogram of Moving Distance (HMD) shows the velocity/acceleration distribution of gesture motion.

To calculate HMD, you can use `calc_histogram.py`.
You can select the measure to compute by `--measure` or `-m` option (default: velocity).  
In addition, this script supports histogram visualization. To enable visualization, use `--visualize` or `-v` option.

```sh
# Compute velocity histogram
python calc_distance.py -g your_prediction_dir -m velocity -w 0.05  # You can change the bin width of the histogram

# Compute acceleration histogram
python calc_distance.py -g your_prediction_dir -m acceleration -w 0.05
```

Note: `calc_distance.py` computes HMD for both original and predicted gestures. The HMD of the original gestures will be stored in `result/original` by default.

### Hellingere distance

Hellinger distance indicates how close two histograms are to each other.

To calculate Hellinger distance, you can use `hellinger_distance.py` script.
