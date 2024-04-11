# 23-sleep-changepoint-detection
2023 Kaggle Competition Alice_Kai_Bofei

This is a Kaggle Competition in 2023, which can be found at this [link](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/overview).

Due to the size of the data files, they are not included in this repo. However a data folder is still included in this repo for completeness.

This is a team project, in which my contribution includes:
> •	Converted timestamps to new features, including a sinusoidal, and a numerical time feature.<
> •	Employed a LSTM + Attention NN based on shifting-window sequences of multivariate time series data (magnitude, and direction of accelerations, and said time features); trained the NN to perform a sequence level four-class classification task (sequence purely awake, purely asleep, including wake-up, including fall-asleep), as a precursing step to identifying the change points (awake-to-asleep, and the opposite), and achieved a validation accuracy of 96%.<
> •	Identified the change points with innovation by scoring accumulatively according to the said classification results (1 point for awake-to-asleep, -1 for the opposite, 0 for the other classes), which, given adequate precursing classification, would yield a spike plot with score vs time. <
