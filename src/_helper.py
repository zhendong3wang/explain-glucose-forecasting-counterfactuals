import copy
import csv
import os
import random as python_random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    """
    Load data into desired formats for training/validation/testing, including preprocessing.
    """

    def __init__(self, horizon, back_horizon):
        self.horizon = horizon
        self.back_horizon = back_horizon
        self.scaler = list()
        self.historical_values = list()  # first by patient idx, then by col_idx

    def preprocessing(
        self,
        lst_train_arrays,
        lst_test_arrays,
        # train_mode=True, # flag for train_mode (split into train/val), test_mode (no split)
        train_size=0.8,
        normalize=False,
        sequence_stride=6,
        target_col=0,
    ):
        self.normalize = normalize
        self.sequence_stride = sequence_stride
        self.target_col = target_col
        train_arrays = copy.deepcopy(lst_train_arrays)
        test_arrays = copy.deepcopy(lst_test_arrays)

        # count valid timesteps for each individual series
        # train_array.shape = n_timesteps x n_features
        self.valid_steps_train = [train_array.shape[0] for train_array in train_arrays]
        train_lst, val_lst, test_lst = list(), list(), list()
        for idx in range(len(train_arrays)):
            bg_sample_train = train_arrays[idx]
            bg_sample_test = test_arrays[idx]
            valid_steps_sample = self.valid_steps_train[idx]

            train = bg_sample_train[: int(train_size * valid_steps_sample), :].copy()
            val = bg_sample_train[int(train_size * valid_steps_sample) :, :].copy()
            test = bg_sample_test[:, :].copy()

            if self.normalize:
                scaler_cols = list()
                # train.shape = n_train_timesteps x n_features
                for col_idx in range(train.shape[1]):
                    scaler = MinMaxScaler(feature_range=(0, 1), clip=False)
                    train[:, col_idx] = remove_extra_dim(
                        scaler.fit_transform((add_extra_dim(train[:, col_idx])))
                    )
                    val[:, col_idx] = remove_extra_dim(
                        scaler.transform(add_extra_dim(val[:, col_idx]))
                    )
                    test[:, col_idx] = remove_extra_dim(
                        scaler.transform(add_extra_dim(test[:, col_idx]))
                    )
                    scaler_cols.append(scaler)  # by col_idx, each feature
                self.scaler.append(scaler_cols)  # by pat_idx, each patient

            lst_hist_values = list()
            for col_idx in range(train.shape[1]):
                all_train_col = np.concatenate((train[:, col_idx], val[:, col_idx]))
                # decimals = 1, 2 OR 3?
                unique_values = np.unique(np.round(all_train_col, decimals=2))
                lst_hist_values.append(unique_values)
            self.historical_values.append(lst_hist_values)

            train_lst.append(train)
            val_lst.append(val)
            test_lst.append(test)

        (
            self.X_train,
            self.Y_train,
            self.train_idxs,
        ) = self.create_sequences(
            train_lst,
            self.horizon,
            self.back_horizon,
            self.sequence_stride,
            self.target_col,
        )
        (
            self.X_val,
            self.Y_val,
            self.val_idxs,
        ) = self.create_sequences(
            val_lst,
            self.horizon,
            self.back_horizon,
            self.sequence_stride,
            self.target_col,
        )
        (
            self.X_test,
            self.Y_test,
            self.test_idxs,
        ) = self.create_sequences(
            test_lst,
            self.horizon,
            self.back_horizon,
            self.sequence_stride,
            self.target_col,
        )

    @staticmethod
    def create_sequences(
        series_lst, horizon, back_horizon, sequence_stride, target_col=0
    ):
        Xs, Ys, sample_idxs = list(), list(), list()

        cnt_nans = 0
        for idx, series in enumerate(series_lst):
            len_series = series.shape[0]
            if len_series < (horizon + back_horizon):
                print(
                    f"Warning: not enough timesteps to split for sample {idx}, len: {len_series}, horizon: {horizon}, back: {back_horizon}."
                )

            for i in range(0, len_series - back_horizon - horizon, sequence_stride):
                input_series = series[i : (i + back_horizon)]
                output_series = series[
                    (i + back_horizon) : (i + back_horizon + horizon), [target_col]
                ]
                # TODO: add future plans as additional variables (?)
                if np.isfinite(input_series).all() and np.isfinite(output_series).all():
                    Xs.append(input_series)
                    Ys.append(output_series)
                    # record the sample index when splitting
                    sample_idxs.append(idx)
                else:
                    cnt_nans += 1
                    if cnt_nans % 100 == 0:
                        print(f"{cnt_nans} strides skipped due to NaN values.")

        return np.array(Xs), np.array(Ys), np.array(sample_idxs)


def load_dataset(dataset, data_path):
    if dataset == "ohiot1dm":
        # load into list of training arrays, and a standalone list for test
        lst_arrays_train = load_ohio_data(data_path, "all_train.csv")
        lst_arrays_test = load_ohio_data(data_path, "all_test.csv")
    elif dataset == "simulated":
        # load into list of training arrays, and a standalone list for test
        lst_arrays_train = load_sim_data(data_path, "all_train.csv")
        lst_arrays_test = load_sim_data(data_path, "all_test.csv")
    else:
        print("Not implemented: load_dataset.")
    return lst_arrays_train, lst_arrays_test


def load_ohio_data(data_path, file_name="all_train.csv"):
    # load all the patients, combined
    data = pd.read_csv(data_path + "data_OhioT1DM/" + file_name)

    from functools import reduce
    from operator import or_ as union

    def idx_union(mylist):
        idx = reduce(union, (index for index in mylist))
        return idx

    idx_missing = data.loc[data["missing"] != -1].index
    idx_missing_union = idx_union([idx_missing - 1, idx_missing])

    data = data.drop(idx_missing_union)
    data_bg = data[
        [
            "index_new",
            "patient_id",
            "glucose",
            "basal",
            "bolus",
            "carbs",
            "exercise_intensity",
        ]
    ]
    data_bg["time"] = data_bg[["index_new"]].apply(
        lambda x: pd.to_datetime(x, errors="coerce", format="%Y-%m-%d %H:%M:%S")
    )
    data_bg = data_bg.drop("index_new", axis=1)

    data_bg["bolus"][data_bg["bolus"] == -1] = 0
    data_bg["carbs"][data_bg["carbs"] == -1] = 0
    data_bg["exercise_intensity"][data_bg["exercise_intensity"] == -1] = 0
    data_bg["glucose"][data_bg["glucose"] == -1] = np.NaN

    lst_patient_id = [
        540,
        544,
        552,
        567,
        584,
        596,
        559,
        563,
        570,
        575,
        588,
        591,
    ]
    lst_arrays = list()
    for pat_id in lst_patient_id:
        lst_arrays.append(
            np.asarray(
                data_bg[data_bg["patient_id"] == pat_id][
                    [
                        "glucose",
                        "basal",
                        "bolus",
                        "carbs",
                        "exercise_intensity",
                    ]
                ]
            )
        )
    return lst_arrays


def load_sim_data(data_path, file_name="all_train.csv"):
    data = pd.read_csv(data_path + "data_simulation/" + file_name)

    data_bg = data[["patient_id", " Time", "CGM", "CHO", "insulin"]]
    data_bg["time"] = data_bg[[" Time"]].apply(
        lambda x: pd.to_datetime(x, errors="coerce", format="%Y-%m-%d %H:%M:%S")
    )
    data_bg = data_bg.drop(" Time", axis=1)

    lst_patient_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    lst_arrays = list()
    for pat_id in lst_patient_id:
        lst_arrays.append(
            np.asarray(
                data_bg[data_bg["patient_id"] == pat_id][["CGM", "CHO", "insulin"]]
            )
        )
    return lst_arrays


# remove an extra dimension
def remove_extra_dim(input_array):
    # 2d to 1d
    if len(input_array.shape) == 2:
        return np.reshape(input_array, (-1))
    # 3d to 2d (remove the last empty dim)
    elif len(input_array.shape) == 3:
        return np.squeeze(np.asarray(input_array), axis=-1)
    else:
        print("Not implemented.")


# add an extra dimension
def add_extra_dim(input_array):
    # 1d to 2d
    if len(input_array.shape) == 1:
        return np.reshape(input_array, (-1, 1))
    # 2d to 3d
    elif len(input_array.shape) == 2:
        return np.asarray(input_array)[:, :, np.newaxis]
    else:
        print("Not implemented.")


###################### Evaluation metrics ########################
def forecast_metrics(dataset, Y_pred, inverse_transform=True):
    Y_test_original, Y_pred_original = list(), list()
    if inverse_transform:
        for i in range(dataset.X_test.shape[0]):
            idx = dataset.test_idxs[i]
            scaler = dataset.scaler[idx]

            Y_test_original.append(
                scaler[dataset.target_col].inverse_transform(dataset.Y_test[i])
            )
            Y_pred_original.append(
                scaler[dataset.target_col].inverse_transform(Y_pred[i])
            )

        Y_test_original = np.array(Y_test_original)
        Y_pred_original = np.array(Y_pred_original)
    else:
        Y_test_original = dataset.Y_test
        Y_pred_original = Y_pred

    def smape(Y_test, Y_pred):
        # src: https://github.com/ServiceNow/N-BEATS/blob/c746a4f13ffc957487e0c3279b182c3030836053/common/metrics.py
        def smape_sample(actual, forecast):
            return 200 * np.mean(
                np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast))
            )

        return np.mean([smape_sample(Y_test[i], Y_pred[i]) for i in range(len(Y_pred))])

    def rmse(Y_test, Y_pred):
        return np.sqrt(np.mean((Y_pred - Y_test) ** 2))

    mean_smape = smape(Y_test_original, Y_pred_original)
    mean_rmse = rmse(Y_test_original, Y_pred_original)

    return mean_smape, mean_rmse


def cf_metrics(
    desired_max_lst,
    desired_min_lst,
    X_test,
    cf_samples,
    z_preds,
    change_idx,
    hist_inputs,
):
    validity = validity_ratio(
        pred_values=z_preds,
        desired_max_lst=desired_max_lst,
        desired_min_lst=desired_min_lst,
    )
    proximity = euclidean_distance(
        X=X_test, cf_samples=cf_samples, change_idx=change_idx
    )
    compactness = compactness_score(
        X=X_test, cf_samples=cf_samples, change_idx=change_idx
    )
    cumsum_auc, cumsum_valid_steps, cumsum_counts = cumulative_valid_steps(
        pred_values=z_preds, max_bounds=desired_max_lst, min_bounds=desired_min_lst
    )

    # Auxiliary metrics for validating the historical values mechanism
    proximity_hist = mae_distance_hist(
        hist_inputs=hist_inputs, cf_samples=cf_samples, change_idx=change_idx
    )
    compactness_hist = compact_score_hist(
        hist_inputs=hist_inputs, cf_samples=cf_samples, change_idx=change_idx
    )

    return (
        validity,
        proximity,
        compactness,
        cumsum_valid_steps,
        cumsum_counts,
        cumsum_auc,
        proximity_hist,
        compactness_hist,
    )


# validity ratio
def validity_ratio(pred_values, desired_max_lst, desired_min_lst):
    validity_lst = np.logical_and(
        pred_values <= desired_max_lst, pred_values >= desired_min_lst
    ).mean(axis=1)
    return validity_lst.mean()


def cumulative_valid_steps(pred_values, max_bounds, min_bounds):
    input_array = np.logical_and(pred_values <= max_bounds, pred_values >= min_bounds)
    until_steps_valid = np.empty(input_array.shape[0])
    n_samples, n_steps_total, _ = pred_values.shape
    for i in range(input_array.shape[0]):
        step_counts = 0
        for step in range(input_array.shape[1]):
            if input_array[i, step] == True:
                step_counts += 1
                until_steps_valid[i] = step_counts
            elif input_array[i, step] == False:
                until_steps_valid[i] = step_counts
                break
            else:
                print("Wrong input: cumulative_valid_steps.")

    valid_steps, counts = np.unique(until_steps_valid, return_counts=True)
    cumsum_counts = np.flip(np.cumsum(np.flip(counts)))
    # remove the valid_step=0 (no valid cf preds) in the trapz calculation
    valid_steps, cumsum_counts = fillna_cumsum_counts(
        n_steps_total, valid_steps, cumsum_counts
    )

    cumsum_auc = np.trapz(
        cumsum_counts[1:] / n_samples, valid_steps[1:] / n_steps_total
    )

    return cumsum_auc, valid_steps, cumsum_counts


def fillna_cumsum_counts(n_steps_total, valid_steps, cumsum_counts):
    df = pd.DataFrame(
        [{key: val for key, val in zip(valid_steps, cumsum_counts)}],
        columns=list(range(0, n_steps_total + 1)),
    )
    df = df.sort_index(ascending=True, axis=1)
    # backfill the previous valid steps
    df = df.fillna(method="backfill", axis=1)
    # fill 0s for the right hand nas
    df = df.fillna(method=None, value=0)
    valid_steps, cumsum_counts = df.columns.to_numpy(), df.values[0]
    return valid_steps, cumsum_counts


def euclidean_distance(X, cf_samples, change_idx, average=True):
    X = X[:, :, change_idx]
    cf_samples = cf_samples[:, :, change_idx]
    paired_distances = np.linalg.norm(X - cf_samples, axis=1)
    # return the average of compactness for each sample
    return paired_distances.mean(axis=1).mean() if average else paired_distances


# originally from: https://github.com/isaksamsten/wildboar/blob/859758884677ba32a601c53a5e2b9203a644aa9c/src/wildboar/metrics/_counterfactual.py#L279
def compactness_score(X, cf_samples, change_idx):
    X = X[:, :, change_idx]
    cf_samples = cf_samples[:, :, change_idx]
    # absolute tolerance atol=0.01, 0.001, OR 0.0001?
    c = np.isclose(X, cf_samples, atol=0.001)
    compact_lst = np.mean(c, axis=1)
    # return the average of compactness for each sample
    return compact_lst.mean(axis=1).mean()


def mae_distance_hist(hist_inputs, cf_samples, change_idx):
    mae_scores = list()
    for i in range(cf_samples.shape[0]):  # for each sample, i
        hist_sample = hist_inputs[i]
        min_ae_lst = list()
        for j in change_idx:  # for each feature, j
            cf_repeated = np.repeat(
                cf_samples[i, :, j][:, np.newaxis], len(hist_sample[j]), axis=1
            )
            min_dist = np.nanmin(np.abs(cf_repeated - hist_sample[j]), axis=1)
            min_ae_lst.append(min_dist.mean())
        mae_per_sample = np.mean(min_ae_lst)
        mae_scores.append(mae_per_sample)
    return np.mean(mae_scores)


def compact_score_hist(hist_inputs, cf_samples, change_idx):
    compact_scores = list()
    for i in range(cf_samples.shape[0]):  # for each sample, i
        hist_sample = hist_inputs[i]
        c_lst = list()
        for j in change_idx:  # for each feature, j
            cf_repeated = np.repeat(
                cf_samples[i, :, j][:, np.newaxis], len(hist_sample[j]), axis=1
            )
            # absolute tolerance atol; defined for hist_inputs (decimals=2)
            atol = 0.005
            c_per_step = np.any(
                np.abs(cf_repeated - hist_sample[j]) <= atol, axis=1
            ).astype(np.float32)
            c_lst.append(c_per_step.mean())
        compact_per_sample = np.mean(c_lst)
        compact_scores.append(compact_per_sample)
    return np.mean(compact_scores)


###################### Utils ########################


class ResultWriter:
    def __init__(self, file_name, dataset_name):
        self.file_name = file_name
        self.dataset_name = dataset_name

    def write_head(self):
        # write the head in csv file
        with open(self.file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "dataset",
                    "random_seed",
                    "forecast_model",
                    "test_group",
                    "horizon",
                    "forecast_smape",
                    "forecast_rmse",
                    "pred_margin_weight",
                    "step_weights",
                    "clip_mechanism",
                    "hist_values_input",
                    "validity_ratio",
                    "proximity",
                    "compactness",
                    "step_validity_auc",
                    "proximity_hist",
                    "compactness_hist",
                ]
            )

    def write_result(
        self,
        random_seed,
        method_name,
        test_group,
        horizon,
        forecast_smape,
        forecast_rmse,
        pred_margin_weight,
        step_weights,
        clip_mechanism,
        hist_values_input,
        validity_ratio,
        proximity,
        compactness,
        step_validity_auc,
        proximity_hist,
        compactness_hist,
    ):
        with open(self.file_name, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.dataset_name,
                    random_seed,
                    method_name,
                    test_group,
                    horizon,
                    forecast_smape,
                    forecast_rmse,
                    pred_margin_weight,
                    step_weights,
                    clip_mechanism,
                    hist_values_input,
                    validity_ratio,
                    proximity,
                    compactness,
                    step_validity_auc,
                    proximity_hist,
                    compactness_hist,
                ]
            )


# Method: Fix the random seeds to get consistent models
def reset_seeds(seed_value=39):
    # ref: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    # necessary for starting Numpy generated random numbers in a well-defined initial state.
    np.random.seed(seed_value)
    # necessary for starting core Python generated random numbers in a well-defined state.
    python_random.seed(seed_value)
    # set_seed() will make random number generation
    tf.random.set_seed(seed_value)
