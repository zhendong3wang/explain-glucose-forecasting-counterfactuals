#!/usr/bin/env python
# coding: utf-8
import logging
import os
import time
import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import tensorflow as tf

os.environ["TF_DETERMINISTIC_OPS"] = "1"
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

warnings.filterwarnings(action="ignore", message="Setting attributes")

from _helper import (
    DataLoader,
    ResultWriter,
    add_extra_dim,
    cf_metrics,
    forecast_metrics,
    load_dataset,
    remove_extra_dim,
    reset_seeds,
)
from bg_forecastcf import BGForecastCF


def main():
    parser = ArgumentParser(
        description="Run this script to evaluate ForecastCF method."
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset that the experiment is running on."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        required=True,
        help="The horizon of the forecasting task.",
    )
    parser.add_argument(
        "--back-horizon",
        type=int,
        required=True,
        help="The back horizon of the forecasting task",
    )
    parser.add_argument(
        "--test-group",
        type=str,
        default=None,
        help="Extract random 100 samples from test group, i.e., 'hyper'/'hypo'; default None.",
    )
    parser.add_argument(
        "--fraction-std",
        type=float,
        default=1,
        help="Fraction of standard deviation into creating the bound, e.g., 0.5, 1, 1.5, 2, ...",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=39,
        help="Random seed parameter, default 39.",
    )
    parser.add_argument("--output", type=str, help="Output file name.")
    A = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.getLogger("matplotlib.font_manager").disabled = True
    logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}.")

    logger.info(f"===========Desired trend parameters=============")
    center = "last"
    desired_shift, poly_order = 0, 1
    fraction_std = A.fraction_std
    logger.info(f"center: {center}, desired_shift: {desired_shift};")
    logger.info(f"fraction_std:{fraction_std};")
    logger.info(f"desired_change:'sample_based', poly_order:{poly_order}.")

    # the target column param should be user-defined
    TARGET_COL = 0
    # the index of the changeable columns
    if A.dataset == "ohiot1dm":
        CHANGE_COLS = [1, 2, 3, 4]
    elif A.dataset == "simulated":
        CHANGE_COLS = [1, 2]
    else:
        CHANGE_COLS = None

    RANDOM_STATE = A.random_seed
    result_writer = ResultWriter(file_name=A.output, dataset_name=A.dataset)

    logger.info(f"===========Random seed setup=============")
    logger.info(f"Random seed: {RANDOM_STATE}.")
    logger.info(f"Result writer is ready, writing to {A.output}...")
    # If `A.output` file already exists, no need to write head (directly append)
    if not os.path.isfile(A.output):
        result_writer.write_head()

    ###############################################
    # ## 1. Load data
    ###############################################
    data_path = "./data/"
    lst_arrays, lst_arrays_test = load_dataset(A.dataset, data_path)
    logger.info(f"The shape of loaded train: {len(lst_arrays)}*{lst_arrays[0].shape}")
    logger.info(f"The shape of test: {len(lst_arrays_test)}*{lst_arrays_test[0].shape}")

    # ### 1.1 Data pre-processing
    back_horizon, horizon = A.back_horizon, A.horizon
    logger.info(f"===========Forecasting setup=============")
    logger.info(f"Back horizon: {back_horizon}, horizon: {horizon}.")
    logger.info(f"Sequence stride (for moving window splits): {horizon}.")
    logger.info(f"Target column: {TARGET_COL}.")

    dataset = DataLoader(horizon, back_horizon)
    dataset.preprocessing(
        lst_train_arrays=lst_arrays,
        lst_test_arrays=lst_arrays_test,
        train_size=0.8,
        normalize=True,
        sequence_stride=horizon,
        target_col=TARGET_COL,
    )

    print(dataset.X_train.shape, dataset.Y_train.shape)
    print(dataset.X_val.shape, dataset.Y_val.shape)
    print(dataset.X_test.shape, dataset.Y_test.shape)

    for model_name in ["wavenet", "gru"]:
        # reset seeds for numpy, tensorflow, python random package and python environment seed
        reset_seeds(RANDOM_STATE)
        n_in_features = dataset.X_train.shape[2]
        n_out_features = 1

        ###############################################
        # ## 2.0 Forecasting model
        ###############################################
        # reset seeds for numpy, tensorflow, python random package and python environment seed
        reset_seeds(RANDOM_STATE)
        if model_name in ["wavenet", "seq2seq"]:
            forecast_model = build_tfts_model(
                model_name, back_horizon, horizon, n_in_features
            )
        elif model_name == "gru":
            forecast_model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Input(shape=(back_horizon, n_in_features)),
                    # Shape [batch, time, features] => [batch, time, gru_units]
                    tf.keras.layers.GRU(100, activation="tanh", return_sequences=True),
                    tf.keras.layers.GRU(100, activation="tanh", return_sequences=False),
                    # Shape => [batch, time, features]
                    tf.keras.layers.Dense(horizon, activation="linear"),
                    tf.keras.layers.Reshape((horizon, n_out_features)),
                ]
            )

            # Definition of the objective function and the optimizer
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
            forecast_model.compile(optimizer=optimizer, loss="mae")
        else:
            print("Not implemented: model_name.")

        # Define the early stopping criteria
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.0001, patience=10, restore_best_weights=True
        )

        # Train the model
        reset_seeds(RANDOM_STATE)
        forecast_model.fit(
            dataset.X_train,
            dataset.Y_train,
            epochs=200,
            batch_size=64,
            validation_data=(dataset.X_val, dataset.Y_val),
            callbacks=[early_stopping],
        )

        # Predict on the testing set (forecast)
        Y_preds = forecast_model.predict(dataset.X_test)
        mean_smape, mean_rmse = forecast_metrics(dataset, Y_preds)

        logger.info(
            f"[[{model_name}]] model trained, with test sMAPE score {mean_smape:0.4f}; test RMSE score: {mean_rmse:0.4f}."
        )

        ###############################################
        # ## 2.1 CF search
        ###############################################
        hyper_bound, hypo_bound = 180, 70
        logger.info(f"===========CF generation setup=============")
        logger.info(f"hyper bound value: {hyper_bound}, hypo bound: {hypo_bound}.")

        event_labels = list()
        for i in range(len(Y_preds)):
            scaler = dataset.scaler[dataset.test_idxs[i]][TARGET_COL]
            Y_preds_original = scaler.inverse_transform(Y_preds[i])

            if np.any(Y_preds_original >= hyper_bound):
                event_labels.append("hyper")
            elif np.any(Y_preds_original <= hypo_bound):
                event_labels.append("hypo")
            else:
                event_labels.append("normal")
        hyper_indices = np.argwhere(np.array(event_labels) == "hyper").reshape(-1)
        hypo_indices = np.argwhere(np.array(event_labels) == "hypo").reshape(-1)

        logger.info(f"hyper_indices shape: {hyper_indices.shape}")
        logger.info(f"hypo_indices shape: {hypo_indices.shape}")

        # use a subset of the test
        rand_test_size = 100
        if A.test_group == "hyper":
            if len(hyper_indices) >= rand_test_size:
                np.random.seed(RANDOM_STATE)
                rand_test_idx = np.random.choice(
                    hyper_indices, rand_test_size, replace=False
                )
            else:
                rand_test_idx = hyper_indices
        elif A.test_group == "hypo":
            if len(hypo_indices) >= rand_test_size:
                np.random.seed(RANDOM_STATE)
                rand_test_idx = np.random.choice(
                    hypo_indices, rand_test_size, replace=False
                )
            else:
                rand_test_idx = hypo_indices
        else:
            rand_test_idx = np.arange(dataset.X_test.shape[0])

        X_test = dataset.X_test[rand_test_idx]
        Y_test = dataset.Y_test[rand_test_idx]
        logger.info(
            f"Generating CFs for {len(rand_test_idx)} samples in total, for {A.test_group} test group..."
        )

        # loss calculation ==> min/max bounds
        desired_max_lst, desired_min_lst = list(), list()
        hist_inputs = list()

        # define the desired center to reach in two hours (24 timesteps for OhioT1DM)
        # then we need to cut the first 6 steps to generate the desired bounds
        desired_steps = 24 if A.dataset == "ohiot1dm" else 20
        if A.test_group == "hyper":
            desired_center_2h = hyper_bound - 10  # -10 for a fluctuating bound
        elif A.test_group == "hypo":
            desired_center_2h = hypo_bound + 10  # +10 for a fluctuating bound
        else:
            logger.warning(
                f"Group not identified: {A.test_group}, use a default center"
            )
            desired_center_2h = (hyper_bound + hypo_bound) / 2
        logger.info(f"desired center {desired_center_2h} in {desired_steps} timesteps.")

        for i in range(len(X_test)):
            idx = dataset.test_idxs[rand_test_idx[i]]
            scaler = dataset.scaler[idx]

            desired_center_scaled = scaler[TARGET_COL].transform(
                np.array(desired_center_2h).reshape(-1, 1)
            )[0][0]
            logger.info(
                f"desired_center: {desired_center_2h}; after scaling: {desired_center_scaled:0.4f}"
            )

            # desired trend bounds: use the `center` parameter from the input sequence as the starting point
            desired_max_scaled, desired_min_scaled = generate_bounds(
                center=center,  # Use the parameters defined at the beginning of the script
                shift=desired_shift,
                desired_center=desired_center_scaled,
                poly_order=poly_order,
                horizon=horizon,
                fraction_std=fraction_std,
                input_series=X_test[i, :, TARGET_COL],
                desired_steps=desired_steps,
            )
            # TODO: remove the ones that already satisfy the bounds here, OR afterwards?
            desired_max_lst.append(desired_max_scaled)
            desired_min_lst.append(desired_min_scaled)
            hist_inputs.append(dataset.historical_values[idx])

        ###############################################
        # ## 2.2 runtime recording
        ###############################################
        # create a dict for step_weights, prediction margin, clip_mechanism, and hist_input
        cf_model = BGForecastCF(
            max_iter=100,
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
            pred_margin_weight=0.9,  # focus 0.9 the prediction bound calculation (then `1-pred_margin_weight` on the input weighted steps)
            step_weights="unconstrained",
            random_state=RANDOM_STATE,
            target_col=TARGET_COL,
            only_change_idx=CHANGE_COLS,
        )
        if model_name in ["wavenet", "seq2seq", "gru"]:
            cf_model.fit(forecast_model)
        else:
            print("Not implemented: cf_model.fit.")
        start_time = time.time()
        cf_samples, losses, _ = cf_model.transform(
            X_test,
            desired_max_lst,
            desired_min_lst,
            clip_range_inputs=None,
            hist_value_inputs=None,
        )
        end_time = time.time()
        elapsed_time1 = end_time - start_time
        logger.info(f"Elapsed time - ForecastCF: {elapsed_time1:0.4f}.")

        cf_model2 = BGForecastCF(
            max_iter=100,
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
            pred_margin_weight=0.9,  # focus 0.9 the prediction bound calculation (then `1-pred_margin_weight` on the input weighted steps)
            step_weights="unconstrained",
            random_state=RANDOM_STATE,
            target_col=TARGET_COL,
            only_change_idx=CHANGE_COLS,
        )
        cf_model2.fit(forecast_model)

        # This can be an user-defined param; normalized low and high for each variable
        # set clip_range = [0, 1] right now, for 5 input variables
        clip_ranges = [[[0.0, 1.0]] * 5] * len(X_test)
        start_time = time.time()
        cf_samples2, losses2, _ = cf_model2.transform(
            X_test,
            desired_max_lst,
            desired_min_lst,
            clip_range_inputs=clip_ranges,
            hist_value_inputs=None,
        )
        end_time = time.time()
        elapsed_time2 = end_time - start_time
        logger.info(f"Elapsed time: {elapsed_time2:0.4f}.")

        cf_model3 = BGForecastCF(
            max_iter=100,
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
            pred_margin_weight=0.9,  # focus 0.9 the prediction bound calculation (then `1-pred_margin_weight` on the input weighted steps)
            step_weights="meal_time",
            random_state=RANDOM_STATE,
            target_col=TARGET_COL,
            only_change_idx=CHANGE_COLS,
        )
        cf_model3.fit(forecast_model)

        # This can be an user-defined param; normalized low and high for each variable
        clip_ranges = [[[0.0, 1.0]] * 5] * len(X_test)
        start_time = time.time()
        cf_samples3, losses3, _ = cf_model3.transform(
            X_test,
            desired_max_lst,
            desired_min_lst,
            clip_range_inputs=clip_ranges,
            hist_value_inputs=None,
        )
        end_time = time.time()
        elapsed_time3 = end_time - start_time
        logger.info(f"Elapsed time: {elapsed_time3:0.4f}.")

        cf_model4 = BGForecastCF(
            max_iter=100,
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
            pred_margin_weight=0.9,  # focus 0.9 the prediction bound calculation (then `1-pred_margin_weight` on the input weighted steps)
            step_weights="meal_time",
            random_state=RANDOM_STATE,
            target_col=TARGET_COL,
            only_change_idx=CHANGE_COLS,
        )
        cf_model4.fit(forecast_model)

        # This can be an user-defined param; normalized low and high for each variable
        clip_ranges = [[[0.0, 1.0]] * 5] * len(X_test)
        start_time = time.time()
        cf_samples4, losses4, _ = cf_model4.transform(
            X_test,
            desired_max_lst,
            desired_min_lst,
            clip_range_inputs=clip_ranges,
            hist_value_inputs=hist_inputs,
        )
        end_time = time.time()
        elapsed_time4 = end_time - start_time
        logger.info(f"Elapsed time - : {elapsed_time4:0.4f}.")

        ###############################################
        # ## 2.3 CF evaluation
        ###############################################
        # input_indices = range(0, back_horizon)
        # label_indices = range(back_horizon, back_horizon + horizon)
        cf_samples_lst = [cf_samples, cf_samples2, cf_samples3, cf_samples4]
        CF_MODEL_NAMES = [
            "ForecastCF",
            "ForecastCF-clip",
            "ForecastCF-meal",
            "ForecastCF-hist",
        ]

        for i in range(len(cf_samples_lst)):
            z_preds = forecast_model.predict(cf_samples_lst[i])

            (
                validity,
                proximity,
                compactness,
                cumsum_valid_steps,
                cumsum_counts,
                cumsum_auc,
                proximity_hist,
                compactness_hist,
            ) = cf_metrics(
                desired_max_lst=desired_max_lst,
                desired_min_lst=desired_min_lst,
                X_test=X_test,
                cf_samples=cf_samples_lst[i],
                z_preds=z_preds,
                change_idx=CHANGE_COLS,  # The change index during CF generation
                hist_inputs=hist_inputs,  # Input the historical input values
            )

            logger.info(f"Done for CF search: [[{CF_MODEL_NAMES[i]}]].")
            logger.info(f"validity: {validity}, step_validity_auc: {cumsum_auc}.")
            logger.info(f"valid_steps: {cumsum_valid_steps}, counts:{cumsum_counts}.")
            logger.info(f"proximity: {proximity}, compactness: {compactness}.")
            logger.info(
                f"proximity_hist:{proximity_hist}, compactness_hist:{compactness_hist}"
            )

            # TODO: use proper variable names in writing CSV
            pred_margin_weight = 0.9
            step_weights = "meal" if i in [2, 3] else "unconstrained"
            clip_mechanism = "yes" if i in [1, 2, 3] else "no"
            hist_values_input = "yes" if i in [3] else "no"
            result_writer.write_result(
                random_seed=RANDOM_STATE,
                method_name=model_name,
                test_group=A.test_group,
                horizon=horizon,
                forecast_smape=mean_smape,
                forecast_rmse=mean_rmse,
                pred_margin_weight=pred_margin_weight,
                step_weights=step_weights,
                clip_mechanism=clip_mechanism,
                hist_values_input=hist_values_input,
                validity_ratio=validity,
                proximity=proximity,
                compactness=compactness,
                step_validity_auc=cumsum_auc,
                proximity_hist=proximity_hist,
                compactness_hist=compactness_hist,
            )
    logger.info("Done.")


def build_tfts_model(model_name, back_horizon, horizon, n_in_features=1):
    import tfts

    inputs = tf.keras.layers.Input([back_horizon, n_in_features])
    if model_name == "wavenet":
        backbone = tfts.AutoModel(
            model_name,
            predict_length=horizon,
            custom_model_params={
                "filters": 256,
                "skip_connect_circle": True,
            },
        )
    elif model_name == "seq2seq":
        backbone = tfts.AutoModel(
            "seq2seq",
            predict_length=horizon,
            custom_model_params={"rnn_size": 256, "dense_size": 256},
        )
    else:
        print("Not implemented: build_tfts_model.")
    outputs = backbone(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="mae")

    return model


def polynomial_values(shift, change_percent, poly_order, horizon, desired_steps=None):
    """
    shift: e.g., +0.1 (110% of the start value)
    change_percent: e.g., 0.1 (10% increase)
    poly_order: e.g., order 1, or 2, ...
    horizon: the forecasting horizon
    desired_steps: the desired timesteps for the change_percent to finally happen (can be larger than horizon)
    """
    if horizon == 1:
        return np.asarray([shift + change_percent])
    desired_steps = desired_steps if desired_steps else horizon

    p_orders = [shift]  # intercept
    p_orders.extend([0 for i in range(poly_order)])
    p_orders[-1] = change_percent / ((desired_steps - 1) ** poly_order)

    p = np.polynomial.Polynomial(p_orders)
    p_coefs = list(reversed(p.coef))
    value_lst = np.asarray([np.polyval(p_coefs, i) for i in range(desired_steps)])

    return value_lst[:horizon]


def generate_bounds(
    center,
    shift,
    desired_center,
    poly_order,
    horizon,
    fraction_std,
    input_series,
    desired_steps,
):
    if center == "last":
        start_value = input_series[-1]
    elif center == "median":
        start_value = np.median(input_series)
    elif center == "mean":
        start_value = np.mean(input_series)
    elif center == "min":
        start_value = np.min(input_series)
    elif center == "max":
        start_value = np.max(input_series)
    else:
        print("Center: not implemented.")

    std = np.std(input_series)
    # Calculate the change_percent based on the desired center (in 2 hours)
    change_percent = (desired_center - start_value) / start_value
    # Create a default fluctuating range for the upper and lower bound if std is too small
    fluct_range = fraction_std * std if fraction_std * std >= 0.025 else 0.025
    upper = add_extra_dim(
        start_value
        * (
            1
            + polynomial_values(
                shift, change_percent, poly_order, horizon, desired_steps
            )
            + fluct_range
        )
    )
    lower = add_extra_dim(
        start_value
        * (
            1
            + polynomial_values(
                shift, change_percent, poly_order, horizon, desired_steps
            )
            - fluct_range
        )
    )

    return upper, lower


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
    main()
