import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors


class BGForecastCF:
    """Explanations by generating a counterfacutal sample for a desired forecasting outcome.
    References
    ----------
    Counterfactual Explanations for Time Series Forecasting,
    Wang, Z., Miliou, I., Samsten, I., Papapetrou, P., 2023.
    in: International Conference on Data Mining (ICDM 2023)
    """

    def __init__(
        self,
        *,
        tolerance=1e-6,
        max_iter=100,
        optimizer=None,
        pred_margin_weight=1.0,  # weighted_steps_weight = 1 - pred_margin_weight
        step_weights="local",
        target_col=0,
        # `only_change_idx` param should be a list of idx, e.g., [2,3]; default None, which uses all the rest except target_col
        only_change_idx=None,
        # only_change_idx=[2, 3],
        random_state=None,
    ):
        """
        Parameters
        ----------
        probability : float, optional
            The desired probability assigned by the model
        tolerance : float, optional
            The maximum difference between the desired and assigned probability
        optimizer :
            Optimizer with a defined learning rate
        max_iter : int, optional
            The maximum number of iterations
        """
        self.optimizer_ = (
            tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
            if optimizer is None
            else optimizer
        )
        self.mse_loss_ = tf.keras.losses.MeanSquaredError()
        self.tolerance_ = tf.constant(tolerance)
        self.max_iter = max_iter

        # Weights of the different loss components
        self.pred_margin_weight = pred_margin_weight
        self.weighted_steps_weight = 1 - self.pred_margin_weight

        self.step_weights = step_weights
        self.random_state = random_state

        self.target_col = target_col
        # default value for only_change_idx, [1]
        self.only_change_idx = only_change_idx

        self.MISSING_MAX_BOUND = np.inf
        self.MISSING_MIN_BOUND = -np.inf

    def fit(self, model):
        """Fit a new counterfactual explainer to the model parameters
        ----------
        model : keras.Model
            The model
        """
        self.model_ = model
        return self

    def predict(self, x):
        """Compute the difference between the desired and actual forecasting predictions
        ---------
        x : Variable
            Variable of the sample
        """

        return self.model_(x)

    # The "forecast_margin_loss" is designed to measure the prediction probability to the desired decision boundary
    def margin_mse(self, prediction, max_bound, min_bound):
        masking_vector = tf.logical_not(
            tf.logical_and(prediction <= max_bound, prediction >= min_bound)
        )
        unmasked_preds = tf.boolean_mask(prediction, masking_vector)

        if unmasked_preds.shape == 0:
            return 0

        mse_loss_ = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.SUM
        )

        if tf.reduce_any(max_bound != self.MISSING_MAX_BOUND):
            dist_max = mse_loss_(max_bound, unmasked_preds)
        else:
            dist_max = 0

        if tf.reduce_any(min_bound != self.MISSING_MIN_BOUND):
            dist_min = mse_loss_(min_bound, unmasked_preds)
        else:
            dist_min = 0

        return dist_max + dist_min

    # An auxiliary weighted MAPE loss function to measure the proximity with step_weights
    def weighted_ape(
        self,
        original_per_feature,
        cf_per_feature,
        step_weights,
    ):
        # note: the output can be arbitrarily high when `original_per_feature` is small (which is specific to the metric)
        # src: https://github.com/scikit-learn/scikit-learn/blob/80598905e517759b4696c74ecc35c6e2eb508cff/sklearn/metrics/_regression.py#L296
        denum = tf.math.maximum(
            tf.math.abs(original_per_feature), tf.keras.backend.epsilon()
        )
        ape_score = tf.math.abs(original_per_feature - cf_per_feature) / denum
        weighted_ape = tf.math.multiply(ape_score, step_weights)
        return tf.math.reduce_mean(weighted_ape)

    # additional input of step_weights
    def compute_loss(
        self,
        original_sample,
        z_search,
        step_weights,
        max_bound,
        min_bound,
        n_iter=None,
    ):
        loss = tf.zeros(shape=())
        pred = self.model_(z_search)

        forecast_margin_loss = self.margin_mse(pred, max_bound, min_bound)
        loss += self.pred_margin_weight * forecast_margin_loss

        # weighted_ape for each changeable variable
        for z_idx in self.z_change_idx:
            weighted_steps_loss = self.weighted_ape(
                tf.cast(original_sample[:, :, z_idx], tf.float32),
                tf.cast(z_search[:, :, z_idx], tf.float32),
                tf.cast(step_weights[:, :, z_idx], tf.float32),
            )
            loss += self.weighted_steps_weight * weighted_steps_loss

        return loss, forecast_margin_loss, weighted_steps_loss

    def transform(
        self,
        x,
        max_bound_lst=None,
        min_bound_lst=None,
        clip_range_inputs=None,
        hist_value_inputs=None,
    ):
        """Generate counterfactual explanations
        x : array-like of shape [n_samples, n_timestep, n_dims]
            The samples
        """
        # TODO: make the parameter check more properly
        try:
            print(
                f"Validating threshold input: {len(max_bound_lst)==x.shape[0] or len(min_bound_lst)==x.shape[0]}"
            )
        except:
            print("Wrong parameter inputs, at least one threshold should be provided.")

        result_samples = np.empty(x.shape)
        losses = np.empty(x.shape[0])
        # `weights_all` needed for debugging
        weights_all = np.empty((x.shape[0], 1, x.shape[1], x.shape[2]))

        for i in range(x.shape[0]):
            # if i % 25 == 0:
            print(f"{i} samples been transformed.")

            x_sample = np.expand_dims(x[i], axis=0)
            if self.step_weights == "unconstrained":
                step_weights = np.zeros(x_sample.shape)
            elif self.step_weights == "uniform":
                step_weights = np.ones(x_sample.shape)
            elif self.step_weights in ["meal", "meal_time"]:
                step_weights = get_meal_weights(x_sample)
            # if self defined arrays as input
            elif isinstance(self.step_weights, np.ndarray):
                step_weights = self.step_weights
            else:
                raise NotImplementedError(
                    "step_weights not implemented, please choose 'unconstrained', 'meal_time' or 'uniform'."
                )

            # Check the condition of desired CF: upper and lower bound
            max_bound = (
                max_bound_lst[i] if max_bound_lst != None else self.MISSING_MAX_BOUND
            )
            min_bound = (
                min_bound_lst[i] if min_bound_lst != None else self.MISSING_MIN_BOUND
            )
            clip_ranges = clip_range_inputs[i] if clip_range_inputs else None
            print(f"clip_ranges:{clip_ranges is not None}")
            hist_values = hist_value_inputs[i] if hist_value_inputs else None
            print(f"hist_values:{hist_values is not None}")
            cf_sample, loss = self._transform_sample(
                x_sample, step_weights, max_bound, min_bound, clip_ranges, hist_values
            )

            result_samples[i] = cf_sample
            losses[i] = loss
            weights_all[i, :, :, :] = step_weights

        return result_samples, losses, weights_all

    def _transform_sample(
        self, x, step_weights, max_bound, min_bound, clip_ranges=None, hist_input=None
    ):
        """Generate counterfactual explanations
        x : array-like of shape [n_samples, n_timestep, n_dims]
            The samples
        """
        # split z_variables into z_static_idx + z_change_idx
        z_variables = list()
        if self.target_col is not None and not self.only_change_idx:
            z_static_idx = [self.target_col]
            z_change_idx = [i for i in list(range(x.shape[2])) if i not in z_static_idx]
        elif self.only_change_idx:
            # only_change_idx should be a list of idx, e.g., [2,3], TODO: add validate() function
            z_change_idx = self.only_change_idx
            z_static_idx = [i for i in list(range(x.shape[2])) if i not in z_change_idx]
        else:
            print("Not implemented, target_col/only_change_idx.")

        print(f"z_change_idx, z_static_idx:{z_change_idx, z_static_idx}")
        self.z_change_idx, self.z_static_idx = z_change_idx, z_static_idx
        # iterate over all features to create tf variables (with z_change_idx + z_static_idx)
        for dim in range(x.shape[2]):
            if clip_ranges:
                clip_low, clip_high = clip_ranges[dim]
                z = tf.Variable(
                    np.expand_dims(x[:, :, dim], axis=2),
                    dtype=tf.float32,
                    # Extra clipping step -> project constraints after apply_gradients()
                    constraint=lambda x: tf.clip_by_value(x, clip_low, clip_high),
                    name="var" + str(dim),
                )
            else:
                z = tf.Variable(
                    np.expand_dims(x[:, :, dim], axis=2),
                    dtype=tf.float32,
                    name="var" + str(dim),
                )
            z_variables.append(z)

        it = 0
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch([z_variables[i] for i in self.z_change_idx])
            loss, forecast_margin_loss, weighted_steps_loss = self.compute_loss(
                x,
                tf.concat(z_variables, axis=2),
                step_weights,
                max_bound,
                min_bound,
                n_iter=it,
            )
        print(f"watched variables:{[var.name for var in tape.watched_variables()]}")

        pred = self.model_(tf.concat(z_variables, axis=2))

        # uncomment for debug
        print(
            f"iter: {it}, current loss: {loss}, forecast_margin_loss: {forecast_margin_loss}, weighted_steps_loss: {weighted_steps_loss} \ndesired range: {min_bound, max_bound} \npred:{tf.reshape(pred, [-1])} \n",
        )
        while (tf.reduce_any(pred > max_bound) or tf.reduce_any(pred < min_bound)) and (
            it < self.max_iter if self.max_iter else True
        ):
            # Get gradients of loss wrt the sample
            z_change_vars = [z_variables[i] for i in self.z_change_idx]
            grads = tape.gradient(loss, z_change_vars)

            # Update the weights of the sample; one grad update per feature, for z_each in z_variables
            self.optimizer_.apply_gradients(zip(grads, z_change_vars))

            # # historical_input_constraints mechanism:
            if hist_input:
                threshold_close = 0.001
                for z_idx in self.z_change_idx:
                    for step in range(z_variables[z_idx].shape[1]):
                        min_dist = tf.reduce_min(
                            tf.abs(z_variables[z_idx][:, step, :] - hist_input[z_idx])
                        )
                        if min_dist <= threshold_close:
                            step_weights[:, step, z_idx] = 1

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch([z_variables[i] for i in self.z_change_idx])
                loss, forecast_margin_loss, weighted_steps_loss = self.compute_loss(
                    x,
                    tf.concat(z_variables, axis=2),
                    step_weights,
                    max_bound,
                    min_bound,
                    n_iter=it,
                )
            it += 1

            pred = self.model_(tf.concat(z_variables, axis=2))

        # uncomment for debug
        print(
            f"iter: {it}, current loss: {loss}, forecast_margin_loss: {forecast_margin_loss}, weighted_steps_loss: {weighted_steps_loss} \ndesired range: {min_bound, max_bound} \npred:{tf.reshape(pred, [-1])} \n",
        )

        res = tf.concat(z_variables, axis=2).numpy()
        return res, float(loss)


def get_meal_weights(x_sample, activity_threshold=0):
    # for all the variables in x_sample => 0 - weights for all positive values (i.e., larger than the threshold); more effective for bolus insulin and carbs intake

    # custom_step_weights has the same dimension as all the input variables (index needed);
    # but then only the weights for `z_change_idx` will be called
    custom_step_weights = (
        np.asarray(x_sample <= activity_threshold, dtype=np.float32) * 1
    )
    return custom_step_weights
