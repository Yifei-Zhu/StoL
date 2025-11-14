import math

def get_lambda_gp_schedule(schedule, total_time):
    """
    Returns the specified lambda_gp adjustment function.

    Parameters:
    - schedule (dict): Configuration dictionary loaded from a YAML file, including:
        - enable (bool): Whether to enable gradient penalty (default: False).
        - type (str): Adjustment strategy type. Options:
            - "linear": Linear increase
            - "exponential": Exponential growth
            - "cosine": Cosine annealing
            - "fixed": Fixed value
        - warmup_time (int): Warmup steps before applying gradient penalty (default: 10000).
        - total_time (int): Total training steps (used for "linear" and "cosine", default: 50000).
        - lambda_gp_final (float): Final gradient penalty weight (default: 10.0).
        - scale (int): Growth scale for exponential strategy (default: 5000).
        - gp_ealy_stop (dict): Gradient penalty early stop configuration, including:
            - enable (bool): Whether to enable early stopping logic (default: False).
            - stop_condition_steps (int): Number of steps to check for stability (default: 20000).
            - stop_gradient_threshold (float): Gradient stability threshold (default: 1e-3).

    Returns:
    - A function that takes the current training step and gradient norm as inputs
      and returns the lambda_gp value.
    """
    enable_gradient_penalty = schedule.get("enable", False)
    if not enable_gradient_penalty:
        # If gradient penalty is disabled, return a function that always returns 0.0
        def lambda_gp_disabled(current_step, grad_norm=None):
            return 0.0
        return lambda_gp_disabled

    schedule_type = schedule.get("type", "linear")
    warmup_time = schedule.get("warmup_time", 10000)
    lambda_gp_final = schedule.get("lambda_gp_final", 10.0)
    scale = schedule.get("scale", 5000)

    # Early stop configuration
    gp_ealy_stop = schedule.get("gp_ealy_stop", {})
    enable_early_stop = gp_ealy_stop.get("enable", False)
    stop_condition_steps = gp_ealy_stop.get("stop_condition_steps", 20000)
    stop_gradient_threshold = gp_ealy_stop.get("stop_gradient_threshold", 1e-3)

    # Gradient history for early stopping
    gradient_history = []

    def should_stop_gradient_penalty():
        # Only apply early stop logic if enabled
        if not enable_early_stop:
            return False
        # Check if recent gradient changes are below the threshold
        if len(gradient_history) < stop_condition_steps:
            return False
        recent_changes = gradient_history[-stop_condition_steps:]
        return all(change < stop_gradient_threshold for change in recent_changes)

    if schedule_type == "linear":
        def lambda_gp_linear(current_step, grad_norm=None):
            if grad_norm is not None:
                gradient_history.append(abs(grad_norm))
            if should_stop_gradient_penalty():
                return 0.0
            if current_step < warmup_time:
                return 0.0
            progress = (current_step - warmup_time) / (total_time - warmup_time)
            return lambda_gp_final * min(progress, 1.0)
        return lambda_gp_linear

    elif schedule_type == "exponential":
        def lambda_gp_exponential(current_step, grad_norm=None):
            if grad_norm is not None:
                gradient_history.append(abs(grad_norm))
            if should_stop_gradient_penalty():
                return 0.0
            if current_step < warmup_time:
                return 0.0
            effective_step = current_step - warmup_time
            return lambda_gp_final * (1 - math.exp(-effective_step / scale))
        return lambda_gp_exponential

    elif schedule_type == "cosine":
        def lambda_gp_cosine(current_step, grad_norm=None):
            if grad_norm is not None:
                gradient_history.append(abs(grad_norm))
            if should_stop_gradient_penalty():
                return 0.0
            if current_step < warmup_time:
                return 0.0
            progress = (current_step - warmup_time) / (total_time - warmup_time)
            return lambda_gp_final * 0.5 * (1 + math.cos(math.pi * (1 - min(progress, 1.0))))
        return lambda_gp_cosine

    elif schedule_type == "fixed":
        def lambda_gp_fixed(current_step, grad_norm=None):
            if grad_norm is not None:
                gradient_history.append(abs(grad_norm))
            if should_stop_gradient_penalty():
                return 0.0
            if current_step < warmup_time:
                return 0.0
            return lambda_gp_final
        return lambda_gp_fixed

    else:
        raise ValueError(f"Unsupported schedule type: {schedule_type}")
