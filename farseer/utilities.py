"""Seer utility functions for working with fitted models

This module provides utility functions similar to Facebook Prophet's utilities.
"""

import pandas as pd


def regressor_coefficients(m):
    """Extract regressor coefficients from a fitted model.

    For additive regressors, the coefficient represents the incremental impact
    on `y` of a unit increase in the regressor. For multiplicative regressors,
    the incremental impact is equal to `trend(t)` multiplied by the coefficient.

    Coefficients are measured on the original scale of the training data.

    Parameters
    ----------
    m : Seer
        Fitted Seer model object

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - regressor: Name of the regressor
        - regressor_mode: Whether the regressor is additive or multiplicative
        - center: The mean of the regressor if it was standardized (otherwise 0)
        - std: The std dev of the regressor if it was standardized (otherwise 1)
        - coef: Coefficient value on the original scale
        - prior_scale: Prior scale used for this regressor

    Raises
    ------
    AssertionError
        If no regressors are found in the model

    Examples
    --------
    >>> from seer import Seer
    >>> from seer.utilities import regressor_coefficients
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create data with regressors
    >>> df = pd.DataFrame({
    ...     'ds': pd.date_range('2020-01-01', periods=100),
    ...     'y': np.random.randn(100) + 10,
    ...     'regressor1': np.random.randn(100),
    ...     'regressor2': np.random.randn(100)
    ... })
    >>>
    >>> # Fit model with regressors
    >>> m = Seer()
    >>> m.add_regressor('regressor1')
    >>> m.add_regressor('regressor2', mode='multiplicative')
    >>> m.fit(df)
    >>>
    >>> # Extract coefficients
    >>> coefs = regressor_coefficients(m)
    >>> print(coefs)
    """
    # Get model parameters
    params = m.params()

    # Check if model has regressors
    if "regressors" not in params or len(params["regressors"]) == 0:
        raise AssertionError("No extra regressors found.")

    # Extract regressor information
    regressors = params["regressors"]
    beta = params.get("beta", [])
    regressor_blocks = params.get("regressor_blocks", [])
    y_scale = params.get("y_scale", 1.0)

    coefs = []

    for regr_info in regressors:
        name = regr_info["name"]
        mode = regr_info["mode"]
        mu = regr_info.get("mu", 0.0)
        std = regr_info.get("std", 1.0)
        prior_scale = regr_info.get("prior_scale", 10.0)

        # Find the beta coefficient index for this regressor
        # Look for matching regressor block
        beta_idx = None
        for block in regressor_blocks:
            if block["name"] == name:
                beta_idx = block["start"]
                break

        if beta_idx is not None and beta_idx < len(beta):
            beta_val = beta[beta_idx]

            # Scale coefficient back to original data scale
            # For additive: coef = beta * y_scale / std
            # For multiplicative: coef = beta / std
            if mode.lower() == "additive":
                coef = beta_val * y_scale / std
            else:
                coef = beta_val / std

            coefs.append(
                {
                    "regressor": name,
                    "regressor_mode": mode,
                    "center": mu,
                    "std": std,
                    "coef": coef,
                    "prior_scale": prior_scale,
                }
            )

    return pd.DataFrame(coefs)
