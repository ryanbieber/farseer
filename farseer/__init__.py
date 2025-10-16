"""Farseer: Fast Bayesian time series forecasting

Python wrapper around the Rust-based Farseer library for time series forecasting.

Examples:
    >>> from farseer import Farseer
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     'ds': pl.date_range(datetime(2020, 1, 1), datetime(2020, 4, 9), interval='1d'),
    ...     'y': range(100)
    ... })
    >>> model = Farseer()
    >>> model.fit(df)
    >>> forecast = model.predict(model.make_future_dataframe(periods=30))

    # Pandas DataFrames are also supported (automatically converted to polars):
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'ds': pd.date_range('2020-01-01', periods=100),
    ...     'y': range(100)
    ... })
    >>> model = Farseer()
    >>> model.fit(df)  # Automatically converted to polars
"""

# This file provides Python-level enhancements to the Rust Farseer class
# The base Farseer class is defined in the Rust library (src/lib.rs)
# and compiled via PyO3/maturin

from typing import Union, Optional, List

# Try importing polars, it's required
try:
    import polars as pl
except ImportError:
    raise ImportError("polars is required. Install with: pip install polars")

# Try importing pandas, it's optional for backward compatibility
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

# Import the Rust module
# Maturin builds this as farseer._farseer or farseer.farseer depending on configuration
# We'll try both approaches for compatibility
try:
    # Try importing the Rust extension module directly
    from . import farseer as _rust_farseer

    _Farseer = _rust_farseer.Farseer
    __version__ = getattr(_rust_farseer, "__version__", "0.1.0")
except ImportError:
    try:
        # Try the alternative naming convention
        from . import _farseer as _rust_farseer

        _Farseer = _rust_farseer.Farseer
        __version__ = getattr(_rust_farseer, "__version__", "0.1.0")
    except ImportError as e:
        raise ImportError(
            f"Could not import farseer Rust module. Please build and install with 'maturin develop'. Error: {e}"
        ) from e

# Import utilities
from .utilities import regressor_coefficients


def _pandas_to_polars(df):
    """Convert pandas DataFrame to polars DataFrame"""
    if not HAS_PANDAS:
        raise TypeError("pandas is not installed, cannot convert pandas DataFrame")
    return pl.from_pandas(df)


def _polars_to_pandas(df: pl.DataFrame):
    """Convert polars DataFrame to pandas DataFrame"""
    if not HAS_PANDAS:
        raise ImportError("pandas is not installed, cannot convert to pandas DataFrame")
    result = df.to_pandas()

    # Ensure datetime columns use datetime64[ns] for consistency with Prophet
    if "ds" in result.columns and pd.api.types.is_datetime64_any_dtype(result["ds"]):
        result["ds"] = result["ds"].astype("datetime64[ns]")

    return result


def _ensure_polars(df) -> pl.DataFrame:
    """Ensure input is a polars DataFrame, converting from pandas if necessary"""
    if isinstance(df, pl.DataFrame):
        return df
    elif HAS_PANDAS and isinstance(df, pd.DataFrame):
        return _pandas_to_polars(df)
    else:
        raise TypeError(
            f"Expected polars.DataFrame or pandas.DataFrame, got {type(df)}"
        )


class Farseer(_Farseer):
    """
    Farseer forecaster with scikit-learn-like interface.

    Supports both polars and pandas DataFrames (pandas will be converted to polars internally).

    Parameters
    ----------
    growth : str, default 'linear'
        'linear', 'logistic', or 'flat' trend
    n_changepoints : int, default 25
        Number of potential changepoints
    changepoint_range : float, default 0.8
        Proportion of history in which trend changepoints will be estimated
    changepoint_prior_scale : float, default 0.05
        Regularization parameter for changepoints
    yearly_seasonality : bool, default True
        Fit yearly seasonality
    weekly_seasonality : bool, default True
        Fit weekly seasonality
    daily_seasonality : bool, default False
        Fit daily seasonality
    seasonality_mode : str, default 'additive'
        'additive' or 'multiplicative'
    interval_width : float, default 0.8
        Width of the uncertainty intervals (0.8 = 80%)

    Examples
    --------
    >>> from seer import Seer
    >>> import polars as pl
    >>>
    >>> # Create sample data with polars
    >>> df = pl.DataFrame({
    ...     'ds': pl.date_range(datetime(2020, 1, 1), datetime(2020, 4, 9), interval='1d'),
    ...     'y': range(100)
    ... })
    >>>
    >>> # Fit model
    >>> model = Farseer()
    >>> model.fit(df)
    >>>
    >>> # Make predictions
    >>> future = model.make_future_dataframe(periods=30)
    >>> forecast = model.predict(future)
    """

    def fit(self, df: Union[pl.DataFrame, "pd.DataFrame"], **kwargs) -> "Farseer":
        """
        Fit the Seer model.

        Parameters
        ----------
        df : polars.DataFrame or pandas.DataFrame
            DataFrame with columns 'ds' (date) and 'y' (value).
            Optional 'cap' column for logistic growth.
            If pandas DataFrame is provided, it will be converted to polars.

        Returns
        -------
        self : Farseer
            Fitted model
        """
        # Convert to polars if needed
        df_polars = _ensure_polars(df)

        # Validate input
        if "ds" not in df_polars.columns or "y" not in df_polars.columns:
            raise ValueError("DataFrame must have 'ds' and 'y' columns")

        # Make a copy to avoid modifying the original
        df_copy = df_polars.clone()

        # Ensure ds is datetime - handle both Date and Datetime types
        ds_dtype = df_copy["ds"].dtype
        if ds_dtype == pl.Date:
            # Convert Date to Datetime
            df_copy = df_copy.with_columns(pl.col("ds").cast(pl.Datetime))
        elif ds_dtype != pl.Datetime:
            # Try to parse as string
            df_copy = df_copy.with_columns(
                pl.col("ds").str.strptime(pl.Datetime, format="%Y-%m-%d")
            )

        # Convert to pandas for Rust interop (temporary until Rust supports polars directly)
        df_pandas = df_copy.to_pandas()

        # Ensure ds is datetime64[ns] for proper seasonality calculations
        # Polars uses nanosecond precision, pandas might not convert properly
        if "ds" in df_pandas.columns:
            df_pandas["ds"] = df_pandas["ds"].astype("datetime64[ns]")

        # Call Rust fit method (handles datetime conversion internally)
        super().fit(df_pandas)
        return self

    def predict(
        self, df: Union[pl.DataFrame, "pd.DataFrame", None] = None
    ) -> pl.DataFrame:
        """
        Predict using the Seer model.

        Parameters
        ----------
        df : polars.DataFrame, pandas.DataFrame, or None
            DataFrame with 'ds' column. If None, uses training data.
            If pandas DataFrame is provided, it will be converted to polars.

        Returns
        -------
        forecast : polars.DataFrame
            DataFrame with predictions. Output schema matches Facebook Prophet:
            - ds: dates
            - trend: trend component
            - yhat_lower: lower uncertainty bound for predictions
            - yhat_upper: upper uncertainty bound for predictions
            - trend_lower: lower uncertainty bound for trend
            - trend_upper: upper uncertainty bound for trend
            - additive_terms: sum of additive seasonal components
            - additive_terms_lower: lower uncertainty bound for additive terms
            - additive_terms_upper: upper uncertainty bound for additive terms
            - weekly: weekly seasonality (zeros if disabled)
            - weekly_lower: lower uncertainty bound for weekly
            - weekly_upper: upper uncertainty bound for weekly
            - yearly: yearly seasonality (zeros if disabled)
            - yearly_lower: lower uncertainty bound for yearly
            - yearly_upper: upper uncertainty bound for yearly
            - multiplicative_terms: sum of multiplicative seasonal components
            - multiplicative_terms_lower: lower uncertainty bound for multiplicative terms
            - multiplicative_terms_upper: upper uncertainty bound for multiplicative terms
            - yhat: final predicted values
        """
        if df is None:
            # Call Rust predict with None
            forecast_pandas = super().predict(None)
        else:
            # Convert to polars if needed
            df_polars = _ensure_polars(df)

            # Make a copy to avoid modifying the original
            df_copy = df_polars.clone()

            # Ensure ds is datetime - handle both Date and Datetime types
            ds_dtype = df_copy["ds"].dtype
            if ds_dtype == pl.Date:
                # Convert Date to Datetime
                df_copy = df_copy.with_columns(pl.col("ds").cast(pl.Datetime))
            elif ds_dtype != pl.Datetime:
                # Try to parse as string
                df_copy = df_copy.with_columns(
                    pl.col("ds").str.strptime(pl.Datetime, format="%Y-%m-%d")
                )

            # Convert to pandas for Rust interop (temporary)
            df_pandas = df_copy.to_pandas()

            # Ensure ds is datetime64[ns] for proper seasonality calculations
            # Polars uses nanosecond precision, pandas might not convert properly
            if "ds" in df_pandas.columns:
                df_pandas["ds"] = df_pandas["ds"].astype("datetime64[ns]")

            # Call Rust predict method (handles datetime conversion internally)
            forecast_pandas = super().predict(df_pandas)

        # Convert result back to polars
        forecast_polars = pl.from_pandas(forecast_pandas)

        # Ensure ds is datetime with nanosecond precision for compatibility with Prophet
        # and proper conversion to pandas datetime64[ns]
        ds_dtype = forecast_polars["ds"].dtype
        if ds_dtype == pl.String:
            # Rust returned strings, need to parse them
            # Try date-only format first (most common case)
            try:
                forecast_polars = forecast_polars.with_columns(
                    pl.col("ds").str.strptime(
                        pl.Datetime("ns"), format="%Y-%m-%d", strict=False
                    )
                )
            except Exception:
                # If that fails, try with time component
                forecast_polars = forecast_polars.with_columns(
                    pl.col("ds").str.strptime(
                        pl.Datetime("ns"), format="%Y-%m-%d %H:%M:%S", strict=False
                    )
                )
        elif ds_dtype != pl.Datetime("ns"):
            # Not string or nanosecond datetime, cast to nanosecond precision
            # This ensures proper conversion to pandas datetime64[ns]
            forecast_polars = forecast_polars.with_columns(
                pl.col("ds").cast(pl.Datetime("ns"))
            )

        return forecast_polars

    def make_future_dataframe(
        self, periods: int, freq: str = "D", include_history: bool = True
    ) -> pl.DataFrame:
        """
        Create a dataframe for future predictions.

        Parameters
        ----------
        periods : int
            Number of periods to forecast
        freq : str, default 'D'
            Frequency string: 'D' for daily, 'H' for hourly, etc.
        include_history : bool, default True
            Whether to include historical dates

        Returns
        -------
        future : polars.DataFrame
            DataFrame with 'ds' column for future dates
        """
        # Call Rust method
        future_pandas = super().make_future_dataframe(periods, freq, include_history)

        # Convert to polars
        future_polars = pl.from_pandas(future_pandas)

        # Ensure ds is datetime with nanosecond precision for compatibility with Prophet
        future_polars = future_polars.with_columns(pl.col("ds").cast(pl.Datetime("ns")))

        return future_polars

    def add_seasonality(
        self,
        name: str,
        period: float,
        fourier_order: int,
        prior_scale: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> "Farseer":
        """
        Add a custom seasonality component.

        Parameters
        ----------
        name : str
            Name of the seasonality component
        period : float
            Period of the seasonality in days (e.g., 30.5 for monthly)
        fourier_order : int
            Number of Fourier components to use
        prior_scale : float, optional
            Regularization parameter (default: 10.0)
        mode : str, optional
            'additive' or 'multiplicative' (default: model's seasonality_mode)

        Returns
        -------
        self : Farseer
            Model instance for method chaining
        """
        super().add_seasonality(name, period, fourier_order, prior_scale, mode)
        return self

    def add_holidays(
        self,
        name: str,
        dates: List[str],
        lower_window: Optional[int] = None,
        upper_window: Optional[int] = None,
        prior_scale: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> "Farseer":
        """
        Add custom holiday effects.

        Parameters
        ----------
        name : str
            Name of the holiday
        dates : list of str
            List of holiday dates in 'YYYY-MM-DD' format
        lower_window : int, optional
            Days before the holiday to include (default: 0)
        upper_window : int, optional
            Days after the holiday to include (default: 0)
        prior_scale : float, optional
            Regularization parameter (default: 10.0)
        mode : str, optional
            'additive' or 'multiplicative' (default: 'additive')

        Returns
        -------
        self : Farseer
            Model instance for method chaining

        Examples
        --------
        >>> model = Farseer()
        >>> model.add_holidays(
        ...     'christmas',
        ...     ['2020-12-25', '2021-12-25', '2022-12-25'],
        ...     lower_window=-2,
        ...     upper_window=2
        ... )
        """
        super().add_holidays(name, dates, lower_window, upper_window, prior_scale, mode)
        return self

    def add_country_holidays(self, country_name: str) -> "Farseer":
        """
        Add country-specific holidays.

        Parameters
        ----------
        country_name : str
            Name of the country (e.g., 'US', 'UK', 'CA')

        Returns
        -------
        self : Farseer
            Model instance for method chaining

        Examples
        --------
        >>> model = Farseer()
        >>> model.add_country_holidays('US')
        """
        super().add_country_holidays(country_name)
        return self

    def save(self, path: str) -> None:
        """
        Save the model to a JSON file.

        Parameters
        ----------
        path : str
            Path to save the model
        """
        json_str = self.to_json()
        with open(path, "w") as f:
            f.write(json_str)

    @classmethod
    def load(cls, path: str) -> "Farseer":
        """
        Load a model from a JSON file.

        Parameters
        ----------
        path : str
            Path to the saved model

        Returns
        -------
        model : Farseer
            Loaded model instance
        """
        with open(path, "r") as f:
            json_str = f.read()
        return cls.from_json(json_str)

    def plot(
        self,
        forecast: Union[pl.DataFrame, "pd.DataFrame"],
        ax=None,
        history: Optional[Union[pl.DataFrame, "pd.DataFrame"]] = None,
        **kwargs,
    ):
        """
        Plot the forecast with matplotlib.

        Parameters
        ----------
        forecast : polars.DataFrame or pandas.DataFrame
            Forecast dataframe from predict()
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure
        history : polars.DataFrame or pandas.DataFrame, optional
            Historical data to plot
        **kwargs : dict
            Additional arguments passed to matplotlib

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object
        """
        import matplotlib.pyplot as plt

        # Convert to pandas for plotting
        if isinstance(forecast, pl.DataFrame):
            forecast_pd = forecast.to_pandas()
        else:
            forecast_pd = forecast

        if history is not None:
            if isinstance(history, pl.DataFrame):
                history_pd = history.to_pandas()
            else:
                history_pd = history
        else:
            history_pd = None

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot historical data if provided
        if history_pd is not None and "y" in history_pd.columns:
            ax.plot(
                history_pd["ds"], history_pd["y"], "k.", label="Observed", markersize=4
            )

        # Plot forecast
        ax.plot(
            forecast_pd["ds"],
            forecast_pd["yhat"],
            label="Forecast",
            color="blue",
            linewidth=2,
        )
        ax.fill_between(
            forecast_pd["ds"],
            forecast_pd["yhat_lower"],
            forecast_pd["yhat_upper"],
            alpha=0.2,
            color="blue",
            label="Uncertainty interval",
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return ax

    def plot_components(
        self, forecast: Union[pl.DataFrame, "pd.DataFrame"], figsize=(10, 8)
    ):
        """
        Plot the forecast components (trend, seasonalities).

        Parameters
        ----------
        forecast : polars.DataFrame or pandas.DataFrame
            Forecast dataframe from predict()
        figsize : tuple, optional
            Figure size (width, height)

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        import matplotlib.pyplot as plt

        # Convert to pandas for plotting
        if isinstance(forecast, pl.DataFrame):
            forecast_pd = forecast.to_pandas()
        else:
            forecast_pd = forecast

        # Determine which components to plot
        components = []
        if "trend" in forecast_pd.columns:
            components.append(("trend", "Trend"))
        if "yearly" in forecast_pd.columns:
            components.append(("yearly", "Yearly Seasonality"))
        if "weekly" in forecast_pd.columns:
            components.append(("weekly", "Weekly Seasonality"))

        n_components = len(components)
        if n_components == 0:
            raise ValueError("No components found in forecast")

        fig, axes = plt.subplots(n_components, 1, figsize=figsize)

        if n_components == 1:
            axes = [axes]

        for ax, (col, title) in zip(axes, components):
            ax.plot(forecast_pd["ds"], forecast_pd[col], linewidth=1.5)
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date")
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()

        return fig


__all__ = ["Farseer", "regressor_coefficients", "__version__"]
