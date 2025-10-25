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

from typing import Union, Optional, List, Dict
import warnings
from multiprocessing import Pool, cpu_count

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
    n_changepoints : int, default 15
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
        condition_name: Optional[str] = None,
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
        condition_name : str, optional
            Column name of a boolean in your dataframe which  specifies when this
            seasonality is active. When the column is True, the seasonality is
            applied; when False, it is not. This allows seasonality to be
            conditional on arbitrary features.

        Returns
        -------
        self : Farseer
            Model instance for method chaining
        """
        super().add_seasonality(
            name, period, fourier_order, prior_scale, mode, condition_name
        )
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
    def from_json(cls, json_str: str) -> "Farseer":
        """
        Deserialize model from JSON string.

        Parameters
        ----------
        json_str : str
            JSON string representation of the model

        Returns
        -------
        model : Farseer
            Deserialized model instance
        """
        # NOTE: Due to PyO3 limitations, the loaded model will be a Rust Farseer object,
        # not the Python wrapper. This means Polars DataFrame support is not available
        # for loaded models - you must use pandas DataFrames.
        # The core functionality (fit/predict) works identically.
        return _Farseer.from_json(json_str)

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


# Multi-series support with multiprocessing
def _fit_single_series(args):
    """
    Helper function to fit a single series in a subprocess.
    Returns (series_id, model_json, df_for_future, error) tuple.

    This function is at module level to support multiprocessing pickling.
    We return the training dataframe so we can recreate the model with full context.
    """
    series_id, df, model_params = args

    try:
        # Create a new model instance with the same parameters
        model = Farseer(**model_params)

        # Fit the model
        model.fit(df)

        # Serialize to JSON for pickling
        model_json = model.to_json()

        # Also return the df (converted to dict for pickling)
        df_dict = df.to_pandas().to_dict(orient="list")

        return (series_id, model_json, df_dict, None)
    except Exception as e:
        # Return the error instead of raising it
        return (series_id, None, None, str(e))


def _predict_single_series(args):
    """
    Helper function to predict for a single series in a subprocess.
    Returns (series_id, forecast, error) tuple.

    This function is at module level to support multiprocessing pickling.
    """
    series_id, model_json, future_df = args

    try:
        # Deserialize the model from JSON
        # Note: from_json returns Rust Farseer, not Python wrapper
        # So we need to convert Polars to Pandas
        model = Farseer.from_json(model_json)

        # Convert future_df to pandas for loaded model compatibility
        if isinstance(future_df, pl.DataFrame):
            future_pd = future_df.to_pandas()
        else:
            future_pd = future_df

        # Make predictions (returns pandas DataFrame from loaded model)
        forecast_pd = model.predict(future_pd)

        # Convert back to Polars
        forecast = pl.from_pandas(forecast_pd)

        # Add series_id column to forecast
        forecast = forecast.with_columns(pl.lit(series_id).alias("series_id"))

        return (series_id, forecast, None)
    except Exception as e:
        # Return the error instead of raising it
        import traceback

        return (series_id, None, str(e) + "\n" + traceback.format_exc())


class FarseerMultiSeries:
    """
    Multi-series forecasting with parallel fitting using multiprocessing.

    This class handles multiple time series identified by a 'series_id' column,
    fitting each series in parallel using separate processes to avoid GIL
    limitations with Stan optimization.

    Parameters
    ----------
    n_processes : int, optional
        Number of parallel processes to use. If None, uses cpu_count().
    **model_params
        Parameters to pass to each Farseer model instance.

    Examples
    --------
    >>> from farseer import FarseerMultiSeries
    >>> import polars as pl
    >>>
    >>> # Create multi-series data
    >>> df = pl.DataFrame({
    ...     'series_id': ['A', 'A', 'A', 'B', 'B', 'B'],
    ...     'ds': pl.date_range(datetime(2020, 1, 1), datetime(2020, 1, 6), interval='1d'),
    ...     'y': [1, 2, 3, 4, 5, 6]
    ... })
    >>>
    >>> # Fit all series in parallel
    >>> multi_model = FarseerMultiSeries(n_processes=2)
    >>> results = multi_model.fit(df, series_col='series_id')
    >>>
    >>> # Check for errors
    >>> if results['errors']:
    ...     print(f"Failed series: {list(results['errors'].keys())}")
    >>>
    >>> # Make predictions for all successful series
    >>> forecasts = multi_model.predict(periods=10)
    """

    def __init__(self, n_processes: Optional[int] = None, **model_params):
        """
        Initialize the multi-series forecaster.

        Parameters
        ----------
        n_processes : int, optional
            Number of parallel processes. Defaults to cpu_count().
        **model_params
            Parameters passed to each Farseer instance (growth, seasonality, etc.)
        """
        self.n_processes = n_processes or cpu_count()
        self.model_params = model_params
        self.models: Dict[str, Farseer] = {}
        self.model_jsons: Dict[str, str] = {}  # Store JSON for serialization
        self.errors: Dict[str, str] = {}
        self.series_col = "series_id"
        self._input_was_pandas = False  # Track input type to return same type

    def fit(
        self,
        df: Union[pl.DataFrame, "pd.DataFrame"],
        series_col: str = "series_id",
    ) -> Dict[str, Union[Dict[str, Farseer], Dict[str, str]]]:
        """
        Fit models for all series in parallel.

        Parameters
        ----------
        df : polars.DataFrame or pandas.DataFrame
            DataFrame with columns 'ds', 'y', and series_col.
        series_col : str, default 'series_id'
            Name of the column containing series identifiers.

        Returns
        -------
        results : dict
            Dictionary with keys:
            - 'models': Dict[str, Farseer] - Successfully fitted models by series_id
            - 'errors': Dict[str, str] - Error messages by series_id for failed fits
            - 'n_success': int - Number of successfully fitted series
            - 'n_failed': int - Number of failed series

        Examples
        --------
        >>> results = multi_model.fit(df, series_col='store_id')
        >>> print(f"Fitted {results['n_success']} series successfully")
        >>> if results['errors']:
        ...     for series_id, error in results['errors'].items():
        ...         print(f"Series {series_id} failed: {error}")
        """
        # Track input type to return same type later
        self._input_was_pandas = HAS_PANDAS and isinstance(df, pd.DataFrame)

        # Convert to polars if needed
        df_polars = _ensure_polars(df)
        self.series_col = series_col

        # Validate required columns
        if series_col not in df_polars.columns:
            raise ValueError(f"Column '{series_col}' not found in DataFrame")
        if "ds" not in df_polars.columns or "y" not in df_polars.columns:
            raise ValueError("DataFrame must have 'ds' and 'y' columns")

        # Get unique series
        series_ids = df_polars[series_col].unique().to_list()

        # Split data by series
        series_data = []
        for series_id in series_ids:
            series_df = df_polars.filter(pl.col(series_col) == series_id)
            # Drop the series_id column before fitting
            series_df = series_df.drop(series_col)
            series_data.append((series_id, series_df, self.model_params))

        # Fit models in parallel using multiprocessing
        with Pool(processes=self.n_processes) as pool:
            results = pool.map(_fit_single_series, series_data)

        # Separate successful models from errors
        self.models = {}
        self.model_jsons = {}
        self.errors = {}

        for series_id, model_json, df_dict, error in results:
            if error is None:
                # Recreate the model from JSON
                model = Farseer.from_json(model_json)
                # Re-fit with the original data to restore full Python wrapper state
                # Convert df_dict back to DataFrame
                if HAS_PANDAS:
                    df_restored = pl.from_pandas(pd.DataFrame(df_dict))
                else:
                    # If pandas not available, create from dict directly
                    df_restored = pl.DataFrame(df_dict)

                # Create fresh model and fit
                model = Farseer(**self.model_params)
                model.fit(df_restored)

                self.models[series_id] = model
                # Also store the JSON for later use in predict
                self.model_jsons[series_id] = model_json
            else:
                self.errors[series_id] = error

        # Print summary
        n_success = len(self.models)
        n_failed = len(self.errors)

        if n_failed > 0:
            warnings.warn(
                f"Fitted {n_success}/{len(series_ids)} series successfully. "
                f"{n_failed} series failed. Check results['errors'] for details.",
                UserWarning,
            )

        return {
            "models": self.models,
            "errors": self.errors,
            "n_success": n_success,
            "n_failed": n_failed,
        }

    def predict(
        self,
        periods: Optional[int] = None,
        future_df: Optional[Union[pl.DataFrame, "pd.DataFrame"]] = None,
        freq: str = "D",
        include_history: bool = True,
    ) -> Dict[str, Union[pl.DataFrame, Dict[str, str]]]:
        """
        Make predictions for all successfully fitted series in parallel.

        Parameters
        ----------
        periods : int, optional
            Number of periods to forecast. Required if future_df is None.
        future_df : polars.DataFrame or pandas.DataFrame, optional
            DataFrame with 'ds' and series_col columns. If None, will generate
            future dates using make_future_dataframe for each series.
        freq : str, default 'D'
            Frequency for generated future dates (if future_df is None).
        include_history : bool, default True
            Include historical dates in predictions (if future_df is None).

        Returns
        -------
        results : dict
            Dictionary with keys:
            - 'forecasts': polars.DataFrame or pandas.DataFrame - Combined forecasts
              with series_id column. Returns the same type as the input to fit().
            - 'errors': Dict[str, str] - Error messages by series_id for failed predictions
            - 'n_success': int - Number of successful predictions
            - 'n_failed': int - Number of failed predictions

        Examples
        --------
        >>> # Predict with auto-generated future dates
        >>> results = multi_model.predict(periods=30, freq='D')
        >>>
        >>> # Predict with custom future dataframe
        >>> future = pl.DataFrame({
        ...     'series_id': ['A', 'A', 'B', 'B'],
        ...     'ds': [date1, date2, date1, date2]
        ... })
        >>> results = multi_model.predict(future_df=future)
        """
        if not self.models:
            raise ValueError("No models have been fitted. Call fit() first.")

        # Prepare prediction tasks
        prediction_tasks = []

        if future_df is not None:
            # Use provided future dataframe
            future_polars = _ensure_polars(future_df)

            if self.series_col not in future_polars.columns:
                raise ValueError(f"Column '{self.series_col}' not found in future_df")

            for series_id in self.models.keys():
                # Filter future dataframe for this series
                series_future = future_polars.filter(
                    pl.col(self.series_col) == series_id
                ).drop(self.series_col)

                # Use stored JSON for multiprocessing
                model_json = self.model_jsons[series_id]
                prediction_tasks.append((series_id, model_json, series_future))
        else:
            # Generate future dataframes for each series
            if periods is None:
                raise ValueError("Either periods or future_df must be provided")

            for series_id, model in self.models.items():
                series_future = model.make_future_dataframe(
                    periods=periods, freq=freq, include_history=include_history
                )
                # Use stored JSON for multiprocessing
                model_json = self.model_jsons[series_id]
                prediction_tasks.append((series_id, model_json, series_future))

        # Make predictions in parallel
        with Pool(processes=self.n_processes) as pool:
            results = pool.map(_predict_single_series, prediction_tasks)

        # Separate successful forecasts from errors
        forecasts = []
        errors = {}

        for series_id, forecast, error in results:
            if error is None:
                forecasts.append(forecast)
            else:
                errors[series_id] = error

        # Combine all forecasts
        combined_forecast = None
        if forecasts:
            combined_forecast = pl.concat(forecasts)

            # Convert to pandas if the original fit input was pandas
            if self._input_was_pandas and HAS_PANDAS:
                combined_forecast = _polars_to_pandas(combined_forecast)

        n_success = len(forecasts)
        n_failed = len(errors)

        if n_failed > 0:
            warnings.warn(
                f"Predicted {n_success}/{len(self.models)} series successfully. "
                f"{n_failed} series failed. Check results['errors'] for details.",
                UserWarning,
            )

        return {
            "forecasts": combined_forecast,
            "errors": errors,
            "n_success": n_success,
            "n_failed": n_failed,
        }

    def get_model(self, series_id: str) -> Optional[Farseer]:
        """
        Get the fitted model for a specific series.

        Parameters
        ----------
        series_id : str
            The series identifier.

        Returns
        -------
        model : Farseer or None
            The fitted model, or None if the series failed to fit.
        """
        return self.models.get(series_id)

    def get_error(self, series_id: str) -> Optional[str]:
        """
        Get the error message for a failed series.

        Parameters
        ----------
        series_id : str
            The series identifier.

        Returns
        -------
        error : str or None
            The error message, or None if the series fitted successfully.
        """
        return self.errors.get(series_id)


__all__ = ["Farseer", "FarseerMultiSeries", "regressor_coefficients", "__version__"]
