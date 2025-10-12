"""Seer: Fast Bayesian time series forecasting

Python wrapper around the Rust-based Seer library for time series forecasting.

Example:
    >>> from seer import Seer
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'ds': pd.date_range('2020-01-01', periods=100),
    ...     'y': range(100)
    ... })
    >>> model = Seer()
    >>> model.fit(df)
    >>> forecast = model.predict(model.make_future_dataframe(periods=30))
"""

# This file provides Python-level enhancements to the Rust Seer class
# The base Seer class and __version__ are defined in the Rust library (src/lib.rs)
# and compiled via PyO3/maturin

# When tests import "from seer import Seer", they get the Rust-compiled module
# This __init__.py file is for the python/seer package which contains test scripts
# The actual package users install gets the Rust module directly

# For the test scripts in this directory, we'll create a pass-through
import sys
import os

# Add parent directory to path so we can import the installed seer module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import from the actual installed module
try:
    import seer as _installed_seer
    Seer = _installed_seer.Seer
    __version__ = getattr(_installed_seer, '__version__', '0.1.0')
except ImportError as e:
    raise ImportError(
        f"Could not import seer module. Please build and install with 'maturin develop'. Error: {e}"
    ) from e

__all__ = ['Seer', '__version__']

class Seer(_Seer):
    """
    Seer forecaster with scikit-learn-like interface.
    
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
    >>> import pandas as pd
    >>> 
    >>> # Create sample data
    >>> df = pd.DataFrame({
    ...     'ds': pd.date_range('2020-01-01', periods=100),
    ...     'y': range(100)
    ... })
    >>> 
    >>> # Fit model
    >>> model = Seer()
    >>> model.fit(df)
    >>> 
    >>> # Make predictions
    >>> future = model.make_future_dataframe(periods=30)
    >>> forecast = model.predict(future)
    """
    
    def fit(self, df: pd.DataFrame, **kwargs) -> 'Seer':
        """
        Fit the Seer model.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns 'ds' (date) and 'y' (value).
            Optional 'cap' column for logistic growth.
            
        Returns
        -------
        self : Seer
            Fitted model
        """
        # Validate input
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must have 'ds' and 'y' columns")
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure ds is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_copy['ds']):
            df_copy['ds'] = pd.to_datetime(df_copy['ds'])
        
        # Call Rust fit method (handles datetime conversion internally)
        super().fit(df_copy)
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using the Seer model.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'ds' column
            
        Returns
        -------
        forecast : pd.DataFrame
            DataFrame with predictions including:
            - ds: dates
            - yhat: predicted values
            - yhat_lower: lower uncertainty bound
            - yhat_upper: upper uncertainty bound
            - trend: trend component
            - yearly: yearly seasonality (if enabled)
            - weekly: weekly seasonality (if enabled)
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure ds is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_copy['ds']):
            df_copy['ds'] = pd.to_datetime(df_copy['ds'])
        
        # Call Rust predict method (handles datetime conversion internally)
        forecast = super().predict(df_copy)
        
        # Convert ds back to datetime
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        return forecast
    
    def add_seasonality(
        self, 
        name: str, 
        period: float, 
        fourier_order: int, 
        prior_scale: Optional[float] = None,
        mode: Optional[str] = None
    ) -> 'Seer':
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
        self : Seer
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
        mode: Optional[str] = None
    ) -> 'Seer':
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
        self : Seer
            Model instance for method chaining
            
        Examples
        --------
        >>> model = Seer()
        >>> model.add_holidays(
        ...     'christmas',
        ...     ['2020-12-25', '2021-12-25', '2022-12-25'],
        ...     lower_window=-2,
        ...     upper_window=2
        ... )
        """
        super().add_holidays(name, dates, lower_window, upper_window, prior_scale, mode)
        return self
    
    def add_country_holidays(self, country_name: str) -> 'Seer':
        """
        Add country-specific holidays.
        
        Parameters
        ----------
        country_name : str
            Name of the country (e.g., 'US', 'UK', 'CA')
            
        Returns
        -------
        self : Seer
            Model instance for method chaining
            
        Examples
        --------
        >>> model = Seer()
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
        with open(path, 'w') as f:
            f.write(json_str)
    
    @classmethod
    def load(cls, path: str) -> 'Seer':
        """
        Load a model from a JSON file.
        
        Parameters
        ----------
        path : str
            Path to the saved model
            
        Returns
        -------
        model : Seer
            Loaded model instance
        """
        with open(path, 'r') as f:
            json_str = f.read()
        return cls.from_json(json_str)
    
    def plot(self, forecast: pd.DataFrame, ax=None, history: Optional[pd.DataFrame] = None, **kwargs):
        """
        Plot the forecast with matplotlib.
        
        Parameters
        ----------
        forecast : pd.DataFrame
            Forecast dataframe from predict()
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure
        history : pd.DataFrame, optional
            Historical data to plot
        **kwargs : dict
            Additional arguments passed to matplotlib
            
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot historical data if provided
        if history is not None and 'y' in history.columns:
            ax.plot(history['ds'], history['y'], 'k.', label='Observed', markersize=4)
        
        # Plot forecast
        ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue', linewidth=2)
        ax.fill_between(
            forecast['ds'],
            forecast['yhat_lower'],
            forecast['yhat_upper'],
            alpha=0.2,
            color='blue',
            label='Uncertainty interval'
        )
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return ax
    
    def plot_components(self, forecast: pd.DataFrame, figsize=(10, 8)):
        """
        Plot the forecast components (trend, seasonalities).
        
        Parameters
        ----------
        forecast : pd.DataFrame
            Forecast dataframe from predict()
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        import matplotlib.pyplot as plt
        
        # Determine which components to plot
        components = []
        if 'trend' in forecast.columns:
            components.append(('trend', 'Trend'))
        if 'yearly' in forecast.columns:
            components.append(('yearly', 'Yearly Seasonality'))
        if 'weekly' in forecast.columns:
            components.append(('weekly', 'Weekly Seasonality'))
        
        n_components = len(components)
        if n_components == 0:
            raise ValueError("No components found in forecast")
        
        fig, axes = plt.subplots(n_components, 1, figsize=figsize)
        
        if n_components == 1:
            axes = [axes]
        
        for ax, (col, title) in zip(axes, components):
            ax.plot(forecast['ds'], forecast[col], linewidth=1.5)
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
            
        axes[-1].set_xlabel('Date')
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        
        return fig

__all__ = ['Seer', '__version__']