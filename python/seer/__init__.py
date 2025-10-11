"""Seer: Fast Bayesian time series forecasting"""

from seer import Seer as _Seer, __version__
import pandas as pd
from typing import Optional, Union

class Seer(_Seer):
    """
    Seer forecaster with scikit-learn-like interface.
    
    Parameters
    ----------
    growth : str, default 'linear'
        'linear', 'logistic', or 'flat' trend
    changepoints : list, optional
        List of dates at which to include potential changepoints
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
            DataFrame with columns 'ds' (date) and 'y' (value)
            
        Returns
        -------
        self : Seer
            Fitted model
        """
        # Validate input
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must have 'ds' and 'y' columns")
        
        # Ensure ds is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df = df.copy()
            df['ds'] = pd.to_datetime(df['ds'])
        
        # Convert to string for Rust
        df_rust = df.copy()
        df_rust['ds'] = df_rust['ds'].astype(str)
        
        super().fit(df_rust)
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
            DataFrame with predictions
        """
        # Ensure ds is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df = df.copy()
            df['ds'] = pd.to_datetime(df['ds'])
        
        # Convert to string for Rust
        df_rust = df.copy()
        df_rust['ds'] = df_rust['ds'].astype(str)
        
        forecast = super().predict(df_rust)
        
        # Convert ds back to datetime
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        return forecast
    
    def plot(self, forecast: pd.DataFrame, ax=None, **kwargs):
        """Plot the forecast with matplotlib."""
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot forecast
        ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue')
        ax.fill_between(
            forecast['ds'],
            forecast['yhat_lower'],
            forecast['yhat_upper'],
            alpha=0.3,
            color='blue',
            label='Uncertainty'
        )
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_components(self, forecast: pd.DataFrame):
        """Plot the forecast components."""
        import matplotlib.pyplot as plt
        
        components = ['trend']
        if 'yearly' in forecast.columns:
            components.append('yearly')
        if 'weekly' in forecast.columns:
            components.append('weekly')
        
        n_components = len(components)
        fig, axes = plt.subplots(n_components, 1, figsize=(10, 3 * n_components))
        
        if n_components == 1:
            axes = [axes]
        
        for ax, component in zip(axes, components):
            ax.plot(forecast['ds'], forecast[component])
            ax.set_ylabel(component.capitalize())
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        
        return fig

__all__ = ['Seer', '__version__']