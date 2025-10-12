#!/usr/bin/env python3
"""
Quick Start Example

The simplest possible example to get started with Seer.
Copy and modify this for your own use case.
"""

import pandas as pd
import numpy as np
from seer import Seer

# 1. Prepare your data
#    Must have 'ds' (date) and 'y' (value) columns
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=365, freq='D'),
    'y': np.random.randn(365).cumsum() + 100
})

# 2. Create model
model = Seer()

# 3. Fit to historical data
model.fit(df)

# 4. Make future dataframe
future = model.make_future_dataframe(periods=30)  # 30 days ahead

# 5. Generate forecast
forecast = model.predict(future)

# 6. View results
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

# 7. (Optional) Save model
model_json = model.to_json()
# Later: model = Seer.from_json(model_json)

# That's it! See other examples for advanced features.
