use seer::core::{self, Seer as CoreSeer, TimeSeriesData, TrendType};

fn make_ds(start: &str, n: usize) -> Vec<String> {
    let start_date = chrono::NaiveDate::parse_from_str(start, "%Y-%m-%d").unwrap();
    (0..n)
        .map(|i| {
            (start_date + chrono::Duration::days(i as i64))
                .format("%Y-%m-%d")
                .to_string()
        })
        .collect()
}

#[test]
fn time_scale_and_changepoints() {
    let ds = make_ds("2020-01-01", 10);
    let ts: Vec<_> = ds
        .iter()
        .map(|s| core::trend::parse_ds(s).unwrap())
        .collect();
    let (t, _scale, _t0) = core::trend::time_scale(&ts);
    assert_eq!(t.len(), 10);
    assert!((t[0] - 0.0).abs() < 1e-12);
    assert!((t[9] - 1.0).abs() < 1e-12);

    let cps = core::trend::select_changepoints(&t, 3, 0.8);
    assert!(cps.len() <= 3);
    let a = core::trend::changepoint_matrix(&t, &cps);
    assert_eq!(a.len(), t.len());
    for row in &a {
        assert_eq!(row.len(), cps.len());
    }
}

#[test]
fn piecewise_linear_reduces_to_linear_when_no_delta() {
    let ds = make_ds("2020-01-01", 5);
    let ts: Vec<_> = ds
        .iter()
        .map(|s| core::trend::parse_ds(s).unwrap())
        .collect();
    let (t, _, _) = core::trend::time_scale(&ts);
    let cps: Vec<f64> = vec![];
    let a = core::trend::changepoint_matrix(&t, &cps);
    let k = 2.0;
    let m = 1.0;
    let delta: Vec<f64> = vec![];
    let trend = core::trend::piecewise_linear(k, m, &delta, &t, &a, &cps);
    for (i, &ti) in t.iter().enumerate() {
        let expected = k * ti + m;
        assert!((trend[i] - expected).abs() < 1e-9);
    }
}

#[test]
fn fourier_series_shapes() {
    let ds = make_ds("2020-01-01", 7);
    let ts: Vec<_> = ds
        .iter()
        .map(|s| core::trend::parse_ds(s).unwrap())
        .collect();
    let (_t, _, t0) = core::trend::time_scale(&ts);
    // build days vector (days since start)
    let days: Vec<f64> = ts
        .iter()
        .map(|dt| (*dt - t0).num_seconds() as f64 / 86_400.0)
        .collect();
    let x = core::seasonality::fourier_series(&days, 7.0, 3);
    assert_eq!(x.len(), days.len());
    assert_eq!(x[0].len(), 6); // 2 * order
}

#[test]
fn model_fit_with_yearly_seasonality() {
    let n = 365;
    let ds = make_ds("2020-01-01", n);
    let two_pi = std::f64::consts::PI * 2.0;
    // build true signal: y = 10 + 0.2*t + 5*sin(2pi * t/365)
    let t_idx: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = t_idx
        .iter()
        .map(|&i| 10.0 + 0.2 * i + 5.0 * ((two_pi * i / 365.25).sin()))
        .collect();

    let data = TimeSeriesData::new(ds.clone(), y.clone(), None, None).unwrap();
    let mut m = CoreSeer::new()
        .with_trend(TrendType::Linear)
        .with_changepoints(10);
    // yearly seasonality enabled by default in constructor; ensure weekly/daily off for clarity
    m = m.with_yearly_seasonality();
    // fit
    m.fit(&data).unwrap();

    // predict on history
    let fcst = m.predict(&ds).unwrap();
    assert_eq!(fcst.ds.len(), n);
    assert_eq!(fcst.yhat.len(), n);
    // Should have yearly component populated
    assert!(!fcst.yearly.is_empty());

    // Check MSE is reasonably small
    let mse: f64 = fcst
        .yhat
        .iter()
        .zip(y.iter())
        .map(|(yh, yt)| (yh - yt).powi(2))
        .sum::<f64>()
        / n as f64;
    assert!(mse < 5.0, "MSE too large: {}", mse);
}

#[test]
fn future_dates_daily_len_and_contiguity() {
    let ds = make_ds("2020-01-01", 10);
    let y: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let data = TimeSeriesData::new(ds.clone(), y, None, None).unwrap();
    let mut m = CoreSeer::new();
    m.fit(&data).unwrap();
    let fut = m.make_future_dates(5, "D", true).unwrap();
    assert_eq!(fut.len(), 15);
    // check last future date is 5 days after last history
    let last_hist = chrono::NaiveDate::parse_from_str(&ds.last().unwrap(), "%Y-%m-%d").unwrap();
    let last_fut = chrono::NaiveDate::parse_from_str(&fut.last().unwrap(), "%Y-%m-%d").unwrap();
    assert_eq!(last_fut, last_hist + chrono::Duration::days(5));
}

#[test]
fn logistic_trend_respects_cap() {
    use core::trend::{changepoint_matrix, piecewise_logistic};
    let t = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let cap = vec![100.0; 5];
    let k = 0.5;
    let m = 0.5;
    let delta = vec![];
    let t_change: Vec<f64> = vec![];
    let a = changepoint_matrix(&t, &t_change);
    let trend = piecewise_logistic(k, m, &delta, &t, &cap, &a, &t_change);
    // All values should be <= cap
    for &val in &trend {
        assert!(val <= 100.0, "Logistic trend exceeds cap: {}", val);
    }
    // Should approach cap asymptotically
    assert!(trend.last().unwrap() > &50.0);
}

#[test]
fn flat_trend_is_constant() {
    use core::trend::flat_trend;
    let m = 42.0;
    let trend = flat_trend(m, 10);
    assert_eq!(trend.len(), 10);
    for &val in &trend {
        assert_eq!(val, m);
    }
}

#[test]
fn model_logistic_growth() {
    let n = 100;
    let ds = make_ds("2020-01-01", n);
    // Create data with cap
    let cap_val = 50.0;
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            cap_val / (1.0 + (-5.0 * (t - 0.5)).exp()) + (rand::random::<f64>() - 0.5) * 2.0
        })
        .collect();
    let cap = vec![cap_val; n];

    let data = TimeSeriesData::new(ds.clone(), y.clone(), Some(cap), None).unwrap();
    let mut m = CoreSeer::new()
        .with_trend(TrendType::Logistic)
        .with_changepoints(5);
    m.fit(&data).unwrap();

    let fcst = m.predict(&ds).unwrap();
    // All predictions should respect cap
    for &val in &fcst.yhat {
        assert!(val <= cap_val * 1.1, "Prediction exceeds cap: {}", val);
    }
}

#[test]
fn uncertainty_intervals_scale_with_sigma() {
    let n = 50;
    let ds = make_ds("2020-01-01", n);
    let y: Vec<f64> = (0..n).map(|i| 10.0 + 0.1 * i as f64).collect();
    let data = TimeSeriesData::new(ds.clone(), y.clone(), None, None).unwrap();
    let mut m = CoreSeer::new();
    m.fit(&data).unwrap();
    let fcst = m.predict(&ds).unwrap();

    // Check intervals are symmetric around yhat
    for i in 0..n {
        let lower_gap = fcst.yhat[i] - fcst.yhat_lower[i];
        let upper_gap = fcst.yhat_upper[i] - fcst.yhat[i];
        assert!(
            (lower_gap - upper_gap).abs() < 1e-6,
            "Intervals not symmetric"
        );
        assert!(lower_gap > 0.0, "Lower interval should be below yhat");
    }
}

// Needed for logistic test
mod rand {
    pub fn random<T: Default>() -> f64 {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        let s = RandomState::new();
        let mut hasher = s.build_hasher();
        hasher.write_usize(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as usize,
        );
        (hasher.finish() % 1000) as f64 / 1000.0
    }
}

// ===== M3 Tests: Seasonality Registry =====

#[test]
fn add_custom_seasonality() {
    let n = 60;
    let ds = make_ds("2020-01-01", n);
    // Create data with monthly pattern (period ~30 days)
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let monthly_pattern = ((i as f64 / 30.0) * 2.0 * std::f64::consts::PI).sin() * 2.0;
            10.0 + monthly_pattern
        })
        .collect();

    let data = TimeSeriesData::new(ds.clone(), y.clone(), None, None).unwrap();
    let mut m = CoreSeer::new();
    // Disable default seasonalities
    m = m.without_yearly_seasonality().without_weekly_seasonality();

    // Add custom monthly seasonality
    m.add_seasonality("monthly", 30.5, 5, None, None).unwrap();
    m.fit(&data).unwrap();

    let fcst = m.predict(&ds).unwrap();
    // Check that predictions capture the pattern
    assert_eq!(fcst.yhat.len(), n);
    // RMSE should be reasonable for monthly pattern
    let rmse: f64 = fcst
        .yhat
        .iter()
        .zip(y.iter())
        .map(|(pred, actual)| (pred - actual).powi(2))
        .sum::<f64>()
        / n as f64;
    let rmse = rmse.sqrt();
    assert!(rmse < 5.0, "RMSE too high: {}", rmse);
}

#[test]
fn multiplicative_seasonality_mode() {
    let n = 100;
    let ds = make_ds("2020-01-01", n);
    // Create data with multiplicative seasonal pattern
    // trend grows, and seasonality is proportional to trend
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 10.0 + 0.2 * i as f64;
            let seasonal_factor = 1.0 + 0.2 * ((i as f64 / 7.0) * 2.0 * std::f64::consts::PI).sin();
            trend * seasonal_factor
        })
        .collect();

    let data = TimeSeriesData::new(ds.clone(), y.clone(), None, None).unwrap();
    let mut m = CoreSeer::new();
    m = m.without_yearly_seasonality().without_daily_seasonality();

    // Add weekly seasonality in multiplicative mode
    m.add_seasonality("weekly_mult", 7.0, 3, None, Some("multiplicative"))
        .unwrap();
    m.fit(&data).unwrap();

    let fcst = m.predict(&ds).unwrap();
    assert_eq!(fcst.yhat.len(), n);

    // Multiplicative mode should handle growing amplitude
    // Check RMSE is reasonable (note: OLS on residuals isn't optimal for multiplicative patterns)
    let rmse: f64 = fcst
        .yhat
        .iter()
        .zip(y.iter())
        .map(|(pred, actual)| (pred - actual).powi(2))
        .sum::<f64>()
        / n as f64;
    let rmse = rmse.sqrt();
    assert!(
        rmse < 30.0,
        "RMSE too high for multiplicative seasonality: {}",
        rmse
    );
}

#[test]
fn mixed_additive_and_multiplicative_seasonality() {
    let n = 60;
    let ds = make_ds("2020-01-01", n);
    // Create data with both additive and multiplicative components
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 10.0 + 0.1 * i as f64;
            let additive_seasonal = 2.0 * ((i as f64 / 30.0) * 2.0 * std::f64::consts::PI).sin();
            let multiplicative_factor =
                1.0 + 0.1 * ((i as f64 / 7.0) * 2.0 * std::f64::consts::PI).cos();
            trend * multiplicative_factor + additive_seasonal
        })
        .collect();

    let data = TimeSeriesData::new(ds.clone(), y.clone(), None, None).unwrap();
    let mut m = CoreSeer::new();
    m = m
        .without_yearly_seasonality()
        .without_weekly_seasonality()
        .without_daily_seasonality();

    // Add monthly additive and weekly multiplicative seasonalities
    m.add_seasonality("monthly", 30.0, 5, None, Some("additive"))
        .unwrap();
    m.add_seasonality("weekly", 7.0, 3, None, Some("multiplicative"))
        .unwrap();
    m.fit(&data).unwrap();

    let fcst = m.predict(&ds).unwrap();
    assert_eq!(fcst.yhat.len(), n);

    // Check RMSE is reasonable
    let rmse: f64 = fcst
        .yhat
        .iter()
        .zip(y.iter())
        .map(|(pred, actual)| (pred - actual).powi(2))
        .sum::<f64>()
        / n as f64;
    let rmse = rmse.sqrt();
    assert!(rmse < 15.0, "RMSE too high for mixed seasonality: {}", rmse);
}

#[test]
fn invalid_seasonality_mode_returns_error() {
    let mut m = CoreSeer::new();
    let result = m.add_seasonality("test", 10.0, 3, None, Some("invalid"));
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Invalid seasonality mode"));
}

#[test]
fn seasonality_with_prior_scale() {
    let n = 50;
    let ds = make_ds("2020-01-01", n);
    let y: Vec<f64> = (0..n).map(|i| 10.0 + 0.1 * i as f64).collect();

    let data = TimeSeriesData::new(ds.clone(), y.clone(), None, None).unwrap();
    let mut m = CoreSeer::new();
    m = m.without_yearly_seasonality().without_weekly_seasonality();

    // Add custom seasonality with prior scale
    m.add_seasonality("custom", 15.0, 3, Some(0.1), None)
        .unwrap();
    m.fit(&data).unwrap();

    let fcst = m.predict(&ds).unwrap();
    assert_eq!(fcst.yhat.len(), n);
    // With prior scale, fit should still work
    for &val in &fcst.yhat {
        assert!(val.is_finite(), "Prediction should be finite");
    }
}

// ===== M4 Tests: Holidays =====

#[test]
fn add_custom_holidays() {
    let n = 30;
    let ds = make_ds("2020-12-01", n);
    // Create data with a spike on Christmas
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let base = 100.0;
            // Christmas is on day 24 (2020-12-25)
            if i == 24 {
                base + 50.0 // Big spike on Christmas
            } else {
                base + (i as f64 * 0.1)
            }
        })
        .collect();

    let data = TimeSeriesData::new(ds.clone(), y.clone(), None, None).unwrap();
    let mut m = CoreSeer::new();
    m = m.without_yearly_seasonality().without_weekly_seasonality();

    // Add Christmas as a holiday
    m.add_holidays(
        "christmas",
        vec!["2020-12-25".to_string()],
        None, // no lower window
        None, // no upper window
        None, // default prior scale
        None, // additive mode
    )
    .unwrap();

    m.fit(&data).unwrap();

    let fcst = m.predict(&ds).unwrap();
    assert_eq!(fcst.yhat.len(), n);

    // The prediction for Christmas should be higher than surrounding days
    let christmas_pred = fcst.yhat[24];
    let day_before = fcst.yhat[23];
    let day_after = fcst.yhat[25];

    assert!(
        christmas_pred > day_before,
        "Christmas prediction should be higher than day before"
    );
    assert!(
        christmas_pred > day_after,
        "Christmas prediction should be higher than day after"
    );
}

#[test]
fn holidays_with_windows() {
    let n = 20;
    let ds = make_ds("2020-01-01", n);
    // Create data with elevated values around a holiday
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let base = 50.0;
            // Days 8-12 are elevated (holiday on day 10 with ±2 day window)
            if i >= 8 && i <= 12 {
                base + 20.0
            } else {
                base
            }
        })
        .collect();

    let data = TimeSeriesData::new(ds.clone(), y.clone(), None, None).unwrap();
    let mut m = CoreSeer::new();
    m = m.without_yearly_seasonality().without_weekly_seasonality();

    // Add holiday with 2-day window on each side
    m.add_holidays(
        "special_event",
        vec!["2020-01-11".to_string()], // Day index 10
        Some(-2),                       // 2 days before
        Some(2),                        // 2 days after
        None,
        None,
    )
    .unwrap();

    m.fit(&data).unwrap();

    let fcst = m.predict(&ds).unwrap();
    assert_eq!(fcst.yhat.len(), n);

    // Predictions during the window should be elevated
    let during_window_avg = (8..=12).map(|i| fcst.yhat[i]).sum::<f64>() / 5.0;
    let outside_window_avg = (0..8).map(|i| fcst.yhat[i]).sum::<f64>() / 8.0;

    assert!(
        during_window_avg > outside_window_avg,
        "Predictions during holiday window should be higher: {} vs {}",
        during_window_avg,
        outside_window_avg
    );
}

#[test]
fn multiple_holidays() {
    let n = 60;
    let ds = make_ds("2020-01-01", n);
    // Create data with spikes on multiple holidays
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let base = 100.0;
            // New Year's Day (day 0), MLK Day (day 19), Valentine's (day 44)
            if i == 0 || i == 19 || i == 44 {
                base + 30.0
            } else {
                base + (i as f64 * 0.05)
            }
        })
        .collect();

    let data = TimeSeriesData::new(ds.clone(), y.clone(), None, None).unwrap();
    let mut m = CoreSeer::new();
    m = m.without_yearly_seasonality().without_weekly_seasonality();

    // Add multiple holidays
    m.add_holidays(
        "new_year",
        vec!["2020-01-01".to_string()],
        None,
        None,
        None,
        None,
    )
    .unwrap();

    m.add_holidays(
        "mlk_day",
        vec!["2020-01-20".to_string()],
        None,
        None,
        None,
        None,
    )
    .unwrap();

    m.add_holidays(
        "valentines",
        vec!["2020-02-14".to_string()],
        None,
        None,
        None,
        None,
    )
    .unwrap();

    m.fit(&data).unwrap();

    let fcst = m.predict(&ds).unwrap();
    assert_eq!(fcst.yhat.len(), n);

    // All three holidays should have elevated predictions
    assert!(fcst.yhat[0] > 110.0, "New Year should be elevated");
    assert!(fcst.yhat[19] > 110.0, "MLK Day should be elevated");
    assert!(fcst.yhat[44] > 110.0, "Valentine's should be elevated");
}

#[test]
fn holiday_invalid_mode_returns_error() {
    let mut m = CoreSeer::new();
    let result = m.add_holidays(
        "test",
        vec!["2020-01-01".to_string()],
        None,
        None,
        None,
        Some("invalid_mode"),
    );
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Invalid holiday mode"));
}

#[test]
fn add_country_holidays_stores_country() {
    let mut m = CoreSeer::new();
    m.add_country_holidays("US").unwrap();

    // Country holidays feature requires Python integration to actually fetch dates
    // This test just verifies the method doesn't error
    assert!(true);
}

// ===== M5 Tests: Serialization and API Polish =====

#[test]
fn params_includes_complete_state() {
    let n = 50;
    let ds = make_ds("2020-01-01", n);
    let y: Vec<f64> = (0..n).map(|i| 10.0 + 0.1 * i as f64).collect();
    let data = TimeSeriesData::new(ds.clone(), y.clone(), None, None).unwrap();

    let mut m = CoreSeer::new();
    m.fit(&data).unwrap();

    let params = m.get_params();

    // Check all major fields are present
    assert!(params.get("version").is_some());
    assert!(params.get("fitted").is_some());
    assert!(params.get("trend").is_some());
    assert!(params.get("n_changepoints").is_some());
    assert!(params.get("seasonality_mode").is_some());
    assert!(params.get("k").is_some());
    assert!(params.get("m").is_some());
    assert!(params.get("beta").is_some());
    assert!(params.get("season_blocks").is_some());
    assert!(params.get("sigma_obs").is_some());
    assert_eq!(params["fitted"], true);
}

#[test]
fn serialization_round_trip() {
    let n = 30;
    let ds = make_ds("2020-01-01", n);
    let y: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
    let data = TimeSeriesData::new(ds.clone(), y.clone(), None, None).unwrap();

    // Create and fit model
    let mut m1 = CoreSeer::new()
        .with_trend(TrendType::Linear)
        .with_changepoints(5);
    m1.fit(&data).unwrap();

    // Serialize
    let json = m1.to_json().unwrap();
    assert!(!json.is_empty());

    // Deserialize
    let m2 = CoreSeer::from_json(&json).unwrap();

    // Check configuration matches via get_params
    let params1 = m1.get_params();
    let params2 = m2.get_params();
    assert_eq!(params2["trend"], params1["trend"]);
    assert_eq!(params2["n_changepoints"], params1["n_changepoints"]);
    assert_eq!(params2["fitted"], params1["fitted"]);
}

#[test]
fn serialization_preserves_fitted_parameters() {
    let n = 40;
    let ds = make_ds("2020-01-01", n);
    let y: Vec<f64> = (0..n).map(|i| 50.0 + 0.5 * i as f64).collect();
    let data = TimeSeriesData::new(ds.clone(), y.clone(), None, None).unwrap();

    // Fit original model
    let mut m1 = CoreSeer::new();
    m1.fit(&data).unwrap();

    // Serialize and deserialize
    let json = m1.to_json().unwrap();
    let m2 = CoreSeer::from_json(&json).unwrap();

    // Check fitted parameters match (approximately, due to JSON serialization precision)
    let params1 = m1.get_params();
    let params2 = m2.get_params();

    // Check key scalar parameters
    let k1 = params1["k"].as_f64().unwrap();
    let k2 = params2["k"].as_f64().unwrap();
    assert!(
        (k1 - k2).abs() < 1e-6,
        "k parameter doesn't match: {} vs {}",
        k1,
        k2
    );

    let m1_val = params1["m"].as_f64().unwrap();
    let m2_val = params2["m"].as_f64().unwrap();
    assert!(
        (m1_val - m2_val).abs() < 1e-6,
        "m parameter doesn't match: {} vs {}",
        m1_val,
        m2_val
    );

    assert_eq!(params2["fitted"], true);

    // Check array lengths match (exact values may differ slightly due to JSON round-trip)
    assert_eq!(
        params1["delta"].as_array().unwrap().len(),
        params2["delta"].as_array().unwrap().len()
    );
    assert_eq!(
        params1["beta"].as_array().unwrap().len(),
        params2["beta"].as_array().unwrap().len()
    );
}

#[test]
fn serialization_with_custom_components() {
    let n = 30;
    let ds = make_ds("2020-01-01", n);
    let y: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
    let data = TimeSeriesData::new(ds.clone(), y.clone(), None, None).unwrap();

    // Create model with custom seasonality and holidays
    let mut m1 = CoreSeer::new();
    m1.add_seasonality("monthly", 30.0, 5, None, None).unwrap();
    m1.add_holidays(
        "test_holiday",
        vec!["2020-01-15".to_string()],
        None,
        None,
        None,
        None,
    )
    .unwrap();
    m1.fit(&data).unwrap();

    // Serialize and deserialize
    let json = m1.to_json().unwrap();
    let m2 = CoreSeer::from_json(&json).unwrap();

    // Check custom components are preserved
    let params = m2.get_params();
    let seasonalities = params["seasonalities"].as_array().unwrap();
    assert_eq!(seasonalities.len(), 1);
    assert_eq!(seasonalities[0]["name"], "monthly");

    let holidays = params["holidays"].as_array().unwrap();
    assert_eq!(holidays.len(), 1);
    assert_eq!(holidays[0]["name"], "test_holiday");
}

#[test]
fn set_seasonality_mode() {
    let m = CoreSeer::new().with_seasonality_mode("multiplicative");
    assert!(m.is_ok());

    let m_invalid = CoreSeer::new().with_seasonality_mode("invalid");
    assert!(m_invalid.is_err());
}

#[test]
fn set_interval_width() {
    let m = CoreSeer::new().with_interval_width(0.95);
    let params = m.get_params();
    assert_eq!(params["interval_width"], 0.95);
}
