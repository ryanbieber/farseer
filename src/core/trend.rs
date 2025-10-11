use chrono::{NaiveDate, NaiveDateTime, Duration};

/// Convert ds string ("%Y-%m-%d" or "%Y-%m-%d %H:%M:%S") to NaiveDateTime
pub fn parse_ds(ds: &str) -> Option<NaiveDateTime> {
    if let Ok(d) = NaiveDateTime::parse_from_str(ds, "%Y-%m-%d %H:%M:%S") {
        Some(d)
    } else if let Ok(d) = NaiveDate::parse_from_str(ds, "%Y-%m-%d") {
        Some(d.and_hms_opt(0, 0, 0).unwrap())
    } else {
        None
    }
}

/// Given history timestamps, return (t values normalized to [0,1], t_scale seconds, start time)
pub fn time_scale(history: &[NaiveDateTime]) -> (Vec<f64>, f64, NaiveDateTime) {
    let t0 = history.first().cloned().unwrap();
    let t_last = *history.last().unwrap();
    let span = t_last - t0;
    let span_s = span.num_microseconds().unwrap_or(1) as f64 / 1_000_000.0;
    let t: Vec<f64> = history
        .iter()
        .map(|ts| {
            let dt = *ts - t0;
            let s = dt.num_microseconds().unwrap_or(0) as f64 / 1_000_000.0;
            if span_s > 0.0 { s / span_s } else { 0.0 }
        })
        .collect();
    (t, span_s.max(1e-12), t0)
}

/// Select n changepoints uniformly from first changepoint_range proportion of history
pub fn select_changepoints(t: &[f64], n: usize, changepoint_range: f64) -> Vec<f64> {
    if t.len() <= 2 || n == 0 { return Vec::new(); }
    let end = (
        (t.len() as f64 - 1.0) * changepoint_range
    ).floor() as usize;
    if end <= 1 { return Vec::new(); }
    let mut cps = Vec::new();
    let step = (end as f64) / (n as f64 + 1.0);
    for i in 1..=n {
        let idx = (i as f64 * step).round() as usize;
        let idx = idx.clamp(1, end - 1);
        cps.push(t[idx]);
    }
    cps.sort_by(|a,b| a.partial_cmp(b).unwrap());
    cps.dedup_by(|a,b| (*a - *b).abs() < 1e-12);
    cps
}

/// Build changepoint indicator matrix A (T x S), where A[i,j] = 1 if t[i] >= t_change[j]
pub fn changepoint_matrix(t: &[f64], t_change: &[f64]) -> Vec<Vec<f64>> {
    let t_len = t.len();
    let s = t_change.len();
    let mut a = vec![vec![0.0; s]; t_len];
    let mut cp_idx = 0;
    let mut active = vec![0.0; s];
    for i in 0..t_len {
        while cp_idx < s && t[i] >= t_change[cp_idx] {
            active[cp_idx] = 1.0;
            cp_idx += 1;
        }
        a[i].clone_from_slice(&active);
    }
    a
}

/// Ordinary least squares to estimate k, m for linear trend y ~ k*t + m
pub fn ols_linear_trend(t: &[f64], y: &[f64]) -> (f64, f64) {
    let n = t.len() as f64;
    if n == 0.0 { return (0.0, 0.0); }
    let sum_t: f64 = t.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_tt: f64 = t.iter().map(|v| v * v).sum();
    let sum_ty: f64 = t.iter().zip(y).map(|(ti, yi)| ti * yi).sum();
    let denom = n * sum_tt - sum_t * sum_t;
    if denom.abs() < 1e-12 {
        // fallback: flat
        let m = sum_y / n;
        return (0.0, m);
    }
    let k = (n * sum_ty - sum_t * sum_y) / denom;
    let m = (sum_y - k * sum_t) / n;
    (k, m)
}

/// Piecewise linear trend with changepoints
pub fn piecewise_linear(k: f64, m: f64, delta: &[f64], t: &[f64], a: &[Vec<f64>], t_change: &[f64]) -> Vec<f64> {
    // trend = (k + A*delta) * t + (m + A*(-t_change .* delta))
    let s = t_change.len();
    let mut out = Vec::with_capacity(t.len());
    for (i, &ti) in t.iter().enumerate() {
        let ai = &a[i];
        let mut a_delta = 0.0;
        let mut a_tc_delta = 0.0;
        for j in 0..s {
            a_delta += ai[j] * delta[j];
            a_tc_delta += ai[j] * (-t_change[j] * delta[j]);
        }
        out.push((k + a_delta) * ti + (m + a_tc_delta));
    }
    out
}

/// Flat trend: constant baseline
pub fn flat_trend(m: f64, n: usize) -> Vec<f64> {
    vec![m; n]
}

/// Logistic gamma: offset adjustments for piecewise continuity
/// gamma[i] = (t_change[i] - m_prev) * (1 - k_s[i] / k_s[i+1])
pub fn logistic_gamma(k: f64, m: f64, delta: &[f64], t_change: &[f64]) -> Vec<f64> {
    let s = t_change.len();
    let mut k_s = vec![k]; // k_s[0] = k
    for &d in delta.iter() {
        k_s.push(k_s.last().unwrap() + d);
    }
    let mut gamma = vec![0.0; s];
    let mut m_pr = m;
    for i in 0..s {
        let ratio = if k_s[i+1].abs() > 1e-12 { 1.0 - k_s[i] / k_s[i+1] } else { 0.0 };
        gamma[i] = (t_change[i] - m_pr) * ratio;
        m_pr += gamma[i];
    }
    gamma
}

/// Piecewise logistic trend with changepoints
/// trend = cap / (1 + exp(-(k + A*delta)*(t - (m + A*gamma))))
pub fn piecewise_logistic(k: f64, m: f64, delta: &[f64], t: &[f64], cap: &[f64], a: &[Vec<f64>], t_change: &[f64]) -> Vec<f64> {
    let gamma = logistic_gamma(k, m, delta, t_change);
    let s = t_change.len();
    let mut out = Vec::with_capacity(t.len());
    for (i, &ti) in t.iter().enumerate() {
        let ai = &a[i];
        let mut a_delta = 0.0;
        let mut a_gamma = 0.0;
        for j in 0..s {
            a_delta += ai[j] * delta[j];
            a_gamma += ai[j] * gamma[j];
        }
        let k_eff = k + a_delta;
        let m_eff = m + a_gamma;
        let arg = -k_eff * (ti - m_eff);
        let val = cap[i] / (1.0 + arg.exp());
        out.push(val);
    }
    out
}

/// Generate future dates after the last timestamp with specified frequency
/// Supported frequencies:
/// - "H" or "h": Hourly
/// - "D" or "d": Daily (default)
/// - "W" or "w": Weekly
/// - "M" or "m": Monthly (approximate - 30 days)
/// - "Y" or "y": Yearly (approximate - 365 days)
pub fn future_dates(last: NaiveDateTime, periods: usize, freq: &str) -> Vec<String> {
    match freq {
        "H" | "h" => {
            // Hourly frequency
            (1..=periods)
                .map(|i| (last + Duration::hours(i as i64)).format("%Y-%m-%d %H:%M:%S").to_string())
                .collect()
        },
        "D" | "d" => {
            // Daily frequency (default)
            (1..=periods)
                .map(|i| (last + Duration::days(i as i64)).format("%Y-%m-%d").to_string())
                .collect()
        },
        "W" | "w" => {
            // Weekly frequency
            (1..=periods)
                .map(|i| (last + Duration::weeks(i as i64)).format("%Y-%m-%d").to_string())
                .collect()
        },
        "M" | "m" => {
            // Monthly frequency (approximate - 30 days)
            (1..=periods)
                .map(|i| (last + Duration::days((i * 30) as i64)).format("%Y-%m-%d").to_string())
                .collect()
        },
        "Y" | "y" => {
            // Yearly frequency (approximate - 365 days)
            (1..=periods)
                .map(|i| (last + Duration::days((i * 365) as i64)).format("%Y-%m-%d").to_string())
                .collect()
        },
        _ => {
            // Default to daily for unknown frequencies
            (1..=periods)
                .map(|i| (last + Duration::days(i as i64)).format("%Y-%m-%d").to_string())
                .collect()
        }
    }
}

/// Generate daily future dates after the last timestamp
/// Deprecated: Use future_dates with freq="D" instead
pub fn future_dates_daily(last: NaiveDateTime, periods: usize) -> Vec<String> {
    future_dates(last, periods, "D")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ds_date() {
        let dt = parse_ds("2020-01-15");
        assert!(dt.is_some());
        let dt = dt.unwrap();
        assert_eq!(dt.format("%Y-%m-%d").to_string(), "2020-01-15");
    }

    #[test]
    fn test_parse_ds_datetime() {
        let dt = parse_ds("2020-01-15 12:30:45");
        assert!(dt.is_some());
        let dt = dt.unwrap();
        assert_eq!(dt.format("%Y-%m-%d %H:%M:%S").to_string(), "2020-01-15 12:30:45");
    }

    #[test]
    fn test_parse_ds_invalid() {
        assert!(parse_ds("invalid").is_none());
        assert!(parse_ds("2020/01/15").is_none());
    }

    #[test]
    fn test_time_scale() {
        let d1 = parse_ds("2020-01-01").unwrap();
        let d2 = parse_ds("2020-01-02").unwrap();
        let d3 = parse_ds("2020-01-03").unwrap();
        let history = vec![d1, d2, d3];
        
        let (t, t_scale, t0) = time_scale(&history);
        
        assert_eq!(t.len(), 3);
        assert_eq!(t[0], 0.0);
        assert_eq!(t[2], 1.0);
        assert!(t[1] > 0.0 && t[1] < 1.0);
        assert!(t_scale > 0.0);
        assert_eq!(t0, d1);
    }

    #[test]
    fn test_select_changepoints() {
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let cps = select_changepoints(&t, 5, 0.8);
        
        assert!(cps.len() <= 5);
        for cp in &cps {
            assert!(*cp >= 0.0 && *cp <= 0.8);
        }
    }

    #[test]
    fn test_changepoint_matrix() {
        let t = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let t_change = vec![0.25, 0.75];
        let a = changepoint_matrix(&t, &t_change);
        
        assert_eq!(a.len(), 5);
        assert_eq!(a[0], vec![0.0, 0.0]);  // t=0.0, before all changepoints
        assert_eq!(a[1], vec![1.0, 0.0]);  // t=0.25, at first changepoint
        assert_eq!(a[2], vec![1.0, 0.0]);  // t=0.5, after first
        assert_eq!(a[3], vec![1.0, 1.0]);  // t=0.75, at second changepoint
        assert_eq!(a[4], vec![1.0, 1.0]);  // t=1.0, after both
    }

    #[test]
    fn test_ols_linear_trend() {
        let t = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![10.0, 12.0, 14.0, 16.0, 18.0];  // y = 10 + 2*t
        
        let (k, m) = ols_linear_trend(&t, &y);
        
        assert!((k - 2.0).abs() < 1e-10);
        assert!((m - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_piecewise_linear_no_changepoints() {
        let k = 2.0;
        let m = 10.0;
        let delta = vec![];
        let t = vec![0.0, 0.5, 1.0];
        let a = vec![vec![]; 3];  // Empty rows for no changepoints
        let t_change = vec![];
        
        let trend = piecewise_linear(k, m, &delta, &t, &a, &t_change);
        
        assert_eq!(trend.len(), 3);
        assert!((trend[0] - 10.0).abs() < 1e-10);
        assert!((trend[1] - 11.0).abs() < 1e-10);
        assert!((trend[2] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_flat_trend() {
        let trend = flat_trend(42.0, 5);
        assert_eq!(trend.len(), 5);
        assert!(trend.iter().all(|&x| (x - 42.0).abs() < 1e-10));
    }

    #[test]
    fn test_logistic_gamma() {
        let k = 0.5;
        let m = 0.5;
        let delta = vec![0.1, -0.05];
        let t_change = vec![0.3, 0.7];
        
        let gamma = logistic_gamma(k, m, &delta, &t_change);
        
        assert_eq!(gamma.len(), 2);
        // Gamma values ensure continuity
        assert!(gamma[0].abs() < 10.0);  // Reasonable magnitude
        assert!(gamma[1].abs() < 10.0);
    }

    #[test]
    fn test_piecewise_logistic() {
        let k = 0.5;
        let m = 0.5;
        let delta = vec![];
        let t = vec![0.0, 0.5, 1.0];
        let cap = vec![100.0, 100.0, 100.0];
        let a = vec![vec![]; 3];  // Empty rows for no changepoints
        let t_change = vec![];
        
        let trend = piecewise_logistic(k, m, &delta, &t, &cap, &a, &t_change);
        
        assert_eq!(trend.len(), 3);
        // All values should be below cap
        assert!(trend.iter().all(|&x| x < 100.0));
        // Trend should be monotonic for simple logistic
        assert!(trend[0] < trend[1]);
        assert!(trend[1] < trend[2]);
    }

    #[test]
    fn test_future_dates_daily() {
        let start = parse_ds("2020-01-31").unwrap();
        let dates = future_dates_daily(start, 3);
        
        assert_eq!(dates.len(), 3);
        assert_eq!(dates[0], "2020-02-01");
        assert_eq!(dates[1], "2020-02-02");
        assert_eq!(dates[2], "2020-02-03");
    }

    #[test]
    fn test_future_dates_hourly() {
        let start = parse_ds("2020-01-31 22:00:00").unwrap();
        let dates = future_dates(start, 3, "H");
        
        assert_eq!(dates.len(), 3);
        assert_eq!(dates[0], "2020-01-31 23:00:00");
        assert_eq!(dates[1], "2020-02-01 00:00:00");
        assert_eq!(dates[2], "2020-02-01 01:00:00");
    }

    #[test]
    fn test_future_dates_weekly() {
        let start = parse_ds("2020-01-31").unwrap();
        let dates = future_dates(start, 2, "W");
        
        assert_eq!(dates.len(), 2);
        assert_eq!(dates[0], "2020-02-07");
        assert_eq!(dates[1], "2020-02-14");
    }

    #[test]
    fn test_future_dates_monthly() {
        let start = parse_ds("2020-01-15").unwrap();
        let dates = future_dates(start, 2, "M");
        
        assert_eq!(dates.len(), 2);
        assert_eq!(dates[0], "2020-02-14");  // 30 days later
        assert_eq!(dates[1], "2020-03-15");  // 60 days later
    }

    #[test]
    fn test_future_dates_yearly() {
        let start = parse_ds("2020-01-15").unwrap();
        let dates = future_dates(start, 2, "Y");
        
        assert_eq!(dates.len(), 2);
        assert_eq!(dates[0], "2021-01-14");  // 365 days later (leap year not accounted for)
        assert_eq!(dates[1], "2022-01-14");  // 730 days later
    }

    #[test]
    fn test_future_dates_fallback() {
        let start = parse_ds("2020-01-31").unwrap();
        let dates = future_dates(start, 2, "unknown");
        
        // Should fallback to daily
        assert_eq!(dates.len(), 2);
        assert_eq!(dates[0], "2020-02-01");
        assert_eq!(dates[1], "2020-02-02");
    }
}
