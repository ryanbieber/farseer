use std::f64::consts::PI;

/// Generate Fourier series features for given times (t in model units of days) and period (in days)
pub fn fourier_series(t: &[f64], period: f64, order: usize) -> Vec<Vec<f64>> {
    let cols = 2 * order;
    let mut x = vec![vec![0.0; cols]; t.len()];
    for (i, &ti) in t.iter().enumerate() {
        for k in 1..=order {
            let c = 2.0 * PI * (k as f64) * ti / period;
            x[i][2 * (k - 1)] = c.sin();
            x[i][2 * (k - 1) + 1] = c.cos();
        }
    }
    x
}

/// Combine multiple seasonality blocks column-wise
pub fn hstack(blocks: &[Vec<Vec<f64>>]) -> Vec<Vec<f64>> {
    if blocks.is_empty() {
        return Vec::new();
    }
    let rows = blocks[0].len();
    let total_cols: usize = blocks
        .iter()
        .map(|b| if b.is_empty() { 0 } else { b[0].len() })
        .sum();
    let mut out = vec![vec![0.0; total_cols]; rows];
    for i in 0..rows {
        let mut col = 0;
        for block in blocks {
            if block.is_empty() {
                continue;
            }
            for &v in &block[i] {
                out[i][col] = v;
                col += 1;
            }
        }
    }
    out
}

/// Generate holiday indicator features
/// Returns a matrix where each column represents a holiday occurrence
pub fn holiday_features(
    timestamps: &[chrono::NaiveDateTime],
    holiday_dates: &[String],
    lower_window: i32,
    upper_window: i32,
) -> Vec<Vec<f64>> {
    if holiday_dates.is_empty() {
        return vec![vec![]; timestamps.len()];
    }

    // Parse holiday dates
    let mut parsed_holidays = Vec::new();
    for date_str in holiday_dates {
        if let Some(date) = crate::core::trend::parse_ds(date_str) {
            parsed_holidays.push(date.date());
        }
    }

    if parsed_holidays.is_empty() {
        return vec![vec![]; timestamps.len()];
    }

    // Number of features per holiday occurrence (1 + windows)
    let window_range = (upper_window - lower_window + 1) as usize;
    let n_features = parsed_holidays.len() * window_range;

    let mut features = vec![vec![0.0; n_features]; timestamps.len()];

    for (i, ts) in timestamps.iter().enumerate() {
        let ts_date = ts.date();

        for (h_idx, &holiday_date) in parsed_holidays.iter().enumerate() {
            // Check if timestamp is within the holiday window
            let days_diff = (ts_date - holiday_date).num_days() as i32;

            if days_diff >= lower_window && days_diff <= upper_window {
                // Calculate which feature column this corresponds to
                let window_offset = (days_diff - lower_window) as usize;
                let feature_idx = h_idx * window_range + window_offset;
                features[i][feature_idx] = 1.0;
            }
        }
    }

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fourier_series_shape() {
        let t = vec![0.0, 1.0, 2.0, 3.0];
        let features = fourier_series(&t, 7.0, 3);

        // Should have 2*order columns (sin and cos for each order)
        assert_eq!(features.len(), 4); // 4 time points
        assert_eq!(features[0].len(), 6); // 2 * 3 features
    }

    #[test]
    fn test_fourier_series_periodicity() {
        // At t=0 and t=period, sin components should be ~0, cos components should be ~1
        let period = 7.0;
        let t = vec![0.0, period, 2.0 * period];
        let features = fourier_series(&t, period, 2);

        // First time point (t=0)
        assert!((features[0][0] - 0.0).abs() < 1e-10); // sin(2π*1*0/7)
        assert!((features[0][1] - 1.0).abs() < 1e-10); // cos(2π*1*0/7)

        // At t=period, should repeat
        assert!((features[1][0] - 0.0).abs() < 1e-10);
        assert!((features[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hstack_empty() {
        let result = hstack(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_hstack_single_block() {
        let block = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = hstack(std::slice::from_ref(&block));
        assert_eq!(result, block);
    }

    #[test]
    fn test_hstack_multiple_blocks() {
        let block1 = vec![vec![1.0, 2.0], vec![5.0, 6.0]];
        let block2 = vec![vec![3.0, 4.0], vec![7.0, 8.0]];
        let result = hstack(&[block1, block2]);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result[1], vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_holiday_features_no_holidays() {
        use chrono::NaiveDate;
        let timestamps = vec![
            NaiveDate::from_ymd_opt(2020, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 2)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
        ];

        let features = holiday_features(&timestamps, &[], 0, 0);
        assert_eq!(features.len(), 2);
        assert!(features[0].is_empty());
    }

    #[test]
    fn test_holiday_features_exact_match() {
        use chrono::NaiveDate;
        let timestamps = vec![
            NaiveDate::from_ymd_opt(2020, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 15)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 20)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
        ];
        let holidays = vec!["2020-01-15".to_string()];

        let features = holiday_features(&timestamps, &holidays, 0, 0);

        assert_eq!(features.len(), 3);
        assert_eq!(features[0][0], 0.0); // Not a holiday
        assert_eq!(features[1][0], 1.0); // Holiday!
        assert_eq!(features[2][0], 0.0); // Not a holiday
    }

    #[test]
    fn test_holiday_features_with_windows() {
        use chrono::NaiveDate;
        let timestamps = vec![
            NaiveDate::from_ymd_opt(2020, 1, 14)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(), // 1 day before
            NaiveDate::from_ymd_opt(2020, 1, 15)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(), // Holiday
            NaiveDate::from_ymd_opt(2020, 1, 16)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(), // 1 day after
            NaiveDate::from_ymd_opt(2020, 1, 17)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(), // 2 days after
        ];
        let holidays = vec!["2020-01-15".to_string()];

        // Window: 1 day before to 1 day after
        let features = holiday_features(&timestamps, &holidays, -1, 1);

        assert_eq!(features.len(), 4);
        // Should have 3 features (window range = -1 to 1 = 3 days)
        assert_eq!(features[0].len(), 3);

        // Day before should activate first window feature
        assert_eq!(features[0][0], 1.0);
        // Holiday day should activate middle window feature
        assert_eq!(features[1][1], 1.0);
        // Day after should activate last window feature
        assert_eq!(features[2][2], 1.0);
        // 2 days after should have no features active
        assert!(features[3].iter().all(|&x| x == 0.0));
    }
}
