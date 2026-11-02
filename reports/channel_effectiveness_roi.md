# Channel Effectiveness & ROAS Analysis

## Scope and method

This analysis reuses the fitted Marketing Mix Model and its saved outputs. It does not retrain or alter the MMM. Channel contribution is taken from `channel_summary.csv`; ROAS is incremental revenue divided by advertising spend. Marginal ROAS and response curves are evaluated using the saved response model. Bootstrap intervals are taken from the moving-block bootstrap output generated in Notebook 08.

## Important financial limitation

The dataset does not provide channel-level profit margins or incremental profit. Therefore, this report uses revenue-based ROAS and does not claim to calculate profit ROI. ROAS above 1 means estimated incremental revenue exceeds ad spend, not necessarily that the channel is profitable after product, fulfilment and other costs.

## Channel comparison

| Channel | Spend (LKR) | Incremental revenue (LKR) | ROAS | Marginal ROAS | ROAS 90% interval |
|---|---:|---:|---:|---:|---:|
| Outdoor | 94,057,802 | 575,825,832 | 6.12 | 1.91 | 1.91–7.30 |
| YouTube | 79,913,729 | 332,995,042 | 4.17 | 1.16 | 0.10–6.70 |
| Google Search | 119,353,135 | 486,988,018 | 4.08 | 1.27 | 0.15–4.78 |
| TV | 156,964,242 | 154,091,107 | 0.98 | 0.48 | 0.00–2.22 |
| Facebook/Instagram | 158,343,617 | 113,512,941 | 0.72 | 0.41 | 0.00–2.56 |
| Radio | 60,107,999 | 0 | 0.00 | 0.00 | 0.00–7.34 |

## Key findings

- Highest point-estimate ROAS: **Outdoor** (6.12).
- Lowest point-estimate ROAS: **Radio** (0.00).
- Highest marginal ROAS at current spend: **Outdoor** (1.91).
- ROAS is an average return over the observed period; marginal ROAS estimates the return from the next unit of spend. They answer different questions.
- The response curves show the modelled change in incremental monthly revenue as sustained monthly spend changes. A declining slope indicates diminishing marginal returns.

## Recommendations

- **Outdoor — Consider gradual testing of additional spend.** Average and marginal revenue returns are above 1, with relatively strong bootstrap support. Increase in controlled steps and monitor. Point ROAS = 6.12; marginal ROAS = 1.91; bootstrap probability ROAS > 1 = 99.0%.
- **YouTube — Consider gradual testing of additional spend.** Average and marginal revenue returns are above 1, with relatively strong bootstrap support. Increase in controlled steps and monitor. Point ROAS = 4.17; marginal ROAS = 1.16; bootstrap probability ROAS > 1 = 89.5%.
- **Google Search — Consider gradual testing of additional spend.** Average and marginal revenue returns are above 1, with relatively strong bootstrap support. Increase in controlled steps and monitor. Point ROAS = 4.08; marginal ROAS = 1.27; bootstrap probability ROAS > 1 = 85.4%.
- **TV — Review and consider reducing or reallocating spend.** Both average and next-LKR revenue returns are below 1. Validate business objectives and uncertainty before reducing. Point ROAS = 0.98; marginal ROAS = 0.48; bootstrap probability ROAS > 1 = 46.1%.
- **Facebook/Instagram — Review and consider reducing or reallocating spend.** Both average and next-LKR revenue returns are below 1. Validate business objectives and uncertainty before reducing. Point ROAS = 0.72; marginal ROAS = 0.41; bootstrap probability ROAS > 1 = 60.5%.
- **Radio — Investigate before reallocating.** The bootstrap assigns meaningful probability to a zero effect; check data quality, channel activity and model uncertainty. Point ROAS = 0.00; marginal ROAS = 0.00; bootstrap probability ROAS > 1 = 55.2%.

## Limitations

- The analysis is conditional on the fitted MMM specification and available data.
- The project has 36 monthly observations, so estimates and bootstrap intervals should be interpreted cautiously.
- Bootstrap intervals quantify uncertainty under the notebook's resampling procedure and fixed model settings; they do not capture every source of model uncertainty.
- A modelled channel contribution is an attribution estimate, not proof of causation.
- Budget changes should be tested gradually and monitored against future actual outcomes.

## Generated files

- `outputs/channel_analysis/channel_performance.csv`
- `outputs/channel_analysis/diminishing_returns_analysis.csv`
- `outputs/channel_analysis/channel_recommendations.csv`
- `reports/figures/channel_roas_with_uncertainty.png`
- `reports/figures/channel_spend_vs_contribution.png`
- `reports/figures/channel_diminishing_returns.png`
- `reports/figures/channel_marginal_roas_scenarios.png`