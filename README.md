<div align="center">

<h1>Marketing Mix Modelling & Budget Optimisation</h1>

<p>
  <strong>Measure advertising impact, understand channel effectiveness, and maximise ROI under a fixed marketing budget.</strong>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-Analytics-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Scikit--learn-Modelling-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="Scikit-learn" />
  <img src="https://img.shields.io/badge/SciPy-Optimisation-8CAAE6?style=flat-square&logo=scipy&logoColor=white" alt="SciPy" />
  <img src="https://img.shields.io/badge/Statsmodels-Statistics-4051B5?style=flat-square" alt="Statsmodels" />
  <img src="https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=flat-square&logo=powerbi&logoColor=black" alt="Power BI" />
</p>

</div>

---

<h2 align="center">About the Project</h2>

**Marketing Mix Modelling & Budget Optimisation** is a group analytics project designed to measure the impact of advertising expenditure across six marketing channels.

The project covers the complete analytical workflow—from data cleaning and exploratory analysis to Marketing Mix Modelling (MMM), channel contribution analysis, ROI/ROAS evaluation, budget optimisation, dashboard development, and final business recommendations.

The model accounts for advertising carryover, diminishing returns, time delays, seasonality, promotions, holidays, inflation, and other relevant external factors. Its response functions support an optimisation algorithm that allocates a fixed advertising budget across channels to maximise predicted incremental revenue or profit where the available data allows.

---

<h2 align="center">Core Features</h2>

<table>
<tr>
<td width="50%" valign="top">

### Data Engineering & EDA

- Data-quality assessment and cleaning
- Missing-value, duplicate, data-type, and outlier handling
- Sales and advertising-spend trend analysis
- Correlation and external-factor analysis
- Time-based feature engineering
- Chronological train/test preparation

</td>
<td width="50%" valign="top">

### Marketing Mix Modelling

- Baseline and advanced regression models
- Advertising carryover through adstock
- Diminishing returns through saturation curves
- Promotion, holiday, economic, and seasonal controls
- Time-aware model validation
- Channel contribution and response functions

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Channel Effectiveness & ROI

- Channel-level sales contribution estimates
- ROI and ROAS calculations where appropriate
- Channel-efficiency comparisons and rankings
- High- and low-performing channel identification
- Diminishing-returns analysis
- Actionable spend recommendations

</td>
<td width="50%" valign="top">

### Budget Optimisation

- Fixed-budget allocation across six channels
- SciPy-based constrained optimisation
- Minimum and maximum channel-spend limits
- Current-versus-optimised allocation comparison
- Alternative budget scenario analysis
- Estimated revenue and efficiency improvements

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Interactive Dashboard

- KPI and executive-summary views
- Sales and advertising-spend trends
- Channel contribution and ROI/ROAS visuals
- Diminishing-returns curves
- Current-versus-optimised budget comparison
- Interactive filters and scenario exploration

</td>
<td width="50%" valign="top">

### Reporting & Documentation

- Reproducible methodology and assumptions
- Model evaluation and diagnostic results
- Integrated findings from every analytical stage
- Limitations and references
- Data-driven business recommendations
- Documented individual team contributions

</td>
</tr>
</table>

---

<h2 align="center">Project Objectives</h2>

- Prepare a clean, reliable, analysis-ready marketing dataset.
- Understand sales trends and advertising expenditure across six channels.
- Quantify each advertising channel's contribution to sales revenue.
- Model delayed advertising effects using adstock transformations.
- Model diminishing returns using saturation transformations.
- Control for relevant external and seasonal factors.
- Compare baseline and advanced regression-based MMM approaches.
- Evaluate channel effectiveness using contribution, ROI, and ROAS.
- Optimise a fixed advertising budget under realistic constraints.
- Present findings through Power BI and a professional final report.

---

<h2 align="center">Technology Stack</h2>

### Data Processing

- Python
- Pandas
- NumPy

### Modelling & Statistics

- Scikit-learn
- Statsmodels
- SciPy

### Visualisation & Business Intelligence

- Matplotlib
- Seaborn
- Power BI

### Working Environment

- Jupyter Notebook
- Git and GitHub

---

<h2 align="center">Analytical Architecture</h2>

```text
Provided Marketing Dataset
          |
          | Sales · Channel Spend · Promotions · Holidays · Economic Factors
          v
  Data Engineering & Preprocessing
          | Cleaning · Validation · Feature Engineering · Chronological Split
          v
  Exploratory Data Analysis
          | Trends · Correlations · Seasonality · External-Factor Analysis
          v
  Marketing Mix Modelling
          | Baseline Regression · Adstock · Saturation · Control Variables
          v
  Model Evaluation
          | RMSE · MAE · R² · Residuals · Time-Aware Validation
          v
  Channel Response Functions
          |------------------------------|
          v                              v
  ROI/ROAS Analysis              Budget Optimisation
          |                      Constraints · Scenarios
          |------------------------------|
                         v
                Power BI Dashboard
                         |
                         v
            Final Report & Recommendations
```

---

<h2 align="center">Completed Project Components</h2>

### Data Engineering, Preprocessing & EDA

The raw marketing data was validated, cleaned, transformed, and organised chronologically. Exploratory analysis examined sales trends, channel spending, correlations, seasonality, promotions, holidays, inflation, and other external influences.

### Marketing Mix Modelling

Baseline and enhanced regression models were developed and evaluated. Adstock transformations captured carryover effects, saturation functions represented diminishing returns, and control variables separated media impact from external demand factors.

### Channel Effectiveness & ROI Analysis

Model contributions were converted into channel-level performance measures. ROI/ROAS comparisons, efficiency rankings, response curves, and spend recommendations provide a clear view of channel effectiveness.

### Budget Optimisation

A constrained SciPy optimisation model uses the estimated response functions to redistribute a fixed advertising budget. It respects channel spending bounds and compares current allocations with optimised scenarios.

### Dashboard & Visualisation

The Power BI dashboard combines KPIs, sales and spending trends, channel contributions, ROI/ROAS comparisons, response curves, and budget scenarios in an interactive business-facing view.

### Reporting & Recommendations

The final analysis consolidates the methodology, model evaluation, channel findings, optimisation results, limitations, and evidence-based recommendations.

---

<h2 align="center">Project Workflow</h2>

| Stage | Input | Output |
|---:|---|---|
| Data Engineering & EDA | Raw marketing data | Clean data, engineered features, quality summary, chronological split |
| Marketing Mix Modelling | Validated data and features | Model estimates, channel contributions, response functions |
| Channel Effectiveness | Validated MMM outputs | ROI/ROAS analysis, rankings, response insights |
| Budget Optimisation | Response functions and constraints | Optimised allocations and scenario comparisons |
| Dashboard | Approved analytical outputs | Interactive Power BI views |
| Reporting | Methods, results, visuals, and limitations | Final recommendations and project documentation |

---

<h2 align="center">Modelling Requirements</h2>

### Time-Aware Evaluation

Training, validation, and testing preserve chronological order to prevent future-data leakage.

### Adstock

Adstock transformations represent the delayed and continuing influence of advertising after the original expenditure period. Parameter selection is documented and reproducible.

### Saturation & Diminishing Returns

Saturation functions represent the declining incremental benefit of additional advertising expenditure. The resulting response curves support contribution analysis and optimisation.

### External Factors

The model controls for available promotions, holidays, special events, inflation, economic conditions, seasonality, long-term trends, and other relevant non-media variables.

### Model Evaluation

| Evaluation Area | Measures |
|---|---|
| Prediction error | RMSE and MAE |
| Explained variation | R² |
| Error behaviour | Residual diagnostics |
| Temporal reliability | Performance across time periods |
| Business validity | Plausible coefficients, contributions, and response curves |

---

<h2 align="center">ROI & ROAS Framework</h2>

Financial metrics are used consistently, with revenue, cost, margin, and attribution assumptions clearly stated.

| Metric | Intended Use |
|---|---|
| **ROAS** | Compare advertising-attributed revenue with advertising expenditure |
| **ROI** | Compare return with investment cost, preferably using profit when reliable margin data is available |

Where profit margins or incremental revenue cannot be estimated reliably, the limitation is stated to avoid overstating financial conclusions.

---

<h2 align="center">Budget Optimisation Rules</h2>

- Preserve the specified total advertising budget.
- Allocate spending across all eligible channels.
- Respect minimum and maximum channel-spend constraints.
- Use MMM response functions instead of assuming constant returns.
- Produce a feasible allocation for every reported scenario.
- Compare the optimised allocation with the current allocation.
- Report the expected incremental outcome and relevant assumptions.

---

<h2 align="center">Dashboard Requirements</h2>

| Dashboard Area | Content |
|---|---|
| Executive overview | Total sales, advertising spend, ROI/ROAS, and key findings |
| Trend analysis | Sales and channel expenditure over time |
| Channel performance | Contributions, effectiveness, efficiency, and rankings |
| Response analysis | Adstock effects and diminishing-returns curves |
| Budget comparison | Current and optimised channel allocations |
| Scenario analysis | Expected outcomes under alternative budgets and constraints |
| Interactivity | Date, channel, and scenario filters where appropriate |

---

<h2 align="center">Expected Deliverables</h2>

| Deliverable | Description |
|---|---|
| Cleaned dataset and EDA | Analysis-ready data, quality summary, features, trends, and visualisations |
| Marketing Mix Model | Evaluated model with adstock, saturation, controls, and channel contributions |
| Channel effectiveness analysis | ROI/ROAS comparisons, efficiency rankings, and recommendations |
| Budget optimisation algorithm | Constrained allocation model with current-versus-optimised scenarios |
| Interactive dashboard | Power BI report integrating validated project outputs |
| Final report | Methodology, findings, limitations, references, and recommendations |

---

<h2 align="center">Quality & Collaboration Guidelines</h2>

- Use consistent channel, variable, metric, and model-version names.
- Keep calculations reproducible and document important assumptions.
- Validate every output before handing it to another team.
- Record the source and version of shared datasets and model outputs.
- Use charts with clear titles, labels, units, and legends.
- Distinguish observed values from model estimates and scenario forecasts.
- Report limitations honestly and avoid unsupported causal claims.
- Coordinate before changing shared work or downstream interfaces.
- Review integrated results as a group before final submission.

---

<div align="center">

## Marketing Mix Modelling & Budget Optimisation

Transforming marketing data into measurable channel insights and evidence-based budget decisions.

**End-to-End Analytics · Marketing Effectiveness · Budget Optimisation**

</div>
