# Trader Performance vs. Market Sentiment Analysis

A comprehensive machine learning analysis exploring the relationship between Bitcoin market sentiment (Fear/Greed Index) and trader performance metrics from Hyperliquid exchange data.

## Project Overview

This project investigates how market sentiment affects trader profitability and identifies trading patterns under different sentiment conditions. Using data from the Crypto Fear & Greed Index and Hyperliquid trader execution data, we build predictive models to forecast trade profitability based on market sentiment and trading characteristics.

## Objectives

1. **Explore Sentiment-Profitability Relationship**: Analyze how market sentiment (Fear vs. Greed) impacts trader profitability
2. **Identify Trading Patterns**: Discover distinct trading behaviors under different sentiment conditions
3. **Build Predictive Models**: Develop machine learning models to predict trade profitability
4. **Model Comparison**: Evaluate multiple algorithms to identify the best performer

## Data Sources

### fear_greed_index.csv
- **Source**: Crypto Fear & Greed Index
- **Contents**: Daily market sentiment classification (Fear/Greed)
- **Date Range**: Historical Bitcoin market sentiment data

### historical_data.csv
- **Source**: Hyperliquid Exchange
- **Contents**: Trader execution data including:
  - Account information
  - Symbol (trading pair)
  - Execution price
  - Position size
  - Closed PnL (Profit/Loss)
  - Leverage used
  - Trade side (Buy/Sell)
  - Execution timestamp

## Project Structure

```
Financial Trading Analyticals/
├── trader_sentiment_analysis.ipynb   # Main analysis notebook
├── data/
│   ├── fear_greed_index.csv         # Market sentiment data
│   └── historical_data.csv          # Trader execution data
└── README.md                        # Project documentation
```

## Analysis Workflow

### 1. Data Preparation
- Load sentiment and trader execution datasets
- Merge datasets using temporal alignment (merge_asof)
- Handle missing values and data cleaning
- Data validation and shape confirmation

### 2. Feature Engineering
- **Target Variable**: Binary classification (Profitable vs. Loss)
- **Features Created**:
  - Sentiment encoding (Fear/Greed classification)
  - Side encoding (Buy/Sell)
  - Time-based features (Hour, Day of Week, Month)
  - Absolute position size
  - PnL ratio
  - Leverage metrics

### 3. Exploratory Data Analysis (EDA)
- **Profitability by Sentiment**: Compare win rates and average PnL across sentiment conditions
- **Trading Behavior**: Analyze leverage and position size variations
- **Feature Correlations**: Identify which features most influence profitability
- **Visualizations**:
  - PnL distribution by sentiment (box plots, histograms)
  - Win rate comparisons
  - Leverage and position size analysis
  - Feature importance charts

### 4. Predictive Modeling
Evaluated 6 different machine learning algorithms:

| Model | Accuracy | F1-Score | Best Use Case |
|-------|----------|----------|---------------|
| **Random Forest** ⭐ | 0.8497 | 0.8166 | **Best Overall** - Balanced performance |
| Decision Tree | 0.8173 | 0.7698 | Interpretability |
| K-Nearest Neighbors | 0.7609 | 0.7026 | Simple baseline |
| Gradient Boosting | 0.6936 | 0.5926 | Sequential learning |
| AdaBoost | 0.6381 | 0.5682 | Ensemble boost |
| Logistic Regression | 0.5887 | 0.0000 | Linear baseline |

### 5. Ensemble Method
- Created Voting Classifier combining top 3 models
- Improved robustness through model combination
- Enhanced prediction reliability

## Key Findings

### Sentiment Impact on Profitability
- Market sentiment shows measurable correlation with trader outcomes
- Fear periods vs. Greed periods exhibit different performance patterns
- Win rates vary significantly across sentiment classifications

### Feature Importance
Top predictive features for trade profitability:
1. **Sentiment** - Market Fear/Greed classification
2. **Leverage** - Position leverage ratio
3. **Side** - Trade direction (Buy/Sell)
4. **Position Size** - Trade volume
5. **Time Features** - Hour, day, and month

### Model Performance
- **Best Model**: Random Forest (F1: 0.8166, Accuracy: 0.8497)
- **Train/Test Split**: 80/20 stratified
- **Precision**: 81.97% - Reliable profitability predictions
- **Recall**: 81.35% - Captures most profitable trades

## Installation & Setup

### Requirements
- Python 3.7+
- Jupyter Notebook or JupyterLab

### Dependencies
```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

### Quick Start

1. **Clone/Download Project**
```bash
cd "Financial Trading Analyticals"
```

2. **Install Dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

3. **Run Notebook**
```bash
jupyter notebook trader_sentiment_analysis.ipynb
```

## Usage

### Running the Full Analysis

1. Open `trader_sentiment_analysis.ipynb` in Jupyter
2. Execute cells sequentially:
   - **Cells 1-6**: Import libraries and load data
   - **Cells 7-11**: Data cleaning and feature engineering
   - **Cells 12-17**: Exploratory Data Analysis (EDA)
   - **Cells 18-21**: Basic model training (Logistic Regression)
   - **Cells 22-30**: Multi-model comparison and evaluation
   - **Cells 31-34**: Ensemble methods and final recommendations

### Making Predictions on New Data

```python
# Load best model
best_model = RandomForestClassifier(...)

# Prepare features
new_data = X_test  # Your test data

# Make predictions
predictions = best_model.predict(new_data)
probabilities = best_model.predict_proba(new_data)
```

## Model Comparison Details

### Random Forest (Winner)
- **Accuracy**: 84.97%
- **Precision**: 81.97%
- **Recall**: 81.35%
- **F1-Score**: 0.8166
- **ROC-AUC**: 0.9243
- **Training Time**: 3.15s
- **Strengths**: Handles non-linear relationships, robust to outliers
- **Best for**: Production deployment

### Decision Tree
- **Accuracy**: 81.73%
- **F1-Score**: 0.7698
- **Strengths**: Highly interpretable, fast training
- **Use case**: When explainability is critical

## Visualizations

The notebook generates several key visualizations:

1. **PnL Distribution**: Box plots and histograms by sentiment
2. **Win Rate Comparison**: Bar charts showing profitability percentages
3. **Leverage Analysis**: Position size and leverage by sentiment
4. **Feature Correlation**: Heatmap of feature relationships
5. **Model Comparison**: Performance metrics across algorithms
6. **Confusion Matrix**: Detailed prediction accuracy breakdown
7. **Feature Importance**: Ranking of influential features

## Performance Metrics

### Classification Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate among predictions
- **Recall**: Coverage of actual positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve

### Dataset Statistics
- **Total Trades**: ~1000+ records
- **Profitable Trades**: Class distribution varies by sentiment
- **Training Set**: 80% of data
- **Test Set**: 20% of data (stratified)

## Next Steps & Recommendations

### For Immediate Use
1. Deploy Random Forest model for trade profitability prediction
2. Monitor model performance on new data
3. Validate predictions against actual outcomes

### For Model Improvement
1. **Feature Engineering**: Add technical indicators (RSI, MACD, Bollinger Bands)
2. **Data Enhancement**: Incorporate order book imbalance, volume profile
3. **Hyperparameter Tuning**: Grid search on best models
4. **Cross-Validation**: 5-fold cross-validation for robust estimates
5. **Ensemble Optimization**: Fine-tune voting weights

### For Production Deployment
1. Save best model to disk for inference
2. Build real-time sentiment-based trading signals
3. Implement continuous model monitoring
4. Create alert system for sentiment shifts
5. A/B test model predictions vs. baseline strategies

## Insights for Traders

- **Sentiment Timing**: Adjust position sizing based on sentiment levels
- **Risk Management**: Leverage utilization correlates with sentiment
- **Trade Selection**: Sentiment-aware trade filtering improves outcomes
- **Time Patterns**: Certain hours/days show stronger sentiment effects

## Limitations & Caveats

- Analysis reflects historical patterns; past performance ≠ future results
- Market conditions may change, requiring model retraining
- Sentiment index may not capture all market dynamics
- Sample data from specific time period; needs validation across markets
- Model assumes feature distributions remain stable

## Contributing

To extend this analysis:

1. Add more data (multi-year historical)
2. Incorporate additional sentiment indicators
3. Analyze account-level performance patterns
4. Build real-time prediction pipeline
5. Test different market conditions (bull/bear markets)

## License

Private project for PrimeTrade assessment purposes.

## Contact & Questions

For questions about this analysis, refer to the notebook documentation and comments within each cell.

---

**Last Updated**: May 2026  
**Analysis Status**: Complete - Production Ready  
**Best Model**: Random Forest (F1: 0.8166)
