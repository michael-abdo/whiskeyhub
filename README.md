# WhiskeyHub ML Analysis Project

A machine learning analysis project for whiskey recommendation systems, featuring data sparsity analysis, linear regression modeling, and hybrid recommendation system planning.

## 📁 Project Structure

```
whiskeyhub_project/
├── README.md                   # This file
├── .gitignore                 # Git ignore rules
│
├── 📊 data/                   # Raw data files
│   ├── WhiskeyHubMySQL_6_13_2025/      # Primary dataset (empty tables)
│   └── WhiskeyHubMySQL_6_13_2025_pt2/  # Working dataset with data
│       ├── flights.csv         # Flight/session data (376 rows)
│       ├── flight_pours.csv    # Pour details (415 rows)
│       ├── flight_notes.csv    # Tasting notes (3,348 rows)
│       └── whishkeys.csv       # Whiskey metadata (2,987 rows)
│
├── 🔬 scripts/                # Analysis scripts
│   ├── db_connect.py          # Data loading and merging
│   ├── sparsity_analysis.py   # Data density analysis
│   └── linear_model.py        # Linear regression model
│
├── 📈 results/                # Analysis outputs
│   ├── full_joined.csv        # Merged dataset (3,348 rows)
│   ├── sparsity_analysis_results.txt
│   ├── linear_model_results.txt
│   ├── linear_model_predictions.png
│   ├── linear_model_feature_importance.png
│   └── linear_model_residuals.png
│
├── 📚 docs/                   # Documentation
│   ├── analysis_summary.md    # Key findings summary
│   ├── results_interpretation.md  # Business insights
│   ├── hybrid_recommendation_plan.md  # Implementation plan
│   ├── project_description.md  # Original job requirements
│   └── whiskeyhub_mvp_cli_plan.md     # MVP plan
│
├── 🎨 demo/                   # Frontend demo application
│   ├── index.html             # Main landing page
│   ├── recommendations.html   # Recommendation interface
│   ├── rating-prediction.html # Rating prediction
│   ├── flavor-profile.html    # Custom flavor matching
│   ├── sensitivity-analysis.html  # User preference analysis
│   ├── gift-guide.html        # Gift recommendations
│   ├── ml-simulator.js        # ML simulation logic
│   ├── whiskey-database.js    # Mock whiskey data
│   └── screenshots_simple/    # UI screenshots
│
├── ⚙️ config/                 # Configuration files
│   └── .tmux_session_info.json
│
└── 📎 misc/                   # Miscellaneous files
    └── zach.txt
```

## 🚀 Quick Start

### Run Data Analysis
```bash
# 1. Merge data from CSV files
cd scripts
python3 db_connect.py

# 2. Analyze data sparsity
python3 sparsity_analysis.py

# 3. Train linear regression model
python3 linear_model.py
```

### View Results
- **Data Analysis**: Check `results/` directory for outputs
- **Visualizations**: View PNG files for model performance charts
- **Documentation**: Read `docs/analysis_summary.md` for key findings

### Explore Demo
- Open `demo/index.html` in a browser
- Interactive demo showcasing recommendation features

## 📊 Key Findings

- **Data Density**: 58.43% (exceptional for recommendation systems)
- **Model Performance**: R² = 0.765, RMSE = 0.613
- **Top Predictors**: Complexity (+0.772), Finish Duration (+0.612)
- **Recommendations**: Hybrid collaborative + content-based approach

## 🛠 Requirements

```bash
pip install pandas sqlalchemy scikit-learn matplotlib seaborn
```

## 📝 Next Steps

See `docs/hybrid_recommendation_plan.md` for detailed implementation roadmap for production recommendation system.

## 🔗 Repository

GitHub: https://github.com/michael-abdo/whiskeyhub