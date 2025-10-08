ExoStack: Ultimate Exoplanet Classification & Habitability Prediction

ExoStack — a NASA-based machine learning system that predicts exoplanet types and habitability potential using advanced feature engineering, optimized models, and an ensemble stacking architecture.

<p align="center"> <img src="https://upload.wikimedia.org/wikipedia/commons/e/e5/Exoplanet_Comparison_Kepler-20.jpg" width="700"/> </p>
🚀 Features
🎯 1. Advanced Feature Engineering

Over 20 domain-specific engineered features for precise star–planet interaction modeling.

Category	Example Features
Ratios & Geometry	planet_star_radius_ratio, radius_ratio_squared, depth_radius_consistency
Transit & Light Curve	transit_duration_ratio_log, transit_signal_log, central_transit
Physical Properties	stellar_density, temp_diff, period_category
Signal Quality	snr_log, high_snr (binary)
🤖 2. Hyperparameter-Optimized Models

Each model is tuned with SMOTEENN balancing for class accuracy and generalization.

LightGBM – 500 estimators, depth=12

XGBoost – Optimized hyperparameters

CatBoost – Newly added

Random Forest – 600 trees

Extra Trees – 600 trees

Gradient Boosting – 400 estimators

<p align="center"> <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Machine_learning_diagram_en.svg" width="600"/> </p>
🧠 3. Ultimate Stacking Ensemble

Combines the top 4 models using a LightGBM meta-learner and 10-fold cross-validation for robust generalization.

<p align="center"> <img src="https://upload.wikimedia.org/wikipedia/commons/b/b9/Stacking_generalization.png" width="500"/> </p>
🔭 4. Intelligent Prediction Function
from exostack import predict_exoplanet_ultimate

result = predict_exoplanet_ultimate(input_data)
print(result)


Generates not only predictions, but also scientific diagnostics:

🔬 Scientific Analysis

🌍 Habitability Assessment

🪐 Planet Type Classification

🌟 Star Feature Analysis

📈 SNR Quality Control

🧩 5. Example Scenarios
Type	Description
🌍 Earth-like	Balanced temperature, habitable zone
🔥 Hot Jupiter	Massive gas giant with short orbit
🌏 Super-Earth	Rocky, potentially habitable planet
💙 Mini-Neptune	Medium density, water-rich atmosphere
<p align="center"> <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Exoplanet_illustration.jpg" width="650"/> </p>
📊 Performance

Expected Accuracy: 85% – 95%

Why So Powerful?

✅ NASA-verified dataset
✅ 20+ astrophysical engineered features
✅ SMOTEENN data balancing
✅ Hyperparameter optimization
✅ Mutual info feature selection
✅ Multi-model ensemble learning

<p align="center"> <img src="https://upload.wikimedia.org/wikipedia/commons/2/23/Confusion_matrix_diagram.svg" width="450"/> </p>
🧪 Usage
from exostack import predict_exoplanet_ultimate

result = predict_exoplanet_ultimate(input_data)
print(result)


Output Includes:

Planet type

Habitability score

Prediction confidence

Model contribution analysis

📁 Project Structure
exostack/
│
├── models/
│   ├── lightgbm.pkl
│   ├── xgboost.pkl
│   ├── catboost.pkl
│   ├── random_forest.pkl
│   ├── extra_trees.pkl
│   ├── gradient_boost.pkl
│   └── ensemble.pkl
│
├── scaler.pkl
├── selector.pkl
├── metadata.json
└── README.md

🧬 Scientific Sources

NASA Exoplanet Archive

Kepler Mission Data

“Feature Engineering for Exoplanet Detection”, ApJ 2023

🧑‍💻 Contributing

Fork the repository

Create your feature branch (feature/*)

Commit your changes

Submit a pull request

You’ll be listed in the Contributors section 🌟

🪐 License

MIT License — free for open-source use and distribution.

🌟 Author

Emre Uludaşdemir
AstroData Scientist | Machine Learning Engineer
📧 uludasdemire@mef.edu.tr

🌐 LinkedIn
 | GitHub

<p align="center"> <img src="https://upload.wikimedia.org/wikipedia/commons/d/dc/Kepler_space_telescope_artwork.jpg" width="500"/> </p>
