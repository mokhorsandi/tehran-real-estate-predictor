# Tehran Apartment Rent Predictor

A machine learning web application for predicting apartment rental prices in Tehran with flexible rent/deposit configurations.

## Features

- 🏠 **Neighborhood Selection**: Choose from 342 Tehran neighborhoods
- 📊 **Smart Predictions**: ML model with 87% accuracy (R² = 0.87)
- 💰 **Flexible Payments**: Interactive slider to adjust rent vs deposit ratio
- 📈 **Visual Breakdown**: See payment structure at a glance

## How to Run

### Prerequisites
```bash
pip install streamlit pandas numpy scikit-learn
```

### Launch the App
```bash
cd "/Users/yashakhorsandi/Desktop/پروژه دیوار"
streamlit run rent_app.py
```

The app will open in your browser at `http://localhost:8501`

## How It Works

### Payment Structure
The app calculates total monthly cost using:
```
Total Monthly Cost = Monthly Rent + (Deposit × 3%)
```

### Slider Functionality
- **100% Rent**: Maximum monthly payment, zero deposit
- **50% Mix**: Balanced rent and deposit  
- **0% Rent**: Zero monthly payment, maximum deposit (full rahn)

**Example**: For 30M Toman total monthly cost:
- **100%**: 30M/month + 0 deposit
- **50%**: 15M/month + 500M deposit  
- **0%**: 0/month + 1,000M deposit

## Model Performance

- **Algorithm**: Random Forest Regressor
- **Test R²**: 0.872
- **Test MAE**: 8.65M Toman
- **Features**: 29 (location, size, amenities, NLP-derived)
- **Training Data**: 44,060 Tehran apartments

## Files

- `rent_app.py` - Main Streamlit application
- `rf_model.pkl` - Trained Random Forest model
- `feature_names.pkl` - List of model features
- `neighborhood_stats.csv` - Neighborhood coordinates
- `model_metadata.pkl` - Model performance metrics

## Project Structure

```
پروژه دیوار/
├── rent_app.py              # Streamlit web app
├── tehran_rent_analysis.ipynb  # Data analysis & model training
├── rf_model.pkl             # Trained model
├── real_estate_ads.csv      # Original dataset
├── tehran_rent_cleaned.csv  # Processed dataset
└── README_APP.md            # This file
```

## Usage

1. Select a neighborhood from the dropdown
2. Adjust property features (size, age, amenities)
3. Use the slider to configure rent/deposit split
4. View predicted costs and payment breakdown

## Technical Details

**Input Features**:
- Location: Neighborhood, lat/lon coordinates
- Property: Size, age, amenities
- Text-derived: Features extracted from Persian descriptions
- Proximity: Metro, parks, shopping centers

**Model Training**: See `tehran_rent_analysis.ipynb` for full pipeline

## License

Data Science Project - Tehran Real Estate Analysis 2025
