# Tehran Rent Predictor Web App 🏠

Interactive web application to predict apartment rental prices in Tehran with adjustable rent/credit combinations.

## Quick Start

### 1. Activate virtual environment
```bash
source .venv/bin/activate
```

### 2. Run the app
```bash
streamlit run rent_predictor_app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

### 🏘️ Neighborhood Selection
- Choose from 342 Tehran neighborhoods
- Each neighborhood has unique price characteristics

### 🏗️ Property Features
- **Basic:** Size, age, location coordinates
- **Amenities:** Parking, elevator, storage, balcony, etc.
- **Premium:** Master room, luxurious finishes, VIP building
- **Location:** Near metro, park, shopping centers
- **Style:** Modern, traditional, renovated, duplex

### 💰 Rent/Credit Slider
- **100% Rent:** Pure monthly rent, no deposit
- **50/50 Mix:** Balanced rent + deposit
- **0% Rent:** Maximum deposit, minimal monthly rent

The slider uses the 3% conversion rule (standard in Tehran):
- Deposit (credit) × 0.03 = Monthly equivalent
- Example: 1 billion Toman deposit = 30 million monthly

## Model Performance

- **Algorithm:** Random Forest Regressor
- **Test R²:** 0.87 (explains 87% of variance)
- **Test MAE:** ~8.65 million Toman
- **Training Data:** 62,944 Tehran apartment rentals
- **Features:** 29 property characteristics

## Files

### Your Apps (Don't Mix These!)
- `app.py` - **Sales** prediction app (original, unchanged)
- `rent_predictor_app.py` - **Rent** prediction app (new)

### Model Files (Rent)
- `tehran_rent_model.pkl` - Trained Random Forest model
- `rent_feature_names.pkl` - Feature list
- `rent_neighborhoods.pkl` - 342 neighborhoods
- `rent_model_metadata.pkl` - Model performance metrics

### Model Files (Sales - your original)
- `tehran_price_model.pkl` - Sales prediction model
- `rf_model.pkl` - Random Forest for sales
- `feature_names.pkl` - Sales features
- `model_metadata.pkl` - Sales metadata

### Notebooks
- `tehran_rent_analysis.ipynb` - Rent data analysis & modeling
- `tehran_price_model.ipynb` - Sales data analysis (original)

## Example Usage

1. **Quick Estimate:**
   - Select neighborhood: "Elahieh"
   - Size: 100 m²
   - Age: 5 years
   - Check: Parking, Elevator, Renovated
   - See prediction instantly

2. **Rent/Credit Calculator:**
   - Get base prediction
   - Move slider to 50%
   - See split: 50% as monthly rent, 50% as deposit equivalent

3. **Compare Scenarios:**
   - Try different neighborhoods
   - Toggle amenities to see price impact
   - Adjust rent/credit mix for affordability

## Tips

- Deposit amounts are typically 10-30x monthly rent in Tehran
- 3% is the standard conversion rate (may vary by landlord)
- Premium neighborhoods + amenities significantly increase prices
- The model works best for typical apartments (not extreme luxury or budget)

## Support

If you see errors about missing model files, make sure you ran the last cell in `tehran_rent_analysis.ipynb` to save all model files.
