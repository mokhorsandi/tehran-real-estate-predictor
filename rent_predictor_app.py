import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Tehran Rent Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load model and data
@st.cache_resource
def load_model():
    with open('tehran_rent_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('rent_feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('rent_model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    with open('rent_neighborhoods.pkl', 'rb') as f:
        neighborhoods = pickle.load(f)
    return model, features, metadata, neighborhoods

model, feature_names, metadata, neighborhoods = load_model()

# Inflation constant (2024 -> 2025)
# Based on Iran's average inflation rate (~40-50% annually)
INFLATION_RATE_2024_2025 = 1.45  # 45% increase from 2024 to 2025

# Title and info
st.title("🏠 Tehran Apartment Rent Predictor")
st.markdown(f"**Model Performance:** R² = {metadata['test_r2']:.3f} | MAE = {metadata['test_mae']:,.0f} Toman (2024 prices)")

st.markdown("""
Estimate monthly rent for apartments in Tehran. Predictions are adjusted for 2025 inflation.
Adjust the rent/credit mix slider to see different payment structures.
""")

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📋 Property Details")
    
    neighborhood = st.selectbox(
        "📍 Neighborhood",
        options=neighborhoods,
        index=0
    )
    
    st.subheader("Basic Features")
    bedrooms = st.selectbox("🛏️ Bedrooms", options=[1, 2, 3, 4, 5, 6], index=1)
    building_size = st.slider("Building Size (m²)", 20, 500, 80, 5)
    property_age = st.slider("Property Age (years)", 0, 100, 10, 1)
    location_latitude = st.number_input("Latitude", value=35.7, format="%.6f")
    location_longitude = st.number_input("Longitude", value=51.4, format="%.6f")
    
    # Inflation adjustment
    st.markdown("---")
    inflation_multiplier = st.slider(
        "💹 Inflation Adjustment (2024 → 2025)",
        min_value=1.0,
        max_value=2.0,
        value=INFLATION_RATE_2024_2025,
        step=0.05,
        help=f"Default {INFLATION_RATE_2024_2025:.0%} reflects typical Tehran rent inflation"
    )
    
    st.subheader("Amenities")
    col_a, col_b = st.columns(2)
    
    with col_a:
        has_parking_text = st.checkbox("Parking", value=True)
        has_elevator_text = st.checkbox("Elevator", value=False)
        has_storage_text = st.checkbox("Storage", value=True)
        has_balcony_text = st.checkbox("Balcony", value=True)
        is_furnished = st.checkbox("Furnished", value=False)
        is_renovated = st.checkbox("Renovated", value=False)
        has_view = st.checkbox("View", value=False)
        has_garden = st.checkbox("Garden", value=False)
        is_duplex = st.checkbox("Duplex", value=False)
        north_facing = st.checkbox("North Facing", value=False)
    
    with col_b:
        south_facing = st.checkbox("South Facing", value=False)
        has_master_room = st.checkbox("Master Room", value=False)
        luxurious = st.checkbox("Luxurious", value=False)
        quiet_area = st.checkbox("Quiet Area", value=False)
        near_metro = st.checkbox("Near Metro", value=False)
        near_park = st.checkbox("Near Park", value=False)
        near_shopping = st.checkbox("Near Shopping", value=False)
        vip_building = st.checkbox("VIP Building", value=False)
        modern = st.checkbox("Modern", value=False)
        traditional = st.checkbox("Traditional", value=False)
    
    st.subheader("Description Details")
    title_length = st.slider("Title Length (chars)", 0, 200, 32, 1)
    description_length = st.slider("Description Length (chars)", 0, 2000, 264, 10)
    description_words = st.slider("Description Words", 0, 500, 47, 1)

with col2:
    st.header("💰 Price Prediction")
    
    # Adjust building size based on bedrooms (rough heuristic)
    # More bedrooms typically means larger space
    size_adjustment = 1.0 + (bedrooms - 2) * 0.1  # 2-bedroom is baseline
    adjusted_size = building_size * size_adjustment
    
    # Create input features
    input_data = {
        'building_size': adjusted_size,  # Adjusted for bedrooms
        'location_latitude': location_latitude,
        'location_longitude': location_longitude,
        'property_age': property_age,
        'has_parking_text': int(has_parking_text),
        'has_elevator_text': int(has_elevator_text),
        'has_storage_text': int(has_storage_text),
        'has_balcony_text': int(has_balcony_text),
        'is_furnished': int(is_furnished),
        'is_renovated': int(is_renovated),
        'has_view': int(has_view),
        'has_garden': int(has_garden),
        'is_duplex': int(is_duplex),
        'north_facing': int(north_facing),
        'south_facing': int(south_facing),
        'has_master_room': int(has_master_room),
        'luxurious': int(luxurious),
        'quiet_area': int(quiet_area),
        'near_metro': int(near_metro),
        'near_park': int(near_park),
        'near_shopping': int(near_shopping),
        'vip_building': int(vip_building),
        'modern': int(modern),
        'traditional': int(traditional),
        'floor_from_text': 0,  # Default
        'amenity_count': sum([has_parking_text, has_elevator_text, has_storage_text, 
                              has_balcony_text, is_furnished, is_renovated, has_view,
                              has_garden, is_duplex, north_facing, south_facing,
                              has_master_room, luxurious, quiet_area, near_metro,
                              near_park, near_shopping, vip_building, modern, traditional]),
        'title_length': title_length,
        'description_length': description_length,
        'description_words': description_words
    }
    
    # Create DataFrame with correct feature order
    input_df = pd.DataFrame([input_data])
    
    # Ensure all features from training are present
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Reorder to match training features
    input_df = input_df[feature_names]
    
    # Make prediction
    predicted_total_monthly_2024 = model.predict(input_df)[0]
    
    # Apply inflation adjustment for 2025
    predicted_total_monthly = predicted_total_monthly_2024 * inflation_multiplier
    
    # Display prediction
    st.metric(
        label="💵 Estimated Total Monthly Cost (2025)",
        value=f"{predicted_total_monthly:,.0f} Toman",
        delta=f"+{(inflation_multiplier-1)*100:.0f}% inflation adjusted"
    )
    
    with st.expander("📊 Show 2024 vs 2025 Comparison"):
        col_2024, col_2025 = st.columns(2)
        with col_2024:
            st.metric("2024 Price", f"{predicted_total_monthly_2024:,.0f} Toman")
        with col_2025:
            st.metric("2025 Price", f"{predicted_total_monthly:,.0f} Toman")
        st.caption(f"Inflation multiplier: {inflation_multiplier:.2f}x")
    
    st.markdown("---")
    
    # Rent/Credit Mix Slider
    st.subheader("⚖️ Adjust Rent/Credit Mix")
    st.markdown("""
    Adjust the slider to change the proportion between monthly rent and upfront deposit (credit).
    - **100% Rent:** All payment as monthly rent, no deposit
    - **50% Rent:** Balanced mix of rent and deposit
    - **0% Rent:** All payment as deposit, minimal monthly rent
    """)
    
    rent_percentage = st.slider(
        "Rent Percentage (%)",
        min_value=0,
        max_value=100,
        value=100,
        step=5,
        help="Slide to adjust the mix between monthly rent and upfront deposit"
    )
    
    # Calculate rent and credit based on slider
    # Formula: total_monthly_rent = rent_value + (credit_value * 0.03)
    # We need to solve for rent_value and credit_value given the percentage
    
    if rent_percentage == 100:
        # Pure rent, no credit
        estimated_rent = predicted_total_monthly
        estimated_credit = 0
    elif rent_percentage == 0:
        # Pure credit, minimal rent
        # total = 0 + credit * 0.03
        # credit = total / 0.03
        estimated_credit = predicted_total_monthly / 0.03
        estimated_rent = 0
    else:
        # Mixed: allocate based on percentage
        # rent portion = predicted_total_monthly * (rent_percentage / 100)
        # credit portion = predicted_total_monthly * (1 - rent_percentage / 100)
        # For credit portion: credit_value = credit_portion / 0.03
        rent_portion = predicted_total_monthly * (rent_percentage / 100)
        credit_portion = predicted_total_monthly * ((100 - rent_percentage) / 100)
        
        estimated_rent = rent_portion
        estimated_credit = credit_portion / 0.03
    
    # Display breakdown
    st.markdown("### 💳 Payment Breakdown")
    
    col_rent, col_credit = st.columns(2)
    
    with col_rent:
        st.metric(
            label="📅 Monthly Rent",
            value=f"{estimated_rent:,.0f} Toman",
            delta=f"{rent_percentage}% of total"
        )
    
    with col_credit:
        st.metric(
            label="💰 Upfront Deposit (Credit)",
            value=f"{estimated_credit:,.0f} Toman",
            delta=f"{100-rent_percentage}% equivalent"
        )
    
    # Verification
    verification = estimated_rent + (estimated_credit * 0.03)
    st.caption(f"✓ Total monthly equivalent: {verification:,.0f} Toman")
    
    # Additional info
    st.markdown("---")
    st.info(f"""
    **How to read this:**
    - **Bedrooms:** {bedrooms} bedroom apartment (impacts size estimate)
    - **Monthly Rent:** The amount you pay every month
    - **Upfront Deposit:** The lump sum paid at the start (returned when you leave)
    - **3% Rule:** In Tehran, deposit is typically converted to monthly rent at 3% (credit × 0.03)
    - **Inflation:** Prices adjusted by {inflation_multiplier:.0%} for 2025 (model trained on 2024 data)
    
    **Example:** If deposit is 1,000,000,000 Toman, its monthly equivalent is 30,000,000 Toman (3%)
    """)

# Footer
st.markdown("---")
st.caption(f"""
**About this model:**
- Trained on {metadata['n_samples']:,} Tehran apartment rentals (2024 data)
- Features: {metadata['n_features']} property characteristics
- Test MAE: {metadata['test_mae']:,.0f} Toman (2024 baseline)
- Predictions adjusted by {INFLATION_RATE_2024_2025:.0%} inflation for 2025
- Bedroom impact: Estimated through size adjustment ({bedrooms} bedrooms)
""")
