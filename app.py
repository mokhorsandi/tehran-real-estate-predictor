"""
Tehran Real Estate Price Prediction App
Mobile-friendly version with simplified UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip

# Page config - centered layout works better on mobile
st.set_page_config(
    page_title="Tehran Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 20px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load model and metadata
@st.cache_resource
def load_model():
    with gzip.open('tehran_price_model.pkl.gz', 'rb') as f:
        model = joblib.load(f)
    metadata = joblib.load('model_metadata.pkl')
    return model, metadata

@st.cache_data
def load_neighborhood_data():
    return pd.read_csv('neighborhood_stats.csv')

# Load resources
try:
    model, metadata = load_model()
    neighborhoods_df = load_neighborhood_data()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Title
st.title("🏠 Tehran Property Price Predictor")
st.caption(f"Trained on 86,000+ listings | Accuracy: {metadata['test_r2']:.0%}" if model_loaded else "")

if model_loaded:
    
    # Tehran boundary
    TEHRAN_LAT_MIN, TEHRAN_LAT_MAX = 35.55, 35.85
    TEHRAN_LON_MIN, TEHRAN_LON_MAX = 51.10, 51.65
    
    st.markdown("---")
    
    # STEP 1: Location
    st.subheader("📍 Step 1: Location")
    
    # Neighborhood dropdown (easier on mobile than map)
    neighborhood_list = sorted(neighborhoods_df['neighbourhood'].tolist())
    selected_hood = st.selectbox(
        "Select neighborhood:",
        neighborhood_list,
        index=neighborhood_list.index('tehranpars-sharghi') if 'tehranpars-sharghi' in neighborhood_list else 0
    )
    
    # Get coordinates from selected neighborhood
    hood_data = neighborhoods_df[neighborhoods_df['neighbourhood'] == selected_hood].iloc[0]
    final_lat = hood_data['lat']
    final_lon = hood_data['lon']
    
    st.info(f"📍 **{selected_hood}** - Median: {hood_data['median_price']/1e9:.1f}B Toman")
    
    # Optional: Manual coordinates
    with st.expander("🔧 Advanced: Enter coordinates manually"):
        col_lat, col_lon = st.columns(2)
        with col_lat:
            manual_lat = st.number_input("Latitude", min_value=35.55, max_value=35.85, value=float(final_lat), format="%.4f")
        with col_lon:
            manual_lon = st.number_input("Longitude", min_value=51.10, max_value=51.65, value=float(final_lon), format="%.4f")
        
        if st.checkbox("Use manual coordinates"):
            final_lat = manual_lat
            final_lon = manual_lon
            st.success(f"Using: {final_lat:.4f}, {final_lon:.4f}")
    
    st.markdown("---")
    
    # STEP 2: Property Details
    st.subheader("🏠 Step 2: Property Details")
    
    # Size and rooms on same row
    col1, col2 = st.columns(2)
    with col1:
        size_sqm = st.number_input("Size (sqm)", min_value=20, max_value=1000, value=100)
    with col2:
        rooms = st.selectbox("Bedrooms", [0, 1, 2, 3, 4, 5], index=2)
    
    # Floor and age
    col3, col4 = st.columns(2)
    with col3:
        floor = st.number_input("Floor", min_value=0, max_value=30, value=3)
        total_floors = st.number_input("Total floors", min_value=1, max_value=50, value=5)
    with col4:
        construction_year = st.number_input("Year built (Shamsi)", min_value=1370, max_value=1403, value=1395)
        building_age = 1403 - construction_year
        st.caption(f"🏗️ Age: {building_age} years")
    
    st.markdown("---")
    
    # STEP 3: Amenities (simplified for mobile)
    st.subheader("🔧 Step 3: Amenities")
    
    col_a, col_b = st.columns(2)
    with col_a:
        has_elevator = st.checkbox("🛗 Elevator", value=True)
        has_parking = st.checkbox("🅿️ Parking", value=True)
        has_storage = st.checkbox("📦 Storage", value=True)
        has_balcony = st.checkbox("🌿 Balcony")
        has_lobby = st.checkbox("🏛️ Lobby")
        is_north_facing = st.checkbox("⬆️ North Facing")
    
    with col_b:
        is_luxury = st.checkbox("✨ Luxury")
        is_renovated = st.checkbox("🔨 Renovated")
        is_well_lit = st.checkbox("☀️ Well Lit")
        has_view = st.checkbox("🏙️ Has View")
        is_newly_built = st.checkbox("🆕 Newly Built")
        has_roof_garden = st.checkbox("🌳 Roof Garden")
    
    # Hidden defaults
    has_caretaker = False
    has_video_intercom = False
    has_remote_door = False
    is_fully_equipped = False
    is_never_used = False
    has_good_layout = False
    has_single_deed = False
    has_built_in_closet = False
    
    st.markdown("---")
    
    # ESTIMATE BUTTON
    if st.button("💰 ESTIMATE PRICE", type="primary"):
        
        # Build features
        feature_values = {
            'size_sqm': size_sqm,
            'rooms': rooms,
            'floor': floor,
            'total_floors': total_floors,
            'building_age_years': building_age,
            'construction_year_int': construction_year,
            'latitude': final_lat,
            'longitude': final_lon,
            'has_elevator': int(has_elevator),
            'has_parking': int(has_parking),
            'has_storage': int(has_storage),
            'has_balcony': int(has_balcony),
            'is_north_facing': int(is_north_facing),
            'is_south_facing': 0,
            'is_east_facing': 0,
            'is_west_facing': 0,
            'is_luxury': int(is_luxury),
            'is_renovated': int(is_renovated),
            'is_well_lit': int(is_well_lit),
            'has_view': int(has_view),
            'is_fully_equipped': int(is_fully_equipped),
            'is_never_used': int(is_never_used),
            'is_newly_built': int(is_newly_built),
            'has_good_layout': int(has_good_layout),
            'has_single_deed': int(has_single_deed),
            'has_roof_garden': int(has_roof_garden),
            'has_lobby': int(has_lobby),
            'has_caretaker': int(has_caretaker),
            'has_video_intercom': int(has_video_intercom),
            'has_built_in_closet': int(has_built_in_closet),
            'is_two_sided': 0,
            'is_dead_end': 0,
            'is_rebuilt': 0,
            'has_remote_door': int(has_remote_door)
        }
        
        # Create array
        X_pred = np.array([[feature_values[f] for f in metadata['feature_names']]])
        
        # Predict
        log_pred = model.predict(X_pred)[0]
        prediction = np.expm1(log_pred)
        
        # Results
        st.markdown("---")
        st.header("💰 Estimated Price")
        
        # Main price - big and clear
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin: 10px 0;">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">{prediction/1e9:.2f} Billion Toman</h1>
            <p style="color: #ddd; margin: 5px 0;">≈ {prediction/size_sqm/1e6:.1f}M per sqm</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Range
        mape = metadata['test_mape'] / 100
        lower = prediction * (1 - mape)
        upper = prediction * (1 + mape)
        
        st.success(f"**Range:** {lower/1e9:.2f} - {upper/1e9:.2f} Billion Toman (±{metadata['test_mape']:.0f}%)")
        
        # Comparison with neighborhood
        hood_median = hood_data['median_price']
        diff = ((prediction - hood_median) / hood_median) * 100
        
        if diff > 0:
            st.info(f"📊 **{diff:+.0f}%** compared to {selected_hood} median ({hood_median/1e9:.1f}B)")
        else:
            st.info(f"📊 **{diff:.0f}%** compared to {selected_hood} median ({hood_median/1e9:.1f}B)")
        
        # Summary
        st.caption(f"📐 {size_sqm}sqm | 🛏️ {rooms} rooms | 🏢 Floor {floor}/{total_floors} | 🏗️ {building_age} years old")

else:
    st.error("⚠️ Could not load model. Please try again later.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ | Data: Divar.ir 1403 | [GitHub](https://github.com/mokhorsandi/tehran-real-estate-predictor)")
