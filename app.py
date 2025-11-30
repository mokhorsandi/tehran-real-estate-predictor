"""
Tehran Real Estate Price Prediction App
========================================
Interactive Streamlit app for predicting property prices in Tehran.
Select a location on the map, choose property features, and get an instant price estimate!
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

# Page config
st.set_page_config(
    page_title="Tehran Property Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load model and metadata
@st.cache_resource
def load_model():
    model = joblib.load('tehran_price_model.pkl')
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
st.markdown("""
Predict real estate prices in Tehran based on location, size, and amenities.
This model was trained on **85,000+ property listings** from Divar.ir (1403).
""")

if model_loaded:
    # Sidebar for model info
    with st.sidebar:
        st.header("📊 Model Information")
        st.metric("Model Type", metadata['model_type'])
        st.metric("R² Score", f"{metadata['test_r2']:.2%}")
        st.metric("MAPE", f"{metadata['test_mape']:.1f}%")
        st.metric("Training Samples", f"{metadata['training_samples']:,}")
        
        st.markdown("---")
        st.markdown("### Top Features")
        st.markdown("""
        1. 📐 Size (sqm)
        2. 🛏️ Number of Rooms
        3. 🏛️ Has Lobby
        4. 🛗 Has Elevator
        5. 🅿️ Has Parking
        """)
    
    # Main layout - two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📍 Select Location")
        
        # Neighborhood selection
        neighborhood_list = sorted(neighborhoods_df['neighbourhood'].tolist())
        selected_neighborhood = st.selectbox(
            "Choose a neighborhood",
            ["-- Select or click on map --"] + neighborhood_list
        )
        
        # Get coordinates from neighborhood or use default
        if selected_neighborhood != "-- Select or click on map --":
            hood_data = neighborhoods_df[neighborhoods_df['neighbourhood'] == selected_neighborhood].iloc[0]
            default_lat = hood_data['lat']
            default_lon = hood_data['lon']
            st.info(f"📍 **{selected_neighborhood}** - Median price: {hood_data['median_price']/1e9:.1f}B Toman")
        else:
            default_lat = 35.7219
            default_lon = 51.3347
        
        # Create map
        m = folium.Map(location=[default_lat, default_lon], zoom_start=12)
        
        # Add neighborhood markers
        for _, row in neighborhoods_df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=3,
                color='blue',
                fill=True,
                popup=f"{row['neighbourhood']}: {row['median_price']/1e9:.1f}B"
            ).add_to(m)
        
        # Add selected location marker
        if selected_neighborhood != "-- Select or click on map --":
            folium.Marker(
                location=[default_lat, default_lon],
                popup=selected_neighborhood,
                icon=folium.Icon(color='red', icon='home')
            ).add_to(m)
        
        # Display map
        map_data = st_folium(m, height=400, width=500)
        
        # Get clicked location
        if map_data and map_data.get('last_clicked'):
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lon = map_data['last_clicked']['lng']
            st.success(f"📍 Selected: ({clicked_lat:.4f}, {clicked_lon:.4f})")
            lat = clicked_lat
            lon = clicked_lon
        elif selected_neighborhood != "-- Select or click on map --":
            lat = default_lat
            lon = default_lon
        else:
            lat = 35.7219
            lon = 51.3347
    
    with col2:
        st.header("🏠 Property Features")
        
        # Core features
        col_a, col_b = st.columns(2)
        with col_a:
            size_sqm = st.number_input("Size (sqm)", min_value=20, max_value=1000, value=100)
            rooms = st.selectbox("Rooms", [0, 1, 2, 3, 4, 5], index=2)
            floor = st.number_input("Floor", min_value=0, max_value=30, value=3)
        
        with col_b:
            total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=5)
            construction_year = st.number_input("Construction Year (Shamsi)", min_value=1370, max_value=1403, value=1395)
            building_age = 1403 - construction_year
        
        st.markdown("---")
        st.subheader("Amenities")
        
        col_c, col_d, col_e = st.columns(3)
        with col_c:
            has_elevator = st.checkbox("🛗 Elevator", value=True)
            has_parking = st.checkbox("🅿️ Parking", value=True)
            has_storage = st.checkbox("📦 Storage", value=True)
            has_balcony = st.checkbox("🌿 Balcony", value=True)
        
        with col_d:
            has_lobby = st.checkbox("🏛️ Lobby")
            has_roof_garden = st.checkbox("🌳 Roof Garden")
            has_caretaker = st.checkbox("👨‍🔧 Caretaker")
            has_video_intercom = st.checkbox("📹 Video Intercom")
        
        with col_e:
            is_north_facing = st.checkbox("⬆️ North Facing")
            is_luxury = st.checkbox("✨ Luxury")
            is_renovated = st.checkbox("🔨 Renovated")
            is_well_lit = st.checkbox("☀️ Well Lit")
        
        # Additional features
        st.markdown("---")
        st.subheader("Additional Features")
        col_f, col_g = st.columns(2)
        with col_f:
            has_view = st.checkbox("🏙️ Has View")
            is_fully_equipped = st.checkbox("🍳 Fully Equipped")
            is_never_used = st.checkbox("🆕 Never Used")
            is_newly_built = st.checkbox("🏗️ Newly Built")
        with col_g:
            has_good_layout = st.checkbox("📐 Good Layout")
            has_single_deed = st.checkbox("📄 Single Deed")
            has_built_in_closet = st.checkbox("🚪 Built-in Closet")
            has_remote_door = st.checkbox("🚗 Remote Door")
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        # Prepare features in the correct order
        feature_values = {
            'size_sqm': size_sqm,
            'rooms': rooms,
            'floor': floor,
            'total_floors': total_floors,
            'building_age_years': building_age,
            'construction_year_int': construction_year,
            'latitude': lat,
            'longitude': lon,
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
        
        # Create feature array in correct order
        X_pred = np.array([[feature_values[f] for f in metadata['feature_names']]])
        
        # Make prediction (model uses log transform)
        log_prediction = model.predict(X_pred)[0]
        prediction = np.expm1(log_prediction)
        
        # Display results
        st.markdown("---")
        st.header("💰 Price Prediction")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric("Estimated Price", f"{prediction/1e9:.2f} Billion Toman")
        
        with col_res2:
            price_per_sqm = prediction / size_sqm
            st.metric("Price per sqm", f"{price_per_sqm/1e6:.1f} Million Toman")
        
        with col_res3:
            # Compare with neighborhood median if available
            if selected_neighborhood != "-- Select or click on map --":
                hood_median = hood_data['median_price']
                diff_pct = ((prediction - hood_median) / hood_median) * 100
                st.metric("vs Neighborhood Median", f"{diff_pct:+.1f}%")
        
        # Confidence range (using MAPE)
        mape = metadata['test_mape'] / 100
        lower_bound = prediction * (1 - mape)
        upper_bound = prediction * (1 + mape)
        
        st.info(f"""
        **Estimated Range**: {lower_bound/1e9:.2f} - {upper_bound/1e9:.2f} Billion Toman
        
        *Based on model MAPE of {metadata['test_mape']:.1f}%*
        """)

else:
    st.error("Please run the model training notebook first to generate the required files.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with ❤️ using Streamlit | Data from Divar.ir (1403)</p>
    <p>Model: {metadata.get('model_type', 'XGBoost')} | R² = {metadata['test_r2']:.1%} | {len(metadata['feature_names'])} Features</p>
</div>
""", unsafe_allow_html=True)
