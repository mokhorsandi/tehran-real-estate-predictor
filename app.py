"""
Tehran Real Estate Price Prediction App
========================================
Click ANYWHERE on the map to select location, then get a price estimate.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip
import folium
from streamlit_folium import st_folium

# Page config
st.set_page_config(
    page_title="Tehran Property Price Predictor",
    page_icon="🏠",
    layout="wide"
)

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

if model_loaded:
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Info")
        st.metric("Accuracy (R²)", f"{metadata['test_r2']:.1%}")
        st.metric("Avg Error", f"±{metadata['test_mape']:.0f}%")
        
        st.markdown("---")
        st.markdown("### 🔝 Price Factors")
        st.markdown("""
        1. 📍 Location
        2. 📐 Size  
        3. 🏗️ Age
        4. 🛗 Elevator
        5. 🅿️ Parking
        """)
    
    # Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📍 Step 1: Select Location")
        st.caption("Click anywhere on the map OR enter coordinates manually")
        
        # Manual coordinate input
        manual_col1, manual_col2 = st.columns(2)
        with manual_col1:
            input_lat = st.number_input("Latitude", min_value=35.55, max_value=35.85, value=35.72, step=0.001, format="%.4f")
        with manual_col2:
            input_lon = st.number_input("Longitude", min_value=51.20, max_value=51.60, value=51.39, step=0.001, format="%.4f")
        
        # Create map - simple, clickable
        m = folium.Map(
            location=[input_lat, input_lon],
            zoom_start=11,
            tiles='cartodbpositron'
        )
        
        # Add click instruction
        folium.LatLngPopup().add_to(m)
        
        # Add current marker
        folium.Marker(
            location=[input_lat, input_lon],
            popup=f"Current: {input_lat:.4f}, {input_lon:.4f}",
            icon=folium.Icon(color='red', icon='home')
        ).add_to(m)
        
        # Add neighborhood markers (smaller, non-blocking)
        for _, row in neighborhoods_df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=3,
                color='blue',
                fill=True,
                fillOpacity=0.4,
                weight=1,
                popup=f"{row['neighbourhood']}"
            ).add_to(m)
        
        # Display map
        map_result = st_folium(m, height=350, width=None)
        
        # Get clicked coordinates
        if map_result and map_result.get('last_clicked'):
            clicked_lat = map_result['last_clicked']['lat']
            clicked_lon = map_result['last_clicked']['lng']
            st.success(f"✅ Clicked: **{clicked_lat:.4f}, {clicked_lon:.4f}**")
            st.caption("👆 Copy these values to the Latitude/Longitude inputs above, then click Estimate")
            final_lat = clicked_lat
            final_lon = clicked_lon
        else:
            final_lat = input_lat
            final_lon = input_lon
        
        # Show which coordinates will be used
        st.info(f"📍 Using: **Lat {final_lat:.4f}, Lon {final_lon:.4f}**")
        
        # Find nearest neighborhood
        distances = np.sqrt(
            (neighborhoods_df['lat'] - final_lat)**2 + 
            (neighborhoods_df['lon'] - final_lon)**2
        )
        nearest_idx = distances.idxmin()
        nearest = neighborhoods_df.loc[nearest_idx]
        st.caption(f"Nearest neighborhood: **{nearest['neighbourhood']}** (median: {nearest['median_price']/1e9:.1f}B)")
    
    with col2:
        st.subheader("🏠 Step 2: Property Details")
        
        # Core features
        c1, c2 = st.columns(2)
        with c1:
            size_sqm = st.number_input("Size (sqm)", min_value=20, max_value=1000, value=100)
            rooms = st.selectbox("Bedrooms", [0, 1, 2, 3, 4, 5], index=2)
            floor = st.number_input("Floor", min_value=0, max_value=30, value=3)
        
        with c2:
            total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=5)
            construction_year = st.number_input("Year Built (Shamsi)", min_value=1370, max_value=1403, value=1395)
            building_age = 1403 - construction_year
            st.caption(f"Age: {building_age} years")
        
        # Amenities
        st.markdown("**Amenities:**")
        amenity_cols = st.columns(4)
        with amenity_cols[0]:
            has_elevator = st.checkbox("🛗 Elevator", value=True)
            has_parking = st.checkbox("🅿️ Parking", value=True)
            has_storage = st.checkbox("📦 Storage", value=True)
        with amenity_cols[1]:
            has_balcony = st.checkbox("🌿 Balcony")
            has_lobby = st.checkbox("🏛️ Lobby")
            is_north_facing = st.checkbox("⬆️ North Facing")
        with amenity_cols[2]:
            is_luxury = st.checkbox("✨ Luxury")
            is_renovated = st.checkbox("🔨 Renovated")
            is_well_lit = st.checkbox("☀️ Well Lit")
        with amenity_cols[3]:
            has_view = st.checkbox("🏙️ View")
            is_newly_built = st.checkbox("🆕 New Build")
            has_roof_garden = st.checkbox("🌳 Roof Garden")
        
        # Extra (hidden defaults)
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
    if st.button("💰 ESTIMATE PRICE", type="primary", use_container_width=True):
        
        # Build feature dict
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
        
        # Create array in correct order
        X_pred = np.array([[feature_values[f] for f in metadata['feature_names']]])
        
        # Predict
        log_pred = model.predict(X_pred)[0]
        prediction = np.expm1(log_pred)
        
        # Results
        st.markdown("---")
        st.header("💰 Price Estimate")
        
        res_cols = st.columns(3)
        with res_cols[0]:
            st.metric("Total Price", f"{prediction/1e9:.2f} Billion Toman")
        with res_cols[1]:
            st.metric("Per sqm", f"{prediction/size_sqm/1e6:.1f} Million Toman")
        with res_cols[2]:
            hood_median = nearest['median_price']
            diff = ((prediction - hood_median) / hood_median) * 100
            st.metric(f"vs {nearest['neighbourhood']}", f"{diff:+.0f}%")
        
        # Range
        mape = metadata['test_mape'] / 100
        lower = prediction * (1 - mape)
        upper = prediction * (1 + mape)
        st.success(f"**Estimated Range:** {lower/1e9:.2f} - {upper/1e9:.2f} Billion Toman")
        
        st.caption(f"📍 Location: {final_lat:.4f}, {final_lon:.4f} | 📐 {size_sqm}sqm | 🛏️ {rooms} rooms | 🏗️ {building_age}yr old")

else:
    st.error("Model files not found.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ | Data: Divar.ir 1403 | [GitHub](https://github.com/mokhorsandi/tehran-real-estate-predictor)")
