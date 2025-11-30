"""
Tehran Real Estate Price Prediction App
========================================
Interactive Streamlit app for predicting property prices in Tehran.
Click anywhere on the map to select location, enter property features, and get an instant price estimate!
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

# Initialize session state for coordinates
if 'lat' not in st.session_state:
    st.session_state.lat = 35.7219  # Default: Tehran center
if 'lon' not in st.session_state:
    st.session_state.lon = 51.3947

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
st.markdown("""
**Click anywhere on the map** to select a location, then enter property details to get a price estimate.
Trained on **86,000+ listings** from Divar.ir (1403).
""")

if model_loaded:
    # Sidebar for model info
    with st.sidebar:
        st.header("📊 Model Info")
        st.metric("Model", metadata['model_type'])
        st.metric("Accuracy (R²)", f"{metadata['test_r2']:.1%}")
        st.metric("Avg Error (MAPE)", f"{metadata['test_mape']:.1f}%")
        
        st.markdown("---")
        st.markdown("### 📍 Selected Location")
        st.write(f"**Lat:** {st.session_state.lat:.4f}")
        st.write(f"**Lon:** {st.session_state.lon:.4f}")
        
        st.markdown("---")
        st.markdown("### 🔝 Top Price Factors")
        st.markdown("""
        1. 📍 Location (North > South)
        2. 📐 Size (sqm)
        3. 🏗️ Building Age
        4. 🛗 Elevator
        5. 🅿️ Parking
        """)
    
    # Layout: Map on left, Features on right
    col_map, col_features = st.columns([1, 1])
    
    with col_map:
        st.subheader("📍 Click on Map to Select Location")
        
        # Quick neighborhood selector
        neighborhood_list = ["-- Or select a neighborhood --"] + sorted(neighborhoods_df['neighbourhood'].tolist())
        selected_hood = st.selectbox("Quick select:", neighborhood_list, key="hood_select")
        
        if selected_hood != "-- Or select a neighborhood --":
            hood_data = neighborhoods_df[neighborhoods_df['neighbourhood'] == selected_hood].iloc[0]
            st.session_state.lat = hood_data['lat']
            st.session_state.lon = hood_data['lon']
            st.success(f"📍 {selected_hood} - Median: {hood_data['median_price']/1e9:.1f}B Toman")
        
        # Create map centered on current selection
        m = folium.Map(
            location=[st.session_state.lat, st.session_state.lon], 
            zoom_start=11,
            tiles='cartodbpositron'
        )
        
        # Add current selection marker (red)
        folium.Marker(
            location=[st.session_state.lat, st.session_state.lon],
            popup=f"Selected: {st.session_state.lat:.4f}, {st.session_state.lon:.4f}",
            icon=folium.Icon(color='red', icon='home')
        ).add_to(m)
        
        # Add neighborhood reference points (small blue dots)
        for _, row in neighborhoods_df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=4,
                color='#3388ff',
                fill=True,
                fillOpacity=0.6,
                popup=f"{row['neighbourhood']}: {row['median_price']/1e9:.1f}B",
                weight=1
            ).add_to(m)
        
        # Display map and capture clicks
        map_output = st_folium(
            m, 
            height=450, 
            width=None,
            returned_objects=["last_clicked"]
        )
        
        # Update coordinates when map is clicked
        if map_output and map_output.get('last_clicked'):
            new_lat = map_output['last_clicked']['lat']
            new_lon = map_output['last_clicked']['lng']
            # Only update if it's a new click
            if new_lat != st.session_state.lat or new_lon != st.session_state.lon:
                st.session_state.lat = new_lat
                st.session_state.lon = new_lon
                st.rerun()
        
        # Show current coordinates
        st.info(f"🎯 **Selected:** Lat {st.session_state.lat:.4f}, Lon {st.session_state.lon:.4f}")
    
    with col_features:
        st.subheader("🏠 Property Details")
        
        # Core features in two columns
        c1, c2 = st.columns(2)
        with c1:
            size_sqm = st.number_input("Size (sqm)", min_value=20, max_value=1000, value=100)
            rooms = st.selectbox("Bedrooms", [0, 1, 2, 3, 4, 5], index=2)
            floor = st.number_input("Floor", min_value=0, max_value=30, value=3)
        
        with c2:
            total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=50, value=5)
            construction_year = st.number_input("Construction Year (Shamsi)", min_value=1370, max_value=1403, value=1395)
            building_age = 1403 - construction_year
            st.caption(f"🏗️ Building Age: {building_age} years")
        
        st.markdown("---")
        
        # Amenities in expandable section
        with st.expander("🔧 Amenities & Features", expanded=True):
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                has_elevator = st.checkbox("🛗 Elevator", value=True)
                has_parking = st.checkbox("🅿️ Parking", value=True)
                has_storage = st.checkbox("📦 Storage", value=True)
                has_balcony = st.checkbox("🌿 Balcony", value=False)
                has_lobby = st.checkbox("🏛️ Lobby", value=False)
            
            with col_b:
                is_north_facing = st.checkbox("⬆️ North Facing", value=False)
                is_luxury = st.checkbox("✨ Luxury", value=False)
                is_renovated = st.checkbox("🔨 Renovated", value=False)
                is_well_lit = st.checkbox("☀️ Well Lit", value=False)
                has_view = st.checkbox("🏙️ Has View", value=False)
            
            with col_c:
                is_newly_built = st.checkbox("🆕 Newly Built", value=False)
                has_roof_garden = st.checkbox("🌳 Roof Garden", value=False)
                has_caretaker = st.checkbox("👨‍🔧 Caretaker", value=False)
                has_video_intercom = st.checkbox("📹 Video Intercom", value=False)
                has_remote_door = st.checkbox("🚗 Remote Door", value=False)
        
        # Hidden features (less common)
        is_fully_equipped = False
        is_never_used = False
        has_good_layout = False
        has_single_deed = False
        has_built_in_closet = False
    
    st.markdown("---")
    
    # Big Estimate Button
    if st.button("💰 ESTIMATE PRICE", type="primary", use_container_width=True):
        
        # Prepare all features
        feature_values = {
            'size_sqm': size_sqm,
            'rooms': rooms,
            'floor': floor,
            'total_floors': total_floors,
            'building_age_years': building_age,
            'construction_year_int': construction_year,
            'latitude': st.session_state.lat,
            'longitude': st.session_state.lon,
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
        
        # Make prediction
        log_prediction = model.predict(X_pred)[0]
        prediction = np.expm1(log_prediction)
        
        # Display results in a nice box
        st.markdown("---")
        st.header("💰 Estimated Price")
        
        result_cols = st.columns(3)
        
        with result_cols[0]:
            st.metric(
                label="Total Price",
                value=f"{prediction/1e9:.2f}B Toman",
                delta=None
            )
        
        with result_cols[1]:
            price_per_sqm = prediction / size_sqm
            st.metric(
                label="Price per sqm",
                value=f"{price_per_sqm/1e6:.1f}M Toman"
            )
        
        with result_cols[2]:
            # Find nearest neighborhood for comparison
            distances = np.sqrt(
                (neighborhoods_df['lat'] - st.session_state.lat)**2 + 
                (neighborhoods_df['lon'] - st.session_state.lon)**2
            )
            nearest_idx = distances.idxmin()
            nearest_hood = neighborhoods_df.loc[nearest_idx]
            hood_median = nearest_hood['median_price']
            diff_pct = ((prediction - hood_median) / hood_median) * 100
            st.metric(
                label=f"vs {nearest_hood['neighbourhood']}",
                value=f"{diff_pct:+.0f}%",
                delta=f"median: {hood_median/1e9:.1f}B"
            )
        
        # Confidence range
        mape = metadata['test_mape'] / 100
        lower = prediction * (1 - mape)
        upper = prediction * (1 + mape)
        
        st.success(f"""
        **📊 Estimated Range:** {lower/1e9:.2f} - {upper/1e9:.2f} Billion Toman  
        *Based on model accuracy (±{metadata['test_mape']:.0f}%)*
        """)
        
        # Property summary
        with st.expander("📋 Property Summary"):
            st.write(f"""
            - **Location:** {st.session_state.lat:.4f}, {st.session_state.lon:.4f}
            - **Size:** {size_sqm} sqm | **Rooms:** {rooms} | **Floor:** {floor}/{total_floors}
            - **Age:** {building_age} years (built {construction_year})
            - **Amenities:** {'Elevator, ' if has_elevator else ''}{'Parking, ' if has_parking else ''}{'Storage, ' if has_storage else ''}{'Lobby, ' if has_lobby else ''}{'Balcony' if has_balcony else ''}
            """)

else:
    st.error("⚠️ Model files not found. Please ensure tehran_price_model.pkl.gz and model_metadata.pkl are in the app directory.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    Built with ❤️ using Streamlit | Data from Divar.ir (1403) | 
    <a href="https://github.com/mokhorsandi/tehran-real-estate-predictor">GitHub</a>
</div>
""", unsafe_allow_html=True)
