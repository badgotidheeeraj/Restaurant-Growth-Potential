
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Restaurant Growth Potential System")

# ==========================================
# 2. DATA GENERATION (MOCK DATA BASED ON SCHEMA)
# ==========================================
# @st.cache_data
# def generate_mock_data(n=200):
#     np.random.seed(42)
#     data = {
#         'RestaurantID': [f'R{str(i).zfill(3)}' for i in range(1, n+1)],
#         'RestaurantName': [f'Auckland Eatery {i}' for i in range(1, n+1)],
#         'CuisineType': np.random.choice(['Burgers', 'Pizza', 'Asian', 'Cafe', 'Fine Dining'], n),
#         'Segment': np.random.choice(['QSR', 'Casual Dining', 'Cafe', 'Premium'], n),
#         'Subregion': np.random.choice(['North Shore', 'CBD', 'South Auckland', 'West Auckland'], n),
#         'GrowthFactor': np.random.uniform(0.99, 1.05, n),
#         'AOV': np.random.uniform(29.79, 47.23, n),
#         'MonthlyOrders': np.random.randint(500, 5000, n),
#         'COGSRate': np.random.uniform(0.20, 0.40, n),
#         'OPEXRate': np.random.uniform(0.20, 0.55, n),
#         'CommissionRate': np.random.uniform(0.15, 0.30, n),
#         'DeliveryRadiusKM': np.random.uniform(3, 18, n),
#         'DeliveryCostOrder': np.random.uniform(0.89, 5.31, n),
#     }
#     df = pd.DataFrame(data)
    
#     # Calculate channel shares
#     df['InStoreShare'] = np.random.uniform(0.2, 0.8, n)
#     df['UE_share'] = np.random.uniform(0.1, 0.5, n)
#     df['DD_share'] = np.random.uniform(0.05, 0.3, n)
#     df['SD_share'] = 1.0 - (df['InStoreShare'] + df['UE_share'] + df['DD_share'])
#     df['SD_share'] = df['SD_share'].clip(lower=0) 
    
#     # Normalize shares
#     total_share = df[['InStoreShare', 'UE_share', 'DD_share', 'SD_share']].sum(axis=1)
#     for col in ['InStoreShare', 'UE_share', 'DD_share', 'SD_share']:
#         df[col] = df[col] / total_share

#     return df

@st.cache_data
def load_actual_data():
    return pd.read_csv("SkyCity Auckland Restaurants & Bars.csv")
# ==========================================
# 3. DATA SCIENCE METHODOLOGY (PIPELINE)
# ==========================================
@st.cache_data
def process_and_cluster(df):
    processed_df = df.copy()
    
    # --- Feature Engineering & KPIs ---
    processed_df['Scale'] = processed_df['MonthlyOrders'] * processed_df['GrowthFactor']
    processed_df['CostDiscipline'] = 1 - (processed_df['COGSRate'] + processed_df['OPEXRate'])
    processed_df['AggregatorDependence'] = processed_df['UE_share'] + processed_df['DD_share']
    processed_df['ExpansionHeadroom'] = processed_df['DeliveryRadiusKM'] / (processed_df['MonthlyOrders']/1000)
    processed_df['RevenueQuality'] = processed_df['AOV'] * processed_df['CostDiscipline']
    
    # --- Preprocessing ---
    features = ['Scale', 'CostDiscipline', 'AggregatorDependence', 'ExpansionHeadroom', 'RevenueQuality']
    X = processed_df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- Dimensionality Reduction (PCA) ---
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_scaled)
    processed_df['PCA1'] = pca_components[:, 0]
    processed_df['PCA2'] = pca_components[:, 1]
    
    # --- Clustering (K-Means) ---
    kmeans = KMeans(n_clusters=4, random_state=42)
    processed_df['ClusterID'] = kmeans.fit_predict(X_scaled)
    
    # --- Cluster Interpretation & Labeling ---
    cluster_labels = {
        0: "Stable Local Performers",
        1: "High-Growth / High-Risk",
        2: "Aggregator-Dependent Low Margin",
        3: "Scalable Self-Delivery Leaders"
    }
    processed_df['ClusterLabel'] = processed_df['ClusterID'].map(cluster_labels)
    
    # --- Strategy Recommendations ---
    strategies = {
        "Stable Local Performers": "Hold / stabilize",
        "High-Growth / High-Risk": "Optimize costs & processes",
        "Aggregator-Dependent Low Margin": "Rebalance channels to self-delivery/in-store",
        "Scalable Self-Delivery Leaders": "Aggressive expansion"
    }
    processed_df['Recommendation'] = processed_df['ClusterLabel'].map(strategies)
    
    # --- Growth Potential Index (GPI Scoring) ---
    # Normalize features to 0-100 range for scoring
    gpi_components = (X_scaled - X_scaled.min(axis=0)) / (X_scaled.max(axis=0) - X_scaled.min(axis=0))
    # Weights: +Scale, +CostDiscipline, -AggregatorDependence, +ExpansionHeadroom, +RevenueQuality
    weights = np.array([0.25, 0.25, -0.15, 0.15, 0.20])
    gpi_raw = np.dot(gpi_components, weights)
    # Scale final GPI to 0-100
    processed_df['GrowthPotentialScore'] = ((gpi_raw - gpi_raw.min()) / (gpi_raw.max() - gpi_raw.min()) * 100).round(1)
    
    return processed_df

# ==========================================
# 4. STREAMLIT UI & DASHBOARD
# ==========================================
df_raw = load_actual_data()
df = process_and_cluster(df_raw)

# Sidebar Filters
st.sidebar.title("Strategic Filtering")
subregion_filter = st.sidebar.multiselect("Subregion", options=df['Subregion'].unique(), default=df['Subregion'].unique())
segment_filter = st.sidebar.multiselect("Segment", options=df['Segment'].unique(), default=df['Segment'].unique())
cuisine_filter = st.sidebar.multiselect("Cuisine Type", options=df['CuisineType'].unique(), default=df['CuisineType'].unique())

# Apply Filters
filtered_df = df[
    (df['Subregion'].isin(subregion_filter)) &
    (df['Segment'].isin(segment_filter)) &
    (df['CuisineType'].isin(cuisine_filter))
]

st.title("Restaurant Growth Potential Modeling")
st.markdown("### Strategic Classification System for SkyCity Auckland")

# High-Level Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Restaurants", len(filtered_df))
col2.metric("Avg Growth Potential Score", f"{filtered_df['GrowthPotentialScore'].mean():.1f}/100")
col3.metric("Highest Growth Segment", filtered_df.groupby('Segment')['GrowthPotentialScore'].mean().idxmax())
col4.metric("Avg Aggregator Dependence", f"{filtered_df['AggregatorDependence'].mean()*100:.1f}%")

st.markdown("---")

# Main Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["Cluster Map & Analysis", "Restaurant Deep Dive", "Data View"])

with tab1:
    st.subheader("Restaurant Archetype Clusters (PCA View)")
    fig_pca = px.scatter(
        filtered_df, x="PCA1", y="PCA2", 
        color="ClusterLabel", 
        hover_data=["RestaurantName", "GrowthPotentialScore", "Recommendation"],
        size="Scale",
        title="Latent Structure of Restaurant Growth Potential",
        labels={"PCA1": "Cost & Margin Pressure", "PCA2": "Growth Momentum & Logicstics"}
    )
    st.plotly_chart(fig_pca, width='stretch')
    
    st.subheader("Cluster Performance Profiles")
    cluster_stats = filtered_df.groupby("ClusterLabel")[
        ['GrowthPotentialScore', 'CostDiscipline', 'AggregatorDependence', 'Scale']
    ].mean().reset_index()
    
    st.dataframe(
        cluster_stats.style.background_gradient(cmap='viridis', subset=['GrowthPotentialScore', 'Scale'])
    )

with tab2:
    st.subheader("Individual Restaurant Strategy Profile")
    selected_restaurant = st.selectbox("Select a Restaurant to analyze:", filtered_df['RestaurantName'].sort_values())
    
    rest_data = filtered_df[filtered_df['RestaurantName'] == selected_restaurant].iloc[0]
    
    r_col1, r_col2 = st.columns([1, 2])
    
    with r_col1:
        st.markdown(f"### {rest_data['RestaurantName']}")
        st.markdown(f"**Cluster:** {rest_data['ClusterLabel']}")
        st.markdown(f"**Strategy:** {rest_data['Recommendation']}")
        st.metric("Growth Potential Score (GPI)", f"{rest_data['GrowthPotentialScore']}/100")
        
        st.markdown("**Operational Specs:**")
        st.markdown(f"- **Subregion**: {rest_data['Subregion']}")
        st.markdown(f"- **Segment**: {rest_data['Segment']}")
        st.markdown(f"- **Monthly Orders**: {rest_data['MonthlyOrders']}")
        st.markdown(f"- **AOV**: ${rest_data['AOV']:.2f}")

    with r_col2:
        # Radar Chart for Feature Contributions
        categories = ['Scale', 'Cost Discipline', 'Expansion Headroom', 'Revenue Quality', '(Inv) Aggregator Dependence']
        
        # Normalize stats for this specific restaurant for radar chart (0 to 1)
        r_scale = (rest_data['Scale'] / df['Scale'].max())
        r_cost = (rest_data['CostDiscipline'] / df['CostDiscipline'].max())
        r_headroom = (rest_data['ExpansionHeadroom'] / df['ExpansionHeadroom'].max())
        r_rev_qual = (rest_data['RevenueQuality'] / df['RevenueQuality'].max())
        r_agg_dep = 1 - (rest_data['AggregatorDependence'] / df['AggregatorDependence'].max())
        
        fig_radar = go.Figure(data=go.Scatterpolar(
          r=[r_scale, r_cost, r_headroom, r_rev_qual, r_agg_dep],
          theta=categories,
          fill='toself',
          name=rest_data['RestaurantName']
        ))
        fig_radar.update_layout(
          polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
          showlegend=False,
          title=f"Growth Drivers: {rest_data['RestaurantName']}"
        )
        st.plotly_chart(fig_radar, width='stretch')

with tab3:
    st.subheader("Raw Data & Classified Outputs")
    st.dataframe(filtered_df.drop(columns=['PCA1', 'PCA2', 'ClusterID']))

