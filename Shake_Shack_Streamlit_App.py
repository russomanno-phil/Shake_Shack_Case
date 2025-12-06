import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Comprehensive SHAK Analysis Report",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🍔 Comprehensive Shake Shack (SHAK) Analysis Report")
st.caption("Data based on 'shak_state_locations_2024.csv' with Census ACS 2023 population data and CBP QSR establishment data.")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Jump to Section:",
    [
        "1. Data Overview",
        "2. Demand Analysis",
        "3. White Space & Expansion",
        "4. US Maps",
        "5. Scenario Analysis",
        "6. Competitive Analysis",
        "7. Financial Metrics",
        "8. Penetration Scatter Plots"
    ]
)

# --- Data Loading and Preparation ---
@st.cache_data
def load_and_prepare_data(file_path):
    """Loads location data and adds Census/QSR data for analysis."""
    
    # Load location data
    df = pd.read_csv(file_path)
    
    # Remove DOMESTIC summary row
    df = df[df['State'] != 'DOMESTIC'].copy()
    
    # Ensure numeric columns
    for col in ['Company Operated', 'Licensed', 'Total']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Rename for consistency with notebook
    df = df.rename(columns={
        'State': 'state',
        'Company Operated': 'company_operated',
        'Total': 'current_company_shacks'
    })
    
    # State abbreviation mapping
    state_abbrev_map = {
        'Alabama': 'AL', 'Arizona': 'AZ', 'California': 'CA', 'Colorado': 'CO', 
        'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC', 
        'Florida': 'FL', 'Georgia': 'GA', 'Illinois': 'IL', 'Indiana': 'IN', 
        'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Missouri': 'MO', 
        'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New York': 'NY', 
        'North Carolina': 'NC', 'Ohio': 'OH', 'Oregon': 'OR', 'Pennsylvania': 'PA', 
        'Rhode Island': 'RI', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 
        'Virginia': 'VA', 'Washington': 'WA', 'Wisconsin': 'WI'
    }
    df['state_abbrev'] = df['state'].map(state_abbrev_map)
    
    # Census ACS 2023 data (population, median income, area, pop_density)
    # Data sourced from US Census Bureau ACS 2023 1-Year Estimates
    census_data = {
        'Alabama': {'population': 5108468, 'median_income': 62212, 'area_sq_mi': 52423},
        'Arizona': {'population': 7431344, 'median_income': 77315, 'area_sq_mi': 114006},
        'California': {'population': 38965193, 'median_income': 95521, 'area_sq_mi': 163707},
        'Colorado': {'population': 5877610, 'median_income': 89302, 'area_sq_mi': 104185},
        'Connecticut': {'population': 3617176, 'median_income': 90213, 'area_sq_mi': 5543},
        'Delaware': {'population': 1031890, 'median_income': 79325, 'area_sq_mi': 2489},
        'District of Columbia': {'population': 678972, 'median_income': 101027, 'area_sq_mi': 68},
        'Florida': {'population': 22610726, 'median_income': 71711, 'area_sq_mi': 65758},
        'Georgia': {'population': 11029227, 'median_income': 74580, 'area_sq_mi': 59441},
        'Illinois': {'population': 12549689, 'median_income': 78433, 'area_sq_mi': 57914},
        'Indiana': {'population': 6862199, 'median_income': 67173, 'area_sq_mi': 36420},
        'Kansas': {'population': 2940546, 'median_income': 70139, 'area_sq_mi': 82282},
        'Kentucky': {'population': 4526154, 'median_income': 60183, 'area_sq_mi': 40411},
        'Louisiana': {'population': 4573749, 'median_income': 57852, 'area_sq_mi': 52378},
        'Maryland': {'population': 6180253, 'median_income': 98461, 'area_sq_mi': 12406},
        'Massachusetts': {'population': 6996483, 'median_income': 96505, 'area_sq_mi': 10554},
        'Michigan': {'population': 10037261, 'median_income': 68505, 'area_sq_mi': 96714},
        'Minnesota': {'population': 5737915, 'median_income': 84313, 'area_sq_mi': 86943},
        'Missouri': {'population': 6196156, 'median_income': 65920, 'area_sq_mi': 69707},
        'Nevada': {'population': 3194176, 'median_income': 72341, 'area_sq_mi': 110577},
        'New Hampshire': {'population': 1402054, 'median_income': 90845, 'area_sq_mi': 9349},
        'New Jersey': {'population': 9290841, 'median_income': 97126, 'area_sq_mi': 8723},
        'New York': {'population': 19571216, 'median_income': 81386, 'area_sq_mi': 54555},
        'North Carolina': {'population': 10835491, 'median_income': 66186, 'area_sq_mi': 53819},
        'Ohio': {'population': 11785935, 'median_income': 65718, 'area_sq_mi': 44826},
        'Oregon': {'population': 4233358, 'median_income': 76632, 'area_sq_mi': 98379},
        'Pennsylvania': {'population': 12961683, 'median_income': 73170, 'area_sq_mi': 46054},
        'Rhode Island': {'population': 1095962, 'median_income': 78402, 'area_sq_mi': 1545},
        'Tennessee': {'population': 7126489, 'median_income': 64035, 'area_sq_mi': 42143},
        'Texas': {'population': 30503301, 'median_income': 73035, 'area_sq_mi': 268596},
        'Utah': {'population': 3417734, 'median_income': 86833, 'area_sq_mi': 84899},
        'Virginia': {'population': 8715698, 'median_income': 87249, 'area_sq_mi': 42775},
        'Washington': {'population': 7812880, 'median_income': 91306, 'area_sq_mi': 71303},
        'Wisconsin': {'population': 5910955, 'median_income': 72458, 'area_sq_mi': 65498}
    }
    
    # CBP 2022 QSR establishments (NAICS 7225) - Limited-service restaurants
    qsr_data = {
        'Alabama': 8008, 'Arizona': 11537, 'California': 78519, 'Colorado': 11401,
        'Connecticut': 7470, 'Delaware': 1910, 'District of Columbia': 2046,
        'Florida': 38160, 'Georgia': 19571, 'Illinois': 22996, 'Indiana': 11411,
        'Kansas': 4995, 'Kentucky': 6830, 'Louisiana': 7906, 'Maryland': 10235,
        'Massachusetts': 14287, 'Michigan': 16609, 'Minnesota': 8760, 'Missouri': 10380,
        'Nevada': 5998, 'New Hampshire': 2978, 'New Jersey': 18252, 'New York': 44227,
        'North Carolina': 18885, 'Ohio': 20329, 'Oregon': 9147, 'Pennsylvania': 22069,
        'Rhode Island': 2505, 'Tennessee': 12123, 'Texas': 51283, 'Utah': 5289,
        'Virginia': 15724, 'Washington': 14815, 'Wisconsin': 10121
    }
    
    # Add census data
    df['population'] = df['state'].map(lambda x: census_data.get(x, {}).get('population', 0))
    df['median_income'] = df['state'].map(lambda x: census_data.get(x, {}).get('median_income', 0))
    df['area_sq_mi'] = df['state'].map(lambda x: census_data.get(x, {}).get('area_sq_mi', 1))
    df['pop_density'] = df['population'] / df['area_sq_mi']
    
    # Add QSR data
    df['qsr_establishments'] = df['state'].map(qsr_data)
    df['qsr_per_100k'] = df['qsr_establishments'] / (df['population'] / 100_000)
    
    # Core penetration metric
    df['shacks_per_million'] = df['current_company_shacks'] / (df['population'] / 1_000_000)
    
    return df

@st.cache_data
def calculate_expansion_model(df):
    """Calculate demand scores, white space, and recommended additions."""
    df = df.copy()
    
    # Mature benchmark (NY, NJ)
    mature_states = ["New York", "New Jersey"]
    mature = df[df["state"].isin(mature_states)]
    mature_density = mature["shacks_per_million"].mean()
    target_density = 0.7 * mature_density
    
    # Potential shacks based on target density
    df["potential_shacks_density"] = (df["population"] / 1_000_000) * target_density
    
    # Normalize features for demand score
    df["income_z"] = (df["median_income"] - df["median_income"].mean()) / df["median_income"].std()
    df["density_z"] = (df["pop_density"] - df["pop_density"].mean()) / df["pop_density"].std()
    df["qsr_z"] = (df["qsr_per_100k"] - df["qsr_per_100k"].mean()) / df["qsr_per_100k"].std()
    
    # Demand score: higher income & density are good, higher QSR is bad
    df["demand_score"] = (
        0.5 * df["income_z"] +
        0.3 * df["density_z"] -
        0.2 * df["qsr_z"]
    )
    
    # Convert to multiplier
    df["demand_multiplier"] = 1 + 0.15 * df["demand_score"].clip(-2, 2)
    
    # Final potential
    df["potential_shacks"] = df["potential_shacks_density"] * df["demand_multiplier"]
    
    # White space calculation
    df["white_space"] = (df["potential_shacks"] - df["current_company_shacks"]).clip(lower=0)
    
    current_total = int(df["current_company_shacks"].sum())
    target_total = 1500
    incremental_needed = max(target_total - current_total, 0)
    
    if incremental_needed > 0 and df["white_space"].sum() > 0:
        df["white_space_share"] = df["white_space"] / df["white_space"].sum()
        df["recommended_adds"] = (df["white_space_share"] * incremental_needed).round()
    else:
        df["white_space_share"] = 0
        df["recommended_adds"] = 0
    
    df["recommended_total"] = df["current_company_shacks"] + df["recommended_adds"]
    
    return df, mature_density, target_density, current_total, incremental_needed

# Load data
DATA_FILE = "shak_state_locations_2024.csv"

try:
    df_raw = load_and_prepare_data(DATA_FILE)
    df, mature_density, target_density, current_total, incremental_needed = calculate_expansion_model(df_raw)
except FileNotFoundError:
    st.error(f"Error: '{DATA_FILE}' not found. Please ensure it's in the same directory.")
    st.stop()

# State centroids for maps
state_centroids = {
    "AL": (32.8, -86.6), "AK": (63.0, -150.0), "AZ": (34.2, -111.6),
    "AR": (34.9, -92.4), "CA": (37.1, -120.4), "CO": (39.0, -105.5),
    "CT": (41.6, -72.7), "DE": (39.0, -75.5), "FL": (28.1, -82.5),
    "GA": (32.7, -83.2), "HI": (20.8, -157.5), "ID": (44.2, -114.4),
    "IL": (40.0, -89.2), "IN": (40.3, -86.1), "IA": (42.0, -93.2),
    "KS": (38.5, -98.3), "KY": (37.6, -85.3), "LA": (31.0, -92.0),
    "ME": (45.3, -69.0), "MD": (39.0, -76.7), "MA": (42.2, -71.8),
    "MI": (44.2, -85.4), "MN": (46.1, -94.3), "MS": (32.7, -89.7),
    "MO": (38.5, -92.5), "MT": (47.0, -109.6), "NE": (41.5, -99.7),
    "NV": (39.0, -116.8), "NH": (43.7, -71.6), "NJ": (40.1, -74.7),
    "NM": (34.3, -106.0), "NY": (42.9, -75.5), "NC": (35.5, -79.5),
    "ND": (47.5, -100.5), "OH": (40.3, -82.8), "OK": (35.6, -97.5),
    "OR": (44.0, -120.5), "PA": (40.9, -77.8), "RI": (41.7, -71.6),
    "SC": (33.8, -80.9), "SD": (44.3, -100.2), "TN": (35.8, -86.4),
    "TX": (31.0, -99.0), "UT": (39.3, -111.7), "VT": (44.1, -72.7),
    "VA": (37.5, -78.8), "WA": (47.5, -120.5), "WV": (38.6, -80.6),
    "WI": (44.6, -89.4), "WY": (43.1, -107.6), "DC": (38.9, -77.0)
}

# =============================================================================
# SECTION 1: DATA OVERVIEW
# =============================================================================
if section == "1. Data Overview":
    st.header("1. Data Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total States with Shacks", len(df))
    with col2:
        st.metric("Current Company-Owned", current_total)
    with col3:
        st.metric("Target Units", 1500)
    with col4:
        st.metric("Incremental Needed", incremental_needed)
    
    st.subheader("Complete Dataset")
    st.dataframe(
        df[['state', 'state_abbrev', 'current_company_shacks', 'Licensed', 'population', 
            'median_income', 'pop_density', 'qsr_per_100k', 'shacks_per_million']].sort_values(
            'current_company_shacks', ascending=False
        ).style.format({
            'population': '{:,.0f}',
            'median_income': '${:,.0f}',
            'pop_density': '{:,.1f}',
            'qsr_per_100k': '{:,.1f}',
            'shacks_per_million': '{:.2f}'
        }),
        use_container_width=True,
        height=600
    )

# =============================================================================
# SECTION 2: DEMAND ANALYSIS
# =============================================================================
elif section == "2. Demand Analysis":
    st.header("2. Demand Score Analysis")
    
    st.markdown("""
    **Demand Score Formula:**
    - 50% weight on Median Income (z-score)
    - 30% weight on Population Density (z-score)  
    - -20% weight on QSR Competition (z-score) — higher competition is negative
    """)
    
    # Chart 1: Demand Score by State (Top 20)
    st.subheader("Demand Score by State (Top 20)")
    top_n = 20
    df_demand = df.sort_values("demand_score", ascending=False).head(top_n)
    
    fig_demand = px.bar(
        df_demand,
        x="state",
        y="demand_score",
        title=f"Demand Score by State (Top {top_n})",
        text="demand_score",
        labels={"state": "State", "demand_score": "Demand Score (z-score composite)"},
    )
    fig_demand.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        marker_color="#005030"
    )
    fig_demand.update_layout(
        xaxis_title="State",
        yaxis_title="Demand Score",
        title_x=0.5,
        bargap=0.2,
        uniformtext_minsize=9,
        uniformtext_mode="hide"
    )
    st.plotly_chart(fig_demand, use_container_width=True)
    
    # Show demand components
    st.subheader("Demand Score Components")
    st.dataframe(
        df[['state', 'median_income', 'pop_density', 'qsr_per_100k', 
            'income_z', 'density_z', 'qsr_z', 'demand_score']].sort_values(
            'demand_score', ascending=False
        ).style.format({
            'median_income': '${:,.0f}',
            'pop_density': '{:,.1f}',
            'qsr_per_100k': '{:,.1f}',
            'income_z': '{:.2f}',
            'density_z': '{:.2f}',
            'qsr_z': '{:.2f}',
            'demand_score': '{:.3f}'
        }),
        use_container_width=True
    )

# =============================================================================
# SECTION 3: WHITE SPACE & EXPANSION
# =============================================================================
elif section == "3. White Space & Expansion":
    st.header("3. White Space & Expansion Recommendations")
    
    st.markdown(f"""
    **Methodology:**
    - Mature benchmark states: New York & New Jersey
    - Mature penetration density: **{mature_density:.2f}** shacks per million
    - Target density (70% of mature): **{target_density:.2f}** shacks per million
    """)
    
    # Chart 2: White Space by State (Top 10)
    st.subheader("White Space by State (Top 10)")
    top10 = df.sort_values("recommended_adds", ascending=False).head(10)
    
    fig_whitespace = px.bar(
        top10,
        x="state",
        y="recommended_adds",
        text="recommended_adds",
        title="White Space by State (Top 10)",
        labels={"recommended_adds": "Recommended New Company-Owned Shacks"},
    )
    fig_whitespace.update_traces(
        texttemplate="%{text:.0f}",
        textposition="outside",
        marker_color="#005030"
    )
    fig_whitespace.update_layout(
        xaxis_title="State",
        yaxis_title="Recommended Adds",
        title_x=0.5,
        bargap=0.15,
        uniformtext_minsize=10,
        uniformtext_mode="hide"
    )
    st.plotly_chart(fig_whitespace, use_container_width=True)
    
    # Expansion recommendations table
    st.subheader("Full Expansion Recommendations")
    st.dataframe(
        df[['state', 'current_company_shacks', 'potential_shacks', 'white_space', 
            'recommended_adds', 'recommended_total']].sort_values(
            'recommended_adds', ascending=False
        ).style.format({
            'current_company_shacks': '{:.0f}',
            'potential_shacks': '{:.1f}',
            'white_space': '{:.1f}',
            'recommended_adds': '{:.0f}',
            'recommended_total': '{:.0f}'
        }),
        use_container_width=True
    )

# =============================================================================
# SECTION 4: US MAPS
# =============================================================================
elif section == "4. US Maps":
    st.header("4. Geographic Analysis — US Maps")
    
    # Prepare map data
    df_map = df.copy()
    df_map["recommended_adds"] = df_map["recommended_adds"].round(0).astype(int)
    df_map["hover_label"] = (
        df_map["state"] + "<br>" +
        "Recommended Adds: " + df_map["recommended_adds"].astype(str)
    )
    df_map["lat"] = df_map["state_abbrev"].map(lambda x: state_centroids.get(x, (0, 0))[0])
    df_map["lon"] = df_map["state_abbrev"].map(lambda x: state_centroids.get(x, (0, 0))[1])
    
    # Map 1: Recommended Net New Units
    st.subheader("Recommended Net New Shake Shack Units — U.S. Map")
    
    df_labels = df_map[df_map["recommended_adds"] > 0].copy()
    
    fig_map1 = go.Figure()
    
    fig_map1.add_trace(
        go.Choropleth(
            locations=df_map["state_abbrev"],
            z=df_map["recommended_adds"],
            locationmode="USA-states",
            colorscale=[
                [0.0, "#E8F6E8"],
                [0.3, "#A8D8A0"],
                [0.6, "#64B67A"],
                [1.0, "#2E8B57"]
            ],
            colorbar=dict(title="Recommended Adds", x=1.03, len=0.8),
            marker_line_color="black",
            marker_line_width=0.7,
            customdata=df_map["hover_label"],
            hovertemplate="%{customdata}<extra></extra>"
        )
    )
    
    fig_map1.add_trace(
        go.Scattergeo(
            lon=df_labels["lon"],
            lat=df_labels["lat"],
            text=df_labels["recommended_adds"],
            mode="text",
            textfont=dict(size=12, color="black"),
            showlegend=False
        )
    )
    
    fig_map1.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="white",
        showlakes=True,
        lakecolor="lightgray"
    )
    
    fig_map1.update_layout(
        title="Recommended Net New Shake Shack Units — U.S. Map",
        title_x=0.5,
        margin=dict(l=20, r=120, t=60, b=20)
    )
    
    st.plotly_chart(fig_map1, use_container_width=True)
    
    # Map 2: Top 10 Recommendations with ranges
    st.subheader("Top 10 State Recommendations (with ranges)")
    
    rec_data = {
        "state": ["Texas", "California", "Florida", "Ohio", "Georgia", 
                  "Illinois", "North Carolina", "Michigan", "Washington", "Arizona"],
        "state_abbrev": ["TX", "CA", "FL", "OH", "GA", "IL", "NC", "MI", "WA", "AZ"],
        "low_rec":  [120, 100, 90, 40, 35, 35, 30, 30, 25, 25],
        "high_rec": [150, 130, 110, 55, 50, 50, 45, 45, 35, 35]
    }
    
    df_rec = pd.DataFrame(rec_data)
    df_rec["mid_rec"] = (df_rec["low_rec"] + df_rec["high_rec"]) / 2
    df_rec["label"] = (
        df_rec["state_abbrev"] + ": " +
        df_rec["low_rec"].astype(str) + "–" +
        df_rec["high_rec"].astype(str)
    )
    
    state_coords = {
        "TX": (31.0, -99.0), "CA": (37.0, -120.0), "FL": (28.0, -82.0),
        "OH": (40.4, -82.8), "GA": (32.5, -83.5), "IL": (40.0, -89.0),
        "NC": (35.5, -79.5), "MI": (44.3, -85.5), "WA": (47.5, -120.5),
        "AZ": (34.2, -111.7)
    }
    
    df_rec["lat"] = df_rec["state_abbrev"].map(lambda s: state_coords[s][0])
    df_rec["lon"] = df_rec["state_abbrev"].map(lambda s: state_coords[s][1])
    
    fig_map2 = go.Figure()
    
    shake_shack_green = [
        [0.0, "#e5f5e0"],
        [0.5, "#7AB800"],
        [1.0, "#005030"]
    ]
    
    fig_map2.add_trace(
        go.Choropleth(
            locations=df_rec["state_abbrev"],
            z=df_rec["mid_rec"],
            locationmode="USA-states",
            colorscale=shake_shack_green,
            colorbar=dict(
                title="Net New Units (midpoint)",
                x=1.04,
                xanchor="left",
                len=0.75
            ),
            customdata=df_rec[["state", "low_rec", "high_rec"]],
            hovertemplate=(
                "%{customdata[0]}<br>" +
                "Recommended net new units: %{customdata[1]}–%{customdata[2]}<extra></extra>"
            ),
            name="Recommended Adds",
            marker_line_color="white",
            marker_line_width=0.5
        )
    )
    
    fig_map2.add_trace(
        go.Scattergeo(
            locationmode="USA-states",
            lon=df_rec["lon"],
            lat=df_rec["lat"],
            text=df_rec["label"],
            mode="text",
            textfont=dict(size=10, color="black"),
            showlegend=False
        )
    )
    
    fig_map2.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="white",
        showcountries=False,
        showlakes=True,
        lakecolor="rgb(240, 240, 240)"
    )
    
    fig_map2.update_layout(
        title_text="Recommended New Company-Owned Shake Shack Units by State: Top 10",
        title_x=0.5,
        margin=dict(l=20, r=90, t=60, b=20)
    )
    
    st.plotly_chart(fig_map2, use_container_width=True)
    
    # Map 3: Side-by-side Penetration vs QSR Competition
    st.subheader("Shake Shack Penetration vs QSR Competition (Side-by-Side)")
    
    shack_colorscale = [[0.0, "#1b7837"], [0.5, "#ffffff"], [1.0, "#d73027"]]
    qsr_colorscale = [[0.0, "#1b7837"], [0.5, "#ffffff"], [1.0, "#d73027"]]
    
    fig_dual = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "choropleth"}, {"type": "choropleth"}]],
        subplot_titles=[
            "Shake Shack Penetration (Shacks per 1M Residents)",
            "QSR Competition (Restaurants per 100k Residents)"
        ],
        horizontal_spacing=0.12
    )
    
    # Left Map — SHAK Penetration
    fig_dual.add_trace(
        go.Choropleth(
            locations=df["state_abbrev"],
            z=df["shacks_per_million"],
            locationmode="USA-states",
            colorscale=shack_colorscale,
            zmin=0, zmax=4,
            colorbar=dict(title="Shacks / 1M", x=0.46, len=0.65),
            hovertext=df["state"],
            hovertemplate="<b>%{hovertext}</b><br>Shacks per 1M: %{z:.2f}<extra></extra>",
            name="SHAK penetration"
        ),
        row=1, col=1
    )
    
    # Right Map — QSR Competition
    fig_dual.add_trace(
        go.Choropleth(
            locations=df["state_abbrev"],
            z=df["qsr_per_100k"],
            locationmode="USA-states",
            colorscale=qsr_colorscale,
            zmin=0, zmax=400,
            colorbar=dict(title="QSR / 100k", x=1.05, len=0.65),
            hovertext=df["state"],
            hovertemplate="<b>%{hovertext}</b><br>QSR per 100k: %{z:.1f}<extra></extra>",
            name="QSR density"
        ),
        row=1, col=2
    )
    
    fig_dual.update_geos(
        scope="usa",
        projection_type="albers usa",
        showlakes=True,
        lakecolor="white"
    )
    
    fig_dual.update_layout(
        title_text="<b>Shake Shack Penetration vs QSR Competition (State-Level)</b>",
        title_x=0.5,
        title_y=0.97,
        margin=dict(l=0, r=40, t=110, b=0),
        font=dict(size=14),
    )
    
    fig_dual.data[0].geo = "geo"
    fig_dual.data[1].geo = "geo2"
    
    st.plotly_chart(fig_dual, use_container_width=True)

# =============================================================================
# SECTION 5: SCENARIO ANALYSIS
# =============================================================================
elif section == "5. Scenario Analysis":
    st.header("5. Scenario Analysis: Path to 1,500 Units")
    
    # Scenario parameters
    start_year = 2025
    end_year = 2035
    years = np.arange(start_year, end_year + 1)
    start_units = 359
    
    # Growth rates
    base_growth = np.array([0.00, 0.13, 0.13, 0.12, 0.12, 0.11, 0.11, 0.10, 0.10, 0.09, 0.09])
    bull_growth = np.array([0.00, 0.18, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10])
    bear_growth = np.array([0.00, 0.08, 0.08, 0.075, 0.075, 0.07, 0.07, 0.067, 0.067, 0.063, 0.063])
    
    def build_path_from_growth(start_units, growth_rates):
        units = [start_units]
        for g in growth_rates[1:]:
            units.append(units[-1] * (1 + g))
        return np.array(units)
    
    base_units = build_path_from_growth(start_units, base_growth)
    bull_units = build_path_from_growth(start_units, bull_growth)
    bear_units = build_path_from_growth(start_units, bear_growth)
    
    # Create scenario dataframe
    paths_df = pd.DataFrame({
        "Year": years,
        "Base Case": base_units.round(0).astype(int),
        "Bull Case": bull_units.round(0).astype(int),
        "Bear Case": bear_units.round(0).astype(int)
    })
    
    st.subheader("Scenario Projections Table")
    st.dataframe(paths_df, use_container_width=True)
    
    # Plotly version of the scenario chart
    st.subheader("Scenario Paths to 1,500 Company-Owned Locations")
    
    fig_scenario = go.Figure()
    
    fig_scenario.add_trace(go.Scatter(
        x=years, y=base_units,
        mode='lines+markers+text',
        name='Base (low-teens growth)',
        line=dict(color='#7AB800', width=3),
        marker=dict(size=10),
        text=[f"{int(v)}" for v in base_units],
        textposition='top center',
        textfont=dict(size=9)
    ))
    
    fig_scenario.add_trace(go.Scatter(
        x=years, y=bull_units,
        mode='lines+markers+text',
        name='Bull (higher growth)',
        line=dict(color='#00A86B', width=3),
        marker=dict(size=10),
        text=[f"{int(v)}" for v in bull_units],
        textposition='top center',
        textfont=dict(size=9)
    ))
    
    fig_scenario.add_trace(go.Scatter(
        x=years, y=bear_units,
        mode='lines+markers+text',
        name='Bear (slower growth)',
        line=dict(color='#CC0000', width=3),
        marker=dict(size=10),
        text=[f"{int(v)}" for v in bear_units],
        textposition='bottom center',
        textfont=dict(size=9)
    ))
    
    # Target line
    fig_scenario.add_hline(y=1500, line_dash="dash", line_color="gray",
                          annotation_text="Mgmt long-term target: 1,500 units",
                          annotation_position="top left")
    
    fig_scenario.update_layout(
        title="Scenario Paths to 1,500 Company-Owned Shake Shack Locations<br><sup>(all scenarios start at 359 units in 2025)</sup>",
        title_x=0.5,
        xaxis_title="Year",
        yaxis_title="Company-Owned Stores",
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_scenario, use_container_width=True)
    
    st.caption("""
    **Note:** Base scenario assumes low-teens annual growth in company-owned units, 
    with higher growth in the bull case and slower build-out in the bear case. 
    Illustrative paths are broadly anchored on management's long-term unit growth 
    framework discussed in the January investor presentation; they are not guidance.
    """)

# =============================================================================
# SECTION 6: COMPETITIVE ANALYSIS
# =============================================================================
elif section == "6. Competitive Analysis":
    st.header("6. Competitive Analysis")
    
    # Chart: Better-Burger Peer Comparison
    st.subheader("Better-Burger / Fast-Casual Peer Comparison (Total Locations)")
    
    peer_data = {
        "brand": [
            "Chipotle Mexican Grill",
            "Five Guys",
            "Shake Shack",
            "Smashburger",
            "BurgerFi"
        ],
        "total_locations": [3800, 1500, 539, 200, 93]
    }
    
    df_peers = pd.DataFrame(peer_data)
    df_peers = df_peers.sort_values("total_locations", ascending=False)
    
    color_map = {
        "Chipotle Mexican Grill": "#7b3f00",
        "Five Guys": "#d52b1e",
        "Shake Shack": "#2cb34a",
        "Smashburger": "#e31b23",
        "BurgerFi": "#7ac143"
    }
    
    fig_peers = px.bar(
        df_peers,
        x="brand",
        y="total_locations",
        title="Better-Burger / Fast-Casual Peer Comparison (Total Locations)",
        labels={"brand": "Brand", "total_locations": "Total Restaurants (Global/System)"},
        color="brand",
        color_discrete_map=color_map,
        text="total_locations"
    )
    
    fig_peers.update_traces(textposition="outside", textfont=dict(size=14))
    
    max_val = df_peers["total_locations"].max()
    fig_peers.update_yaxes(range=[0, max_val * 1.15])
    
    fig_peers.update_layout(
        xaxis_tickangle=-30,
        showlegend=False,
        yaxis=dict(title="Total Restaurants"),
        xaxis=dict(title="")
    )
    
    fig_peers.add_annotation(
        text="*Smashburger and BurgerFi totals are approximate due to conflicting public data.",
        xref="paper", yref="paper",
        x=0, y=-0.25,
        showarrow=False,
        font=dict(size=11, color="gray"),
        align="left"
    )
    
    fig_peers.update_layout(margin=dict(b=100))
    
    st.plotly_chart(fig_peers, use_container_width=True)

# =============================================================================
# SECTION 7: FINANCIAL METRICS
# =============================================================================
elif section == "7. Financial Metrics":
    st.header("7. Financial Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Chart: Restaurant-Level Margins Comparison
        st.subheader("Restaurant-Level Margins: Shake Shack vs Peers")
        
        margin_data = {
            "company": [
                "Chipotle Mexican Grill",
                "Shake Shack",
                "BurgerFi",
                "Smashburger"
            ],
            "restaurant_margin_pct": [26.7, 21.4, 12.0, 9.0],
            "color": ["#8B0000", "#43A047", "#000000", "#D32F2F"]
        }
        
        df_margins = pd.DataFrame(margin_data)
        df_margins_sorted = df_margins.sort_values(by="restaurant_margin_pct", ascending=False)
        
        fig_margins = px.bar(
            df_margins_sorted,
            x="company",
            y="restaurant_margin_pct",
            title="Restaurant-Level Margins: Shake Shack vs Premium Fast-Casual Peers",
            text="restaurant_margin_pct",
            color="company",
            color_discrete_map={row.company: row.color for _, row in df_margins_sorted.iterrows()},
        )
        
        fig_margins.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_margins.update_layout(
            yaxis_title="Restaurant-Level Margin (%)",
            xaxis_title="Company",
            xaxis_tickangle=-30,
            plot_bgcolor="white",
            showlegend=False
        )
        
        st.plotly_chart(fig_margins, use_container_width=True)
    
    with col2:
        # Chart: Shake Shack Margin Trend
        st.subheader("Shake Shack Margin Trend (2022-2024)")
        
        margin_years = [2022, 2023, 2024]
        margins = [17.5, 19.9, 21.4]
        
        fig_margin_trend = go.Figure()
        
        fig_margin_trend.add_trace(go.Scatter(
            x=margin_years,
            y=margins,
            mode='lines+markers+text',
            line=dict(color='#2E8B57', width=3),
            marker=dict(size=12),
            text=[f"{m}%" for m in margins],
            textposition='top center',
            textfont=dict(size=12)
        ))
        
        fig_margin_trend.update_layout(
            title="Shake Shack Restaurant-Level Profit Margin (2022–2024)",
            xaxis_title="Year",
            yaxis_title="Restaurant-Level Profit Margin (%)",
            xaxis=dict(tickmode='array', tickvals=margin_years),
            plot_bgcolor='white',
            yaxis=dict(gridcolor='lightgray', gridwidth=0.5)
        )
        
        st.plotly_chart(fig_margin_trend, use_container_width=True)
    
    # Chart: Revenue Growth vs Store Count
    st.subheader("Revenue Growth vs. Store Count (2020-2024)")
    
    rev_data = pd.DataFrame({
        "year": [2020, 2021, 2022, 2023, 2024],
        "revenue": [523, 739, 900, 1098, 1310],
        "units": [183, 218, 263, 307, 329]
    })
    
    fig_rev = go.Figure()
    
    fig_rev.add_trace(
        go.Bar(
            x=rev_data["year"],
            y=rev_data["revenue"],
            name="Revenue ($M)",
            marker_color="#005030",
            text=rev_data["revenue"],
            textposition='outside'
        )
    )
    
    fig_rev.add_trace(
        go.Scatter(
            x=rev_data["year"],
            y=rev_data["units"],
            name="Company-Operated Units",
            yaxis="y2",
            mode="lines+markers+text",
            marker_color="black",
            line=dict(width=2),
            text=rev_data["units"],
            textposition='top center'
        )
    )
    
    fig_rev.update_layout(
        title="Revenue Growth vs. Store Count",
        title_x=0.5,
        yaxis=dict(title="Revenue ($M)"),
        yaxis2=dict(
            title="Units",
            overlaying="y",
            side="right",
            tickfont=dict(color="black")
        ),
        margin=dict(l=70, r=110, t=80, b=60),
        legend=dict(x=0.02, y=0.98),
        barmode='group'
    )
    
    st.plotly_chart(fig_rev, use_container_width=True)

# =============================================================================
# SECTION 8: PENETRATION SCATTER PLOTS
# =============================================================================
elif section == "8. Penetration Scatter Plots":
    st.header("8. Penetration Analysis — Scatter Plots")
    
    # Chart 1: Penetration vs Population Density
    st.subheader("State-Level Penetration vs Population Density")
    
    fig_pop_density = px.scatter(
        df,
        x="pop_density",
        y="shacks_per_million",
        text="state_abbrev",
        title="State-Level Penetration vs Population Density",
    )
    fig_pop_density.update_traces(textposition="top center", marker_size=12, marker_color="#005030")
    fig_pop_density.update_layout(
        xaxis_title="Population Density (people per sq mi)",
        yaxis_title="Shacks per Million Residents"
    )
    st.plotly_chart(fig_pop_density, use_container_width=True)
    
    st.divider()
    
    # Chart 2: Penetration vs QSR Competition
    st.subheader("SHAK Penetration vs QSR Competition")
    
    fig_qsr = px.scatter(
        df,
        x="qsr_per_100k",
        y="shacks_per_million",
        text="state_abbrev",
        title="SHAK Penetration vs QSR Competition",
    )
    fig_qsr.update_traces(marker_color="red", marker_size=12, textposition="top center")
    fig_qsr.update_layout(
        xaxis_title="QSR Restaurants per 100k Residents",
        yaxis_title="Shacks per Million Residents"
    )
    st.plotly_chart(fig_qsr, use_container_width=True)
    
    st.divider()
    
    # Chart 3: Penetration vs QSR with Trendline
    st.subheader("SHAK Penetration vs QSR Competition (with OLS Trendline)")
    
    # Calculate trendline manually using numpy (no statsmodels required)
    x_data = df["qsr_per_100k"].values
    y_data = df["shacks_per_million"].values
    
    # Linear regression using numpy
    slope, intercept = np.polyfit(x_data, y_data, 1)
    trendline_y = slope * x_data + intercept
    
    # Calculate R-squared
    ss_res = np.sum((y_data - trendline_y) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Create scatter plot without plotly's trendline (avoids statsmodels dependency)
    fig_qsr_trend = go.Figure()
    
    # Add scatter points with color scale
    fig_qsr_trend.add_trace(go.Scatter(
        x=df["qsr_per_100k"],
        y=df["shacks_per_million"],
        mode="markers+text",
        text=df["state_abbrev"],
        textposition="top center",
        marker=dict(
            size=12,
            color=df["shacks_per_million"],
            colorscale=[[0, "#1b7837"], [0.5, "#ffffff"], [1, "#d73027"]],
            colorbar=dict(title="Shacks per 1M", ticks="outside"),
            line=dict(width=1, color="black")
        ),
        name="States",
        hovertemplate="<b>%{text}</b><br>QSR per 100k: %{x:.1f}<br>Shacks per 1M: %{y:.2f}<extra></extra>"
    ))
    
    # Add trendline
    x_sorted = np.sort(x_data)
    y_trend_sorted = slope * x_sorted + intercept
    
    fig_qsr_trend.add_trace(go.Scatter(
        x=x_sorted,
        y=y_trend_sorted,
        mode="lines",
        line=dict(color="rgba(0,0,0,0.5)", width=2, dash="solid"),
        name=f"OLS Trendline (R²={r_squared:.3f})",
        hovertemplate=f"<b>OLS Trendline</b><br>y = {slope:.4f}x + {intercept:.4f}<br>R² = {r_squared:.3f}<extra></extra>"
    ))
    
    fig_qsr_trend.update_layout(
        title="Shake Shack Penetration vs. QSR Competitive Intensity",
        xaxis_title="QSR Restaurants per 100k Residents",
        yaxis_title="Shacks per 1M Residents",
        title_x=0.5,
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig_qsr_trend, use_container_width=True)
    
    # Show trendline info
    st.info(f"""
    **OLS Regression Results:**
    - R² = {r_squared:.3f}
    - Slope = {slope:.4f}
    - Intercept = {intercept:.4f}
    - Equation: Shacks per 1M = {slope:.4f} × (QSR per 100k) + {intercept:.4f}
    """)

# Footer
st.divider()
st.caption("Data sources: Shake Shack 10-K filings, US Census Bureau (ACS 2023, CBP 2022)")