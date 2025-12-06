import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Shake Shack (SHAK) Investment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🍔 Shake Shack (SHAK) Investment Analysis")
st.caption("Phil Russomanno | Millennium Case Study Submission | 12/7/2025")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Jump to Section:",
    [
        "1. Executive Recommendation",
        "2. Analytical Approach",
        "3. Whitespace Analysis",
        "4. Feasibility: Path to 1,500 Units",
        "5. Competitive Context",
        "6. Investment Applications",
        "7. Conclusion",
        "8. Appendix: Supplemental Charts"
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
    
    # CBP 2022 QSR establishments (NAICS 7225)
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
# SECTION 1: EXECUTIVE RECOMMENDATION
# =============================================================================
if section == "1. Executive Recommendation":
    st.header("1. Executive Recommendation")
    
    st.markdown("""
    **My recommendation is that Shake Shack can achieve the stated goal of 1,500 company-operated locations** across geographic regions with large populations, high incomes, and lower QSR saturation. 
    
    The **top 10 target states** are:
    """)
    
    # Key target states callout
    target_states = ["Texas", "California", "Florida", "Ohio", "Georgia", 
                     "Illinois", "North Carolina", "Michigan", "Washington", "Arizona"]
    
    cols = st.columns(5)
    for i, state in enumerate(target_states):
        with cols[i % 5]:
            st.success(f"**{state}**")
    
    st.markdown("""
    These markets represent the largest collections of **demographic depth**, **favorable competition intensity**, 
    and **whitespace opportunities** compared to the mature benchmarks of New York and New Jersey.
    
    Based on the demand scoring model, these states alone can support **more than half of new company-operated Shacks**, 
    representing a substantial share of the ~1,000 additional units needed to reach management's long-term target.
    """)
    
    # Top 10 Recommendations Map
    st.subheader("Recommended New Company-Owned Units by State: Top 10")
    
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
    
    fig_rec_map = go.Figure()
    
    shake_shack_green = [
        [0.0, "#e5f5e0"],
        [0.5, "#7AB800"],
        [1.0, "#005030"]
    ]
    
    fig_rec_map.add_trace(
        go.Choropleth(
            locations=df_rec["state_abbrev"],
            z=df_rec["mid_rec"],
            locationmode="USA-states",
            colorscale=shake_shack_green,
            colorbar=dict(title="Net New Units", x=1.04, len=0.75),
            customdata=df_rec[["state", "low_rec", "high_rec"]],
            hovertemplate="%{customdata[0]}<br>Recommended: %{customdata[1]}–%{customdata[2]}<extra></extra>",
            marker_line_color="white",
            marker_line_width=0.5
        )
    )
    
    fig_rec_map.add_trace(
        go.Scattergeo(
            lon=df_rec["lon"],
            lat=df_rec["lat"],
            text=df_rec["label"],
            mode="text",
            textfont=dict(size=10, color="black"),
            showlegend=False
        )
    )
    
    fig_rec_map.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="white",
        showlakes=True,
        lakecolor="rgb(240, 240, 240)"
    )
    
    fig_rec_map.update_layout(
        title_text="Recommended New Company-Owned Shake Shack Units by State: Top 10",
        title_x=0.5,
        margin=dict(l=20, r=90, t=60, b=20)
    )
    
    st.plotly_chart(fig_rec_map, use_container_width=True)
    
    # Key metrics summary
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Company-Owned", current_total)
    with col2:
        st.metric("Target Units", 1500)
    with col3:
        st.metric("Units Needed", f"~{incremental_needed:,}")
    with col4:
        st.metric("Top 10 States Share", "60%+")

# =============================================================================
# SECTION 2: ANALYTICAL APPROACH
# =============================================================================
elif section == "2. Analytical Approach":
    st.header("2. Overview of Analytical Approach")
    
    st.markdown("""
    To identify the credibility of Shake Shack's potential roadmap, I developed a **data-driven framework** emphasizing population, income, urban density, competitive saturation, and current Shake Shack penetration.
    
    **Data Sources Merged:**
    - Census ACS (population/income)
    - NAICS 7225 concentration (competitive saturation)  
    - Shake Shack's 10-K/Q disclosures
    
    **Three Diagnostic Metrics Created:**
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("""
        **1. Shacks per Million Residents**
        
        A penetration proxy measuring maturity thresholds
        """)
    with col2:
        st.info("""
        **2. QSR per 100k Residents**
        
        A competition density benchmark for market saturation
        """)
    with col3:
        st.info("""
        **3. Demand Score**
        
        Normalized composite of median income, population density, and competition
        """)
    
    st.markdown("""
    These metrics are crucial for identifying the optimal allocation of new Shacks in whitespace markets.
    """)
    
    # Side-by-side Penetration vs QSR Competition Map
    st.subheader("Shake Shack Penetration vs QSR Competition (State-Level)")
    
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
    
    st.caption("""
    **Interpretation:** Green indicates lower values (less penetration / less competition), 
    red indicates higher values. States with low SHAK penetration but low QSR competition 
    represent prime whitespace opportunities.
    """)
    
    # Demand Score Components
    st.subheader("Demand Score Formula")
    st.markdown("""
    The **Demand Score** is calculated as a weighted z-score composite:
    
    ```
    Demand Score = 0.50 × Income_z + 0.30 × Density_z − 0.20 × QSR_z
    ```
    
    - **50% weight** on Median Income (higher income = higher demand)
    - **30% weight** on Population Density (higher density = higher demand)
    - **−20% weight** on QSR Competition (higher competition = lower attractiveness)
    """)
    
    # Demand Score Chart - FIX APPLIED: Increased tick angle and bottom margin
    st.subheader("Demand Score by State (Top 20)")
    top_n = 20
    df_demand = df.sort_values("demand_score", ascending=False).head(top_n)
    
    # Create enhanced bar chart
    fig_demand = go.Figure()
    
    # Color bars based on whether they're in top 10 target states
    target_states = ["Texas", "California", "Florida", "Ohio", "Georgia", 
                     "Illinois", "North Carolina", "Michigan", "Washington", "Arizona"]
    
    colors = ["#005030" if state in target_states else "#7AB800" for state in df_demand["state"]]
    
    fig_demand.add_trace(go.Bar(
        x=df_demand["state"],
        y=df_demand["demand_score"],
        text=df_demand["demand_score"].round(2),
        textposition="outside",
        texttemplate="%{text:.2f}",
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Demand Score: %{y:.3f}<br>Median Income: $%{customdata[0]:,.0f}<br>Pop Density: %{customdata[1]:,.1f}/sq mi<br>QSR per 100k: %{customdata[2]:.1f}<extra></extra>",
        customdata=df_demand[["median_income", "pop_density", "qsr_per_100k"]].values,
        name="Demand Score"
    ))
    
    # Calculate proper y-axis range
    max_score = df_demand["demand_score"].max()
    min_score = df_demand["demand_score"].min()
    
    fig_demand.update_layout(
        title=f"Demand Score by State (Top {top_n})",
        title_x=0.5,
        xaxis_title="State",
        yaxis_title="Demand Score (z-score composite)",
        yaxis=dict(
            range=[min(0, min_score * 1.1), max_score * 1.18],  # Add headroom for labels
            gridcolor="lightgray",
            gridwidth=0.5
        ),
        bargap=0.2,
        uniformtext_minsize=9,
        uniformtext_mode="hide",
        plot_bgcolor="white",
        hovermode="x unified",
        xaxis_tickangle=-65, # FIX: Increased angle to prevent label overlap
        annotations=[
            dict(
                text="<b>■</b> Top 10 Target States  <b style='color:#7AB800'>■</b> Other States",
                xref="paper", yref="paper",
                x=0.5, y=-0.35, # Adjusted y position for annotation due to angle
                showarrow=False,
                font=dict(size=11),
                align="center"
            )
        ],
        margin=dict(b=150) # FIX: Increased bottom margin to accommodate angle
    )
    
    st.plotly_chart(fig_demand, use_container_width=True)

# =============================================================================
# SECTION 3: WHITESPACE ANALYSIS
# =============================================================================
elif section == "3. Whitespace Analysis":
    st.header("3. Whitespace Analysis")
    
    st.markdown(f"""
    The whitespace model quantifies **underpenetration relative to demographic fundamentals**.
    
    **Methodology:**
    - Mature benchmark states: **New York & New Jersey**
    - Mature penetration density: **{mature_density:.2f}** shacks per million
    - Target density (70% of mature): **{target_density:.2f}** shacks per million
    
    **Key Findings:**
    - Texas, Florida and California performed strongly due to strong in-migration, suburban expansion, 
      and low current Shake Shack density
    - Other top 10 states share similar characteristics: affluent consumers, strong population density, 
      and diversified cuisine offerings
    - The model suggests **60% of future company-operated units** should be in the 10 highlighted states, 
      with 40% spread across the broader United States as saturation normalizes
    """)
    
    # Whitespace Map
    st.subheader("Recommended Net New Shake Shack Units — U.S. Map")
    
    df_map = df.copy()
    df_map["recommended_adds"] = df_map["recommended_adds"].round(0).astype(int)
    df_map["hover_label"] = (
        df_map["state"] + "<br>" +
        "Recommended Adds: " + df_map["recommended_adds"].astype(str)
    )
    df_map["lat"] = df_map["state_abbrev"].map(lambda x: state_centroids.get(x, (0, 0))[0])
    df_map["lon"] = df_map["state_abbrev"].map(lambda x: state_centroids.get(x, (0, 0))[1])
    
    df_labels = df_map[df_map["recommended_adds"] > 0].copy()
    
    fig_map = go.Figure()
    
    fig_map.add_trace(
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
    
    fig_map.add_trace(
        go.Scattergeo(
            lon=df_labels["lon"],
            lat=df_labels["lat"],
            text=df_labels["recommended_adds"],
            mode="text",
            textfont=dict(size=12, color="black"),
            showlegend=False
        )
    )
    
    fig_map.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="white",
        showlakes=True,
        lakecolor="lightgray"
    )
    
    fig_map.update_layout(
        title="Recommended Net New Shake Shack Units — U.S. Map",
        title_x=0.5,
        margin=dict(l=20, r=120, t=60, b=20)
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Whitespace Bar Chart (Top 10)
    st.subheader("White Space by State (Top 10)")
    top10 = df.sort_values("recommended_adds", ascending=False).head(10)
    
    fig_whitespace = go.Figure()
    
    fig_whitespace.add_trace(go.Bar(
        x=top10["state"],
        y=top10["recommended_adds"],
        text=top10["recommended_adds"].round(0).astype(int),
        textposition="outside",
        texttemplate="%{text}",
        marker_color="#005030",
        hovertemplate="<b>%{x}</b><br>Recommended Adds: %{y:.0f}<br>Population: %{customdata[0]:,.0f}<br>Current Shacks: %{customdata[1]:.0f}<extra></extra>",
        customdata=top10[["population", "current_company_shacks"]].values,
        name="Recommended Adds"
    ))
    
    # Calculate proper y-axis range to prevent clipping
    max_val = top10["recommended_adds"].max()
    
    fig_whitespace.update_layout(
        title="White Space by State (Top 10)",
        title_x=0.5,
        xaxis_title="State",
        yaxis_title="Recommended Adds",
        yaxis=dict(range=[0, max_val * 1.18]),  # Add 18% headroom for labels
        bargap=0.15,
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        hovermode="x unified",
        plot_bgcolor="white",
        yaxis_gridcolor="lightgray",
        yaxis_gridwidth=0.5
    )
    
    # Add range slider for interactivity
    fig_whitespace.update_xaxes(
        tickangle=-30,
        categoryorder="total descending"
    )
    
    st.plotly_chart(fig_whitespace, use_container_width=True)
    
    # Full Expansion Table
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
# SECTION 4: FEASIBILITY - PATH TO 1,500 UNITS
# =============================================================================
elif section == "4. Feasibility: Path to 1,500 Units":
    st.header("4. Feasibility of 1,500 Unit Target")
    
    st.markdown("""
    The feasibility analysis treats the **1,500-unit goal as a market-implied carrying capacity**.
    
    **Methodology:**
    - Applying 70% of the mature market's penetration to the full US population 
    - Results in a ceiling of **~1,550 units** — confirming the goal is attainable
    
    **Three Scenarios Modeled:**
    - **Base Case:** Management expectations from January ICR conference (low-teens growth)
    - **Bull Case:** Accelerated execution with favorable real estate and kiosk adoption
    - **Bear Case:** Slower rollout due to execution challenges
    
    **Critical Execution Drivers:**
    - Kiosk adaptation
    - Suburban drive-thru rollout
    - Real estate availability
    """)
    
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
    
    # Calculate growth rates for hover
    def calc_yoy_growth(units):
        growth = [0]
        for i in range(1, len(units)):
            growth.append((units[i] - units[i-1]) / units[i-1] * 100)
        return growth
    
    base_yoy = calc_yoy_growth(base_units)
    bull_yoy = calc_yoy_growth(bull_units)
    bear_yoy = calc_yoy_growth(bear_units)
    
    # Scenario Chart - FIX APPLIED: Removed rangeslider to eliminate plot lines under chart
    st.subheader("Scenario Paths to 1,500 Company-Owned Locations")
    
    fig_scenario = go.Figure()
    
    # Base case
    fig_scenario.add_trace(go.Scatter(
        x=years, y=base_units,
        mode='lines+markers',
        name='Base (low-teens growth)',
        line=dict(color='#7AB800', width=3),
        marker=dict(size=10),
        hovertemplate="<b>Base Case - %{x}</b><br>Units: %{y:.0f}<br>YoY Growth: %{customdata:.1f}%<extra></extra>",
        customdata=base_yoy
    ))
    
    # Bull case
    fig_scenario.add_trace(go.Scatter(
        x=years, y=bull_units,
        mode='lines+markers',
        name='Bull (higher growth)',
        line=dict(color='#00A86B', width=3),
        marker=dict(size=10),
        hovertemplate="<b>Bull Case - %{x}</b><br>Units: %{y:.0f}<br>YoY Growth: %{customdata:.1f}%<extra></extra>",
        customdata=bull_yoy
    ))
    
    # Bear case
    fig_scenario.add_trace(go.Scatter(
        x=years, y=bear_units,
        mode='lines+markers',
        name='Bear (slower growth)',
        line=dict(color='#CC0000', width=3),
        marker=dict(size=10),
        hovertemplate="<b>Bear Case - %{x}</b><br>Units: %{y:.0f}<br>YoY Growth: %{customdata:.1f}%<extra></extra>",
        customdata=bear_yoy
    ))
    
    # Target line
    fig_scenario.add_hline(y=1500, line_dash="dash", line_color="gray",
                          annotation_text="Mgmt long-term target: 1,500 units",
                          annotation_position="top left")
    
    # Add milestone annotations
    for scenario, units, color, name in [
        ("Bull", bull_units, "#00A86B", "Bull"),
        ("Base", base_units, "#7AB800", "Base"),
        ("Bear", bear_units, "#CC0000", "Bear")
    ]:
        # Find when scenario crosses 1000 and 1500
        for milestone in [1000, 1500]:
            for i, u in enumerate(units):
                if u >= milestone:
                    fig_scenario.add_annotation(
                        x=years[i],
                        y=units[i],
                        text=f"{name}: {milestone}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=0.8,
                        arrowcolor=color,
                        font=dict(size=9, color=color),
                        ax=0,
                        ay=-25 if name == "Bull" else (25 if name == "Bear" else 0),
                        visible=False  # Hidden by default, shown via buttons
                    )
                    break
    
    fig_scenario.update_layout(
        title="Scenario Paths to 1,500 Company-Owned Shake Shack Locations<br><sup>(all scenarios start at 359 units in 2025)</sup>",
        title_x=0.5,
        xaxis_title="Year",
        yaxis_title="Company-Owned Stores",
        legend=dict(
            x=0.02, 
            y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1
        ),
        hovermode='x unified',
        plot_bgcolor="white",
        yaxis=dict(gridcolor="lightgray", gridwidth=0.5),
        xaxis=dict(
            tickmode='linear',
            tick0=2025,
            dtick=1,
            # rangeslider=dict(visible=True, thickness=0.05), # REMOVED: To prevent extra lines under chart
        )
    )
    
    # Add toggle buttons for scenarios
    fig_scenario.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": [True, True, True]}],
                        label="All Scenarios",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True, False, False]}],
                        label="Base Only",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True, False]}],
                        label="Bull Only",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, False, True]}],
                        label="Bear Only",
                        method="update"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.5,
                xanchor="center",
                y=-0.22,
                yanchor="top"
            )
        ]
    )
    
    st.plotly_chart(fig_scenario, use_container_width=True)
    
    # Scenario Table
    st.subheader("Scenario Projections Table")
    paths_df = pd.DataFrame({
        "Year": years,
        "Base Case": base_units.round(0).astype(int),
        "Bull Case": bull_units.round(0).astype(int),
        "Bear Case": bear_units.round(0).astype(int)
    })
    st.dataframe(paths_df, use_container_width=True)
    
    st.caption("""
    **Note:** Base scenario assumes low-teens annual growth in company-owned units, 
    with higher growth in the bull case and slower build-out in the bear case. 
    Illustrative paths are broadly anchored on management's long-term unit growth 
    framework discussed in the January investor presentation; they are not guidance.
    """)

# =============================================================================
# SECTION 5: COMPETITIVE CONTEXT
# =============================================================================
elif section == "5. Competitive Context":
    st.header("5. Competitive Context")
    
    st.markdown("""
    Shake Shack competes in the **premium fast-casual space** as opposed to mass QSR, making 
    **Five Guys, BurgerFi, Smashburger, and Chipotle** more relevant comparables than McDonald's and Wendy's.
    
    States with heavy QSR clustering (Midwest, parts of Northeast) present higher saturation risk, 
    but demand score analysis suggests the target customer is fairly unpenetrated in **high income, 
    fast-growing markets** such as the Sun Belt and Pacific Coast.
    """)
    
    # Peer Comparison Chart - FIX APPLIED: Increased bottom margin to prevent note overlap
    st.subheader("Better-Burger / Fast-Casual Peer Comparison (Total Locations)")
    
    peer_data = {
        "brand": [
            "Chipotle Mexican Grill",
            "Five Guys",
            "Shake Shack",
            "Smashburger",
            "BurgerFi"
        ],
        "total_locations": [3800, 1500, 539, 200, 93],
        "category": ["Mexican Fast-Casual", "Better Burger", "Better Burger", "Better Burger", "Better Burger"],
        "ownership": ["Public (CMG)", "Private", "Public (SHAK)", "Private (Jollibee)", "Public (BFI)"]
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
    
    fig_peers = go.Figure()
    
    for _, row in df_peers.iterrows():
        fig_peers.add_trace(go.Bar(
            x=[row["brand"]],
            y=[row["total_locations"]],
            text=[f"{row['total_locations']:,}"],
            textposition="outside",
            textfont=dict(size=14),
            marker_color=color_map[row["brand"]],
            name=row["brand"],
            hovertemplate=f"<b>{row['brand']}</b><br>Total Locations: {row['total_locations']:,}<br>Category: {row['category']}<br>Ownership: {row['ownership']}<extra></extra>",
            showlegend=False
        ))
    
    max_val = df_peers["total_locations"].max()
    
    fig_peers.update_layout(
        title="Better-Burger / Fast-Casual Peer Comparison (Total Locations)",
        title_x=0.5,
        xaxis_tickangle=-30,
        yaxis=dict(
            title="Total Restaurants",
            range=[0, max_val * 1.18],  # Add 18% headroom for labels
            gridcolor="lightgray",
            gridwidth=0.5
        ),
        xaxis=dict(title=""),
        plot_bgcolor="white",
        hovermode="x unified"
    )
    
    fig_peers.add_annotation(
        text="*Smashburger and BurgerFi totals are approximate due to conflicting public data.",
        xref="paper", yref="paper",
        x=0, y=-0.3, # FIX: Moved annotation down to prevent overlap
        showarrow=False,
        font=dict(size=11, color="gray"),
        align="left"
    )
    fig_peers.update_layout(margin=dict(b=120)) # FIX: Increased bottom margin
    
    st.plotly_chart(fig_peers, use_container_width=True)
    
    # Margin Comparison
    st.subheader("Restaurant-Level Margins: Shake Shack vs Peers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # FIX APPLIED: Centered title and adjusted bottom margin/removed x-axis title to prevent overlap
        margin_data = {
            "company": [
                "Chipotle Mexican Grill",
                "Shake Shack",
                "BurgerFi",
                "Smashburger"
            ],
            "restaurant_margin_pct": [26.7, 21.4, 12.0, 9.0],
            "color": ["#8B0000", "#43A047", "#000000", "#D32F2F"],
            "year": ["2024", "2024", "2024", "2024"],
            "source": ["10-K Filing", "10-K Filing", "Quarterly Report", "Jollibee Disclosures"]
        }
        
        df_margins = pd.DataFrame(margin_data)
        df_margins_sorted = df_margins.sort_values(by="restaurant_margin_pct", ascending=False)
        
        fig_margins = go.Figure()
        
        for _, row in df_margins_sorted.iterrows():
            fig_margins.add_trace(go.Bar(
                x=[row["company"]],
                y=[row["restaurant_margin_pct"]],
                text=[f"{row['restaurant_margin_pct']}%"],
                textposition="outside",
                marker_color=row["color"],
                name=row["company"],
                hovertemplate=f"<b>{row['company']}</b><br>Margin: {row['restaurant_margin_pct']}%<br>Source: {row['source']} ({row['year']})<extra></extra>",
                showlegend=False
            ))
        
        # Calculate proper y-axis range to prevent clipping
        max_margin = df_margins_sorted["restaurant_margin_pct"].max()
        
        fig_margins.update_layout(
            title="Restaurant-Level Margins vs Peers",
            title_x=0.5, # FIX: Centered title
            yaxis_title="Restaurant-Level Margin (%)",
            xaxis_title="", # FIX: Removed x-axis title
            xaxis_tickangle=-30,
            plot_bgcolor="white",
            yaxis=dict(
                range=[0, max_margin * 1.18],  # Add 18% headroom for labels
                gridcolor="lightgray",
                gridwidth=0.5
            ),
            hovermode="x unified",
            margin=dict(t=80, b=100) # FIX: Increased bottom margin
        )
        
        st.plotly_chart(fig_margins, use_container_width=True)
    
    with col2:
        # FIX APPLIED: Centered title and adjusted bottom margin
        # Shake Shack Margin Trend
        margin_years = [2022, 2023, 2024]
        margins = [17.5, 19.9, 21.4]
        
        # Calculate YoY improvement
        margin_improvement = [0, margins[1] - margins[0], margins[2] - margins[1]]
        
        fig_margin_trend = go.Figure()
        
        fig_margin_trend.add_trace(go.Scatter(
            x=margin_years,
            y=margins,
            mode='lines+markers',
            line=dict(color='#2E8B57', width=3),
            marker=dict(size=14, symbol="circle"),
            hovertemplate="<b>%{x}</b><br>Margin: %{y}%<br>YoY Change: +%{customdata:.1f}pp<extra></extra>",
            customdata=margin_improvement,
            name="Restaurant Margin"
        ))
        
        # Add annotations for data labels (positioned above points)
        for i, (year, margin) in enumerate(zip(margin_years, margins)):
            fig_margin_trend.add_annotation(
                x=year,
                y=margin,
                text=f"<b>{margin}%</b>",
                showarrow=False,
                yshift=20,
                font=dict(size=12, color="#2E8B57")
            )
        
        # Calculate proper y-axis range
        max_margin = max(margins)
        min_margin = min(margins)
        
        fig_margin_trend.update_layout(
            title="Shake Shack Margin Expansion (2022–2024)",
            title_x=0.5, # FIX: Centered title
            xaxis_title="Year",
            yaxis_title="Restaurant-Level Profit Margin (%)",
            xaxis=dict(tickmode='array', tickvals=margin_years),
            plot_bgcolor='white',
            yaxis=dict(
                gridcolor='lightgray', 
                gridwidth=0.5,
                range=[min_margin - 2, max_margin + 3]  # Add headroom for labels
            ),
            hovermode="x unified",
            margin=dict(t=80, b=80) # FIX: Increased bottom margin
        )
        
        # Add cumulative improvement annotation
        total_improvement = margins[-1] - margins[0]
        fig_margin_trend.add_annotation(
            text=f"Total Improvement: +{total_improvement:.1f}pp",
            xref="paper", yref="paper",
            x=0.5, y=-0.18,
            showarrow=False,
            font=dict(size=11, color="#2E8B57", weight="bold"),
            align="center"
        )
        
        st.plotly_chart(fig_margin_trend, use_container_width=True)
    
    st.info("""
    **Key Insight:** Restaurant-level margin benchmarking shows Shake Shack gaining ground (21.4%), 
    but not quite up to Chipotle's level (26.7%). The margin expansion trend from 17.5% to 21.4% 
    over 2022-2024 highlights operational efficiencies.
    """)

# =============================================================================
# SECTION 6: INVESTMENT APPLICATIONS
# =============================================================================
elif section == "6. Investment Applications":
    st.header("6. Investment Applications")
    
    st.markdown("""
    From a **long/short investor's perspective**, the unit economics align with feasibility for long-term expansion.
    """)
    
    # Key valuation metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Market Cap", "$3.42B")
    with col2:
        st.metric("Company-Operated Stores", "359")
    with col3:
        st.metric("Implied Value per Unit", "~$9.5M")
    
    st.markdown("""
    With **strong margins despite input cost increases**, Shake Shack potentially is undervalued 
    from their 52-week high. This creates a favorable setup if Shake Shack can execute on the growth 
    plan management is targeting.
    
    **Key Success Indicators:**
    - Kiosk conversion rates
    - Real estate selection quality
    - Operational throughput improvements
    
    Coupled with base case growth expectations, these present **strong upside potential**.
    """)
    
    # Revenue Growth vs Store Count - FIX APPLIED: Replaced textposition with annotations and adjusted ranges/margins
    st.subheader("Revenue Growth vs. Store Count (2020-2024)")
    
    rev_data = pd.DataFrame({
        "year": [2020, 2021, 2022, 2023, 2024],
        "revenue": [523, 739, 900, 1098, 1310],
        "units": [183, 218, 263, 307, 329]
    })
    
    # Calculate YoY growth for hover info
    rev_data["rev_growth"] = rev_data["revenue"].pct_change() * 100
    rev_data["unit_growth"] = rev_data["units"].pct_change() * 100
    rev_data["rev_per_unit"] = (rev_data["revenue"] / rev_data["units"]).round(2)
    
    fig_rev = go.Figure()
    
    # Revenue bars with enhanced hover
    fig_rev.add_trace(
        go.Bar(
            x=rev_data["year"],
            y=rev_data["revenue"],
            name="Revenue ($M)",
            marker_color="#005030",
            # text=rev_data["revenue"].apply(lambda x: f"${x:,}M"), # REMOVED: Using annotation instead
            # textposition='outside',
            hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,}M<br>YoY Growth: %{customdata[0]:.1f}%<br>Rev/Unit: $%{customdata[1]:.2f}M<extra></extra>",
            customdata=rev_data[["rev_growth", "rev_per_unit"]].fillna(0).values
        )
    )

    # Add annotations for Revenue bars
    for i, row in rev_data.iterrows():
        fig_rev.add_annotation(
            x=row["year"],
            y=row["revenue"],
            yref="y1",
            text=f"<b>${row['revenue']:,}M</b>",
            showarrow=False,
            yshift=10, # Adjusted yshift to prevent overlap
            font=dict(size=11, color="#005030")
        )
    
    # Units line with offset labels to prevent overlap
    fig_rev.add_trace(
        go.Scatter(
            x=rev_data["year"],
            y=rev_data["units"],
            name="Company-Operated Units",
            yaxis="y2",
            mode="lines+markers",
            marker=dict(color="black", size=10),
            line=dict(width=3, color="black"),
            hovertemplate="<b>%{x}</b><br>Units: %{y}<br>YoY Growth: %{customdata:.1f}%<extra></extra>",
            customdata=rev_data["unit_growth"].fillna(0).values
        )
    )
    
    # Add annotations for units (positioned to avoid overlap with bars)
    for i, row in rev_data.iterrows():
        fig_rev.add_annotation(
            x=row["year"],
            y=row["units"],
            yref="y2",
            text=f"<b>{row['units']}</b>",
            showarrow=False,
            yshift=20, # Increased yshift for line labels
            font=dict(size=11, color="black")
        )
    
    # Calculate proper ranges
    max_rev = rev_data["revenue"].max()
    max_units = rev_data["units"].max()
    
    fig_rev.update_layout(
        title="Revenue Growth vs. Store Count",
        title_x=0.5,
        yaxis=dict(
            title="Revenue ($M)",
            range=[0, max_rev * 1.30],  # Increased headroom for bar labels
            gridcolor="lightgray",
            gridwidth=0.5
        ),
        yaxis2=dict(
            title="Units",
            overlaying="y",
            side="right",
            range=[0, max_units * 1.40],  # Increased headroom for line labels
            tickfont=dict(color="black"),
            showgrid=False
        ),
        margin=dict(l=70, r=110, t=80, b=80), # Increased bottom margin
        legend=dict(
            x=0.02, 
            y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1
        ),
        barmode='group',
        plot_bgcolor="white",
        hovermode="x unified"
    )
    
    # Add buttons for interactivity
    fig_rev.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": [True, True]}],
                        label="Both",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True, False]}],
                        label="Revenue Only",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True]}],
                        label="Units Only",
                        method="update"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.5,
                xanchor="center",
                y=-0.15,
                yanchor="top"
            )
        ]
    )
    
    st.plotly_chart(fig_rev, use_container_width=True)
    
    st.caption("**Evidence of scaling economics:** Revenue has grown from $523M to $1.31B (150%+ growth) while units grew from 183 to 329 (80% growth), demonstrating strong same-store sales growth alongside unit expansion.")

# =============================================================================
# SECTION 7: CONCLUSION
# =============================================================================
elif section == "7. Conclusion":
    st.header("7. Conclusion")
    
    st.markdown("""
    The analysis supports the view that **Shake Shack has a credible path to achieving the 1,500 unit target**, 
    and management's excitement is grounded in demographic and competitive fundamentals.
    
    **Strongest Prospective Markets:**
    - Favorable population dynamics
    - Premium income profiles  
    - Low brand penetration
    
    **Recommended Rollout Strategy:**
    - **Primary focus:** Texas, Florida, and California
    - **Complementary expansion:** Sun Belt, select Midwestern states, and Pacific Northwest
    
    **Investment Perspective:**
    - Current per-unit valuation is attractive
    - Meaningful whitespace remains
    - Restaurant-level margins are expanding
    - This provides a **compelling long-duration growth story**
    
    **Key to Success:**
    - Executing a **kiosk-heavy, suburban-focused strategy** is critical to unlocking Shake Shack's full IRR potential
    """)
    
    st.divider()
    
    st.subheader("What I Would Do in the Real Environment")
    
    st.markdown("""
    In the interest of brevity, here are additional measures I would take in a professional environment:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Direct Engagement with Shake Shack Management**
        - How management scores real estate sites
        - Expected mix between drive-thru, in-line, and urban formats
        - Regional differences in construction timelines and COGS inflation
        
        **2. Direct Engagement with Team Analysts**
        - Sanity check estimates and assumptions
        - Dig deeper into valuation model gaps
        - Leverage experience for future analysis
        """)
    
    with col2:
        st.markdown("""
        **3. Leverage Sell-Side and Other Sources**
        - Sentiment about expansion from sell-side
        - Any disconnect between management and execution
        - Information from former Shake Shack regional managers
        
        **4. Validate Demand Score with Local Economics**
        - CBRE retail vacancy rates
        - Foot traffic patterns
        - Household discretionary spend trends
        """)

# =============================================================================
# SECTION 8: APPENDIX - SUPPLEMENTAL CHARTS
# =============================================================================
elif section == "8. Appendix: Supplemental Charts":
    st.header("8. Appendix: Supplemental Charts and Figures")
    
    # Chart 1: State-Level Penetration vs Population Density
    st.subheader("1. State-Level Penetration vs Population Density")
    
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
    
    # Chart 2: SHAK Penetration vs QSR Competition
    st.subheader("2. SHAK Penetration vs QSR Competition")
    
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
    st.subheader("3. SHAK Penetration vs QSR Competition (with OLS Trendline)")
    
    x_data = df["qsr_per_100k"].values
    y_data = df["shacks_per_million"].values
    
    slope, intercept = np.polyfit(x_data, y_data, 1)
    trendline_y = slope * x_data + intercept
    
    ss_res = np.sum((y_data - trendline_y) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    fig_qsr_trend = go.Figure()
    
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
    
    st.info(f"""
    **OLS Regression Results:**
    - R² = {r_squared:.3f}
    - Slope = {slope:.4f}
    - Equation: Shacks per 1M = {slope:.4f} × (QSR per 100k) + {intercept:.4f}
    """)
    
    st.divider()
    
    # Chart 4: Demand Score Components Table
    st.subheader("4. Demand Score Components (Full Data)")
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

# Footer
st.divider()
st.caption("Data sources: Shake Shack 10-K filings, US Census Bureau (ACS 2023, CBP 2022)")