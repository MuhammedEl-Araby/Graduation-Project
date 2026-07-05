import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from scipy.stats import norm
from PIL import Image, ImageDraw, ImageFont
import base64
import os
import time


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SCADA Dynamic Pricing & Smart Meter Simulator",
    page_icon="⚡",
    layout="wide"
)


# =========================================================
# DISABLE STREAMLIT CLEAR-CACHE SHORTCUT BEHAVIOR
# =========================================================

components.html(
    """
    <script>
    (function () {
        const isEditable = (el) => {
            if (!el) return false;
            const tag = (el.tagName || "").toLowerCase();
            return tag === "input" || tag === "textarea" || tag === "select" || el.isContentEditable;
        };

        window.parent.document.addEventListener("keydown", function(e) {
            const key = (e.key || "").toLowerCase();
            const target = e.target;

            if ((e.ctrlKey || e.metaKey) && key === "c") {
                e.stopImmediatePropagation();
                return true;
            }

            if (!e.ctrlKey && !e.metaKey && !e.altKey && key === "c" && !isEditable(target)) {
                e.stopImmediatePropagation();
                return true;
            }
        }, true);
    })();
    </script>
    """,
    height=0
)


# =========================================================
# SAFE IMAGE LOADER
# =========================================================

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None


bg_img = get_base64_image("gettyimages-1395219224.jpg")


# =========================================================
# STYLE
# =========================================================

if bg_img:
    background_css = f"""
    background-image:
    linear-gradient(rgba(0,0,0,0.62), rgba(0,0,0,0.62)),
    url("data:image/jpg;base64,{bg_img}");
    """
else:
    background_css = """
    background: linear-gradient(135deg, #050505, #111827, #1e293b);
    """

page_bg = f"""
<style>

[data-testid="stAppViewContainer"] {{
{background_css}
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stSidebar"] {{
background: rgba(0,0,0,0.45);
}}

h1, h2, h3, h4, h5, h6, p, label, div, span {{
color: white;
}}

.stMetric {{
background: rgba(255,255,255,0.08);
border-radius: 16px;
padding: 12px;
border: 1px solid rgba(255,255,255,0.15);
}}

[data-testid="stDataFrame"] {{
background: rgba(255,255,255,0.05);
border-radius: 14px;
}}

@keyframes pulse {{
0% {{ transform: scale(1); }}
50% {{ transform: scale(1.01); }}
100% {{ transform: scale(1); }}
}}

.stAlert {{
animation: pulse 2.5s infinite;
}}

.scada-card {{
background: rgba(0,0,0,0.42);
border: 1px solid rgba(255,255,255,0.18);
border-radius: 18px;
padding: 18px;
margin-bottom: 12px;
}}

.big-status {{
font-size: 28px;
font-weight: 800;
}}

.manual-box {{
background: rgba(255,255,255,0.08);
border: 1px solid rgba(255,255,255,0.16);
border-radius: 16px;
padding: 18px;
margin-bottom: 14px;
}}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)


# =========================================================
# CONSTANTS
# =========================================================

BASE_RATE = 0.25
PEAK_RATE = 0.80
PENALTY_RATE = 1.20
PREMIUM_PRESERVATION_RATE = 1.60
DISCOUNT_RATE = 0.15
BONUS_RATE = 0.10
LOYALTY_DISCOUNT_RATE = 0.08
GOOD_BEHAVIOR_STEP_DISCOUNT = 0.02
GOOD_BEHAVIOR_MAX_DISCOUNT = 0.12
CHARITY_MEDICAL_DEVICE_DISCOUNT_RATE = 0.25
FAIRNESS_INACTIVE_ADJUSTMENT_RATE = 0.01

# Saudi monthly consumption tariff rates in SAR/kWh
SAUDI_TIER_LIMIT_KWH = 6000
RESIDENTIAL_TIER1_RATE = 0.18
RESIDENTIAL_TIER2_RATE = 0.30
COMMERCIAL_TIER1_RATE = 0.22
COMMERCIAL_TIER2_RATE = 0.32
CHARITY_TIER1_RATE = 0.16
CHARITY_TIER2_RATE = 0.20
SAUDI_VAT_RATE = 0.15
DEFAULT_RESIDENTIAL_METER_FEE = 10.0
DEFAULT_COMMERCIAL_METER_FEE = 15.0
SAUDI_APARTMENT_LOW_KWH = 800
SAUDI_APARTMENT_HIGH_KWH = 1500
SAUDI_VILLA_LOW_KWH = 2000
SAUDI_VILLA_HIGH_KWH = 4000
SAUDI_AVERAGE_HOME_LOW_KWH = 2000
SAUDI_AVERAGE_HOME_HIGH_KWH = 2500
COMPANY_GROWTH_DISCOUNT_RATE = 0.10

PRIORITY_MIN = 1
PRIORITY_MAX = 10

LOW_BASELINE_THRESHOLD_KWH = 2.0
LOW_BASELINE_MAX_REDUCTION_PERCENT = 10

LOAD_CATEGORY_OPTIONS = [
    "Critical / Never Disconnect",
    "Shed First",
    "Luxury Load",
    "Normal Load",
    "Comfort Load"
]


# =========================================================
# DEFAULT APPLIANCE CONFIG
# =========================================================

def default_appliance_config():
    return pd.DataFrame([
        {
            "Appliance": "Lights",
            "Quantity": 10,
            "Power per Unit kW": 0.02,
            "Connected": True,
            "Load Category": "Critical / Never Disconnect",
            "Allow Company Emergency Control": False,
            "Preserve Minimum Units": 10,
            "Disconnectable": False,
            "Critical": True,
            "User Priority": 10,
            "Company Priority": 10
        },
        {
            "Appliance": "Power Sockets",
            "Quantity": 8,
            "Power per Unit kW": 0.15,
            "Connected": True,
            "Load Category": "Shed First",
            "Allow Company Emergency Control": True,
            "Preserve Minimum Units": 0,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 1,
            "Company Priority": 2
        },
        {
            "Appliance": "Water Heater",
            "Quantity": 1,
            "Power per Unit kW": 2.0,
            "Connected": True,
            "Load Category": "Comfort Load",
            "Allow Company Emergency Control": True,
            "Preserve Minimum Units": 0,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 5,
            "Company Priority": 4
        },
        {
            "Appliance": "Hand Dryer",
            "Quantity": 1,
            "Power per Unit kW": 1.8,
            "Connected": True,
            "Load Category": "Luxury Load",
            "Allow Company Emergency Control": True,
            "Preserve Minimum Units": 0,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 2,
            "Company Priority": 3
        },
        {
            "Appliance": "Washing Machine",
            "Quantity": 1,
            "Power per Unit kW": 1.0,
            "Connected": True,
            "Load Category": "Luxury Load",
            "Allow Company Emergency Control": True,
            "Preserve Minimum Units": 0,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 2,
            "Company Priority": 1
        },
        {
            "Appliance": "ACs",
            "Quantity": 6,
            "Power per Unit kW": 1.3,
            "Connected": True,
            "Load Category": "Comfort Load",
            "Allow Company Emergency Control": True,
            "Preserve Minimum Units": 0,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 5,
            "Company Priority": 3
        },
        {
            "Appliance": "Heavy Machines",
            "Quantity": 2,
            "Power per Unit kW": 1.6,
            "Connected": True,
            "Load Category": "Shed First",
            "Allow Company Emergency Control": True,
            "Preserve Minimum Units": 0,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 1,
            "Company Priority": 1
        }
    ])


# =========================================================
# SESSION STATE DEFAULTS
# =========================================================

if "appliance_config" not in st.session_state:
    st.session_state.appliance_config = default_appliance_config()

if "selected_user_policy" not in st.session_state:
    st.session_state.selected_user_policy = "Manual User Priority"

if "refuse_disconnect" not in st.session_state:
    st.session_state.refuse_disconnect = False

if "climate_mode" not in st.session_state:
    st.session_state.climate_mode = "Hot Summer - Cooling Priority"

if "mandatory_reduction_percent" not in st.session_state:
    st.session_state.mandatory_reduction_percent = 20

if "voluntary_reduction_percent" not in st.session_state:
    st.session_state.voluntary_reduction_percent = 35

if "ac_unit_states" not in st.session_state:
    st.session_state.ac_unit_states = {
        "AC 1": True,
        "AC 2": True,
        "AC 3": True,
        "AC 4": True,
        "AC 5": True,
        "AC 6": True
    }

if "deadline_started_at" not in st.session_state:
    st.session_state.deadline_started_at = None

if "deadline_end_at" not in st.session_state:
    st.session_state.deadline_end_at = None

if "deadline_penalty_latched" not in st.session_state:
    st.session_state.deadline_penalty_latched = False

if "deadline_penalty_reason" not in st.session_state:
    st.session_state.deadline_penalty_reason = ""

if "deadline_signature" not in st.session_state:
    st.session_state.deadline_signature = None

if "good_behavior_streak" not in st.session_state:
    st.session_state.good_behavior_streak = 0

if "last_behavior_event_id" not in st.session_state:
    st.session_state.last_behavior_event_id = None

if "previous_requested_usage_by_person" not in st.session_state:
    st.session_state.previous_requested_usage_by_person = {}

if "last_good_behavior_counted_event" not in st.session_state:
    st.session_state.last_good_behavior_counted_event = None

if "company_force_fair_settlement" not in st.session_state:
    st.session_state.company_force_fair_settlement = False
if "person_a_medical_device" not in st.session_state:
    st.session_state.person_a_medical_device = False
if "person_b_medical_device" not in st.session_state:
    st.session_state.person_b_medical_device = False
if "ignored_request_count_by_person" not in st.session_state:
    st.session_state.ignored_request_count_by_person = {}
if "last_ignored_request_event_id" not in st.session_state:
    st.session_state.last_ignored_request_event_id = None
if "override_house_size" not in st.session_state:
    st.session_state.override_house_size = 120
if "override_occupants" not in st.session_state:
    st.session_state.override_occupants = 3

# Stores the latest non-forced scenario so Last Resort can show a simple
# BEFORE vs AFTER comparison at the end of the page.
if "last_resort_before_snapshot" not in st.session_state:
    st.session_state.last_resort_before_snapshot = None

if "last_resort_before_snapshot_id" not in st.session_state:
    st.session_state.last_resort_before_snapshot_id = None


# =========================================================
# AC OVERLAY POSITIONS - FIXED VERSION
# =========================================================

AC_POSITION_VERSION = "v7_fixed_user_pattern_ac6_ac5_ac2_ac3_ac1_ac4_847x658"

DEFAULT_AC_OVERLAY_POSITIONS = {
    # Required apartment pattern:
    #
    # AC6                  AC5                    AC2
    #
    # AC3                                          AC1
    #
    # AC4
    #
    # Coordinates are near red AC symbols and shifted away from AutoCAD text.
    "AC 6": {"x": 132, "y": 112, "label_x": 132, "label_y": 58},
    "AC 5": {"x": 584, "y": 125, "label_x": 584, "label_y": 72},
    "AC 2": {"x": 805, "y": 174, "label_x": 790, "label_y": 118},
    "AC 3": {"x": 48,  "y": 408, "label_x": 62,  "label_y": 348},
    "AC 1": {"x": 805, "y": 444, "label_x": 790, "label_y": 506},
    "AC 4": {"x": 48,  "y": 546, "label_x": 62,  "label_y": 500},
}

if (
    "ac_position_version" not in st.session_state
    or st.session_state.ac_position_version != AC_POSITION_VERSION
):
    st.session_state.ac_overlay_positions = DEFAULT_AC_OVERLAY_POSITIONS.copy()
    st.session_state.ac_position_version = AC_POSITION_VERSION

if "ac_overlay_positions" not in st.session_state:
    st.session_state.ac_overlay_positions = DEFAULT_AC_OVERLAY_POSITIONS.copy()


# =========================================================
# DATAFRAME SAFETY
# =========================================================

def ensure_appliance_columns(df):
    df = df.copy()

    # Migration from older code: keep Critical internally but add a clear protected flag.
    if "Protected / Never Disconnect" not in df.columns:
        if "Critical" in df.columns:
            df["Protected / Never Disconnect"] = df["Critical"].astype(bool)
        else:
            df["Protected / Never Disconnect"] = False

    defaults = {
        "Appliance": "Unknown",
        "Quantity": 0,
        "Power per Unit kW": 0.0,
        "Connected": True,
        "Load Category": "Normal Load",
        "Allow Company Emergency Control": True,
        "Preserve Minimum Units": 0,
        "Protected / Never Disconnect": False,
        "Disconnectable": True,
        "Critical": False,
        "User Priority": 4,
        "Company Priority": 4
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    valid_categories = set(LOAD_CATEGORY_OPTIONS)
    df["Load Category"] = df["Load Category"].apply(
        lambda x: x if x in valid_categories else "Normal Load"
    )

    for col in ["Quantity", "Preserve Minimum Units"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    df["Power per Unit kW"] = pd.to_numeric(
        df["Power per Unit kW"], errors="coerce"
    ).fillna(0).clip(lower=0)

    # Priority is now only 1 to 10. Protected flag is used instead of hidden priority numbers.
    for col in ["User Priority", "Company Priority"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(4).clip(PRIORITY_MIN, PRIORITY_MAX).astype(int)

    for col in ["Connected", "Allow Company Emergency Control", "Protected / Never Disconnect", "Disconnectable", "Critical"]:
        df[col] = df[col].astype(bool)

    # Protected loads are protected by flags, not by ugly priority numbers.
    protected_mask = df["Load Category"].eq("Critical / Never Disconnect") | df["Protected / Never Disconnect"]
    df.loc[protected_mask, "Protected / Never Disconnect"] = True
    df.loc[protected_mask, "Critical"] = True
    df.loc[protected_mask, "Disconnectable"] = False
    df.loc[protected_mask, "Preserve Minimum Units"] = df.loc[protected_mask, ["Preserve Minimum Units", "Quantity"]].max(axis=1)

    return df[list(defaults.keys())]


st.session_state.appliance_config = ensure_appliance_columns(
    st.session_state.appliance_config
)


# =========================================================
# SYNTHETIC DATA GENERATION
# =========================================================

def generate_training_data(n=5000):
    np.random.seed(42)

    lamps = np.random.randint(1, 300, n)
    acs = np.random.randint(0, 80, n)
    washing = np.random.randint(0, 60, n)
    heavy_machines = np.random.randint(0, 80, n)
    occupants = np.random.randint(1, 150, n)
    house_size = np.random.randint(20, 6000, n)

    gaussian_randomness = np.random.normal(0, 0.75, n)

    baseline = (
        0.10 * lamps +
        1.15 * acs +
        0.90 * washing +
        1.55 * heavy_machines +
        0.32 * occupants +
        0.010 * house_size +
        gaussian_randomness
    )

    baseline = np.clip(baseline, 0.5, None)

    df = pd.DataFrame({
        "lamps": lamps,
        "acs": acs,
        "washing_machine": washing,
        "heavy_machines": heavy_machines,
        "occupants": occupants,
        "house_size": house_size,
        "historical_baseline_kwh": baseline
    })

    return df


# =========================================================
# MODEL TRAINING
# =========================================================

def train_model():
    df = generate_training_data()

    X = df.drop(columns=["historical_baseline_kwh"])
    y = df["historical_baseline_kwh"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }

    return model, metrics, df


if "trained_model_bundle" not in st.session_state:
    st.session_state.trained_model_bundle = train_model()

model, metrics, training_df = st.session_state.trained_model_bundle

mean_usage = training_df["historical_baseline_kwh"].mean()
std_usage = training_df["historical_baseline_kwh"].std()


# =========================================================
# SCADA GRID PREDICTIVE / PROTECTIVE MAINTENANCE ENGINE
# =========================================================
def generate_grid_maintenance_training_data(n=6000):
    np.random.seed(77)
    age = np.random.uniform(0.5, 38, n)
    live_load_kw = np.random.uniform(0.1, 250, n)
    loading_percent = np.random.uniform(20, 145, n)
    peak_stress_percent = np.random.uniform(0, 100, n)
    voltage_deviation_percent = np.random.uniform(0, 20, n)
    temperature_c = np.random.uniform(22, 118, n)
    vibration_mm_s = np.random.uniform(0.1, 15, n)
    insulation_health_percent = np.random.uniform(5, 100, n)
    breaker_operations = np.random.randint(0, 9000, n)
    fault_events_30d = np.random.poisson(2.0, n)
    humidity_percent = np.random.uniform(15, 100, n)
    maintenance_quality = np.random.uniform(0.55, 1.25, n)
    stress_index = (
        0.025 * loading_percent
        + 0.020 * peak_stress_percent
        + 0.120 * voltage_deviation_percent
        + 0.030 * np.maximum(temperature_c - 55, 0)
        + 0.320 * vibration_mm_s
        + 0.003 * breaker_operations
        + 0.800 * fault_events_30d
        + 0.010 * humidity_percent
        + 0.012 * live_load_kw
        - 0.030 * insulation_health_percent
    ) / maintenance_quality
    condition_score = np.clip(
        100 - 1.35 * age - 2.65 * stress_index - 0.18 * (100 - insulation_health_percent)
        + np.random.normal(0, 2.5, n),
        0,
        100,
    )
    remaining_life_months = np.clip(
        420 - 9.2 * age - 5.7 * stress_index + 0.55 * insulation_health_percent
        + np.random.normal(0, 16, n),
        1,
        480,
    )
    failure_risk_12m = (
        (remaining_life_months <= 12)
        | (condition_score <= 35)
        | (stress_index >= 34)
        | ((fault_events_30d >= 7) & (voltage_deviation_percent >= 8))
    ).astype(int)
    return pd.DataFrame({
        "age_years": age,
        "live_load_kw": live_load_kw,
        "loading_percent": loading_percent,
        "peak_stress_percent": peak_stress_percent,
        "voltage_deviation_percent": voltage_deviation_percent,
        "temperature_c": temperature_c,
        "vibration_mm_s": vibration_mm_s,
        "insulation_health_percent": insulation_health_percent,
        "breaker_operations": breaker_operations,
        "fault_events_30d": fault_events_30d,
        "humidity_percent": humidity_percent,
        "maintenance_quality": maintenance_quality,
        "stress_index": stress_index,
        "condition_score": condition_score,
        "remaining_life_months": remaining_life_months,
        "failure_risk_12m": failure_risk_12m,
    })


def train_grid_maintenance_models():
    df = generate_grid_maintenance_training_data()
    features = [
        "age_years", "live_load_kw", "loading_percent", "peak_stress_percent",
        "voltage_deviation_percent", "temperature_c", "vibration_mm_s",
        "insulation_health_percent", "breaker_operations", "fault_events_30d",
        "humidity_percent", "maintenance_quality"
    ]
    X = df[features]
    y_rul = df["remaining_life_months"]
    y_risk = df["failure_risk_12m"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_rul, test_size=0.2, random_state=42)
    rul_model = RandomForestRegressor(n_estimators=240, max_depth=15, min_samples_leaf=2, random_state=42)
    rul_model.fit(X_train, y_train)
    rul_pred = rul_model.predict(X_test)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_risk, test_size=0.2, random_state=42, stratify=y_risk)
    risk_model = RandomForestClassifier(n_estimators=240, max_depth=14, min_samples_leaf=2, random_state=42)
    risk_model.fit(Xc_train, yc_train)
    risk_pred = risk_model.predict(Xc_test)
    return rul_model, risk_model, {
        "RUL MAE Months": mean_absolute_error(y_test, rul_pred),
        "RUL R2": r2_score(y_test, rul_pred),
        "Risk Accuracy": accuracy_score(yc_test, risk_pred),
    }, df, features


if "grid_maintenance_bundle" not in st.session_state:
    st.session_state.grid_maintenance_bundle = train_grid_maintenance_models()

maintenance_rul_model, maintenance_risk_model, maintenance_metrics, maintenance_training_df, maintenance_features = st.session_state.grid_maintenance_bundle


def calculate_grid_asset_condition(age_years, live_load_kw, loading_percent, peak_stress_percent, voltage_deviation_percent, temperature_c, vibration_mm_s, insulation_health_percent, breaker_operations, fault_events_30d, humidity_percent, maintenance_quality):
    stress_index = (
        0.025 * loading_percent + 0.020 * peak_stress_percent + 0.120 * voltage_deviation_percent
        + 0.030 * max(temperature_c - 55, 0) + 0.320 * vibration_mm_s
        + 0.003 * breaker_operations + 0.800 * fault_events_30d + 0.010 * humidity_percent
        + 0.012 * live_load_kw - 0.030 * insulation_health_percent
    ) / max(maintenance_quality, 0.1)
    condition_score = np.clip(100 - 1.35 * age_years - 2.65 * stress_index - 0.18 * (100 - insulation_health_percent), 0, 100)
    return float(condition_score), float(stress_index)


def classify_grid_maintenance_state(condition_score, risk_probability, remaining_life_months):
    if condition_score <= 25 or risk_probability >= 0.75 or remaining_life_months <= 6:
        return "Functional Failure Zone", "Critical", "Emergency protective maintenance or replacement is required now."
    if condition_score <= 45 or risk_probability >= 0.50 or remaining_life_months <= 12:
        return "Potential Failure P", "Danger", "Schedule predictive maintenance before the P-F interval closes."
    if condition_score <= 65 or risk_probability >= 0.25 or remaining_life_months <= 30:
        return "Degradation Started", "Warning", "Plan protective maintenance and increase SCADA monitoring frequency."
    return "Normal State", "Healthy", "Continue SCADA monitoring and normal inspection cycle."


def simulate_grid_maintenance_curves(condition_score, stress_index, maintenance_day, restoration_strength, degradation_speed, horizon_days):
    days = np.arange(0, horizon_days + 1)
    rate = max(0.018, (100 - condition_score + stress_index) / 2600) * degradation_speed
    no_maintenance = np.clip(condition_score - rate * (days ** 1.32), 0, 100)
    normal_life = np.clip(condition_score - (rate * 0.46) * days, 0, 100)
    predictive = no_maintenance.copy()
    for i, day in enumerate(days):
        if day >= maintenance_day:
            restored = min(100, no_maintenance[i] + restoration_strength)
            predictive[i] = np.clip(restored - (rate * 0.33) * (day - maintenance_day), 0, 100)
    time_based = np.zeros_like(days, dtype=float)
    interval = max(int(maintenance_day / 2), 20)
    last_restore_day = 0
    restored_level = condition_score
    for i, day in enumerate(days):
        if day > 0 and day % interval == 0:
            restored_level = min(100, time_based[i - 1] + restoration_strength * 0.55)
            last_restore_day = day
        time_based[i] = np.clip(restored_level - (rate * 0.52) * (day - last_restore_day), 0, 100)
    breakdown = no_maintenance.copy()
    fail_days = np.where(breakdown <= 45)[0]
    breakdown_day = int(fail_days[0]) if len(fail_days) > 0 else None
    if breakdown_day is not None:
        for i, day in enumerate(days):
            if day >= breakdown_day:
                breakdown[i] = np.clip(35 - 0.18 * (day - breakdown_day), 0, 100)
    curve_df = pd.DataFrame({
        "Day": days,
        "No Maintenance": no_maintenance,
        "Normal Life Expectancy": normal_life,
        "Predictive Maintenance": predictive,
        "Time-Based Maintenance": time_based,
        "Breakdown Maintenance": breakdown,
        "Failure Threshold": np.full_like(days, 45, dtype=float),
        "Potential Failure Threshold": np.full_like(days, 60, dtype=float),
    })
    p_days = curve_df[curve_df["No Maintenance"] <= 60]["Day"]
    f_days = curve_df[curve_df["No Maintenance"] <= 45]["Day"]
    d_days = curve_df[curve_df["No Maintenance"] <= 80]["Day"]
    points = {
        "td": int(d_days.iloc[0]) if len(d_days) > 0 else None,
        "p": int(p_days.iloc[0]) if len(p_days) > 0 else None,
        "f": int(f_days.iloc[0]) if len(f_days) > 0 else None,
        "breakdown": breakdown_day,
    }
    return curve_df, points


def draw_grid_pf_curve(curve_df, points):
    fig = go.Figure()
    td, tp, tf = points["td"], points["p"], points["f"]
    normal = curve_df.copy(); warn = curve_df.copy(); danger = curve_df.copy()
    normal.loc[normal["Day"] > (td if td is not None else 0), "No Maintenance"] = np.nan
    warn.loc[(warn["Day"] < (td if td is not None else 0)) | (warn["Day"] > (tp if tp is not None else curve_df["Day"].max())), "No Maintenance"] = np.nan
    danger.loc[danger["Day"] < (tp if tp is not None else curve_df["Day"].max()), "No Maintenance"] = np.nan
    fig.add_trace(go.Scatter(x=normal["Day"], y=normal["No Maintenance"], mode="lines", name="Normal State", line=dict(color="lime", width=6)))
    fig.add_trace(go.Scatter(x=warn["Day"], y=warn["No Maintenance"], mode="lines", name="Degradation / Warning State", line=dict(color="gold", width=6)))
    fig.add_trace(go.Scatter(x=danger["Day"], y=danger["No Maintenance"], mode="lines", name="Dangerous State", line=dict(color="red", width=6)))
    for label, day, color in [("td", td, "lime"), ("P", tp, "gold"), ("F", tf, "red")]:
        if day is not None:
            yv = float(curve_df.loc[curve_df["Day"] == day, "No Maintenance"].iloc[0])
            fig.add_trace(go.Scatter(x=[day], y=[yv], mode="markers+text", name=label, marker=dict(size=22, color=color, line=dict(width=3, color="black")), text=[label], textposition="top center"))
            fig.add_vline(x=day, line_dash="dot", line_color=color)
    potential_threshold_value = float(curve_df["Potential Failure Threshold"].iloc[0]) if "Potential Failure Threshold" in curve_df.columns else 60.0
    failure_threshold_value = float(curve_df["Failure Threshold"].iloc[0]) if "Failure Threshold" in curve_df.columns else 45.0
    fig.add_hline(y=potential_threshold_value, line_dash="dash", line_color="gold", annotation_text="Potential Failure Threshold")
    fig.add_hline(y=failure_threshold_value, line_dash="dash", line_color="red", annotation_text="Functional Failure Threshold")
    fig.update_layout(title="SCADA P-F Curve: Normal State → Dangerous State → Functional Failure", xaxis_title="Time / Operating Days", yaxis_title="Grid Asset Condition / Performance", template="plotly_dark", height=620, yaxis=dict(range=[0,105]))
    return fig



# =========================================================
# MAINTENANCE CASE STUDY HELPERS
# =========================================================
def get_grid_asset_profile(asset_type):
    profiles = {
        "Distribution Transformer": {
            "stress_multiplier": 1.15,
            "condition_offset": -3.0,
            "rul_multiplier": 0.92,
            "failure_threshold": 45,
            "potential_threshold": 62,
            "default_temperature": 65.0,
            "default_vibration": 2.4,
            "default_insulation": 78.0,
        },
        "Feeder Cable": {
            "stress_multiplier": 0.95,
            "condition_offset": 2.0,
            "rul_multiplier": 1.08,
            "failure_threshold": 42,
            "potential_threshold": 58,
            "default_temperature": 48.0,
            "default_vibration": 0.6,
            "default_insulation": 82.0,
        },
        "Circuit Breaker": {
            "stress_multiplier": 1.35,
            "condition_offset": -5.0,
            "rul_multiplier": 0.85,
            "failure_threshold": 48,
            "potential_threshold": 65,
            "default_temperature": 55.0,
            "default_vibration": 3.8,
            "default_insulation": 74.0,
        },
        "Switchgear": {
            "stress_multiplier": 1.05,
            "condition_offset": -1.0,
            "rul_multiplier": 0.98,
            "failure_threshold": 45,
            "potential_threshold": 60,
            "default_temperature": 58.0,
            "default_vibration": 2.2,
            "default_insulation": 80.0,
        },
        "Main Distribution Panel": {
            "stress_multiplier": 1.00,
            "condition_offset": 0.0,
            "rul_multiplier": 1.00,
            "failure_threshold": 45,
            "potential_threshold": 60,
            "default_temperature": 50.0,
            "default_vibration": 1.4,
            "default_insulation": 84.0,
        },
    }
    return profiles.get(asset_type, profiles["Main Distribution Panel"])


def set_perfect_maintenance_defaults():
    perfect_defaults = {
        "mx_age_years": 1.0,
        "mx_loading_percent": 35.0,
        "mx_peak_stress_percent": 5.0,
        "mx_voltage_deviation_percent": 0.5,
        "mx_temperature_c": 32.0,
        "mx_vibration_mm_s": 0.2,
        "mx_insulation_health_percent": 100.0,
        "mx_breaker_operations": 0,
        "mx_fault_events_30d": 0,
        "mx_humidity_percent": 35.0,
        "mx_maintenance_quality": 1.30,
        "mx_maintenance_day": 365,
        "mx_restoration_strength": 40.0,
        "mx_degradation_speed": 0.35,
        "mx_horizon_days": 730,
    }
    for key, value in perfect_defaults.items():
        st.session_state[key] = value


def build_dynamic_maintenance_feature_impact(feature_row, rul_model, risk_model, feature_columns):
    base_rul = float(rul_model.predict(feature_row[feature_columns])[0])
    base_risk = float(risk_model.predict_proba(feature_row[feature_columns])[0][1])
    rows = []
    for feature in feature_columns:
        perturbed = feature_row.copy()
        current_value = float(perturbed[feature].iloc[0])
        step = max(abs(current_value) * 0.10, 0.5)
        if feature in ["fault_events_30d"]:
            step = max(1, round(step))
        if feature in ["breaker_operations"]:
            step = max(50, round(step))
        perturbed[feature] = current_value + step
        changed_rul = float(rul_model.predict(perturbed[feature_columns])[0])
        changed_risk = float(risk_model.predict_proba(perturbed[feature_columns])[0][1])
        rows.append({
            "SCADA Feature": feature,
            "Current Value": current_value,
            "Sensitivity Step": step,
            "RUL Change Months": changed_rul - base_rul,
            "Failure Risk Change %": (changed_risk - base_risk) * 100,
            "Dynamic Impact Score": abs(changed_rul - base_rul) + abs((changed_risk - base_risk) * 100),
        })
    return pd.DataFrame(rows).sort_values("Dynamic Impact Score", ascending=False)


def style_status_cells(status_df):
    def color_status(value):
        value_text = str(value).lower()
        negative_words = ["not satisfied", "above own baseline", "penalty active", "access denied", "no discount"]
        positive_words = ["satisfied", "within", "discount applied", "no forced", "no penalty", "active"]
        if any(word in value_text for word in negative_words):
            return "background-color:#7f1d1d;color:white;font-weight:bold;"
        if any(word in value_text for word in positive_words):
            return "background-color:#14532d;color:white;font-weight:bold;"
        return ""
    styler = status_df.style
    if hasattr(styler, "map"):
        return styler.map(color_status, subset=["Status"])
    return styler.applymap(color_status, subset=["Status"])

# =========================================================
# HYBRID BASELINE CALCULATION
# =========================================================

def calculate_engineering_baseline(input_df):
    return estimate_saudi_monthly_baseline_from_household(input_df)

def predict_historical_baseline(model, input_df):
    # Saudi monthly baseline is estimated with property size, occupants and device mix.
    # The previous synthetic Random Forest remains for model details, but billing baseline
    # uses the Saudi-oriented engineering estimator to avoid unrealistic low kWh values.
    return max(calculate_engineering_baseline(input_df), 0.5)


# =========================================================
# COMPANY STATISTICAL BASELINE / FRAUD-GUARD ENGINE
# =========================================================
def company_statistical_baseline(household_df, historical_baseline):
    row = household_df.iloc[0].astype(float)
    occupants = max(row["occupants"], 1)
    size = max(row["house_size"], 1)
    area_per_person = size / occupants

    statistical_baseline = calculate_engineering_baseline(household_df)

    if size <= 180:
        expected_band = f"Apartment expected range {SAUDI_APARTMENT_LOW_KWH}-{SAUDI_APARTMENT_HIGH_KWH} kWh/month"
        expected_high = SAUDI_APARTMENT_HIGH_KWH + max(row["acs"] - 2, 0) * 450
    else:
        expected_band = f"Villa expected range {SAUDI_VILLA_LOW_KWH}-{SAUDI_VILLA_HIGH_KWH} kWh/month"
        expected_high = SAUDI_VILLA_HIGH_KWH + max(row["acs"] - 4, 0) * 550 + max(size - 350, 0) * 5

    company_approved_baseline = min(historical_baseline, statistical_baseline)
    inflation_gap = max(historical_baseline - statistical_baseline, 0)
    abnormal_usage_gap = max(statistical_baseline - expected_high, 0)
    fraud_gap = max(inflation_gap, abnormal_usage_gap)
    fraud_risk_percent = min((fraud_gap / max(statistical_baseline, 0.1)) * 100, 100)

    if fraud_risk_percent >= 35:
        fraud_status = "High baseline inflation / abnormal usage risk"
    elif fraud_risk_percent >= 15:
        fraud_status = "Medium baseline review required"
    else:
        fraud_status = "Normal baseline"

    return {
        "Historical Baseline kWh": historical_baseline,
        "Company Statistical Baseline kWh": statistical_baseline,
        "Company Approved Baseline kWh": company_approved_baseline,
        "Expected Saudi Property Band": expected_band,
        "Fraud Gap kWh": fraud_gap,
        "Fraud Risk %": fraud_risk_percent,
        "Fraud Status": fraud_status,
        "Area per Occupant m²": area_per_person,
    }


# =========================================================
# HOUSEHOLD INPUT
# =========================================================

def describe_household_pattern(lamps, acs, washing, heavy, occupants, size):
    messages = []

    if lamps >= 20 and size <= 80:
        messages.append(
            "Lighting count is high for this area, but this can be normal depending on lamp wattage, luminance level, decorative lighting, and room distribution. The full lamp count is calculated."
        )

    if acs >= 4 and size <= 100:
        messages.append(
            "AC count is high for this area, but this may represent multi-zone cooling, poor insulation, office-like usage, or high cooling demand. The full AC count is calculated."
        )

    if washing >= 3:
        messages.append(
            "Multiple washing machines may represent shared housing, service use, or commercial laundry behavior. The full number is calculated."
        )

    if heavy >= 3:
        messages.append(
            "Heavy machines may represent workshop, commercial, or semi-industrial usage. The full number is calculated."
        )

    if occupants >= 8 and size <= 100:
        messages.append(
            "Occupancy is dense for the entered area, but this can be valid for shared accommodation. The full occupant number is calculated."
        )

    if not messages:
        messages.append(
            "Household pattern is accepted. All entered values are included in calculation."
        )

    return messages


def household_input(title, default_lamps, default_acs, default_washing,
                    default_heavy, default_occupants, default_size):

    st.subheader(title)

    lamps = st.number_input(
        f"{title} - Lamps",
        min_value=0,
        value=default_lamps,
        step=1
    )

    acs = st.number_input(
        f"{title} - ACs",
        min_value=0,
        value=default_acs,
        step=1
    )

    washing = st.number_input(
        f"{title} - Washing Machines",
        min_value=0,
        value=default_washing,
        step=1
    )

    heavy = st.number_input(
        f"{title} - Heavy Machines",
        min_value=0,
        value=default_heavy,
        step=1
    )

    occupants = st.number_input(
        f"{title} - Occupants",
        min_value=0,
        value=default_occupants,
        step=1
    )

    size = st.number_input(
        f"{title} - House Size m²",
        min_value=1,
        value=default_size,
        step=1
    )

    for msg in describe_household_pattern(lamps, acs, washing, heavy, occupants, size):
        st.info(msg)

    return pd.DataFrame([{
        "lamps": lamps,
        "acs": acs,
        "washing_machine": washing,
        "heavy_machines": heavy,
        "occupants": occupants,
        "house_size": size
    }])


# =========================================================
# PRIORITY HELPERS
# =========================================================

def priority_meaning_from_row(row, priority_col):
    if bool(row.get("Protected / Never Disconnect", False)):
        return "Protected - Never Disconnect"

    try:
        p = int(row[priority_col])
    except Exception:
        return "Unknown"

    if p == 1:
        return "1 - Disconnect First"
    elif p == 2:
        return "2 - Disconnect Early"
    elif p == 3:
        return "3 - Emergency Early"
    elif p == 4:
        return "4 - Normal Shedding"
    elif p == 5:
        return "5 - Preserve Longer"
    elif p in [6, 7, 8, 9, 10]:
        return f"{p} - Disconnect Late"
    else:
        return f"{p} - Custom Order"


def add_priority_meaning_columns(df):
    df = ensure_appliance_columns(df)
    df["User Priority Meaning"] = df.apply(
        lambda row: priority_meaning_from_row(row, "User Priority"), axis=1
    )
    df["Company Priority Meaning"] = df.apply(
        lambda row: priority_meaning_from_row(row, "Company Priority"), axis=1
    )
    return df


def apply_load_category_priority_rules(appliance_df):
    """
    Important fix:
    - This function no longer overwrites User Priority or Company Priority.
    - User and Company priorities are editable and actually used by shedding.
    - Critical loads are protected by flags, not by hidden priority numbers.
    """
    df = ensure_appliance_columns(appliance_df)

    for idx, row in df.iterrows():
        category = row["Load Category"]

        if category == "Critical / Never Disconnect":
            df.loc[idx, "Critical"] = True
            df.loc[idx, "Protected / Never Disconnect"] = True
            df.loc[idx, "Disconnectable"] = False
            df.loc[idx, "Preserve Minimum Units"] = max(
                float(row["Preserve Minimum Units"]),
                float(row["Quantity"])
            )
        else:
            df.loc[idx, "Critical"] = False
            df.loc[idx, "Protected / Never Disconnect"] = False
            df.loc[idx, "Disconnectable"] = True

    return ensure_appliance_columns(df)


# =========================================================
# LOAD CALCULATION ENGINE
# =========================================================

def calculate_current_connected_load(appliance_df):
    df = ensure_appliance_columns(appliance_df)

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["Power per Unit kW"] = pd.to_numeric(df["Power per Unit kW"], errors="coerce").fillna(0)
    df["Preserve Minimum Units"] = pd.to_numeric(df["Preserve Minimum Units"], errors="coerce").fillna(0)
    df["User Priority"] = pd.to_numeric(df["User Priority"], errors="coerce").fillna(4)
    df["Company Priority"] = pd.to_numeric(df["Company Priority"], errors="coerce").fillna(4)

    df["Quantity"] = df["Quantity"].clip(lower=0)
    df["Power per Unit kW"] = df["Power per Unit kW"].clip(lower=0)
    df["Preserve Minimum Units"] = df["Preserve Minimum Units"].clip(lower=0)

    df["Connected Load kW"] = np.where(
        df["Connected"],
        df["Quantity"] * df["Power per Unit kW"],
        0
    )

    return df


def sync_ac_quantity_with_real_life_page():
    active_ac_count = sum(
        1 for state in st.session_state.ac_unit_states.values() if state
    )

    df = ensure_appliance_columns(st.session_state.appliance_config)

    if "ACs" in df["Appliance"].values:
        df.loc[df["Appliance"] == "ACs", "Quantity"] = active_ac_count
        df.loc[df["Appliance"] == "ACs", "Connected"] = active_ac_count > 0
    else:
        df.loc[len(df)] = {
            "Appliance": "ACs",
            "Quantity": active_ac_count,
            "Power per Unit kW": 1.3,
            "Connected": active_ac_count > 0,
            "Load Category": "Comfort Load",
            "Allow Company Emergency Control": True,
            "Preserve Minimum Units": 0,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 5,
            "Company Priority": 3
        }

    st.session_state.appliance_config = apply_load_category_priority_rules(df)


def apply_household_counts_to_appliance_config(base_config, household_df):
    df = ensure_appliance_columns(base_config)
    row = household_df.iloc[0]

    mapping = {
        "Lights": int(row["lamps"]),
        "ACs": int(row["acs"]),
        "Washing Machine": int(row["washing_machine"]),
        "Heavy Machines": int(row["heavy_machines"]),
    }

    for appliance, qty in mapping.items():
        if appliance in df["Appliance"].values:
            df.loc[df["Appliance"] == appliance, "Quantity"] = qty
            df.loc[df["Appliance"] == appliance, "Connected"] = qty > 0

    return apply_load_category_priority_rules(df)


def smart_meter_shed_load(
    appliance_df,
    requested_reduction_percent,
    policy_mode,
    refuse_disconnect,
    climate_mode,
    mandatory_minimum_percent,
    user_failed_to_respond,
    enforcement_enabled,
    minimum_service_enabled,
    force_mode=False
):
    df = calculate_current_connected_load(appliance_df)
    original_load = df["Connected Load kW"].sum()

    df["Disconnected Units"] = 0.0
    df["Remaining Units"] = df["Quantity"].astype(float)
    df["Shed kW"] = 0.0
    df["Remaining Load kW"] = df["Connected Load kW"].astype(float)
    df["Shedding Order"] = ""
    df["Emergency Shed Rank"] = 10
    df["Effective Shed Priority"] = 10
    df["Shed Explanation"] = ""

    if original_load <= 0:
        return df, 0, 0, 0, "No active load"

    requested_reduction_kw = original_load * requested_reduction_percent / 100
    mandatory_reduction_kw = original_load * mandatory_minimum_percent / 100

    if force_mode:
        target_reduction_kw = max(requested_reduction_kw, mandatory_reduction_kw)
        active_policy = "Company Priority"
        enforcement_status = (
            "Last-resort fair settlement is active. Manual-data no-bias shedding is active. "
            "The actual appliance table values are used. Same-priority exact ties rotate instead of using appliance name."
        )
    elif refuse_disconnect:
        if enforcement_enabled and user_failed_to_respond:
            target_reduction_kw = mandatory_reduction_kw
            active_policy = "Company Priority"
            enforcement_status = "User refused or ignored request. Mandatory emergency reduction was enforced."
        else:
            target_reduction_kw = 0
            active_policy = policy_mode
            enforcement_status = "No load was disconnected. Premium pricing/timer may apply."
    else:
        target_reduction_kw = requested_reduction_kw
        active_policy = policy_mode
        enforcement_status = "Smart meter priority was applied."

    priority_col = "Company Priority" if active_policy == "Company Priority" else "User Priority"

    def emergency_shed_rank(row):
        appliance = str(row.get("Appliance", "")).lower()
        category = str(row.get("Load Category", ""))
        if bool(row.get("Protected / Never Disconnect", False)) or bool(row.get("Critical", False)):
            return 10
        if "heavy" in appliance or "washing" in appliance:
            return 1
        if category == "Shed First":
            return 2
        if category == "Luxury Load":
            return 3
        if category == "Normal Load":
            return 4
        if category == "Comfort Load":
            return 5
        return 6

    df["Emergency Shed Rank"] = df.apply(emergency_shed_rank, axis=1)
    df["Effective Shed Priority"] = pd.to_numeric(df[priority_col], errors="coerce").fillna(4).astype(int)

    df.loc[df["Protected / Never Disconnect"] == True, "Shed Explanation"] = "Not shed: protected / never disconnect load."
    df.loc[df["Critical"] == True, "Shed Explanation"] = "Not shed: critical load."
    df.loc[df["Connected"] == False, "Shed Explanation"] = "Not shed: appliance is not connected."
    df.loc[df["Quantity"] <= 0, "Shed Explanation"] = "Not shed: quantity is zero."

    candidates = df[
        (df["Connected"] == True) &
        (df["Quantity"] > 0) &
        (df["Power per Unit kW"] > 0) &
        (df["Disconnectable"] == True) &
        (df["Critical"] == False) &
        (df["Protected / Never Disconnect"] == False)
    ].copy()

    preserve_minimum_by_idx = {}
    for idx, row in candidates.iterrows():
        preserve_minimum = float(row["Preserve Minimum Units"])
        appliance = row["Appliance"]
        if minimum_service_enabled:
            if climate_mode == "Hot Summer - Cooling Priority" and appliance == "ACs":
                preserve_minimum = max(preserve_minimum, 1)
            if climate_mode == "Cold Winter - Heating Priority" and appliance == "Water Heater":
                preserve_minimum = max(preserve_minimum, 1)
        preserve_minimum_by_idx[idx] = preserve_minimum
        if float(row["Quantity"]) <= preserve_minimum:
            df.loc[idx, "Shed Explanation"] = "Not shed: preserved minimum service leaves no disconnectable units."
        else:
            df.loc[idx, "Shed Explanation"] = "Candidate: available for shedding if its priority tier is needed."

    if target_reduction_kw <= 0:
        df["Remaining Load kW"] = (df["Remaining Units"] * df["Power per Unit kW"]).round(2)
        return df, original_load, original_load, 0, enforcement_status

    shed_so_far = 0.0
    order_counter = 1

    if force_mode:
        while shed_so_far < target_reduction_kw:
            available_rows = []
            for idx, row in candidates.iterrows():
                remaining_units = float(df.loc[idx, "Remaining Units"])
                preserve_minimum = preserve_minimum_by_idx.get(idx, 0)
                power = float(row["Power per Unit kW"])
                if remaining_units > preserve_minimum and power > 0:
                    available_rows.append(idx)
            if not available_rows:
                break

            available_df = df.loc[available_rows].copy()
            best_priority = available_df["Effective Shed Priority"].min()
            tier_df = available_df[available_df["Effective Shed Priority"] == best_priority].copy()

            tier_df["Remaining Disconnectable Units"] = tier_df.index.map(
                lambda i: max(float(df.loc[i, "Remaining Units"]) - preserve_minimum_by_idx.get(i, 0), 0)
            )
            tier_df["Current Remaining Load kW"] = tier_df.index.map(
                lambda i: float(df.loc[i, "Remaining Units"]) * float(df.loc[i, "Power per Unit kW"])
            )
            tier_df["Already Shed kW"] = tier_df.index.map(lambda i: float(df.loc[i, "Shed kW"]))
            tier_df["Already Disconnected Units"] = tier_df.index.map(lambda i: float(df.loc[i, "Disconnected Units"]))
            tier_df["Power Sort"] = tier_df["Power per Unit kW"].astype(float)

            # Electrical order only. Appliance name is intentionally NOT used.
            tier_df = tier_df.sort_values(
                by=["Current Remaining Load kW", "Already Shed kW", "Already Disconnected Units", "Power Sort", "Remaining Disconnectable Units"],
                ascending=[False, True, True, False, False]
            )

            best_row = tier_df.iloc[0]
            tie_df = tier_df[
                (tier_df["Current Remaining Load kW"] == best_row["Current Remaining Load kW"]) &
                (tier_df["Already Shed kW"] == best_row["Already Shed kW"]) &
                (tier_df["Already Disconnected Units"] == best_row["Already Disconnected Units"]) &
                (tier_df["Power Sort"] == best_row["Power Sort"]) &
                (tier_df["Remaining Disconnectable Units"] == best_row["Remaining Disconnectable Units"])
            ].copy()

            tie_names = sorted(tie_df["Appliance"].astype(str).tolist())
            tie_key = "shed_rotation_" + "__".join(tie_names)
            if tie_key not in st.session_state:
                st.session_state[tie_key] = 0
            chosen_position = int(st.session_state[tie_key] % len(tie_df))
            chosen_idx = int(tie_df.index[chosen_position])
            st.session_state[tie_key] += 1

            chosen_power = float(df.loc[chosen_idx, "Power per Unit kW"])
            chosen_priority = int(df.loc[chosen_idx, "Effective Shed Priority"])
            before_remaining_kw = float(df.loc[chosen_idx, "Remaining Units"]) * chosen_power

            df.loc[chosen_idx, "Disconnected Units"] += 1
            df.loc[chosen_idx, "Remaining Units"] -= 1
            df.loc[chosen_idx, "Shed kW"] += chosen_power
            df.loc[chosen_idx, "Remaining Load kW"] = df.loc[chosen_idx, "Remaining Units"] * chosen_power
            if df.loc[chosen_idx, "Shedding Order"] == "":
                df.loc[chosen_idx, "Shedding Order"] = f"{order_counter} - First Shed"
                order_counter += 1

            tie_note = ""
            if len(tie_df) > 1:
                tie_note = f" Exact tie detected between: {', '.join(tie_df['Appliance'].astype(str).tolist())}. Selection used tie-rotation."
            df.loc[chosen_idx, "Shed Explanation"] = (
                f"Shed by force mode from priority {chosen_priority}. "
                f"Before this step, this load had {before_remaining_kw:.2f} kW remaining. "
                f"One unit × {chosen_power:.2f} kW was disconnected." + tie_note
            )
            shed_so_far += chosen_power

    else:
        candidates = candidates.sort_values(by=[priority_col, "Power per Unit kW", "Quantity"], ascending=[True, False, False])
        for idx, row in candidates.iterrows():
            if shed_so_far >= target_reduction_kw:
                break
            quantity = float(row["Quantity"])
            power = float(row["Power per Unit kW"])
            preserve_minimum = preserve_minimum_by_idx.get(idx, 0)
            max_disconnectable_units = max(quantity - preserve_minimum, 0)
            if max_disconnectable_units <= 0 or power <= 0:
                continue
            remaining_needed_kw = target_reduction_kw - shed_so_far
            units_needed = np.ceil(remaining_needed_kw / power)
            units_to_disconnect = min(max_disconnectable_units, units_needed)
            shed_kw = units_to_disconnect * power
            df.loc[idx, "Disconnected Units"] = units_to_disconnect
            df.loc[idx, "Remaining Units"] = quantity - units_to_disconnect
            df.loc[idx, "Shed kW"] = shed_kw
            df.loc[idx, "Remaining Load kW"] = df.loc[idx, "Remaining Units"] * power
            df.loc[idx, "Shedding Order"] = f"{order_counter} - Shed"
            df.loc[idx, "Shed Explanation"] = f"Shed by {priority_col}: priority {int(row[priority_col])}, {units_to_disconnect:.0f} units × {power:.2f} kW = {shed_kw:.2f} kW."
            shed_so_far += shed_kw
            order_counter += 1

    final_load = max(original_load - shed_so_far, 0)
    achieved_reduction_percent = (shed_so_far / original_load) * 100

    for idx, row in df.iterrows():
        if row.get("Shed Explanation", "") == "Candidate: available for shedding if its priority tier is needed." and float(df.loc[idx, "Disconnected Units"]) == 0:
            df.loc[idx, "Shed Explanation"] = "Not shed: target reduction was reached before this load was needed, or a lower-number priority tier satisfied the required kW."

    df["Disconnected Units"] = df["Disconnected Units"].round(2)
    df["Remaining Units"] = df["Remaining Units"].round(2)
    df["Shed kW"] = df["Shed kW"].round(2)
    df["Connected Load kW"] = df["Connected Load kW"].round(2)
    df["Remaining Load kW"] = (df["Remaining Units"] * df["Power per Unit kW"]).round(2)
    return df, original_load, final_load, achieved_reduction_percent, enforcement_status


# =========================================================
# SAUDI TARIFF AND STATIC COMPOUND ENGINE
# =========================================================
def calculate_saudi_tariff_breakdown(kwh, category="Residential", meter_fee=None, include_vat=False):
    kwh = max(float(kwh), 0.0)
    category_key = str(category).strip().lower()

    if category_key == "commercial":
        tier1_rate = COMMERCIAL_TIER1_RATE
        tier2_rate = COMMERCIAL_TIER2_RATE
        default_meter_fee = DEFAULT_COMMERCIAL_METER_FEE
    elif category_key in ["charity", "charitable", "charitable institution"]:
        tier1_rate = CHARITY_TIER1_RATE
        tier2_rate = CHARITY_TIER2_RATE
        default_meter_fee = DEFAULT_RESIDENTIAL_METER_FEE
    else:
        tier1_rate = RESIDENTIAL_TIER1_RATE
        tier2_rate = RESIDENTIAL_TIER2_RATE
        default_meter_fee = DEFAULT_RESIDENTIAL_METER_FEE

    if meter_fee is None:
        meter_fee = default_meter_fee

    tier1_kwh = min(kwh, SAUDI_TIER_LIMIT_KWH)
    tier2_kwh = max(kwh - SAUDI_TIER_LIMIT_KWH, 0)
    tier1_charge = tier1_kwh * tier1_rate
    tier2_charge = tier2_kwh * tier2_rate
    energy_charge = tier1_charge + tier2_charge
    subtotal = energy_charge + meter_fee
    vat = subtotal * SAUDI_VAT_RATE if include_vat else 0.0
    total = subtotal + vat

    return {
        "Category": category,
        "kWh": kwh,
        "Tier 1 kWh": tier1_kwh,
        "Tier 2 kWh": tier2_kwh,
        "Tier 1 Rate SAR/kWh": tier1_rate,
        "Tier 2 Rate SAR/kWh": tier2_rate,
        "Tier 1 Charge SAR": tier1_charge,
        "Tier 2 Charge SAR": tier2_charge,
        "Energy Charge SAR": energy_charge,
        "Meter Fee SAR": meter_fee,
        "VAT SAR": vat,
        "Total Bill SAR": total,
    }


def estimate_saudi_monthly_baseline_from_household(household_df):
    row = household_df.iloc[0].astype(float)
    lamps = row["lamps"]
    acs = row["acs"]
    washing = row["washing_machine"]
    heavy = row["heavy_machines"]
    occupants = max(row["occupants"], 1)
    size = max(row["house_size"], 1)

    # Monthly Saudi-oriented estimator. It anchors apartments near 800-1500 kWh,
    # standard villas near 2000-4000 kWh, then adds device and density effects.
    if size <= 180:
        shell_baseline = 650 + 3.2 * size + 120 * occupants
        property_floor = SAUDI_APARTMENT_LOW_KWH
        property_soft_cap = SAUDI_APARTMENT_HIGH_KWH + max(acs - 2, 0) * 450
    elif size <= 350:
        shell_baseline = 1250 + 4.8 * size + 180 * occupants
        property_floor = SAUDI_VILLA_LOW_KWH
        property_soft_cap = SAUDI_VILLA_HIGH_KWH + max(acs - 4, 0) * 550
    else:
        shell_baseline = 1800 + 5.5 * size + 220 * occupants
        property_floor = SAUDI_VILLA_LOW_KWH + 500
        property_soft_cap = 6500 + max(acs - 5, 0) * 600

    device_baseline = lamps * 6 + acs * 430 + washing * 120 + heavy * 350
    density_factor = 1.0
    area_per_occupant = size / occupants
    if area_per_occupant < 15:
        density_factor += 0.08
    elif area_per_occupant > 90:
        density_factor -= 0.07

    estimated = (shell_baseline + device_baseline) * density_factor
    estimated = max(estimated, property_floor)
    estimated = min(estimated, property_soft_cap)
    return max(estimated, 0.5)


def build_static_compound_summary():
    rows = []

    static_assets = [
        {
            "Compound Asset": "8 Villas",
            "Meters": 8,
            "Units per Meter": 1,
            "Category": "Residential",
            "Monthly kWh per Meter": 3000,
            "Note": "Standard villa assumption inside Saudi range 2,000-4,000 kWh/month",
        },
        {
            "Compound Asset": "3 Apartment Blocks",
            "Meters": 36,
            "Units per Meter": 1,
            "Category": "Residential",
            "Monthly kWh per Meter": 1200,
            "Note": "Static assumption: 12 apartments per block, 1,200 kWh/month per apartment",
        },
        {
            "Compound Asset": "Club House",
            "Meters": 1,
            "Units per Meter": 1,
            "Category": "Commercial",
            "Monthly kWh per Meter": 6500,
            "Note": "Club house treated as commercial with usage crossing 6,000 kWh/month",
        },
    ]

    for item in static_assets:
        breakdown = calculate_saudi_tariff_breakdown(
            item["Monthly kWh per Meter"],
            category=item["Category"],
            include_vat=False,
        )
        meters = item["Meters"]
        rows.append({
            **item,
            "Total Monthly kWh": item["Monthly kWh per Meter"] * meters,
            "Tier 1 kWh per Meter": breakdown["Tier 1 kWh"],
            "Tier 2 kWh per Meter": breakdown["Tier 2 kWh"],
            "Monthly Bill per Meter SAR": breakdown["Total Bill SAR"],
            "Total Monthly Bill SAR": breakdown["Total Bill SAR"] * meters,
        })

    df = pd.DataFrame(rows)
    total_row = {
        "Compound Asset": "TOTAL COMPOUND",
        "Meters": df["Meters"].sum(),
        "Units per Meter": "",
        "Category": "Mixed",
        "Monthly kWh per Meter": "",
        "Total Monthly kWh": df["Total Monthly kWh"].sum(),
        "Tier 1 kWh per Meter": "",
        "Tier 2 kWh per Meter": "",
        "Monthly Bill per Meter SAR": "",
        "Total Monthly Bill SAR": df["Total Monthly Bill SAR"].sum(),
        "Note": "Static compound total",
    }
    return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

# =========================================================
# BILLING ENGINE
# =========================================================

def billing_engine(
    baseline,
    requested_usage,
    final_usage,
    mean_usage,
    grid_stress,
    new_company_growth_mode,
    refused_disconnect,
    achieved_reduction_percent,
    mandatory_reduction_percent,
    fair_conditions,
    deadline_penalty_active,
    forced_fair_settlement_active,
    good_behavior_discount_rate
):
    premium_usage = max(final_usage - baseline, 0)
    normal_usage = min(final_usage, baseline)

    final_tariff = calculate_saudi_tariff_breakdown(final_usage, category="Residential", include_vat=False)
    no_action_tariff = calculate_saudi_tariff_breakdown(requested_usage, category="Residential", include_vat=False)

    no_action_bill = no_action_tariff["Total Bill SAR"]
    bill = final_tariff["Total Bill SAR"]

    bonus = 0
    penalty = 0
    timer_penalty = 0
    premium_charge = 0
    discount = 0
    loyalty_discount = 0
    good_behavior_discount = 0
    penalty_waived = 0
    company_growth_discount = 0
    status = []

    if grid_stress:
        if fair_conditions["marginal_premium"]:
            if premium_usage > 0:
                premium_charge = premium_usage * (RESIDENTIAL_TIER2_RATE + 0.10)
                bill += premium_charge
                status.append("Premium pricing applied only to usage above the company-approved baseline.")
        else:
            if final_usage > baseline:
                premium_charge = final_usage * 0.30
                bill += premium_charge
                status.append("General premium pricing applied because marginal baseline protection is disabled.")

    penalty_should_apply = (
        grid_stress
        and mandatory_reduction_percent > 0
        and achieved_reduction_percent < mandatory_reduction_percent
        and not forced_fair_settlement_active
    )

    if (
        penalty_should_apply
        and fair_conditions["low_baseline_protection"]
        and baseline <= LOW_BASELINE_THRESHOLD_KWH
        and final_usage <= baseline
    ):
        penalty_waived = final_usage * 0.20
        penalty_should_apply = False
        status.append("Penalty waived by low-baseline protection because the customer stayed within normal usage.")

    if penalty_should_apply:
        if fair_conditions["progressive_penalty"]:
            shortfall = mandatory_reduction_percent - achieved_reduction_percent
            shortfall_ratio = shortfall / max(mandatory_reduction_percent, 1)
            penalty = final_tariff["Energy Charge SAR"] * 0.20 * shortfall_ratio
            status.append("Progressive penalty applied based on reduction shortfall.")
        else:
            penalty = final_tariff["Energy Charge SAR"] * 0.20
            status.append("Flat grid stress penalty applied because mandatory reduction was not achieved.")
        bill += penalty

    if deadline_penalty_active and not forced_fair_settlement_active:
        timer_penalty = final_tariff["Energy Charge SAR"] * 0.25
        bill += timer_penalty
        status.append("Deadline/ignored-request penalty applied because the user delayed or ignored company requests during stress.")

    if (
        fair_conditions["grid_support_discount"]
        and grid_stress
        and mandatory_reduction_percent > 0
        and achieved_reduction_percent >= mandatory_reduction_percent
    ):
        discount_value = bill * DISCOUNT_RATE
        bill -= discount_value
        discount += discount_value
        status.append("Grid support discount applied because mandatory reduction was achieved.")

    if (
        fair_conditions["loyalty_discount"]
        and fair_conditions["historical_baseline"]
        and grid_stress
        and final_usage <= baseline
    ):
        loyalty_discount = bill * LOYALTY_DISCOUNT_RATE
        bill -= loyalty_discount
        discount += loyalty_discount
        status.append("Loyalty discount applied because usage stayed at or below the company-approved baseline during peak stress.")

    if good_behavior_discount_rate > 0 and grid_stress and final_usage <= baseline:
        good_behavior_discount = bill * good_behavior_discount_rate
        bill -= good_behavior_discount
        status.append(f"Repeated good-behavior discount applied: {good_behavior_discount_rate * 100:.0f}%.")

    if fair_conditions["growth_bonus"] and new_company_growth_mode:
        company_growth_discount = bill * COMPANY_GROWTH_DISCOUNT_RATE
        bill -= company_growth_discount
        discount += company_growth_discount
        bonus += company_growth_discount
        status.append("New company growth discount applied to the whole bill after charges and penalties.")

    if refused_disconnect and grid_stress and not forced_fair_settlement_active:
        if fair_conditions["customer_autonomy"]:
            status.append("User used manual override. Timer, premium pricing, or emergency enforcement may apply.")

    if forced_fair_settlement_active:
        status.append("Forced fair settlement stabilized the line. Timer and delay penalty are disabled.")

    if not status:
        status.append("Normal Saudi tariff billing condition.")

    final_bill = max(bill, 0)
    amount_saved = max(no_action_bill - final_bill, 0)

    return {
        "Normal Usage kWh": normal_usage,
        "Premium Usage kWh": premium_usage,
        "Saudi Energy Charge SAR": final_tariff["Energy Charge SAR"],
        "Saudi Meter Fee SAR": final_tariff["Meter Fee SAR"],
        "Saudi Tier 1 kWh": final_tariff["Tier 1 kWh"],
        "Saudi Tier 2 kWh": final_tariff["Tier 2 kWh"],
        "Premium Charge": premium_charge,
        "Penalty": penalty,
        "Timer Penalty": timer_penalty,
        "Penalty Waived": penalty_waived,
        "Bonus": bonus,
        "Discount": discount,
        "Loyalty Discount": loyalty_discount,
        "Good Behavior Discount": good_behavior_discount,
        "Company Growth Discount": company_growth_discount,
        "No Action Bill": no_action_bill,
        "Amount Saved": amount_saved,
        "Final Bill": final_bill,
        "Status": " | ".join(status)
    }

# =========================================================
# TIMER ENGINE
# =========================================================

def reset_deadline_timer(clear_penalty=False):
    st.session_state.deadline_started_at = None
    st.session_state.deadline_end_at = None
    st.session_state.deadline_signature = None

    if clear_penalty:
        st.session_state.deadline_penalty_latched = False
        st.session_state.deadline_penalty_reason = ""


def deadline_timer_engine(
    timer_should_run,
    deadline_minutes,
    force_settlement_active,
    timer_signature=None
):
    """
    Robust Streamlit timer:
    - Stores deadline_end_at in session_state.
    - If JS reloads the page after expiry, Python still sees that now >= deadline_end_at.
    - Once expired, penalty is latched until reset or last-resort settlement clears it.
    """
    now = time.time()

    if timer_signature is None:
        timer_signature = "default_timer_signature"

    if force_settlement_active:
        reset_deadline_timer(clear_penalty=True)
        return False, "Timer disabled because last-resort fair settlement is active."

    if st.session_state.deadline_penalty_latched:
        return True, st.session_state.deadline_penalty_reason or "Timer expired. Penalty/enforcement is active."

    if not timer_should_run:
        reset_deadline_timer(clear_penalty=False)
        return False, "Timer inactive."

    deadline_seconds = max(int(deadline_minutes * 60), 1)

    if (
        st.session_state.deadline_signature != timer_signature
        or st.session_state.deadline_started_at is None
        or st.session_state.deadline_end_at is None
    ):
        st.session_state.deadline_signature = timer_signature
        st.session_state.deadline_started_at = now
        st.session_state.deadline_end_at = now + deadline_seconds

    remaining = max(st.session_state.deadline_end_at - now, 0)
    expired = remaining <= 0

    if expired:
        st.session_state.deadline_penalty_latched = True
        st.session_state.deadline_penalty_reason = "Timer expired. Penalty/enforcement is active."
        return True, st.session_state.deadline_penalty_reason

    end_timestamp_ms = int(st.session_state.deadline_end_at * 1000)

    components.html(
        f"""
        <div style="
            padding:14px;
            border-radius:14px;
            border:1px solid rgba(255,255,255,0.25);
            background:rgba(0,0,0,0.45);
            font-family:Arial;
            color:white;
            font-size:20px;">
            <b>Response Deadline Timer:</b>
            <span id="deadline_timer" style="color:#00ffff;font-weight:bold;"></span>
        </div>

        <script>
        const endTime = {end_timestamp_ms};

        function updateTimer() {{
            const now = Date.now();
            let diff = Math.max(0, endTime - now);
            let totalSeconds = Math.floor(diff / 1000);
            let minutes = Math.floor(totalSeconds / 60);
            let seconds = totalSeconds % 60;

            const text = minutes.toString().padStart(2, '0') + ':' + seconds.toString().padStart(2, '0');
            document.getElementById("deadline_timer").innerText = text;

            if (diff <= 0) {{
                document.getElementById("deadline_timer").innerText = "EXPIRED - Penalty/Enforcement Active";
                document.getElementById("deadline_timer").style.color = "red";
                setTimeout(() => window.parent.location.reload(), 1500);
            }}
        }}

        updateTimer();
        setInterval(updateTimer, 1000);
        </script>
        """,
        height=90
    )

    return False, f"Timer running. Remaining seconds: {int(remaining)}"

# =========================================================
# IMAGE OVERLAY ENGINE
# =========================================================

def get_font(size=72):
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "arial.ttf"
    ]

    for font_path in font_candidates:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)

    return ImageFont.load_default()


def draw_centered_text(draw, xy, text, font, fill, outline_fill="black", outline_width=6):
    x, y = xy

    try:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=outline_width)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except Exception:
        w, h = draw.textsize(text, font=font)

    draw.text(
        (x - w / 2, y - h / 2),
        text,
        font=font,
        fill=fill,
        stroke_width=outline_width,
        stroke_fill=outline_fill
    )


def draw_label_box(draw, xy, text, font, fill="yellow"):
    x, y = xy

    try:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=3)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except Exception:
        w, h = draw.textsize(text, font=font)

    padding_x = 10
    padding_y = 6

    rect = [
        x - w / 2 - padding_x,
        y - h / 2 - padding_y,
        x + w / 2 + padding_x,
        y + h / 2 + padding_y
    ]

    draw.rounded_rectangle(
        rect,
        radius=8,
        fill=(0, 0, 0, 190),
        outline=(255, 255, 0, 230),
        width=2
    )

    draw_centered_text(
        draw=draw,
        xy=(x, y),
        text=text,
        font=font,
        fill=fill,
        outline_fill="black",
        outline_width=3
    )


def render_ac_plan_overlay(image_path, ac_states, positions, show_added_labels=False):
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)

    symbol_font = get_font(170)
    label_font = get_font(24)

    for ac_name, is_on in ac_states.items():
        pos = positions.get(
            ac_name,
            {"x": 100, "y": 100, "label_x": 130, "label_y": 130}
        )

        x = int(pos["x"])
        y = int(pos["y"])
        label_x = int(pos.get("label_x", x + 45))
        label_y = int(pos.get("label_y", y + 45))

        symbol = "O" if is_on else "X"
        color = "lime" if is_on else "red"

        draw_centered_text(
            draw=draw,
            xy=(x, y),
            text=symbol,
            font=symbol_font,
            fill=color,
            outline_fill="black",
            outline_width=15
        )

        if show_added_labels:
            draw_label_box(
                draw=draw,
                xy=(label_x, label_y),
                text=ac_name,
                font=label_font,
                fill="yellow"
            )

    return image


# =========================================================
# NAVIGATION
# =========================================================

page = st.radio(
    "Navigation",
    [
        "SCADA Control Center",
        "Smart Meter Override Page",
        "Real Life Simulation",
        "Crisis Live Simulation",
        "Predictive Maintenance",
        "How To Use",
        "AI & Model Details"
    ],
    horizontal=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    if os.path.exists("Alex.jpg"):
        st.image("Alex.jpg")

    st.header("Model Performance")
    st.metric("MAE", f"{metrics['MAE']:.2f} kWh")
    st.metric("Model Score R²", f"{metrics['R2']:.2f}")
    st.metric("Maintenance RUL MAE", f"{maintenance_metrics['RUL MAE Months']:.1f} months")
    st.metric("Maintenance Risk Accuracy", f"{maintenance_metrics['Risk Accuracy']:.2f}")

    st.divider()

    st.header("Tariff Catalogue")
    st.write(f"Normal rate: **{BASE_RATE} SAR/kWh**")
    st.write(f"Peak rate: **{PEAK_RATE} SAR/kWh**")
    st.write(f"Penalty rate: **{PENALTY_RATE} SAR/kWh**")
    st.write(f"Premium preservation rate: **{PREMIUM_PRESERVATION_RATE} SAR/kWh**")
    st.write(f"Grid support discount: **{int(DISCOUNT_RATE * 100)}%**")
    st.write(f"Loyalty baseline discount: **{int(LOYALTY_DISCOUNT_RATE * 100)}%**")
    st.write(f"Growth bonus: **{int(BONUS_RATE * 100)}%**")
    st.write(f"Good behavior discount cap: **{int(GOOD_BEHAVIOR_MAX_DISCOUNT * 100)}%**")
    st.metric("Good Behavior Streak", st.session_state.good_behavior_streak)
    if st.button("Reset good-behavior tracker"):
        st.session_state.good_behavior_streak = 0
        st.session_state.last_behavior_event_id = None
        st.rerun()


# =========================================================
# PREDICTIVE MAINTENANCE PAGE
# =========================================================
if page == "Predictive Maintenance":
    st.title("Machine Learning Predictive & Protective Maintenance")
    st.markdown("""
<div class="scada-card">
<div class="big-status">SCADA Grid Asset Maintenance Simulator</div>
This page is linked to the SCADA smart meter table. Asset type now changes stress, condition thresholds,
remaining useful life, and graph shape. The feature-impact graph is dynamic for the current slider case.
</div>
    """, unsafe_allow_html=True)

    if st.button("Load Perfect Condition Defaults"):
        set_perfect_maintenance_defaults()
        st.success("Perfect-condition maintenance defaults loaded. The next graph uses healthy asset values.")

    live_load_df = calculate_current_connected_load(st.session_state.appliance_config)
    live_connected_load_kw = float(live_load_df["Connected Load kW"].sum())

    left_col, right_col = st.columns([0.34, 0.66])
    with left_col:
        st.subheader("SCADA Case Study Inputs")
        asset_type = st.selectbox(
            "Grid Asset Type",
            ["Distribution Transformer", "Feeder Cable", "Circuit Breaker", "Switchgear", "Main Distribution Panel"],
            index=0,
            key="mx_asset_type",
        )
        asset_profile = get_grid_asset_profile(asset_type)
        use_live_load = st.checkbox("Use current Smart Meter connected load as grid load input", value=True, key="mx_use_live_load")
        if use_live_load:
            live_load_kw = st.number_input(
                "Connected Grid Load kW from SCADA",
                min_value=0.0,
                max_value=None,
                value=float(round(max(live_connected_load_kw, 0.0), 2)),
                step=0.1,
                help="No artificial upper limit. If the SCADA table has high connected kW, this can grow with it.",
                key="mx_live_load_display",
            )
        else:
            live_load_kw = st.number_input("Manual Connected Grid Load kW", min_value=0.0, max_value=None, value=25.0, step=0.5, key="mx_live_load_manual")

        age_years = st.slider("Asset Age (years)", 0.0, 40.0, st.session_state.get("mx_age_years", 12.0), 0.5, key="mx_age_years")
        loading_percent = st.slider("Asset Loading (%)", 0.0, 250.0, st.session_state.get("mx_loading_percent", 82.0), 1.0, key="mx_loading_percent")
        peak_stress_percent = st.slider("Peak Stress / Overload Severity (%)", 0.0, 150.0, st.session_state.get("mx_peak_stress_percent", 45.0), 1.0, key="mx_peak_stress_percent")
        voltage_deviation_percent = st.slider("Voltage Deviation (%)", 0.0, 40.0, st.session_state.get("mx_voltage_deviation_percent", 4.5), 0.1, key="mx_voltage_deviation_percent")
        temperature_c = st.slider("Asset Temperature (°C)", 0.0, 180.0, st.session_state.get("mx_temperature_c", asset_profile["default_temperature"]), 1.0, key="mx_temperature_c")
        vibration_mm_s = st.slider("Vibration / Mechanical Stress (mm/s)", 0.0, 30.0, st.session_state.get("mx_vibration_mm_s", asset_profile["default_vibration"]), 0.1, key="mx_vibration_mm_s")
        insulation_health_percent = st.slider("Insulation / Oil Health (%)", 0.0, 100.0, st.session_state.get("mx_insulation_health_percent", asset_profile["default_insulation"]), 1.0, key="mx_insulation_health_percent")
        breaker_operations = st.slider("Breaker / Switching Operations", 0, 30000, st.session_state.get("mx_breaker_operations", 1200), 50, key="mx_breaker_operations")
        fault_events_30d = st.slider("Fault Events in Last 30 Days", 0, 100, st.session_state.get("mx_fault_events_30d", 2), 1, key="mx_fault_events_30d")
        humidity_percent = st.slider("Humidity (%)", 0.0, 100.0, st.session_state.get("mx_humidity_percent", 60.0), 1.0, key="mx_humidity_percent")
        maintenance_quality = st.slider("Maintenance Quality Factor", 0.30, 1.50, st.session_state.get("mx_maintenance_quality", 0.95), 0.01, key="mx_maintenance_quality")

        st.subheader("Protective Maintenance Controls")
        maintenance_day = st.slider("Planned Predictive Maintenance Day", 1, 1500, st.session_state.get("mx_maintenance_day", 120), 1, key="mx_maintenance_day")
        restoration_strength = st.slider("Maintenance Restoration Strength", 0.0, 80.0, st.session_state.get("mx_restoration_strength", 24.0), 1.0, key="mx_restoration_strength")
        degradation_speed = st.slider("Case Study Degradation Speed", 0.05, 5.00, st.session_state.get("mx_degradation_speed", 1.00), 0.05, key="mx_degradation_speed")
        horizon_days = st.slider("Simulation Horizon Days", 120, 2000, st.session_state.get("mx_horizon_days", 730), 30, key="mx_horizon_days")

    feature_row = pd.DataFrame([{
        "age_years": age_years,
        "live_load_kw": live_load_kw,
        "loading_percent": loading_percent,
        "peak_stress_percent": peak_stress_percent,
        "voltage_deviation_percent": voltage_deviation_percent,
        "temperature_c": temperature_c,
        "vibration_mm_s": vibration_mm_s,
        "insulation_health_percent": insulation_health_percent,
        "breaker_operations": breaker_operations,
        "fault_events_30d": fault_events_30d,
        "humidity_percent": humidity_percent,
        "maintenance_quality": maintenance_quality,
    }])

    raw_rul_months = float(maintenance_rul_model.predict(feature_row[maintenance_features])[0])
    raw_risk_probability = float(maintenance_risk_model.predict_proba(feature_row[maintenance_features])[0][1])
    base_condition_score, base_stress_index = calculate_grid_asset_condition(age_years, live_load_kw, loading_percent, peak_stress_percent, voltage_deviation_percent, temperature_c, vibration_mm_s, insulation_health_percent, breaker_operations, fault_events_30d, humidity_percent, maintenance_quality)

    stress_index = base_stress_index * asset_profile["stress_multiplier"]
    condition_score = float(np.clip(base_condition_score + asset_profile["condition_offset"] - (asset_profile["stress_multiplier"] - 1.0) * 3.5, 0, 100))
    predicted_rul_months = float(np.clip(raw_rul_months * asset_profile["rul_multiplier"] - max(asset_profile["stress_multiplier"] - 1.0, 0) * 8, 1, 600))
    risk_probability = float(np.clip(raw_risk_probability * asset_profile["stress_multiplier"] + max(0, 55 - condition_score) / 160, 0, 1))

    asset_state, severity_level, recommendation = classify_grid_maintenance_state(condition_score, risk_probability, predicted_rul_months)
    curve_df, points = simulate_grid_maintenance_curves(condition_score, stress_index, maintenance_day, restoration_strength, degradation_speed * asset_profile["stress_multiplier"], horizon_days)
    curve_df["Failure Threshold"] = asset_profile["failure_threshold"]
    curve_df["Potential Failure Threshold"] = asset_profile["potential_threshold"]
    profile_p_days = curve_df[curve_df["No Maintenance"] <= asset_profile["potential_threshold"]]["Day"]
    profile_f_days = curve_df[curve_df["No Maintenance"] <= asset_profile["failure_threshold"]]["Day"]
    profile_d_days = curve_df[curve_df["No Maintenance"] <= 80]["Day"]
    points = {
        "td": int(profile_d_days.iloc[0]) if len(profile_d_days) > 0 else None,
        "p": int(profile_p_days.iloc[0]) if len(profile_p_days) > 0 else None,
        "f": int(profile_f_days.iloc[0]) if len(profile_f_days) > 0 else None,
        "breakdown": int(profile_f_days.iloc[0]) if len(profile_f_days) > 0 else None,
    }

    dynamic_impact_df = build_dynamic_maintenance_feature_impact(feature_row, maintenance_rul_model, maintenance_risk_model, maintenance_features)

    with right_col:
        st.subheader("Machine Learning Prediction Result")
        a, b, c, d = st.columns(4)
        a.metric("Asset Condition", f"{condition_score:.1f}/100")
        b.metric("Predicted RUL", f"{predicted_rul_months:.1f} months")
        c.metric("12-Month Failure Risk", f"{risk_probability * 100:.1f}%")
        d.metric("Asset-Type Stress Index", f"{stress_index:.2f}")
        if severity_level == "Critical":
            st.error(f"{asset_type}: {asset_state}. {recommendation}")
        elif severity_level == "Danger":
            st.warning(f"{asset_type}: {asset_state}. {recommendation}")
        elif severity_level == "Warning":
            st.info(f"{asset_type}: {asset_state}. {recommendation}")
        else:
            st.success(f"{asset_type}: {asset_state}. {recommendation}")

        st.dataframe(pd.DataFrame([
            {"Point": "td - Degradation Start", "Day": points["td"], "Meaning": "First measurable degradation from normal operation"},
            {"Point": "P - Potential Failure", "Day": points["p"], "Meaning": "Fault symptoms are visible; predictive maintenance should act here"},
            {"Point": "F - Functional Failure", "Day": points["f"], "Meaning": "Asset can no longer perform the required function"},
            {"Point": "Planned Predictive Maintenance", "Day": maintenance_day, "Meaning": "Selected intervention timing"},
        ]), use_container_width=True)

    st.divider()
    st.subheader("P-F Curve: Asset-Type-Aware Normal / Dangerous / Failure States")
    st.plotly_chart(draw_grid_pf_curve(curve_df, points), use_container_width=True)

    st.subheader("Protective Maintenance Strategy Graph")
    fig_strategy = go.Figure()
    fig_strategy.add_trace(go.Scatter(x=curve_df["Day"], y=curve_df["Normal Life Expectancy"], mode="lines", name="Normal Life Expectancy", line=dict(color="royalblue", width=4)))
    fig_strategy.add_trace(go.Scatter(x=curve_df["Day"], y=curve_df["No Maintenance"], mode="lines", name="No Maintenance", line=dict(color="orangered", width=4, dash="dash")))
    fig_strategy.add_trace(go.Scatter(x=curve_df["Day"], y=curve_df["Predictive Maintenance"], mode="lines", name="Predictive Maintenance", line=dict(color="mediumseagreen", width=5)))
    fig_strategy.add_trace(go.Scatter(x=curve_df["Day"], y=curve_df["Time-Based Maintenance"], mode="lines", name="Time-Based Maintenance", line=dict(color="gray", width=3)))
    fig_strategy.add_trace(go.Scatter(x=curve_df["Day"], y=curve_df["Breakdown Maintenance"], mode="lines", name="Breakdown Maintenance", line=dict(color="firebrick", width=3, dash="dot")))
    fig_strategy.add_trace(go.Scatter(x=curve_df["Day"], y=curve_df["Failure Threshold"], mode="lines", name="Failure Threshold", line=dict(color="red", width=2, dash="dash")))
    fig_strategy.add_trace(go.Scatter(x=curve_df["Day"], y=curve_df["Potential Failure Threshold"], mode="lines", name="Potential Failure Threshold", line=dict(color="gold", width=2, dash="dash")))
    fig_strategy.add_vline(x=maintenance_day, line_width=3, line_dash="dot", line_color="yellow", annotation_text="Predictive Maintenance Action")
    fig_strategy.update_layout(title=f"Protective Maintenance Strategy Comparison - {asset_type}", xaxis_title="Time / Operating Days", yaxis_title="Equipment Condition / Performance", template="plotly_dark", height=660, yaxis=dict(range=[0,105]))
    st.plotly_chart(fig_strategy, use_container_width=True)

    st.subheader("SCADA Maintenance Prediction Dataset Row")
    display_feature_row = feature_row.copy()
    display_feature_row["Asset Type"] = asset_type
    display_feature_row["Predicted RUL Months"] = predicted_rul_months
    display_feature_row["Failure Risk Probability"] = risk_probability
    display_feature_row["Condition Score"] = condition_score
    display_feature_row["Asset-Type Stress Index"] = stress_index
    display_feature_row["Asset State"] = asset_state
    st.dataframe(display_feature_row, use_container_width=True)

    st.subheader("Dynamic ML Feature Impact for This Current Case")
    st.info("This graph changes with the sliders. It is not the static Random Forest global feature importance.")
    fig_dynamic = px.bar(dynamic_impact_df, x="SCADA Feature", y="Dynamic Impact Score", title="Current Case Sensitivity: Which slider changes the prediction most?", text_auto=".2f")
    fig_dynamic.update_layout(template="plotly_dark", height=520)
    st.plotly_chart(fig_dynamic, use_container_width=True)
    st.dataframe(dynamic_impact_df, use_container_width=True)

    with st.expander("Show static model feature importance"):
        feature_importance_df = pd.DataFrame({"SCADA Feature": maintenance_features, "Importance": maintenance_rul_model.feature_importances_}).sort_values("Importance", ascending=False)
        fig_importance = px.bar(feature_importance_df, x="SCADA Feature", y="Importance", title="Static Model Feature Importance", text_auto=".3f")
        fig_importance.update_layout(template="plotly_dark", height=520)
        st.plotly_chart(fig_importance, use_container_width=True)

    st.subheader("Maintenance Curve Data")
    st.dataframe(curve_df, use_container_width=True)
    st.stop()

# =========================================================
# HOW TO USE PAGE
# =========================================================

if page == "How To Use":

    st.title("How To Use The Website")

    st.markdown("""
<div class="manual-box">

## Priority Direction

The priority number is a load-shedding order.

- **1** means disconnect first.
- **2** means disconnect early.
- **4** means normal shedding.
- **5** means preserve longer.
- **Protected / Never Disconnect** means it cannot be disconnected. Protected flag is used instead.

</div>

<div class="manual-box">

## Force Fair Settlement

The last resort settlement button is stronger than user refusal.

If it is ON:

- A user within their own baseline is protected.
- A user above their own baseline is forced back to the fair baseline limit or below it.
- The Last Resort comparison shows two modes: Normal Mode result versus Last Resort Mode result.
- If Last Resort Mode has higher connected kW, it does not mean power was added; it means Normal Mode shed more load than the fair-settlement mode needed.
- Baseline is measured in kWh, while connected appliance load is measured in kW.
- Timer is disabled because the system immediately acts.

</div>

<div class="manual-box">

## Timer

The timer runs only when the user refuses or delays during real stress.

If the timer expires, penalty/enforcement activates.

If forced fair settlement is ON, the timer stops because the company already corrected the line stress.

</div>
    """, unsafe_allow_html=True)

    st.stop()


# =========================================================
# AI DETAILS PAGE
# =========================================================

if page == "AI & Model Details":

    st.title("Model & Dataset Information")

    st.markdown("""
<div class="manual-box">

## Dataset

The dataset is synthetic and generated inside the program.

Features:

- Lamps
- ACs
- Washing machines
- Heavy machines
- Occupants
- House size

Output:

- Historical baseline consumption in kWh

</div>

<div class="manual-box">

## Hybrid Baseline Logic

The simulator uses Random Forest plus engineering-style calculation.

This prevents hidden caps and makes sure every entered number contributes.

</div>

<div class="manual-box">

## MAE and R²

MAE means Mean Absolute Error.

R² means Coefficient of Determination.

Higher R² and lower MAE usually mean better model performance.

</div>
    """, unsafe_allow_html=True)

    st.subheader("Synthetic Dataset Preview")
    st.dataframe(training_df.head(150), use_container_width=True)

    x = np.linspace(
        training_df["historical_baseline_kwh"].min(),
        training_df["historical_baseline_kwh"].max(),
        600
    )

    y = norm.pdf(x, mean_usage, std_usage)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=training_df["historical_baseline_kwh"],
        histnorm="probability density",
        name="Synthetic Baseline Histogram",
        opacity=0.55
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name="Normal Distribution Bell Curve",
        line=dict(width=5, color="cyan")
    ))

    fig.add_vline(
        x=mean_usage,
        line_width=4,
        line_dash="dash",
        line_color="yellow",
        annotation_text="Mean"
    )

    fig.update_layout(
        title="Historical Baseline Consumption Bell Curve",
        xaxis_title="Historical Baseline kWh",
        yaxis_title="Probability Density",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=26),
        height=620
    )

    st.plotly_chart(fig, use_container_width=True)

    st.stop()


# =========================================================
# REAL LIFE SIMULATION PAGE
# =========================================================

if page == "Real Life Simulation":

    st.title("Real Life Simulation")

    st.markdown("""
<div class="scada-card">
<div class="big-status">HVAC Plan Live Overlay</div>
Pattern: AC6 — AC5 — AC2 / AC3 — AC1 / AC4
<br><br>
<b style="color:lime;">O</b> = working AC<br>
<b style="color:red;">X</b> = disconnected AC
</div>
    """, unsafe_allow_html=True)

    st.subheader("AC Operating State")

    ac_cols = st.columns(6)

    for idx, ac_name in enumerate(st.session_state.ac_unit_states.keys()):
        with ac_cols[idx]:
            st.session_state.ac_unit_states[ac_name] = st.toggle(
                ac_name,
                value=st.session_state.ac_unit_states[ac_name],
                key=f"toggle_{ac_name}"
            )

    sync_ac_quantity_with_real_life_page()

    active_ac_count = sum(
        1 for state in st.session_state.ac_unit_states.values() if state
    )
    disconnected_ac_count = len(st.session_state.ac_unit_states) - active_ac_count

    m1, m2, m3 = st.columns(3)
    m1.metric("Working ACs", active_ac_count)
    m2.metric("Disconnected ACs", disconnected_ac_count)
    m3.metric("Synced AC Quantity", active_ac_count)

    image_path = "ACs.png"

    if not os.path.exists(image_path):
        st.error("ACs.png was not found.")
        st.stop()

    show_added_ac_labels = st.checkbox(
        "Show extra AC labels on image",
        value=False
    )

    if st.button("Reset AC positions to corrected apartment pattern"):
        st.session_state.ac_overlay_positions = DEFAULT_AC_OVERLAY_POSITIONS.copy()
        st.session_state.ac_position_version = AC_POSITION_VERSION
        st.success("AC positions were reset.")

    with st.expander("Edit X/O and AC Number Positions"):
        position_rows = []

        for ac_name, pos in st.session_state.ac_overlay_positions.items():
            position_rows.append({
                "AC": ac_name,
                "x": int(pos["x"]),
                "y": int(pos["y"]),
                "label_x": int(pos.get("label_x", pos["x"] + 45)),
                "label_y": int(pos.get("label_y", pos["y"] + 45))
            })

        pos_df = pd.DataFrame(position_rows)

        edited_pos_df = st.data_editor(
            pos_df,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "AC": st.column_config.TextColumn("AC", disabled=True),
                "x": st.column_config.NumberColumn("Symbol X", step=1),
                "y": st.column_config.NumberColumn("Symbol Y", step=1),
                "label_x": st.column_config.NumberColumn("Label X", step=1),
                "label_y": st.column_config.NumberColumn("Label Y", step=1)
            }
        )

        for _, row in edited_pos_df.iterrows():
            st.session_state.ac_overlay_positions[row["AC"]] = {
                "x": int(row["x"]),
                "y": int(row["y"]),
                "label_x": int(row["label_x"]),
                "label_y": int(row["label_y"])
            }

    overlay_image = render_ac_plan_overlay(
        image_path=image_path,
        ac_states=st.session_state.ac_unit_states,
        positions=st.session_state.ac_overlay_positions,
        show_added_labels=show_added_ac_labels
    )

    center_left, center_main, center_right = st.columns([0.02, 0.96, 0.02])

    with center_main:
        st.image(
            overlay_image,
            caption="Live HVAC SCADA Overlay",
            use_container_width=True
        )

    status_df = pd.DataFrame([
        {
            "AC Unit": ac_name,
            "State": "Working" if state else "Disconnected",
            "Symbol": "O" if state else "X"
        }
        for ac_name, state in st.session_state.ac_unit_states.items()
    ])

    st.subheader("Real Life Simulation Status Table")
    st.dataframe(status_df, use_container_width=True)

    st.stop()



# =========================================================
# CRISIS LIVE SIMULATION PAGE
# =========================================================

if page == "Crisis Live Simulation":

    st.title("Crisis Live Simulation")

    st.markdown("""
<div class="scada-card">
<div class="big-status">Live Crisis Simulator</div>
This page tests whether the system has enough <b>controllable</b> load to satisfy a crisis target.
It no longer reports stable just because the shedding function ran. It compares:
<br><br>
<b>Required Shed kW</b> vs <b>Actual Shed kW</b> vs <b>Maximum Controllable Shed kW</b>.
</div>
    """, unsafe_allow_html=True)

    st.info(
        "Crisis stability means the actual shed kW reached the requested crisis target. "
        "If protected loads, preserved minimum service, disconnected appliances, or zero quantities prevent enough shedding, "
        "the page will show: Crisis target is not fully satisfied."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        crisis_reduction = st.slider(
            "Crisis Reduction Target (%)",
            min_value=0,
            max_value=90,
            value=35,
            step=1
        )

    with c2:
        crisis_policy = st.selectbox(
            "Crisis Priority Mode",
            ["Manual User Priority", "Company Priority"],
            index=1
        )

    with c3:
        force_crisis = st.checkbox("Force crisis shedding now", value=True)

    with c4:
        min_service = st.checkbox("Minimum service preservation", value=True)

    st.subheader("Crisis Input Scenario")

    crisis_input_mode = st.radio(
        "Choose crisis test input",
        [
            "Use current Smart Meter Override Page table",
            "Built-in impossible crisis test: protected/zero controllable load",
            "Temporary crisis editor only for this page"
        ],
        index=0,
        help=(
            "Use the built-in impossible test if you specifically want to see the warning: "
            "Crisis target is not fully satisfied. More controllable load is needed."
        )
    )

    base_crisis_config = ensure_appliance_columns(st.session_state.appliance_config.copy())

    if crisis_input_mode == "Built-in impossible crisis test: protected/zero controllable load":
        crisis_input_df = base_crisis_config.copy()
        non_protected_mask = crisis_input_df["Load Category"] != "Critical / Never Disconnect"
        crisis_input_df.loc[non_protected_mask, "Quantity"] = 0
        crisis_input_df.loc[non_protected_mask, "Connected"] = False
        crisis_input_df.loc[non_protected_mask, "Preserve Minimum Units"] = 0
        crisis_input_df = apply_load_category_priority_rules(crisis_input_df)
        st.warning(
            "Impossible crisis test is active: non-protected controllable loads are set to zero/disconnected only inside this page. "
            "This should normally produce the NOT fully satisfied crisis warning."
        )

    elif crisis_input_mode == "Temporary crisis editor only for this page":
        st.info(
            "Edit this temporary crisis table to test edge cases. It does not overwrite the Smart Meter Override Page table."
        )
        crisis_input_df = st.data_editor(
            base_crisis_config,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Appliance": st.column_config.TextColumn("Appliance"),
                "Quantity": st.column_config.NumberColumn("Quantity", min_value=0, step=1),
                "Power per Unit kW": st.column_config.NumberColumn("Power per Unit kW", min_value=0.0, step=0.01),
                "Connected": st.column_config.CheckboxColumn("Connected"),
                "Load Category": st.column_config.SelectboxColumn("Load Category", options=LOAD_CATEGORY_OPTIONS),
                "Allow Company Emergency Control": st.column_config.CheckboxColumn("Allow Company Emergency Control"),
                "Preserve Minimum Units": st.column_config.NumberColumn("Preserve Minimum Units", min_value=0, step=1),
                "Protected / Never Disconnect": st.column_config.CheckboxColumn("Protected / Never Disconnect", disabled=True),
                "Disconnectable": st.column_config.CheckboxColumn("Disconnectable", disabled=True),
                "Critical": st.column_config.CheckboxColumn("Critical", disabled=True),
                "User Priority": st.column_config.NumberColumn("User Priority 1-10", min_value=1, max_value=10, step=1),
                "Company Priority": st.column_config.NumberColumn("Company Priority 1-10", min_value=1, max_value=10, step=1),
            }
        )
        crisis_input_df = apply_load_category_priority_rules(crisis_input_df)

    else:
        crisis_input_df = apply_load_category_priority_rules(base_crisis_config)
        st.success("Using the current Smart Meter Override Page appliance table.")

    crisis_df, crisis_original_kw, crisis_final_kw, crisis_achieved, crisis_msg = smart_meter_shed_load(
        appliance_df=crisis_input_df,
        requested_reduction_percent=crisis_reduction,
        policy_mode=crisis_policy,
        refuse_disconnect=False,
        climate_mode=st.session_state.climate_mode,
        mandatory_minimum_percent=crisis_reduction,
        user_failed_to_respond=True,
        enforcement_enabled=True,
        minimum_service_enabled=min_service,
        force_mode=force_crisis
    )

    crisis_required_shed_kw = crisis_original_kw * crisis_reduction / 100 if crisis_original_kw > 0 else 0
    crisis_actual_shed_kw = max(crisis_original_kw - crisis_final_kw, 0)

    # Feasibility estimate: run a 100% forced shedding pass on the same temporary crisis input.
    max_crisis_df, max_original_kw, max_final_kw, max_achieved, max_msg = smart_meter_shed_load(
        appliance_df=crisis_input_df,
        requested_reduction_percent=100,
        policy_mode=crisis_policy,
        refuse_disconnect=False,
        climate_mode=st.session_state.climate_mode,
        mandatory_minimum_percent=100,
        user_failed_to_respond=True,
        enforcement_enabled=True,
        minimum_service_enabled=min_service,
        force_mode=True
    )
    crisis_max_controllable_shed_kw = max(max_original_kw - max_final_kw, 0)

    crisis_tolerance_kw = 1e-6

    crisis_no_load = crisis_original_kw <= 0
    crisis_no_target = crisis_reduction <= 0
    crisis_physically_possible = crisis_max_controllable_shed_kw + crisis_tolerance_kw >= crisis_required_shed_kw
    crisis_target_met = crisis_actual_shed_kw + crisis_tolerance_kw >= crisis_required_shed_kw
    crisis_stable = (not crisis_no_load) and (crisis_no_target or (crisis_physically_possible and crisis_target_met))

    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Original Load", f"{crisis_original_kw:.2f} kW")
    cm2.metric("Required Shed", f"{crisis_required_shed_kw:.2f} kW")
    cm3.metric("Actual Shed", f"{crisis_actual_shed_kw:.2f} kW")
    cm4.metric("Achieved Reduction", f"{crisis_achieved:.2f}%")

    cm5, cm6, cm7 = st.columns(3)
    cm5.metric("Final Load", f"{crisis_final_kw:.2f} kW")
    cm6.metric("Max Controllable Shed", f"{crisis_max_controllable_shed_kw:.2f} kW")
    cm7.metric("Crisis Target", f"{crisis_reduction:.0f}%")

    if crisis_no_load:
        st.error("Crisis target is not fully satisfied. There is no active connected load to control.")
    elif crisis_no_target:
        st.info("No crisis reduction target was requested. Set the crisis target above 0% to test stability.")
    elif not crisis_physically_possible:
        st.warning(
            "Crisis target is not fully satisfied. More controllable load is needed. "
            f"Required shed is {crisis_required_shed_kw:.2f} kW, but maximum controllable shed is only {crisis_max_controllable_shed_kw:.2f} kW."
        )
    elif not crisis_target_met:
        st.warning(
            "Crisis target is not fully satisfied. The system had enough theoretical controllable load, "
            "but the selected shedding policy did not remove enough load. Check priorities, preservation rules, and force mode."
        )
    else:
        st.success(
            "Crisis target satisfied. Grid is stable in this simulation because actual shed kW reached the required crisis shed kW."
        )

    st.info(crisis_msg)

    st.subheader("Crisis Result Table")
    crisis_display_df = crisis_df.copy()
    if "Remaining Load kW" not in crisis_display_df.columns:
        crisis_display_df["Remaining Load kW"] = crisis_display_df["Remaining Units"] * crisis_display_df["Power per Unit kW"]
    st.dataframe(crisis_display_df, use_container_width=True)

    st.subheader("Crisis kW Feasibility Summary")
    crisis_summary_df = pd.DataFrame([
        {"Metric": "Original Load kW", "Value": crisis_original_kw},
        {"Metric": "Required Shed kW", "Value": crisis_required_shed_kw},
        {"Metric": "Actual Shed kW", "Value": crisis_actual_shed_kw},
        {"Metric": "Maximum Controllable Shed kW", "Value": crisis_max_controllable_shed_kw},
        {"Metric": "Final Load kW", "Value": crisis_final_kw},
    ])
    st.dataframe(crisis_summary_df, use_container_width=True)

    fig_crisis_kw = go.Figure()
    fig_crisis_kw.add_trace(go.Bar(
        x=["Required Shed", "Actual Shed", "Max Controllable Shed"],
        y=[crisis_required_shed_kw, crisis_actual_shed_kw, crisis_max_controllable_shed_kw],
        marker_color=["orange", "red" if not crisis_target_met else "lime", "deepskyblue"],
        text=[round(crisis_required_shed_kw, 2), round(crisis_actual_shed_kw, 2), round(crisis_max_controllable_shed_kw, 2)],
        textposition="auto"
    ))
    fig_crisis_kw.update_layout(
        title="Crisis Feasibility: Required vs Actual vs Maximum Controllable Shed",
        xaxis_title="Crisis Metric",
        yaxis_title="kW",
        template="plotly_dark",
        height=520,
        showlegend=False
    )
    st.plotly_chart(fig_crisis_kw, use_container_width=True)

    st.subheader("Unit Reduction Only")
    fig_crisis_units = go.Figure()
    fig_crisis_units.add_trace(go.Bar(
        x=crisis_display_df["Appliance"],
        y=crisis_display_df["Quantity"],
        name="Before Units",
        marker_color="deepskyblue",
        text=crisis_display_df["Quantity"],
        textposition="auto"
    ))
    fig_crisis_units.add_trace(go.Bar(
        x=crisis_display_df["Appliance"],
        y=crisis_display_df["Disconnected Units"],
        name="Disconnected Units",
        marker_color="red",
        text=crisis_display_df["Disconnected Units"],
        textposition="auto"
    ))
    fig_crisis_units.add_trace(go.Bar(
        x=crisis_display_df["Appliance"],
        y=crisis_display_df["Remaining Units"],
        name="Remaining Units",
        marker_color="lime",
        text=crisis_display_df["Remaining Units"],
        textposition="auto"
    ))
    fig_crisis_units.update_layout(
        title="Live Crisis Unit Reduction",
        barmode="group",
        template="plotly_dark",
        height=600,
        yaxis_title="Units"
    )
    st.plotly_chart(fig_crisis_units, use_container_width=True)

    st.subheader("Shed kW by Appliance")
    fig_shed_kw = px.bar(
        crisis_display_df,
        x="Appliance",
        y="Shed kW",
        title="Crisis Shed kW by Appliance",
        text_auto=".2f"
    )
    fig_shed_kw.update_layout(template="plotly_dark", height=520, yaxis_title="Shed kW")
    st.plotly_chart(fig_shed_kw, use_container_width=True)

    with st.expander("Why this crisis result happened"):
        st.write("Crisis stable:", crisis_stable)
        st.write("Original load kW:", round(crisis_original_kw, 4))
        st.write("Required shed kW:", round(crisis_required_shed_kw, 4))
        st.write("Actual shed kW:", round(crisis_actual_shed_kw, 4))
        st.write("Maximum controllable shed kW:", round(crisis_max_controllable_shed_kw, 4))
        st.write("Physically possible:", crisis_physically_possible)
        st.write("Target met:", crisis_target_met)
        st.write("Minimum service enabled:", min_service)
        st.write("Force crisis shedding now:", force_crisis)
        st.write("Crisis input mode:", crisis_input_mode)
        st.write("Tip: To force the warning, choose the built-in impossible crisis test or edit all non-protected controllable loads to quantity 0 / connected False.")

    st.stop()


# =========================================================
# SMART METER OVERRIDE PAGE
# =========================================================

if page == "Smart Meter Override Page":

    st.title("Smart Meter Override Page")

    st.info(
        "Priority direction: lower number means disconnect earlier. "
        "Higher number means preserve longer. Protected loads cannot be disconnected. Protected flag is used instead."
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.session_state.selected_user_policy = st.selectbox(
            "Priority Control Mode",
            [
                "Manual User Priority",
                "Company Priority"
            ],
            index=0 if st.session_state.selected_user_policy == "Manual User Priority" else 1
        )

    with col_b:
        st.session_state.refuse_disconnect = st.checkbox(
            "I do not want to disconnect anything",
            value=st.session_state.refuse_disconnect
        )

    with col_c:
        climate_options = [
            "Hot Summer - Cooling Priority",
            "Normal Operation",
            "Cold Winter - Heating Priority"
        ]

        st.session_state.climate_mode = st.selectbox(
            "Operating Climate",
            climate_options,
            index=climate_options.index(st.session_state.climate_mode)
            if st.session_state.climate_mode in climate_options else 0
        )

    ha, hb = st.columns(2)
    with ha:
        st.session_state.override_house_size = st.number_input(
            "Smart Meter House Area Size m²",
            min_value=1,
            value=int(st.session_state.override_house_size),
            step=1,
            help="Area is used by the company statistical baseline/fraud guard."
        )
    with hb:
        st.session_state.override_occupants = st.number_input(
            "Smart Meter Occupants",
            min_value=1,
            value=int(st.session_state.override_occupants),
            step=1,
            help="Occupants are used by the company statistical baseline/fraud guard."
        )

    override_household_df = pd.DataFrame([{
        "lamps": int(st.session_state.appliance_config.loc[st.session_state.appliance_config["Appliance"].eq("Lights"), "Quantity"].iloc[0]) if "Lights" in st.session_state.appliance_config["Appliance"].values else 0,
        "acs": int(st.session_state.appliance_config.loc[st.session_state.appliance_config["Appliance"].eq("ACs"), "Quantity"].iloc[0]) if "ACs" in st.session_state.appliance_config["Appliance"].values else 0,
        "washing_machine": int(st.session_state.appliance_config.loc[st.session_state.appliance_config["Appliance"].eq("Washing Machine"), "Quantity"].iloc[0]) if "Washing Machine" in st.session_state.appliance_config["Appliance"].values else 0,
        "heavy_machines": int(st.session_state.appliance_config.loc[st.session_state.appliance_config["Appliance"].eq("Heavy Machines"), "Quantity"].iloc[0]) if "Heavy Machines" in st.session_state.appliance_config["Appliance"].values else 0,
        "occupants": int(st.session_state.override_occupants),
        "house_size": int(st.session_state.override_house_size),
    }])
    override_expected_baseline = calculate_engineering_baseline(override_household_df)
    st.metric("Smart Meter Statistical Expected Baseline", f"{override_expected_baseline:.2f} kWh")

    st.subheader("Edit Appliance Status and Load Category")

    editable_df = ensure_appliance_columns(st.session_state.appliance_config)

    edited_df = st.data_editor(
        editable_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Appliance": st.column_config.TextColumn("Appliance"),
            "Quantity": st.column_config.NumberColumn("Quantity", min_value=0, step=1),
            "Power per Unit kW": st.column_config.NumberColumn("Power per Unit kW", min_value=0.0, step=0.01),
            "Connected": st.column_config.CheckboxColumn("Connected"),
            "Load Category": st.column_config.SelectboxColumn(
                "Load Category",
                options=LOAD_CATEGORY_OPTIONS,
                help="Choose only one category."
            ),
            "Allow Company Emergency Control": st.column_config.CheckboxColumn("Allow Company Emergency Control"),
            "Preserve Minimum Units": st.column_config.NumberColumn("Preserve Minimum Units", min_value=0, step=1),
            "Protected / Never Disconnect": st.column_config.CheckboxColumn("Protected / Never Disconnect", disabled=True),
            "Disconnectable": st.column_config.CheckboxColumn("Internal Disconnectable", disabled=True),
            "Critical": st.column_config.CheckboxColumn("Internal Critical", disabled=True),
            "User Priority": st.column_config.NumberColumn("User Priority 1-10", min_value=1, max_value=10, step=1),
            "Company Priority": st.column_config.NumberColumn("Company Priority 1-10", min_value=1, max_value=10, step=1)
        }
    )

    st.session_state.appliance_config = apply_load_category_priority_rules(edited_df)

    priority_explanation_df = add_priority_meaning_columns(st.session_state.appliance_config)

    st.subheader("Priority Meaning Summary")

    st.dataframe(
        priority_explanation_df[
            [
                "Appliance",
                "Load Category",
                "Allow Company Emergency Control",
                "Protected / Never Disconnect",
                "User Priority Meaning",
                "Company Priority Meaning"
            ]
        ],
        use_container_width=True
    )

    st.divider()

    st.subheader("Current Smart Meter Load Summary")

    load_df = calculate_current_connected_load(st.session_state.appliance_config)

    total_connected_kw = load_df["Connected Load kW"].sum()

    st.metric("Total Connected Load", f"{total_connected_kw:.2f} kW")

    st.dataframe(load_df, use_container_width=True)

    fig = px.bar(
        load_df,
        x="Appliance",
        y="Connected Load kW",
        color="Connected",
        title="Current Connected Load by Appliance",
        text_auto=".2f"
    )

    fig.update_layout(
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=25),
        height=560
    )

    st.plotly_chart(fig, use_container_width=True)

    st.stop()


# =========================================================
# SCADA CONTROL CENTER PAGE
# =========================================================

st.title("SCADA Dynamic Pricing & Smart Meter Control Center")

if st.session_state.get("company_force_fair_settlement", False):
    st.success(
        "GRID STABLE MODE: Last-resort fair settlement is ON. "
        "Peak-event warning is hidden because the system is forcing the line back to a fair stable state."
    )
else:
    st.warning(
        "⚠ PEAK EVENT NOTIFICATION: High electrical demand is active. "
        "The smart meter may request load reduction. Premium Load Preservation Pricing may apply."
    )

st.markdown("""
<div class="scada-card">
<div class="big-status">Integrated Operating Scenario</div>
This dashboard combines baseline calculation, dynamic tariffs, compatible fairness rules,
smart meter override, priority-based load shedding, forced fair settlement,
deadline timer, and premium uninterrupted consumption pricing.
</div>
""", unsafe_allow_html=True)


company_force_fair_settlement = st.session_state.get(
    "company_force_fair_settlement",
    False
)


# =========================================================
# FAIRNESS CONDITIONS
# =========================================================

st.header("SCADA Fairness & Protection Conditions")

fc1, fc2, fc3 = st.columns(3)

with fc1:
    condition_district_reduction = st.checkbox("District-wide load reduction", value=True)
    condition_historical_baseline = st.checkbox("Historical baseline protection", value=True)
    condition_marginal_premium = st.checkbox("Premium only above own baseline", value=True)
    condition_low_baseline_protection = st.checkbox("Low-baseline customer protection", value=True)

with fc2:
    condition_customer_autonomy = st.checkbox("Customer autonomy / manual override", value=True)
    condition_critical_load_protection = st.checkbox("Critical loads never disconnect", value=True)
    condition_minimum_service = st.checkbox("Minimum service preservation", value=True)
    condition_emergency_enforcement = st.checkbox("Physical-grid emergency enforcement", value=True)

with fc3:
    condition_grid_support_discount = st.checkbox("Grid support discount", value=True)
    condition_loyalty_discount = st.checkbox("Loyalty discount below baseline", value=True)
    condition_progressive_penalty = st.checkbox("Progressive penalty by shortfall", value=True)
    condition_growth_bonus = st.checkbox("Company growth bonus when grid is stable", value=False)
    st.checkbox(
        "Company force fair settlement for all users causing line stress",
        value=company_force_fair_settlement,
        key="company_force_fair_settlement",
        help="13th protection button: company last-resort control during real stress."
    )
    company_force_fair_settlement = st.session_state.get("company_force_fair_settlement", False)

fair_conditions = {
    "district_reduction": condition_district_reduction,
    "historical_baseline": condition_historical_baseline,
    "marginal_premium": condition_marginal_premium,
    "low_baseline_protection": condition_low_baseline_protection,
    "customer_autonomy": condition_customer_autonomy,
    "critical_load_protection": condition_critical_load_protection,
    "minimum_service": condition_minimum_service,
    "emergency_enforcement": condition_emergency_enforcement,
    "grid_support_discount": condition_grid_support_discount,
    "loyalty_discount": condition_loyalty_discount,
    "progressive_penalty": condition_progressive_penalty,
    "growth_bonus": condition_growth_bonus
}


# =========================================================
# GRID EVENT CONTROL
# =========================================================

st.header("Live Grid Event Status")

g1, g2, g3 = st.columns(3)

with g1:
    grid_stress = st.checkbox("Real Stress On Line", value=True)

with g2:
    peak_event = st.checkbox("Peak Usage Event", value=True)

with g3:
    new_company_growth_mode = st.checkbox("New Company Growth Mode - cheaper company discount", value=False)

enforcement_enabled = condition_emergency_enforcement

st.subheader("SCADA Reduction Commands")

c1, c2, c3 = st.columns(3)

with c1:
    voluntary_reduction_percent = st.slider(
        "Requested Reduction Signal From Company (%)",
        min_value=0,
        max_value=90,
        value=st.session_state.voluntary_reduction_percent,
        step=1
    )

with c2:
    mandatory_reduction_percent = st.slider(
        "Mandatory Minimum Reduction During Real Stress (%)",
        min_value=0,
        max_value=60,
        value=st.session_state.mandatory_reduction_percent,
        step=1
    )

with c3:
    response_deadline_minutes = st.number_input(
        "Response Deadline Before Enforcement (minutes)",
        min_value=0,
        value=60,
        step=1
    )

st.session_state.voluntary_reduction_percent = voluntary_reduction_percent
st.session_state.mandatory_reduction_percent = mandatory_reduction_percent

user_failed_to_respond = st.checkbox(
    "User ignored repeated company requests until deadline expired",
    value=False
)


# =========================================================
# HOUSEHOLD INPUTS
# =========================================================

st.divider()
st.header("Household Historical Baseline Inputs")

col1, col2 = st.columns(2)

with col1:
    person_a = household_input(
        "Person A: Low Baseline Home",
        default_lamps=3,
        default_acs=0,
        default_washing=0,
        default_heavy=0,
        default_occupants=1,
        default_size=60
    )

with col2:
    person_b = household_input(
        "Person B: Heavy Usage Home",
        default_lamps=20,
        default_acs=4,
        default_washing=4,
        default_heavy=5,
        default_occupants=5,
        default_size=220
    )

historical_baseline_a = predict_historical_baseline(model, person_a)
historical_baseline_b = predict_historical_baseline(model, person_b)
company_baseline_a_info = company_statistical_baseline(person_a, historical_baseline_a)
company_baseline_b_info = company_statistical_baseline(person_b, historical_baseline_b)
baseline_a = company_baseline_a_info["Company Approved Baseline kWh"]
baseline_b = company_baseline_b_info["Company Approved Baseline kWh"]


# =========================================================
# BASELINE METRICS
# =========================================================

st.divider()
st.header("Historical Consumption Baselines + Company Fraud Guard")

m1, m2, m3 = st.columns(3)
m1.metric("Company Approved Baseline - Person A", f"{baseline_a:.2f} kWh", delta=f"Historical {historical_baseline_a:.2f}")
m2.metric("Company Approved Baseline - Person B", f"{baseline_b:.2f} kWh", delta=f"Historical {historical_baseline_b:.2f}")
m3.metric("Population Mean Baseline", f"{mean_usage:.2f} kWh")

fraud_guard_df = pd.DataFrame([
    {"Client": "Person A", **company_baseline_a_info},
    {"Client": "Person B", **company_baseline_b_info},
])
st.subheader("Company Statistical Baseline / Fraud Detection")
st.info("Company-approved baseline protects the utility from inflated historical baselines by comparing history against house area, occupants, and declared appliance/device mix.")
st.dataframe(fraud_guard_df, use_container_width=True)

st.divider()
st.header("Static Compound Monthly Consumption Estimate")
st.info("Static case study: 8 villas, 3 apartment blocks, and 1 club house. Apartment blocks assume 12 apartments per block. Values are fixed for the compound scenario.")
compound_summary_df = build_static_compound_summary()
st.dataframe(compound_summary_df, use_container_width=True)
compound_total_kwh = float(compound_summary_df.loc[compound_summary_df["Compound Asset"] == "TOTAL COMPOUND", "Total Monthly kWh"].iloc[0])
compound_total_bill = float(compound_summary_df.loc[compound_summary_df["Compound Asset"] == "TOTAL COMPOUND", "Total Monthly Bill SAR"].iloc[0])
cm1, cm2 = st.columns(2)
cm1.metric("Compound Total Monthly Consumption", f"{compound_total_kwh:,.0f} kWh")
cm2.metric("Compound Total Monthly Bill", f"{compound_total_bill:,.2f} SAR")
fig_compound = px.bar(
    compound_summary_df[compound_summary_df["Compound Asset"] != "TOTAL COMPOUND"],
    x="Compound Asset",
    y="Total Monthly kWh",
    color="Category",
    title="Static Compound Monthly kWh by Asset",
    text_auto=".0f"
)
fig_compound.update_layout(template="plotly_dark", height=520)
st.plotly_chart(fig_compound, use_container_width=True)


# =========================================================
# PERSON SELECTION AND USAGE
# =========================================================

st.divider()
st.header("Client Selection and Peak Usage")

selected_person = st.radio(
    "Choose Client / Smart Meter for Detailed Smart Meter Result",
    ["Person A", "Person B"],
    horizontal=True
)

u1, u2 = st.columns(2)

with u1:
    requested_usage_a = st.number_input(
        "Person A Requested / Original Usage During Peak Event kWh",
        min_value=0.0,
        value=float(baseline_a),
        step=0.1
    )

with u2:
    requested_usage_b = st.number_input(
        "Person B Requested / Original Usage During Peak Event kWh",
        min_value=0.0,
        value=float(baseline_b + 3),
        step=0.1
    )

mda, mdb = st.columns(2)
with mda:
    st.session_state.person_a_medical_device = st.checkbox(
        "Person A has life-support medical device - charity discount",
        value=st.session_state.person_a_medical_device,
        help="Applies a charity discount and flags the customer as medically protected for billing review."
    )
with mdb:
    st.session_state.person_b_medical_device = st.checkbox(
        "Person B has life-support medical device - charity discount",
        value=st.session_state.person_b_medical_device,
        help="Applies a charity discount and flags the customer as medically protected for billing review."
    )

if selected_person == "Person A":
    selected_baseline = baseline_a
    requested_usage = requested_usage_a
    selected_household_df = person_a
else:
    selected_baseline = baseline_b
    requested_usage = requested_usage_b
    selected_household_df = person_b


# =========================================================
# FAIR SETTLEMENT BOTH USERS
# =========================================================

settlement_rows = []

for client_name, baseline_value, requested_value in [
    ("Person A", baseline_a, requested_usage_a),
    ("Person B", baseline_b, requested_usage_b)
]:
    surplus = max(requested_value - baseline_value, 0)

    if requested_value > 0:
        forced_percent = (surplus / requested_value) * 100
    else:
        forced_percent = 0

    settlement_rows.append({
        "Client": client_name,
        "Baseline kWh": baseline_value,
        "Requested kWh": requested_value,
        "Surplus Above Baseline kWh": surplus,
        "Forced Fair Reduction %": forced_percent,
        "Settlement Status": "High load - force reduction" if surplus > 0 else "Within baseline - no forced reduction"
    })

settlement_df = pd.DataFrame(settlement_rows)


# =========================================================
# TIMER AND FORCE SETTLEMENT LOGIC
# =========================================================

policy_mode = st.session_state.selected_user_policy
refuse_disconnect = st.session_state.refuse_disconnect
climate_mode = st.session_state.climate_mode

# One clean stress flag.
stress_active = grid_stress and peak_event

# Customer baseline relation.
selected_surplus = max(requested_usage - selected_baseline, 0)
customer_above_baseline = requested_usage > selected_baseline
customer_within_or_below_baseline = requested_usage <= selected_baseline

# Last-resort mode means the company emergency settlement button is ON during stress.
# This must disable the timer completely, even if the selected customer is within baseline.
last_resort_mode_active = (
    company_force_fair_settlement
    and stress_active
)

# Forced settlement applies only to customers who are above baseline.
# But timer blocking depends on last_resort_mode_active, not this variable.
forced_fair_settlement_active = (
    last_resort_mode_active
    and selected_surplus > 0
)

# TIMER RULE:
# Timer is allowed ONLY when:
# 1) stress is active
# 2) last-resort mode is OFF
# 3) selected customer is above baseline
#
# Timer must NOT run when:
# - last resort is ON
# - customer is within/below baseline
# - stress or peak is OFF
timer_should_run = (
    stress_active
    and not last_resort_mode_active
    and customer_above_baseline
)

timer_signature = (
    f"{selected_person}|"
    f"stress={stress_active}|"
    f"last_resort={last_resort_mode_active}|"
    f"above_baseline={customer_above_baseline}|"
    f"requested={round(requested_usage, 2)}|"
    f"baseline={round(selected_baseline, 2)}|"
    f"deadline={response_deadline_minutes}"
)

deadline_penalty_active, timer_status = deadline_timer_engine(
    timer_should_run=timer_should_run,
    deadline_minutes=response_deadline_minutes,
    force_settlement_active=last_resort_mode_active,
    timer_signature=timer_signature
)

# =========================================================
# FINAL TIMER PENALTY AUTHORITY
# =========================================================
now_for_timer_check = time.time()

if last_resort_mode_active:
    reset_deadline_timer(clear_penalty=True)
    deadline_penalty_active = False
    timer_status = "Timer disabled because last-resort fair settlement is active."
else:
    if (
        st.session_state.get("deadline_end_at") is not None
        and now_for_timer_check >= st.session_state.deadline_end_at
        and timer_should_run
    ):
        st.session_state.deadline_penalty_latched = True
        st.session_state.deadline_penalty_reason = "Timer expired. Penalty/enforcement is active."

    deadline_penalty_active = bool(st.session_state.get("deadline_penalty_latched", False))

    if deadline_penalty_active:
        timer_status = st.session_state.get(
            "deadline_penalty_reason",
            "Timer expired. Penalty/enforcement is active."
        )

if deadline_penalty_active:
    user_failed_to_respond = True



# =========================================================
# EFFECTIVE REDUCTION LOGIC
# =========================================================

effective_voluntary_reduction_percent = voluntary_reduction_percent
effective_mandatory_reduction_percent = mandatory_reduction_percent

forced_fair_settlement_reason = "Not active"

if (
    fair_conditions["low_baseline_protection"]
    and selected_baseline <= LOW_BASELINE_THRESHOLD_KWH
):
    effective_voluntary_reduction_percent = min(
        effective_voluntary_reduction_percent,
        LOW_BASELINE_MAX_REDUCTION_PERCENT
    )

    effective_mandatory_reduction_percent = min(
        effective_mandatory_reduction_percent,
        LOW_BASELINE_MAX_REDUCTION_PERCENT
    )

if not fair_conditions["district_reduction"]:
    effective_voluntary_reduction_percent = 0

if company_force_fair_settlement and grid_stress and peak_event:
    if selected_surplus <= 0:
        effective_voluntary_reduction_percent = 0
        effective_mandatory_reduction_percent = 0
        forced_fair_settlement_reason = (
            "Selected customer is within own historical baseline. No forced disconnection required."
        )
    else:
        forced_reduction_percent = (selected_surplus / max(requested_usage, 0.001)) * 100

        effective_voluntary_reduction_percent = max(
            effective_voluntary_reduction_percent,
            forced_reduction_percent
        )

        effective_mandatory_reduction_percent = max(
            effective_mandatory_reduction_percent,
            forced_reduction_percent
        )

        forced_fair_settlement_reason = (
            f"ACCESS DENIED ABOVE BASELINE. Selected customer exceeds own baseline by {selected_surplus:.2f} kWh. "
            f"Company forced fair reduction target is {forced_reduction_percent:.2f}%."
        )


# =========================================================
# SMART METER SIMULATION
# =========================================================

st.subheader("Smart Meter Data Source")

smart_meter_data_source = st.radio(
    "Choose which quantities the smart meter should use",
    [
        "Use selected Person A/B quantities + editable kW/priority table",
        "Use Smart Meter Override Page table exactly"
    ],
    index=0,
    horizontal=False,
    help=(
        "Recommended: use Person A/B quantities so the numbers entered in the SCADA Control Center "
        "become the smart-meter quantities, while kW, Load Category, User Priority, and Company Priority "
        "still come from the editable Smart Meter Override Page table. "
        "Choose the exact editable table only if you want to ignore Person A/B appliance counts completely."
    )
)

use_household_counts_in_smart_meter = (
    smart_meter_data_source == "Use selected Person A/B quantities + editable kW/priority table"
)

if use_household_counts_in_smart_meter:
    st.success(
        "Person A/B quantity mode is active: the graph will use the selected client's entered quantities "
        "for Lights, ACs, Washing Machine, and Heavy Machines. kW, categories, and priorities still come "
        "from the Smart Meter Override Page editable table."
    )
    simulation_appliance_config = apply_household_counts_to_appliance_config(
        st.session_state.appliance_config,
        selected_household_df
    )
else:
    st.warning(
        "Exact editable-table mode is active: Person A/B quantities are ignored. "
        "The graph will use only the quantities currently saved in the Smart Meter Override Page table."
    )
    simulation_appliance_config = st.session_state.appliance_config.copy()

simulation_appliance_config = apply_load_category_priority_rules(simulation_appliance_config)

# Hard validation: make sure selected Person quantities really reached the simulation table.
if use_household_counts_in_smart_meter:
    selected_counts_row = selected_household_df.iloc[0]
    expected_quantity_map = {
        "Lights": int(selected_counts_row["lamps"]),
        "ACs": int(selected_counts_row["acs"]),
        "Washing Machine": int(selected_counts_row["washing_machine"]),
        "Heavy Machines": int(selected_counts_row["heavy_machines"]),
    }

    quantity_mismatch_messages = []
    for appliance_name, expected_qty in expected_quantity_map.items():
        if appliance_name in simulation_appliance_config["Appliance"].values:
            actual_qty = int(
                simulation_appliance_config.loc[
                    simulation_appliance_config["Appliance"] == appliance_name,
                    "Quantity"
                ].iloc[0]
            )
            if actual_qty != expected_qty:
                quantity_mismatch_messages.append(
                    f"{appliance_name}: expected {expected_qty}, actual {actual_qty}"
                )

    if quantity_mismatch_messages:
        mismatch_text = "\n- ".join(quantity_mismatch_messages)
        st.error(
            "Quantity sync error detected before shedding:\n\n- " + mismatch_text
        )
    else:
        st.success(
            "Quantity sync check passed: selected Person A/B appliance counts are the quantities used by the smart meter."
        )

st.subheader("Actual Appliance Data Used By Smart Meter Before Shedding")
actual_config_view = simulation_appliance_config[
    [
        "Appliance",
        "Quantity",
        "Power per Unit kW",
        "Connected",
        "Load Category",
        "Preserve Minimum Units",
        "Protected / Never Disconnect",
        "Disconnectable",
        "Critical",
        "User Priority",
        "Company Priority"
    ]
].copy()
st.dataframe(actual_config_view, use_container_width=True)

# Equality diagnostic for Washing Machine and Heavy Machines.
wm_hm_debug = actual_config_view[
    actual_config_view["Appliance"].isin(["Washing Machine", "Heavy Machines"])
].copy()

if len(wm_hm_debug) == 2:
    wm_row = wm_hm_debug[wm_hm_debug["Appliance"] == "Washing Machine"].iloc[0]
    hm_row = wm_hm_debug[wm_hm_debug["Appliance"] == "Heavy Machines"].iloc[0]

    comparison_columns = [
        "Quantity",
        "Power per Unit kW",
        "Connected",
        "Load Category",
        "Preserve Minimum Units",
        "Protected / Never Disconnect",
        "Disconnectable",
        "Critical",
        "User Priority",
        "Company Priority"
    ]

    differences = []
    for col in comparison_columns:
        if str(wm_row[col]) != str(hm_row[col]):
            differences.append(f"{col}: Washing Machine={wm_row[col]} | Heavy Machines={hm_row[col]}")

    if differences:
        diff_text = "\n- ".join(differences)
        st.warning(
            "Washing Machine and Heavy Machines are NOT identical in the actual smart-meter input. "
            "This can explain different shedding behavior:\n\n- " + diff_text
        )
    else:
        st.success(
            "Washing Machine and Heavy Machines are identical in the actual smart-meter input. "
            "If both are selected in the same priority tier and exact electrical tie, the no-name-bias rotation logic will rotate between them."
        )

shed_df, original_load_kw, final_load_kw, achieved_reduction_percent, enforcement_status = smart_meter_shed_load(
    appliance_df=simulation_appliance_config,
    requested_reduction_percent=effective_voluntary_reduction_percent,
    policy_mode="Company Priority" if forced_fair_settlement_active else policy_mode,
    refuse_disconnect=False if forced_fair_settlement_active else refuse_disconnect,
    climate_mode=climate_mode,
    mandatory_minimum_percent=effective_mandatory_reduction_percent,
    user_failed_to_respond=True if forced_fair_settlement_active else user_failed_to_respond,
    enforcement_enabled=(
        forced_fair_settlement_active
        or (
            enforcement_enabled
            and grid_stress
            and peak_event
            and fair_conditions["emergency_enforcement"]
        )
    ),
    minimum_service_enabled=fair_conditions["minimum_service"],
    force_mode=forced_fair_settlement_active
)

if original_load_kw > 0:
    usage_ratio = final_load_kw / original_load_kw
else:
    usage_ratio = 1

final_usage_raw = requested_usage * usage_ratio

if forced_fair_settlement_active:
    final_usage = min(final_usage_raw, selected_baseline)
else:
    final_usage = final_usage_raw

good_behavior_discount_rate = min(
    st.session_state.good_behavior_streak * GOOD_BEHAVIOR_STEP_DISCOUNT,
    GOOD_BEHAVIOR_MAX_DISCOUNT
)

ignored_request_penalty_active = bool(
    user_failed_to_respond and stress_active and not last_resort_mode_active
)
if ignored_request_penalty_active:
    timer_status = "User ignored company request. Immediate delay penalty is active for this calculation."

billing = billing_engine(
    baseline=selected_baseline,
    requested_usage=requested_usage,
    final_usage=final_usage,
    mean_usage=mean_usage,
    grid_stress=grid_stress and peak_event,
    new_company_growth_mode=new_company_growth_mode,
    refused_disconnect=refuse_disconnect,
    achieved_reduction_percent=achieved_reduction_percent,
    mandatory_reduction_percent=effective_mandatory_reduction_percent,
    fair_conditions=fair_conditions,
    deadline_penalty_active=(deadline_penalty_active or ignored_request_penalty_active),
    forced_fair_settlement_active=forced_fair_settlement_active,
    good_behavior_discount_rate=good_behavior_discount_rate
)

selected_medical_device = (
    st.session_state.person_a_medical_device if selected_person == "Person A" else st.session_state.person_b_medical_device
)
medical_charity_discount = 0.0
if selected_medical_device:
    medical_charity_discount = billing["Final Bill"] * CHARITY_MEDICAL_DEVICE_DISCOUNT_RATE
    billing["Final Bill"] = max(billing["Final Bill"] - medical_charity_discount, 0)
    billing["Amount Saved"] += medical_charity_discount
    billing["Status"] += " | Medical life-support charity discount applied."

inactive_fairness_count = sum(1 for key, value in fair_conditions.items() if key != "growth_bonus" and not value)
fairness_config_adjustment = 0.0
if inactive_fairness_count > 0 and grid_stress and peak_event:
    fairness_config_adjustment = billing["Final Bill"] * FAIRNESS_INACTIVE_ADJUSTMENT_RATE * inactive_fairness_count
    billing["Final Bill"] += fairness_config_adjustment
    billing["Status"] += f" | Fairness configuration adjustment active because {inactive_fairness_count} protection condition(s) are unchecked."

ignored_event_id = f"{selected_person}|ignored={user_failed_to_respond}|stress={stress_active}|requested={round(requested_usage,2)}|baseline={round(selected_baseline,2)}"
if ignored_request_penalty_active and st.session_state.last_ignored_request_event_id != ignored_event_id:
    st.session_state.ignored_request_count_by_person[selected_person] = st.session_state.ignored_request_count_by_person.get(selected_person, 0) + 1
    st.session_state.last_ignored_request_event_id = ignored_event_id
ignored_request_count = st.session_state.ignored_request_count_by_person.get(selected_person, 0)
repeated_ignore_penalty = 0.0
if ignored_request_penalty_active and ignored_request_count > 1:
    repeated_ignore_penalty = billing["Final Bill"] * min(0.05 * (ignored_request_count - 1), 0.25)
    billing["Final Bill"] += repeated_ignore_penalty
    billing["Penalty"] += repeated_ignore_penalty
    billing["Status"] += f" | Repeated ignored-request penalty applied after {ignored_request_count} ignored event(s)."

grid_support_discount_applied = bool(
    fair_conditions["grid_support_discount"]
    and grid_stress
    and peak_event
    and effective_mandatory_reduction_percent > 0
    and achieved_reduction_percent >= effective_mandatory_reduction_percent
)

# =========================================================
# SAVE BEFORE-LAST-RESORT SNAPSHOT
# =========================================================
current_snapshot_id = (
    f"{selected_person}|"
    f"requested={round(requested_usage, 3)}|"
    f"baseline={round(selected_baseline, 3)}|"
    f"grid={grid_stress}|peak={peak_event}|"
    f"policy={policy_mode}|"
    f"data_source={smart_meter_data_source}|"
    f"appliance_signature="
    f"{simulation_appliance_config[['Appliance', 'Quantity', 'Power per Unit kW', 'Connected', 'Load Category', 'User Priority', 'Company Priority']].to_json()}"
)

if not last_resort_mode_active:
    st.session_state.last_resort_before_snapshot = {
        "snapshot_id": current_snapshot_id,
        "selected_person": selected_person,
        "requested_usage": float(requested_usage),
        "selected_baseline": float(selected_baseline),
        "original_load_kw": float(original_load_kw),
        "final_load_kw": float(final_load_kw),
        "achieved_reduction_percent": float(achieved_reduction_percent),
        "final_usage": float(final_usage),
        "final_bill": float(billing["Final Bill"]),
        "shed_df": shed_df.copy(),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.last_resort_before_snapshot_id = current_snapshot_id

# =========================================================
# REPEATED GOOD-BEHAVIOR TRACKING
# =========================================================

stress_active = grid_stress and peak_event

customer_above_baseline = requested_usage > selected_baseline
customer_within_baseline = requested_usage <= selected_baseline

previous_requested_usage = st.session_state.previous_requested_usage_by_person.get(
    selected_person,
    requested_usage
)

customer_manually_decreased_request = requested_usage < previous_requested_usage
customer_manually_increased_request = requested_usage > previous_requested_usage
customer_request_unchanged = requested_usage == previous_requested_usage

customer_supported_grid_enough = (
    achieved_reduction_percent >= effective_mandatory_reduction_percent
)

# Good behavior counts ONLY when the user voluntarily decreases requested usage.
voluntary_good_behavior = (
    stress_active
    and customer_manually_decreased_request
    and customer_within_baseline
    and customer_supported_grid_enough
    and not refuse_disconnect
    and not user_failed_to_respond
    and not deadline_penalty_active
    and not forced_fair_settlement_active
    and effective_mandatory_reduction_percent > 0
)

# Bad behavior resets streak.
bad_behavior_event = (
    stress_active
    and (
        customer_above_baseline
        or customer_manually_increased_request
        or refuse_disconnect
        or user_failed_to_respond
        or deadline_penalty_active
        or forced_fair_settlement_active
        or (
            effective_mandatory_reduction_percent > 0
            and achieved_reduction_percent < effective_mandatory_reduction_percent
        )
    )
)

neutral_behavior_event = (
    not stress_active
    or (
        stress_active
        and customer_within_baseline
        and customer_request_unchanged
        and not refuse_disconnect
        and not user_failed_to_respond
        and not deadline_penalty_active
        and not forced_fair_settlement_active
    )
)

behavior_event_id = (
    f"{selected_person}|"
    f"previous_requested={round(previous_requested_usage, 2)}|"
    f"current_requested={round(requested_usage, 2)}|"
    f"baseline={round(selected_baseline, 2)}|"
    f"stress={stress_active}|"
    f"mandatory={round(effective_mandatory_reduction_percent, 2)}|"
    f"refuse={refuse_disconnect}|"
    f"ignored={user_failed_to_respond}|"
    f"deadline={deadline_penalty_active}|"
    f"force={forced_fair_settlement_active}"
)

good_behavior_status_message = "No good-behavior event evaluated."

if behavior_event_id != st.session_state.last_good_behavior_counted_event:

    if voluntary_good_behavior:
        st.session_state.good_behavior_streak += 1
        st.session_state.last_good_behavior_counted_event = behavior_event_id
        good_behavior_status_message = (
            "Good behavior counted: customer voluntarily decreased requested usage, "
            "stayed within baseline, and supported the grid."
        )

    elif bad_behavior_event:
        st.session_state.good_behavior_streak = 0
        st.session_state.last_good_behavior_counted_event = behavior_event_id
        good_behavior_status_message = (
            "Good behavior streak reset: customer increased demand, exceeded baseline, "
            "refused, delayed, needed forced settlement, or failed mandatory support."
        )

    elif neutral_behavior_event:
        st.session_state.last_good_behavior_counted_event = behavior_event_id
        good_behavior_status_message = (
            "No streak change: customer stayed neutral. No voluntary decrease was detected."
        )

    else:
        st.session_state.last_good_behavior_counted_event = behavior_event_id
        good_behavior_status_message = (
            "No streak change: event did not qualify as good or bad behavior."
        )

else:
    good_behavior_status_message = (
        "Same behavior event already evaluated. Streak was not counted again."
    )

# Update previous requested usage AFTER evaluating behavior.
# This is very important.
st.session_state.previous_requested_usage_by_person[selected_person] = requested_usage

# Recalculate displayed good-behavior discount after streak update.
good_behavior_discount_rate = min(
    st.session_state.good_behavior_streak * GOOD_BEHAVIOR_STEP_DISCOUNT,
    GOOD_BEHAVIOR_MAX_DISCOUNT
)

additional_good_behavior_discount = 0.0
if good_behavior_discount_rate > 0 and grid_stress and peak_event and final_usage <= selected_baseline:
    target_good_behavior_discount = billing["Final Bill"] * good_behavior_discount_rate
    additional_good_behavior_discount = max(target_good_behavior_discount - billing["Good Behavior Discount"], 0)
    if additional_good_behavior_discount > 0:
        billing["Final Bill"] = max(billing["Final Bill"] - additional_good_behavior_discount, 0)
        billing["Good Behavior Discount"] += additional_good_behavior_discount
        billing["Discount"] += additional_good_behavior_discount
        billing["Amount Saved"] += additional_good_behavior_discount
        billing["Status"] += " | Good behavior streak discount applied after tracker update."

# =========================================================
# MAIN SCADA METRICS
# =========================================================

st.divider()
st.header("SCADA Live Status")

s1, s2, s3, s4, s5, s6 = st.columns(6)

s1.metric("Original Connected Load", f"{original_load_kw:.2f} kW")
s2.metric("Final Connected Load", f"{final_load_kw:.2f} kW")
s3.metric("Achieved Reduction", f"{achieved_reduction_percent:.2f}%")
s4.metric("Final Usage", f"{final_usage:.2f} kWh")
s5.metric("Final Bill", f"{billing['Final Bill']:.2f} SAR")
s6.metric("Amount Saved", f"{billing['Amount Saved']:.2f} SAR")

e1, e2 = st.columns(2)
e1.metric("Effective Voluntary Reduction", f"{effective_voluntary_reduction_percent:.2f}%")
e2.metric("Effective Mandatory Reduction", f"{effective_mandatory_reduction_percent:.2f}%")

deadline_penalty_active = bool(
    deadline_penalty_active or st.session_state.get("deadline_penalty_latched", False)
)

e3, e4, e5 = st.columns(3)
e3.metric("Forced Fair Settlement", "Active" if forced_fair_settlement_active else "Inactive")

if last_resort_mode_active:
    visible_timer_state = "Disabled by Last Resort"
elif deadline_penalty_active or ignored_request_penalty_active:
    visible_timer_state = "Expired / Ignored"
elif timer_should_run:
    visible_timer_state = "Running"
else:
    visible_timer_state = "Inactive"

e4.metric("Timer Status", visible_timer_state)
e5.metric("Timer Penalty", "Active" if (deadline_penalty_active or ignored_request_penalty_active) else "Inactive")

if st.session_state.get("deadline_penalty_latched", False):
    st.error("Deadline expired. Timer penalty/enforcement is latched active.")

    if st.button("Reset Timer Penalty / Start New Event"):
        reset_deadline_timer(clear_penalty=True)
        st.rerun()

st.metric("Repeated Good-Behavior Discount Rate", f"{good_behavior_discount_rate * 100:.0f}%")
st.info(good_behavior_status_message)

if forced_fair_settlement_active:
    st.error(forced_fair_settlement_reason)
    st.error("ACCESS DENIED: The customer cannot exceed the fair baseline limit during physical line stress.")
    st.success("GRID STABLE: Last-resort fair settlement reduced the above-baseline load. The peak event message is cleared for the stabilized line.")
elif company_force_fair_settlement and selected_surplus <= 0:
    st.success("Force settlement is ON, but this customer is within baseline. No forced action is needed.")

if last_resort_mode_active:
    st.success("Timer disabled: last-resort fair settlement mode is active.")

elif customer_within_or_below_baseline and stress_active:
    st.success("Timer inactive: selected customer is within/below own baseline.")

elif timer_should_run and not deadline_penalty_active:
    st.warning("Timer is running because the selected customer is above baseline during stress.")

if deadline_penalty_active or ignored_request_penalty_active:
    st.error("Deadline expired or company request ignored. Penalty/enforcement is now active.")

st.info(enforcement_status)


# =========================================================
# SMART METER LOAD TABLE
# =========================================================

st.divider()
st.header("Smart Meter Load Shedding Result")

st.dataframe(shed_df, use_container_width=True)

st.subheader("Before / After Appliance Quantities")
qty_view = shed_df[
    [
        "Appliance",
        "Load Category",
        "Quantity",
        "Disconnected Units",
        "Remaining Units",
        "Power per Unit kW",
        "Connected Load kW",
        "Remaining Load kW",
        "Shed kW",
        "Emergency Shed Rank",
        "Shedding Order",
        "Protected / Never Disconnect",
        "User Priority",
        "Company Priority"
    ]
].copy()

qty_view = qty_view.sort_values(
    by=["Emergency Shed Rank", "Shed kW", "Appliance"],
    ascending=[True, False, True]
)

st.dataframe(qty_view, use_container_width=True)

st.subheader("Unit Reduction Only")
fig_units = go.Figure()
fig_units.add_trace(go.Bar(
    x=qty_view["Appliance"],
    y=qty_view["Quantity"],
    name="Before Units",
    marker_color="deepskyblue",
    text=qty_view["Quantity"],
    textposition="auto"
))
fig_units.add_trace(go.Bar(
    x=qty_view["Appliance"],
    y=qty_view["Disconnected Units"],
    name="Disconnected Units",
    marker_color="red",
    text=qty_view["Disconnected Units"],
    textposition="auto"
))
fig_units.add_trace(go.Bar(
    x=qty_view["Appliance"],
    y=qty_view["Remaining Units"],
    name="Remaining Units",
    marker_color="lime",
    text=qty_view["Remaining Units"],
    textposition="auto"
))
fig_units.update_layout(
    title="Before, Disconnected, and Remaining Units",
    barmode="group",
    template="plotly_dark",
    height=560,
    yaxis_title="Units",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)
st.plotly_chart(fig_units, use_container_width=True)

st.subheader("Power Reduction Only")
kw_view = qty_view.copy()
fig_kw = go.Figure()
fig_kw.add_trace(go.Bar(
    x=kw_view["Appliance"],
    y=kw_view["Connected Load kW"],
    name="Original kW",
    marker_color="deepskyblue",
    text=kw_view["Connected Load kW"],
    textposition="auto"
))
fig_kw.add_trace(go.Bar(
    x=kw_view["Appliance"],
    y=kw_view["Shed kW"],
    name="Shed kW",
    marker_color="red",
    text=kw_view["Shed kW"],
    textposition="auto"
))
fig_kw.add_trace(go.Bar(
    x=kw_view["Appliance"],
    y=kw_view["Remaining Load kW"],
    name="Remaining kW",
    marker_color="lime",
    text=kw_view["Remaining Load kW"],
    textposition="auto"
))
fig_kw.update_layout(
    title="Original, Shed, and Remaining Load in kW",
    barmode="group",
    template="plotly_dark",
    height=560,
    yaxis_title="kW",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)
st.plotly_chart(fig_kw, use_container_width=True)

# =========================================================
# SETTLEMENT TABLE
# =========================================================

st.divider()
st.header("Fair Settlement Across Users")

st.dataframe(settlement_df, use_container_width=True)

if company_force_fair_settlement and grid_stress and peak_event:
    high_load_clients = settlement_df[
        settlement_df["Surplus Above Baseline kWh"] > 0
    ]["Client"].tolist()

    if len(high_load_clients) == 0:
        st.success("No user is above their own baseline. No forced settlement is required.")
    elif len(high_load_clients) == 1:
        st.warning(
            f"Only {high_load_clients[0]} is above baseline. Forced settlement applies only to that user."
        )
    else:
        st.error(
            "Both users are above baseline. Forced settlement applies fairly to both users according to their own surplus."
        )



if company_force_fair_settlement and grid_stress and peak_event:
    both_results = []

    for client_name, baseline_value, requested_value, household_df in [
        ("Person A", baseline_a, requested_usage_a, person_a),
        ("Person B", baseline_b, requested_usage_b, person_b)
    ]:
        surplus_value = max(requested_value - baseline_value, 0)
        forced_percent_value = (surplus_value / max(requested_value, 0.001)) * 100 if surplus_value > 0 else 0

        client_config = apply_household_counts_to_appliance_config(
            st.session_state.appliance_config,
            household_df
        )

        client_shed_df, client_original_kw, client_final_kw, client_achieved, client_msg = smart_meter_shed_load(
            appliance_df=client_config,
            requested_reduction_percent=forced_percent_value,
            policy_mode="Company Priority",
            refuse_disconnect=False,
            climate_mode=climate_mode,
            mandatory_minimum_percent=forced_percent_value,
            user_failed_to_respond=True,
            enforcement_enabled=True,
            minimum_service_enabled=fair_conditions["minimum_service"],
            force_mode=surplus_value > 0
        )

        for _, load_row in client_shed_df.iterrows():
            if load_row["Disconnected Units"] > 0 or load_row["Quantity"] > 0 or load_row["Connected Load kW"] > 0:
                both_results.append({
                    "Client": client_name,
                    "Appliance": load_row["Appliance"],
                    "Before Units": load_row["Quantity"],
                    "Disconnected Units": load_row["Disconnected Units"],
                    "After Units": load_row["Remaining Units"],
                    "Shed kW": load_row["Shed kW"],
                    "Effective Shed Priority": load_row.get("Effective Shed Priority", ""),
                    "Shed Explanation": load_row.get("Shed Explanation", ""),
                    "Status": "Forced" if surplus_value > 0 else "Protected within baseline"
                })

    both_settlement_df = pd.DataFrame(both_results)
    st.subheader("Last-Resort Fair Settlement Live Result For Both Users")
    st.dataframe(both_settlement_df, use_container_width=True)

    if not both_settlement_df.empty:
        fig_both = px.bar(
            both_settlement_df,
            x="Appliance",
            y="Disconnected Units",
            color="Client",
            barmode="group",
            title="Disconnected Units by Client Under Fair Settlement",
            text_auto=".0f"
        )
        fig_both.update_layout(template="plotly_dark", height=580)
        st.plotly_chart(fig_both, use_container_width=True)

# =========================================================
# CONDITION EVALUATION
# =========================================================

st.divider()
st.header("SCADA Fairness Condition Evaluation")

condition_rows = []

condition_rows.append({
    "Condition": "District-wide load reduction",
    "Active": fair_conditions["district_reduction"],
    "Rule": f"Utility requests {effective_voluntary_reduction_percent:.2f}% effective reduction",
    "Result": f"Achieved {achieved_reduction_percent:.2f}%",
    "Status": "Satisfied" if achieved_reduction_percent >= effective_voluntary_reduction_percent else "Not satisfied"
})

condition_rows.append({
    "Condition": "Historical baseline protection",
    "Active": fair_conditions["historical_baseline"],
    "Rule": "Customer is judged against own historical baseline",
    "Result": f"Final usage {final_usage:.2f} kWh vs baseline {selected_baseline:.2f} kWh",
    "Status": "Within own baseline" if final_usage <= selected_baseline else "Above own baseline"
})

condition_rows.append({
    "Condition": "Company forced fair settlement",
    "Active": company_force_fair_settlement,
    "Rule": "Users within baseline are protected; users above baseline are forced down immediately",
    "Result": forced_fair_settlement_reason,
    "Status": "Access denied above baseline" if forced_fair_settlement_active else "No forced reduction"
})

condition_rows.append({
    "Condition": "Response deadline timer",
    "Active": timer_should_run,
    "Rule": "Timer runs only when user delays/refuses and force settlement is OFF",
    "Result": timer_status,
    "Status": "Penalty active" if deadline_penalty_active else "No penalty from timer"
})

condition_rows.append({
    "Condition": "Priority direction",
    "Active": True,
    "Rule": "Lower number disconnects earlier; 6-10 disconnect late; Protected flag means never disconnect",
    "Result": "Heavy/luxury loads shed before protected lights",
    "Status": "Active"
})

condition_rows.append({
    "Condition": "Grid support discount",
    "Active": fair_conditions["grid_support_discount"],
    "Rule": f"Achieve at least {effective_mandatory_reduction_percent:.2f}% reduction",
    "Result": f"Achieved {achieved_reduction_percent:.2f}%",
    "Status": "Discount applied" if grid_support_discount_applied else "No discount"
})

condition_rows.append({
    "Condition": "Progressive penalty",
    "Active": fair_conditions["progressive_penalty"],
    "Rule": "Penalty increases according to reduction shortfall",
    "Result": f"Penalty {billing['Penalty']:.2f} SAR",
    "Status": "Applied" if billing["Penalty"] > 0 else "No penalty"
})

condition_rows.append({
    "Condition": "Medical device charity support",
    "Active": selected_medical_device,
    "Rule": f"{CHARITY_MEDICAL_DEVICE_DISCOUNT_RATE * 100:.0f}% charity discount for life-support medical device case",
    "Result": f"Discount {medical_charity_discount:.2f} SAR",
    "Status": "Discount applied" if medical_charity_discount > 0 else "No medical discount"
})

condition_rows.append({
    "Condition": "Ignored company request",
    "Active": ignored_request_penalty_active,
    "Rule": "Immediate delay penalty if user ignored company request during active stress",
    "Result": timer_status,
    "Status": "Penalty active" if ignored_request_penalty_active else "No penalty from ignored request"
})

condition_rows.append({
    "Condition": "Company fraud-guard baseline",
    "Active": True,
    "Rule": "Historical baseline is capped by statistical expected baseline from house size, occupants, and device mix",
    "Result": f"Approved baseline {selected_baseline:.2f} kWh instead of historical {(historical_baseline_a if selected_person == 'Person A' else historical_baseline_b):.2f} kWh",
    "Status": "Satisfied" if selected_baseline <= (historical_baseline_a if selected_person == "Person A" else historical_baseline_b) else "Not satisfied"
})

condition_df = pd.DataFrame(condition_rows)
st.dataframe(style_status_cells(condition_df), use_container_width=True)


# =========================================================
# BILLING DETAILS
# =========================================================

st.divider()
st.header("Billing & Condition Results")

billing_df = pd.DataFrame([{
    "Client": selected_person,
    "Baseline kWh": selected_baseline,
    "Requested Usage kWh": requested_usage,
    "Final Usage kWh": final_usage,
    "Normal Usage kWh": billing["Normal Usage kWh"],
    "Premium Usage kWh": billing["Premium Usage kWh"],
    "Saudi Energy Charge SAR": billing.get("Saudi Energy Charge SAR", 0),
        "Saudi Meter Fee SAR": billing.get("Saudi Meter Fee SAR", 0),
        "Saudi Tier 1 kWh": billing.get("Saudi Tier 1 kWh", 0),
        "Saudi Tier 2 kWh": billing.get("Saudi Tier 2 kWh", 0),
        "Premium Charge SAR": billing["Premium Charge"],
    "Penalty SAR": billing["Penalty"],
    "Timer Penalty SAR": billing["Timer Penalty"],
    "Penalty Waived SAR": billing["Penalty Waived"],
    "Bonus SAR": billing["Bonus"],
    "Discount SAR": billing["Discount"],
    "Loyalty Discount SAR": billing["Loyalty Discount"],
    "Good Behavior Discount SAR": billing["Good Behavior Discount"],
    "Medical Charity Discount SAR": medical_charity_discount,
    "Fairness Config Adjustment SAR": fairness_config_adjustment,
    "No Action Bill SAR": billing["No Action Bill"],
    "Amount Saved SAR": billing["Amount Saved"],
    "Final Bill SAR": billing["Final Bill"],
    "Condition Status": billing["Status"]
}])

st.dataframe(billing_df, use_container_width=True)


# =========================================================
# VISUAL ANALYTICS
# =========================================================

st.divider()
st.header("SCADA Visual Analytics")

tab1, tab2, tab3, tab4 = st.tabs([
    "Load Shedding Graph",
    "Usage & Billing Graph",
    "Baseline Bell Curve",
    "Priority Explanation"
])

with tab1:
    load_view = shed_df.copy()
    if "Remaining Load kW" not in load_view.columns:
        load_view["Remaining Load kW"] = load_view["Remaining Units"] * load_view["Power per Unit kW"]

    load_view = load_view.copy()
    load_view = load_view.sort_values(
        by=["Emergency Shed Rank", "Shed kW", "Appliance"] if "Emergency Shed Rank" in load_view.columns else ["Shed kW", "Appliance"],
        ascending=[True, False, True] if "Emergency Shed Rank" in load_view.columns else [False, True]
    )

    fig_load = go.Figure()
    fig_load.add_trace(go.Bar(
        x=load_view["Appliance"],
        y=load_view["Connected Load kW"],
        name="Original kW",
        marker_color="deepskyblue",
        text=load_view["Connected Load kW"].round(2),
        textposition="auto"
    ))
    fig_load.add_trace(go.Bar(
        x=load_view["Appliance"],
        y=load_view["Shed kW"],
        name="Shed kW",
        marker_color="red",
        text=load_view["Shed kW"].round(2),
        textposition="auto"
    ))
    fig_load.add_trace(go.Bar(
        x=load_view["Appliance"],
        y=load_view["Remaining Load kW"],
        name="Remaining kW",
        marker_color="lime",
        text=load_view["Remaining Load kW"].round(2),
        textposition="auto"
    ))
    fig_load.update_layout(
        title="Original, Shed, and Remaining Load by Appliance",
        xaxis_title="Appliance",
        yaxis_title="kW",
        barmode="group",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=26),
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_load, use_container_width=True)

with tab2:
    usage_fig = go.Figure()

    usage_fig.add_trace(go.Bar(
        x=["Baseline", "Requested Usage", "Final Usage"],
        y=[selected_baseline, requested_usage, final_usage],
        marker_color=["yellow", "orange", "lime"],
        text=[
            round(selected_baseline, 2),
            round(requested_usage, 2),
            round(final_usage, 2)
        ],
        textposition="auto"
    ))

    usage_fig.update_layout(
        title="Baseline vs Requested vs Final Usage",
        xaxis_title="Condition",
        yaxis_title="kWh",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=26),
        height=600
    )

    st.plotly_chart(usage_fig, use_container_width=True)

    bill_parts = pd.DataFrame({
        "Component": [
            "Premium Charge",
            "Penalty",
            "Timer Penalty",
            "Penalty Waived",
            "Bonus",
            "Discount",
            "Loyalty Discount",
            "Good Behavior Discount",
            "Amount Saved",
            "Final Bill"
        ],
        "SAR": [
            billing["Premium Charge"],
            billing["Penalty"],
            billing["Timer Penalty"],
            billing["Penalty Waived"],
            billing["Bonus"],
            billing["Discount"],
            billing["Loyalty Discount"],
            billing["Good Behavior Discount"],
            billing["Amount Saved"],
            billing["Final Bill"]
        ]
    })

    fig_bill = px.bar(
        bill_parts,
        x="Component",
        y="SAR",
        title="Bill Components",
        text_auto=".2f"
    )

    fig_bill.update_layout(
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=26),
        height=600
    )

    st.plotly_chart(fig_bill, use_container_width=True)

with tab3:
    x = np.linspace(
        training_df["historical_baseline_kwh"].min(),
        training_df["historical_baseline_kwh"].max(),
        600
    )

    y = norm.pdf(x, mean_usage, std_usage)
    user_y = norm.pdf(selected_baseline, mean_usage, std_usage)

    z_score = (selected_baseline - mean_usage) / std_usage
    percentile = norm.cdf(z_score) * 100

    fig_curve = go.Figure()

    fig_curve.add_trace(go.Histogram(
        x=training_df["historical_baseline_kwh"],
        histnorm="probability density",
        name="Population Histogram",
        opacity=0.45,
        marker_color="gray"
    ))

    fig_curve.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name="Normal Bell Curve",
        line=dict(width=5, color="cyan")
    ))

    fig_curve.add_trace(go.Scatter(
        x=[selected_baseline],
        y=[user_y],
        mode="markers+text",
        name=f"{selected_person} Position",
        marker=dict(size=20, color="red"),
        text=[f"{selected_person}"],
        textposition="top center"
    ))

    fig_curve.add_vline(
        x=mean_usage,
        line_width=4,
        line_dash="dash",
        line_color="yellow",
        annotation_text="Mean"
    )

    fig_curve.update_layout(
        title="Client Position on Historical Baseline Bell Curve",
        xaxis_title="Baseline Consumption kWh",
        yaxis_title="Probability Density",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=26),
        height=650
    )

    st.plotly_chart(fig_curve, use_container_width=True)

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Z-Score", f"{z_score:.2f}")
    cc2.metric("Percentile", f"{percentile:.2f}%")
    cc3.metric("Population Mean", f"{mean_usage:.2f} kWh")

with tab4:
    priority_df = add_priority_meaning_columns(simulation_appliance_config.copy())

    st.info(
        "Lower priority number means disconnect earlier. "
        "Higher priority number means preserve longer. Protected means never disconnected. Protected flag is used instead."
    )

    st.dataframe(
        priority_df[
            [
                "Appliance",
                "Load Category",
                "Protected / Never Disconnect",
                "User Priority",
                "User Priority Meaning",
                "Company Priority",
                "Company Priority Meaning"
            ]
        ],
        use_container_width=True
    )

    fig_priority = go.Figure()

    fig_priority.add_trace(go.Bar(
        x=priority_df["Appliance"],
        y=priority_df["User Priority"],
        name="User Priority",
        marker_color="lime",
        text=priority_df["User Priority Meaning"],
        textposition="auto"
    ))

    fig_priority.add_trace(go.Bar(
        x=priority_df["Appliance"],
        y=priority_df["Company Priority"],
        name="Company Priority",
        marker_color="orange",
        text=priority_df["Company Priority Meaning"],
        textposition="auto"
    ))

    fig_priority.update_layout(
        title="Readable Priority Explanation",
        xaxis_title="Appliance",
        yaxis_title="Priority Number: Lower Means Disconnect First",
        barmode="group",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=26),
        height=650
    )

    st.plotly_chart(fig_priority, use_container_width=True)


# =========================================================
# LAST RESORT FORCE SETTLEMENT CONTROL
# =========================================================

st.divider()
st.header("Last Resort Company Fair Settlement Control")

st.warning(
    "This is the final emergency fairness control. "
    "When enabled, the company overrides user refusal and premium payment during real stress. "
    "Users within their own baseline are protected. Users above baseline are denied access above the fair limit."
)

st.info("The Last Resort company fair-settlement button is now included as the 13th checkbox under SCADA Fairness & Protection Conditions.")

if company_force_fair_settlement:
    st.error(
        "Forced fair settlement is ON. Timer is disabled. Company has upper hand to protect physical line stress."
    )
else:
    st.info(
        "Forced fair settlement is OFF. User refusal or pay-more behavior can trigger the response deadline timer."
    )


# =========================================================
# SIMPLE LAST RESORT MODE COMPARISON
# =========================================================
# VERY IMPORTANT EXPLANATION:
# This is NOT a physical timeline where the load goes:
#     Before value  -> press Last Resort -> After value
# Instead, this section compares TWO SEPARATE SIMULATION MODES using the same input scenario:
#     1) Normal Mode Result: saved while Last Resort was OFF.
#     2) Last Resort Mode Result: recalculated after Last Resort is ON.
#
# Therefore, Last Resort Mode can sometimes show a HIGHER connected kW than Normal Mode.
# That does not mean Last Resort added power. It means Normal Mode shed more load than the
# fair-settlement mode needed. The main fairness target is Final Usage kWh compared with
# the Historical Baseline kWh, not making connected kW exactly equal to the baseline.
if last_resort_mode_active:
    st.subheader("Normal Mode vs Last Resort Mode Result")

    st.info(
        "Important: this section compares two separate calculations, not a continuous physical timeline. "
        "Normal Mode is the saved result from when Last Resort was OFF. Last Resort Mode is the recalculated result "
        "after the company fair-settlement control is ON. If Last Resort Mode has a higher kW than Normal Mode, "
        "it means Normal Mode disconnected more load than Last Resort needed; it does not mean Last Resort added load."
    )

    before_snapshot = st.session_state.get("last_resort_before_snapshot", None)

    if before_snapshot is None:
        st.warning(
            "No saved Normal Mode result yet. Turn Last Resort OFF once, let the current scenario run, then turn Last Resort ON."
        )
    else:
        if before_snapshot.get("snapshot_id", "") != current_snapshot_id:
            st.warning(
                "The saved Normal Mode result may belong to an older scenario. For the cleanest result, turn Last Resort OFF once "
                "with the current inputs, then turn it ON again."
            )

        normal_mode_load_kw = float(before_snapshot["final_load_kw"])
        last_resort_load_kw = float(final_load_kw)
        normal_mode_usage_kwh = float(before_snapshot["final_usage"])
        last_resort_usage_kwh = float(final_usage)
        baseline_kwh = float(selected_baseline)

        load_difference_kw = last_resort_load_kw - normal_mode_load_kw
        usage_difference_kwh = last_resort_usage_kwh - normal_mode_usage_kwh

        if load_difference_kw > 0:
            load_interpretation = (
                f"Last Resort Mode preserved {load_difference_kw:.2f} kW more connected load than Normal Mode. "
                "This means Normal Mode shed more load than the fair-settlement mode needed."
            )
            load_status_color = "info"
        elif load_difference_kw < 0:
            load_interpretation = (
                f"Last Resort Mode removed {abs(load_difference_kw):.2f} kW more connected load than Normal Mode. "
                "This means Last Resort had to force extra reduction."
            )
            load_status_color = "warning"
        else:
            load_interpretation = "Both modes ended with the same connected load."
            load_status_color = "success"

        if last_resort_usage_kwh <= baseline_kwh:
            usage_interpretation = (
                f"Last Resort fairness target is satisfied: final usage is {last_resort_usage_kwh:.2f} kWh, "
                f"which is at or below the baseline of {baseline_kwh:.2f} kWh."
            )
        else:
            usage_interpretation = (
                f"Last Resort final usage is still above baseline: {last_resort_usage_kwh:.2f} kWh vs baseline {baseline_kwh:.2f} kWh. "
                "This means more controllable load may be required."
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Historical Baseline", f"{baseline_kwh:.2f} kWh")
        c2.metric("Normal Mode Final Usage", f"{normal_mode_usage_kwh:.2f} kWh")
        c3.metric("Last Resort Mode Final Usage", f"{last_resort_usage_kwh:.2f} kWh")

        c4, c5, c6 = st.columns(3)
        c4.metric("Normal Mode Final Load", f"{normal_mode_load_kw:.2f} kW")
        c5.metric("Last Resort Mode Final Load", f"{last_resort_load_kw:.2f} kW")
        c6.metric("Mode Difference", f"{load_difference_kw:+.2f} kW")

        if load_status_color == "warning":
            st.warning(load_interpretation)
        elif load_status_color == "success":
            st.success(load_interpretation)
        else:
            st.info(load_interpretation)

        # Do not use a one-line conditional expression here.
        # Streamlit can display the returned DeltaGenerator object on the page.
        # A normal if/else block shows only the alert message and avoids the long
        # DeltaGenerator(...) text that appeared in the UI.
        if last_resort_usage_kwh <= baseline_kwh:
            st.success(usage_interpretation)
        else:
            st.warning(usage_interpretation)

        st.markdown(
            f"""
            **How to read this result:**

            - **Normal Mode Final Load** = original connected load minus what the normal shedding mode disconnected.
            - **Last Resort Mode Final Load** = original connected load minus what the fair-settlement mode disconnected.
            - These two values are **two separate mode results**, not step-by-step physical continuation.
            - The main Last Resort fairness check is:  
              **Last Resort Final Usage ({last_resort_usage_kwh:.2f} kWh) ≤ Historical Baseline ({baseline_kwh:.2f} kWh)**.
            """
        )

        simple_last_resort_df = pd.DataFrame({
            "State": ["Historical Baseline", "Normal Mode Final Usage", "Last Resort Mode Final Usage"],
            "kWh": [baseline_kwh, normal_mode_usage_kwh, last_resort_usage_kwh]
        })

        fig_usage_comparison = go.Figure()
        fig_usage_comparison.add_trace(go.Bar(
            x=simple_last_resort_df["State"],
            y=simple_last_resort_df["kWh"],
            name="Usage compared with baseline",
            marker_color=["yellow", "deepskyblue", "lime"],
            text=simple_last_resort_df["kWh"].round(2),
            textposition="auto"
        ))
        fig_usage_comparison.update_layout(
            title="Last Resort Fairness Check: Baseline vs Normal Mode vs Last Resort Mode",
            xaxis_title="Scenario Result",
            yaxis_title="Energy Usage (kWh)",
            template="plotly_dark",
            height=520,
            showlegend=False
        )
        st.plotly_chart(fig_usage_comparison, use_container_width=True)

        with st.expander("Optional: connected-load comparison and appliance details"):
            st.info(
                "Connected load is shown in kW. Baseline is shown in kWh. They are related but not the same unit. "
                "Use this optional section only to understand which mode shed more connected appliance power."
            )

            load_mode_df = pd.DataFrame({
                "Mode": ["Normal Mode", "Last Resort Mode"],
                "Final Connected Load kW": [normal_mode_load_kw, last_resort_load_kw]
            })
            fig_load_mode = go.Figure()
            fig_load_mode.add_trace(go.Bar(
                x=load_mode_df["Mode"],
                y=load_mode_df["Final Connected Load kW"],
                marker_color=["deepskyblue", "lime"],
                text=load_mode_df["Final Connected Load kW"].round(2),
                textposition="auto"
            ))
            fig_load_mode.update_layout(
                title="Optional: Connected Load Comparison Between Modes",
                xaxis_title="Mode",
                yaxis_title="Connected Load (kW)",
                template="plotly_dark",
                height=450,
                showlegend=False
            )
            st.plotly_chart(fig_load_mode, use_container_width=True)

            before_shed_df = before_snapshot["shed_df"].copy()
            after_shed_df = shed_df.copy()

            before_compare = before_shed_df[["Appliance", "Disconnected Units", "Shed kW"]].copy().rename(columns={
                "Disconnected Units": "Normal Mode Disconnected Units",
                "Shed kW": "Normal Mode Shed kW"
            })
            after_compare = after_shed_df[["Appliance", "Disconnected Units", "Shed kW"]].copy().rename(columns={
                "Disconnected Units": "Last Resort Mode Disconnected Units",
                "Shed kW": "Last Resort Mode Shed kW"
            })

            appliance_change_df = pd.merge(before_compare, after_compare, on="Appliance", how="outer").fillna(0)
            appliance_change_df["More Shed by Last Resort kW"] = (
                appliance_change_df["Last Resort Mode Shed kW"] - appliance_change_df["Normal Mode Shed kW"]
            ).clip(lower=0).round(2)
            appliance_change_df["Less Shed by Last Resort kW"] = (
                appliance_change_df["Normal Mode Shed kW"] - appliance_change_df["Last Resort Mode Shed kW"]
            ).clip(lower=0).round(2)
            appliance_change_df["More Disconnected by Last Resort Units"] = (
                appliance_change_df["Last Resort Mode Disconnected Units"] - appliance_change_df["Normal Mode Disconnected Units"]
            ).clip(lower=0).round(2)
            appliance_change_df["Less Disconnected by Last Resort Units"] = (
                appliance_change_df["Normal Mode Disconnected Units"] - appliance_change_df["Last Resort Mode Disconnected Units"]
            ).clip(lower=0).round(2)

            changed_only_df = appliance_change_df[
                (appliance_change_df["More Shed by Last Resort kW"] > 0) |
                (appliance_change_df["Less Shed by Last Resort kW"] > 0) |
                (appliance_change_df["More Disconnected by Last Resort Units"] > 0) |
                (appliance_change_df["Less Disconnected by Last Resort Units"] > 0)
            ].copy()

            if changed_only_df.empty:
                st.info("No appliance-level difference was detected between Normal Mode and Last Resort Mode.")
            else:
                st.dataframe(changed_only_df, use_container_width=True)
                st.caption(
                    "Less Shed by Last Resort does not mean negative power. It means Last Resort chose to shed that appliance less than Normal Mode."
                )

        if st.button("Clear saved Normal Mode result"):
            st.session_state.last_resort_before_snapshot = None
            st.session_state.last_resort_before_snapshot_id = None
            st.rerun()

# =========================================================
# FINAL SYSTEM SUMMARY
# =========================================================

st.divider()
st.header("Final SCADA Decision Summary")

if forced_fair_settlement_active:
    st.error(
        "Final Decision: ACCESS DENIED ABOVE BASELINE. "
        "The customer exceeded their own historical baseline during real line stress. "
        "Company forced reduction was applied using company priority, overriding refusal and premium payment."
    )
elif deadline_penalty_active or ignored_request_penalty_active:
    st.error(
        "Final Decision: The response deadline expired or the user ignored the company request. Penalty/enforcement is active while stress remains."
    )
elif grid_stress and peak_event and achieved_reduction_percent < effective_mandatory_reduction_percent:
    st.error(
        "Final Decision: The user did not satisfy the effective minimum physical grid protection requirement."
    )
elif refuse_disconnect and grid_stress and peak_event:
    st.warning(
        "Final Decision: User preserved comfort and refused disconnection. "
        "Premium pricing applies, but timer or emergency enforcement may still activate."
    )
elif achieved_reduction_percent >= effective_mandatory_reduction_percent and grid_stress and peak_event:
    st.success(
        "Final Decision: User supported the grid by reducing enough load. "
        "Discount or positive reliability score can be applied."
    )
else:
    st.info(
        "Final Decision: Normal dynamic pricing mode. No emergency grid protection action required."
    )
