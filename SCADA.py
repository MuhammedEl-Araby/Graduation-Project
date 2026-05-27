
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
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

    # Priority is now only 1 to 10. Protected flag is used instead or displayed.
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
# HYBRID BASELINE CALCULATION
# =========================================================

def calculate_engineering_baseline(input_df):
    row = input_df.iloc[0].astype(float)

    engineering_baseline = (
        0.10 * row["lamps"] +
        1.15 * row["acs"] +
        0.90 * row["washing_machine"] +
        1.55 * row["heavy_machines"] +
        0.32 * row["occupants"] +
        0.010 * row["house_size"]
    )

    return max(engineering_baseline, 0.5)


def predict_historical_baseline(model, input_df):
    ml_prediction = float(model.predict(input_df.astype(float))[0])
    engineering_prediction = calculate_engineering_baseline(input_df)

    row = input_df.iloc[0].astype(float)

    extreme_score = 0

    if row["lamps"] > 300:
        extreme_score += 1
    if row["acs"] > 80:
        extreme_score += 1
    if row["washing_machine"] > 60:
        extreme_score += 1
    if row["heavy_machines"] > 80:
        extreme_score += 1
    if row["occupants"] > 150:
        extreme_score += 1
    if row["house_size"] > 6000:
        extreme_score += 1

    if extreme_score == 0:
        final_prediction = 0.60 * ml_prediction + 0.40 * engineering_prediction
    elif extreme_score <= 2:
        final_prediction = 0.30 * ml_prediction + 0.70 * engineering_prediction
    else:
        final_prediction = engineering_prediction

    return max(final_prediction, 0.5)


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
    - Critical loads are protected by flags, not by 10.
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
    df["Emergency Shed Rank"] = 999
    df["Effective Shed Priority"] = 999
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
            return 999
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

    no_action_bill = (
        min(requested_usage, baseline) * BASE_RATE +
        max(requested_usage - baseline, 0) * (PREMIUM_PRESERVATION_RATE if grid_stress else BASE_RATE)
    )

    bill = normal_usage * BASE_RATE

    bonus = 0
    penalty = 0
    timer_penalty = 0
    premium_charge = 0
    discount = 0
    loyalty_discount = 0
    good_behavior_discount = 0
    penalty_waived = 0
    status = []

    if (
        fair_conditions["growth_bonus"]
        and new_company_growth_mode
        and not grid_stress
    ):
        if final_usage > mean_usage:
            bonus = bill * BONUS_RATE
            bill -= bonus
            status.append("Growth bonus applied because the company wants to increase average demand.")
        else:
            status.append("Growth mode active, but usage is still below desired growth level.")

    if grid_stress:
        if fair_conditions["marginal_premium"]:
            if premium_usage > 0:
                premium_charge = premium_usage * PREMIUM_PRESERVATION_RATE
                bill += premium_charge
                status.append("Premium pricing applied only to usage above the customer's own historical baseline.")
        else:
            if final_usage > baseline:
                premium_charge = final_usage * 0.30
                bill += premium_charge
                status.append("General premium pricing applied because marginal baseline protection is disabled.")

    penalty_should_apply = (
        grid_stress
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
            penalty = final_usage * 0.20 * shortfall_ratio
            status.append("Progressive penalty applied based on reduction shortfall.")
        else:
            penalty = final_usage * 0.20
            status.append("Flat grid stress penalty applied because mandatory reduction was not achieved.")

        bill += penalty

    if deadline_penalty_active and not forced_fair_settlement_active:
        timer_penalty = final_usage * 0.25
        bill += timer_penalty
        status.append("Deadline penalty applied because the user refused or delayed response after the timer expired.")

    if (
        fair_conditions["grid_support_discount"]
        and grid_stress
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
        status.append("Loyalty discount applied because usage stayed at or below historical baseline during peak stress.")

    if good_behavior_discount_rate > 0 and grid_stress and final_usage <= baseline:
        good_behavior_discount = bill * good_behavior_discount_rate
        bill -= good_behavior_discount
        status.append(f"Repeated good-behavior discount applied: {good_behavior_discount_rate * 100:.0f}%.")

    if refused_disconnect and grid_stress and not forced_fair_settlement_active:
        if fair_conditions["customer_autonomy"]:
            status.append("User used manual override. Timer, premium pricing, or emergency enforcement may apply.")

    if forced_fair_settlement_active:
        status.append("Forced fair settlement stabilized the line. Timer and delay penalty are disabled.")

    if not status:
        status.append("Normal billing condition.")

    final_bill = max(bill, 0)
    amount_saved = max(no_action_bill - final_bill, 0)

    return {
        "Normal Usage kWh": normal_usage,
        "Premium Usage kWh": premium_usage,
        "Premium Charge": premium_charge,
        "Penalty": penalty,
        "Timer Penalty": timer_penalty,
        "Penalty Waived": penalty_waived,
        "Bonus": bonus,
        "Discount": discount,
        "Loyalty Discount": loyalty_discount,
        "Good Behavior Discount": good_behavior_discount,
        "No Action Bill": no_action_bill,
        "Amount Saved": amount_saved,
        "Final Bill": final_bill,
        "Status": " | ".join(status)
    }


# =========================================================
# TIMER ENGINE
# =========================================================

def reset_deadline_timer():
    st.session_state.deadline_started_at = None
    st.session_state.deadline_signature = None


def reset_deadline_timer(clear_penalty=False):
    st.session_state.deadline_started_at = None
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
    now = time.time()

    if timer_signature is None:
        timer_signature = "default_timer_signature"

    # Last resort must completely disable timer and clear timer penalty.
    if force_settlement_active:
        reset_deadline_timer(clear_penalty=True)
        return False, "Timer disabled because last-resort fair settlement is active."

    # If penalty already expired before, keep it active.
    if st.session_state.deadline_penalty_latched:
        return True, st.session_state.deadline_penalty_reason or "Timer already expired. Penalty/enforcement is active."

    # If timer should not run, reset timer countdown but do not create penalty.
    if not timer_should_run:
        reset_deadline_timer(clear_penalty=False)
        return False, "Timer inactive."

    # New timer event.
    if st.session_state.deadline_signature != timer_signature:
        st.session_state.deadline_signature = timer_signature
        st.session_state.deadline_started_at = now

    if st.session_state.deadline_started_at is None:
        st.session_state.deadline_started_at = now

    deadline_seconds = max(int(deadline_minutes * 60), 1)
    elapsed = now - st.session_state.deadline_started_at
    remaining = max(deadline_seconds - elapsed, 0)

    expired = remaining <= 0

    # If Python detects expiry, latch it permanently.
    if expired:
        st.session_state.deadline_penalty_latched = True
        st.session_state.deadline_penalty_reason = (
            "Timer expired. Penalty/enforcement is active."
        )
        return True, st.session_state.deadline_penalty_reason

    end_timestamp_ms = int(
        (st.session_state.deadline_started_at + deadline_seconds) * 1000
    )

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

            const text =
                minutes.toString().padStart(2, '0') +
                ":" +
                seconds.toString().padStart(2, '0');

            document.getElementById("deadline_timer").innerText = text;

            if (diff <= 0) {{
                document.getElementById("deadline_timer").innerText =
                    "EXPIRED - Penalty/Enforcement Active";
                document.getElementById("deadline_timer").style.color = "red";

                // Wait a little so user can see EXPIRED, then rerun Streamlit.
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

    st.divider()

    st.header("Tariff Catalogue")
    st.write(f"Normal rate: **{BASE_RATE} EGP/kWh**")
    st.write(f"Peak rate: **{PEAK_RATE} EGP/kWh**")
    st.write(f"Penalty rate: **{PENALTY_RATE} EGP/kWh**")
    st.write(f"Premium preservation rate: **{PREMIUM_PRESERVATION_RATE} EGP/kWh**")
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
- A user above their own baseline is denied access above the fair limit.
- Recommended mode uses Person A/B quantities together with editable Smart Meter kW, categories, and priorities.
- Exact editable-table mode ignores Person A/B quantities and uses the Smart Meter Override Page table exactly.
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
This page shows direct priority shedding without the full billing section.
It respects protected loads, editable User Priority, editable Company Priority, and minimum service.
</div>
    """, unsafe_allow_html=True)

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

    crisis_df, crisis_original_kw, crisis_final_kw, crisis_achieved, crisis_msg = smart_meter_shed_load(
        appliance_df=st.session_state.appliance_config,
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

    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("Original kW", f"{crisis_original_kw:.2f}")
    cm2.metric("Final kW", f"{crisis_final_kw:.2f}")
    cm3.metric("Achieved Reduction", f"{crisis_achieved:.2f}%")

    if crisis_achieved >= crisis_reduction:
        st.success("Crisis target satisfied. Grid is stable in this simulation.")
    else:
        st.warning("Crisis target is not fully satisfied. More controllable load is needed.")

    st.info(crisis_msg)
    st.dataframe(crisis_df, use_container_width=True)

    fig_crisis_units = go.Figure()
    fig_crisis_units.add_trace(go.Bar(
        x=crisis_df["Appliance"],
        y=crisis_df["Quantity"],
        name="Before Units",
        marker_color="deepskyblue"
    ))
    fig_crisis_units.add_trace(go.Bar(
        x=crisis_df["Appliance"],
        y=crisis_df["Remaining Units"],
        name="After Units",
        marker_color="lime"
    ))
    fig_crisis_units.update_layout(
        title="Live Crisis Unit Reduction",
        barmode="group",
        template="plotly_dark",
        height=600
    )
    st.plotly_chart(fig_crisis_units, use_container_width=True)

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

st.header("Grid Event & Company Control Panel")

g1, g2, g3, g4 = st.columns(4)

with g1:
    grid_stress = st.checkbox("Real Stress On Line", value=True)

with g2:
    peak_event = st.checkbox("Peak Usage Event", value=True)

with g3:
    new_company_growth_mode = st.checkbox("New Company Growth Mode", value=False)

with g4:
    enforcement_enabled = st.checkbox("Emergency Enforcement Enabled", value=True)

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

baseline_a = predict_historical_baseline(model, person_a)
baseline_b = predict_historical_baseline(model, person_b)


# =========================================================
# BASELINE METRICS
# =========================================================

st.divider()
st.header("Historical Consumption Baselines")

m1, m2, m3 = st.columns(3)

m1.metric("Historical Baseline - Person A", f"{baseline_a:.2f} kWh")
m2.metric("Historical Baseline - Person B", f"{baseline_b:.2f} kWh")
m3.metric("Population Mean Baseline", f"{mean_usage:.2f} kWh")


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
    deadline_penalty_active=deadline_penalty_active,
    forced_fair_settlement_active=forced_fair_settlement_active,
    good_behavior_discount_rate=good_behavior_discount_rate
)

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
s5.metric("Final Bill", f"{billing['Final Bill']:.2f} EGP")
s6.metric("Amount Saved", f"{billing['Amount Saved']:.2f} EGP")

e1, e2 = st.columns(2)
e1.metric("Effective Voluntary Reduction", f"{effective_voluntary_reduction_percent:.2f}%")
e2.metric("Effective Mandatory Reduction", f"{effective_mandatory_reduction_percent:.2f}%")

e3, e4 = st.columns(2)
e3.metric("Forced Fair Settlement", "Active" if forced_fair_settlement_active else "Inactive")
e4.metric("Timer Penalty", "Active" if deadline_penalty_active else "Inactive")

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

if deadline_penalty_active:
    st.error("Deadline expired. Penalty/enforcement is now active.")

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
kw_view = qty_view[qty_view["Connected Load kW"] > 0].copy()
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
            if load_row["Disconnected Units"] > 0 or load_row["Appliance"] in ["Lights", "Washing Machine", "Heavy Machines", "ACs"]:
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
    "Status": "Discount applied" if billing["Discount"] > 0 else "No discount"
})

condition_rows.append({
    "Condition": "Progressive penalty",
    "Active": fair_conditions["progressive_penalty"],
    "Rule": "Penalty increases according to reduction shortfall",
    "Result": f"Penalty {billing['Penalty']:.2f} EGP",
    "Status": "Applied" if billing["Penalty"] > 0 else "No penalty"
})

condition_df = pd.DataFrame(condition_rows)
st.dataframe(condition_df, use_container_width=True)


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
    "Premium Charge EGP": billing["Premium Charge"],
    "Penalty EGP": billing["Penalty"],
    "Timer Penalty EGP": billing["Timer Penalty"],
    "Penalty Waived EGP": billing["Penalty Waived"],
    "Bonus EGP": billing["Bonus"],
    "Discount EGP": billing["Discount"],
    "Loyalty Discount EGP": billing["Loyalty Discount"],
    "Good Behavior Discount EGP": billing["Good Behavior Discount"],
    "No Action Bill EGP": billing["No Action Bill"],
    "Amount Saved EGP": billing["Amount Saved"],
    "Final Bill EGP": billing["Final Bill"],
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

    load_view = load_view[load_view["Connected Load kW"] > 0].copy()
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
        "EGP": [
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
        y="EGP",
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

st.checkbox(
    "Company force fair settlement for all users causing line stress",
    value=company_force_fair_settlement,
    key="company_force_fair_settlement",
    help=(
        "If enabled, users who stay within their own baseline are not disconnected. "
        "Users who exceed their fair baseline during real stress are forced to reduce immediately, "
        "even if premium payment or refusal is selected."
    )
)

if company_force_fair_settlement:
    st.error(
        "Forced fair settlement is ON. Timer is disabled. Company has upper hand to protect physical line stress."
    )
else:
    st.info(
        "Forced fair settlement is OFF. User refusal or pay-more behavior can trigger the response deadline timer."
    )


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
elif deadline_penalty_active:
    st.error(
        "Final Decision: The response deadline expired. Penalty/enforcement is active because the user delayed or refused while stress remained."
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
