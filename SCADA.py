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

.small-muted {{
opacity: 0.82;
font-size: 14px;
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

LOW_BASELINE_THRESHOLD_KWH = 2.0
LOW_BASELINE_MAX_REDUCTION_PERCENT = 10


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
            "Shed First": False,
            "Comfort Load": False,
            "Luxury Load": False,
            "Critical Load": True,
            "Allow Company Emergency Control": False,
            "Preserve Minimum Units": 10,
            "Disconnectable": False,
            "Critical": True,
            "User Priority": 999,
            "Company Priority": 999
        },
        {
            "Appliance": "Power Sockets",
            "Quantity": 8,
            "Power per Unit kW": 0.15,
            "Connected": True,
            "Shed First": True,
            "Comfort Load": False,
            "Luxury Load": False,
            "Critical Load": False,
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
            "Shed First": False,
            "Comfort Load": True,
            "Luxury Load": False,
            "Critical Load": False,
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
            "Shed First": True,
            "Comfort Load": False,
            "Luxury Load": True,
            "Critical Load": False,
            "Allow Company Emergency Control": True,
            "Preserve Minimum Units": 0,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 1,
            "Company Priority": 3
        },
        {
            "Appliance": "Washing Machine",
            "Quantity": 1,
            "Power per Unit kW": 1.0,
            "Connected": True,
            "Shed First": False,
            "Comfort Load": False,
            "Luxury Load": True,
            "Critical Load": False,
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
            "Shed First": False,
            "Comfort Load": True,
            "Luxury Load": False,
            "Critical Load": False,
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
            "Shed First": True,
            "Comfort Load": False,
            "Luxury Load": True,
            "Critical Load": False,
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

if "ac_overlay_positions" not in st.session_state:
    # Corrected for uploaded ACs.png.
    # Image size: 847 x 658 px.
    # x/y = center of red AC symbol.
    # label_x/label_y = position of yellow AC number label.
    st.session_state.ac_overlay_positions = {
        "AC 1": {"x": 819, "y": 444, "label_x": 727, "label_y": 424},
        "AC 2": {"x": 818, "y": 174, "label_x": 730, "label_y": 255},
        "AC 3": {"x": 20,  "y": 408, "label_x": 79,  "label_y": 481},
        "AC 4": {"x": 20,  "y": 546, "label_x": 78,  "label_y": 618},
        "AC 5": {"x": 584, "y": 125, "label_x": 493, "label_y": 207},
        "AC 6": {"x": 132, "y": 112, "label_x": 213, "label_y": 194},
    }


# =========================================================
# DATAFRAME SAFETY
# =========================================================

def ensure_appliance_columns(df):
    df = df.copy()

    defaults = {
        "Appliance": "Unknown",
        "Quantity": 0,
        "Power per Unit kW": 0.0,
        "Connected": True,
        "Shed First": False,
        "Comfort Load": False,
        "Luxury Load": False,
        "Critical Load": False,
        "Allow Company Emergency Control": True,
        "Preserve Minimum Units": 0,
        "Disconnectable": True,
        "Critical": False,
        "User Priority": 4,
        "Company Priority": 4
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

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
# HYBRID BASELINE CALCULATION - NO LIMITS
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
# HOUSEHOLD INTERPRETATION - NO LIMITS
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

    total_equipment = lamps + acs + washing + heavy

    if total_equipment / max(size, 1) > 0.5:
        messages.append(
            "Overall equipment density is high. This is treated as a valid operating scenario, not an input error."
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

    pattern_messages = describe_household_pattern(
        lamps=lamps,
        acs=acs,
        washing=washing,
        heavy=heavy,
        occupants=occupants,
        size=size
    )

    for msg in pattern_messages:
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
# CHECKBOX PRIORITY LOGIC
# =========================================================

def apply_checkbox_priority_rules(appliance_df):
    df = ensure_appliance_columns(appliance_df)

    for idx, row in df.iterrows():
        critical_load = bool(row["Critical Load"])
        shed_first = bool(row["Shed First"])
        comfort_load = bool(row["Comfort Load"])
        luxury_load = bool(row["Luxury Load"])
        allow_company = bool(row["Allow Company Emergency Control"])

        if critical_load:
            df.loc[idx, "Critical"] = True
            df.loc[idx, "Disconnectable"] = False
            df.loc[idx, "User Priority"] = 999
            df.loc[idx, "Company Priority"] = 999

        elif shed_first:
            df.loc[idx, "Critical"] = False
            df.loc[idx, "Disconnectable"] = True
            df.loc[idx, "User Priority"] = 1
            df.loc[idx, "Company Priority"] = 1 if allow_company else 7

        elif luxury_load:
            df.loc[idx, "Critical"] = False
            df.loc[idx, "Disconnectable"] = True
            df.loc[idx, "User Priority"] = 2
            df.loc[idx, "Company Priority"] = 2 if allow_company else 8

        elif comfort_load:
            df.loc[idx, "Critical"] = False
            df.loc[idx, "Disconnectable"] = True
            df.loc[idx, "User Priority"] = 5
            df.loc[idx, "Company Priority"] = 3 if allow_company else 9

        else:
            df.loc[idx, "Critical"] = False
            df.loc[idx, "Disconnectable"] = True
            df.loc[idx, "User Priority"] = 4
            df.loc[idx, "Company Priority"] = 4 if allow_company else 10

    return df


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
            "Shed First": False,
            "Comfort Load": True,
            "Luxury Load": False,
            "Critical Load": False,
            "Allow Company Emergency Control": True,
            "Preserve Minimum Units": 0,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 5,
            "Company Priority": 3
        }

    st.session_state.appliance_config = apply_checkbox_priority_rules(df)


def smart_meter_shed_load(
    appliance_df,
    requested_reduction_percent,
    policy_mode,
    refuse_disconnect,
    climate_mode,
    mandatory_minimum_percent,
    user_failed_to_respond,
    enforcement_enabled,
    minimum_service_enabled
):
    df = calculate_current_connected_load(appliance_df)

    original_load = df["Connected Load kW"].sum()

    if original_load <= 0:
        df["Disconnected Units"] = 0
        df["Remaining Units"] = 0
        df["Shed kW"] = 0
        return df, 0, 0, 0, "No active load"

    requested_reduction_kw = original_load * requested_reduction_percent / 100
    mandatory_reduction_kw = original_load * mandatory_minimum_percent / 100

    if refuse_disconnect:
        if enforcement_enabled and user_failed_to_respond:
            target_reduction_kw = mandatory_reduction_kw
            active_policy = "Company Emergency Enforcement"
            enforcement_status = "User refused or ignored request. Mandatory reduction was enforced."
        else:
            target_reduction_kw = 0
            active_policy = "User Refused Disconnection"
            enforcement_status = "No load was disconnected. Premium pricing applied."
    else:
        target_reduction_kw = requested_reduction_kw
        active_policy = policy_mode
        enforcement_status = "User smart meter priority was applied."

    if target_reduction_kw <= 0:
        df["Disconnected Units"] = 0
        df["Remaining Units"] = df["Quantity"]
        df["Shed kW"] = 0.0
        final_load = original_load
        achieved_reduction_percent = 0
        return df, original_load, final_load, achieved_reduction_percent, enforcement_status

    df["Disconnected Units"] = 0
    df["Remaining Units"] = df["Quantity"]
    df["Shed kW"] = 0.0

    if active_policy == "Company Emergency Enforcement":
        priority_col = "Company Priority"
    elif policy_mode == "Company Priority":
        priority_col = "Company Priority"
    else:
        priority_col = "User Priority"

    shed_so_far = 0.0

    candidates = df[
        (df["Connected"] == True) &
        (df["Disconnectable"] == True) &
        (df["Critical"] == False)
    ].copy()

    candidates = candidates.sort_values(by=priority_col, ascending=True)

    for idx, row in candidates.iterrows():
        if shed_so_far >= target_reduction_kw:
            break

        quantity = float(row["Quantity"])
        power = float(row["Power per Unit kW"])
        appliance = row["Appliance"]
        preserve_minimum = float(row["Preserve Minimum Units"])

        if minimum_service_enabled:
            if climate_mode == "Hot Summer - Cooling Priority" and appliance == "ACs":
                preserve_minimum = max(preserve_minimum, 1)

            if climate_mode == "Cold Winter - Heating Priority" and appliance == "Water Heater":
                preserve_minimum = max(preserve_minimum, 1)

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

        shed_so_far += shed_kw

    final_load = max(original_load - shed_so_far, 0)
    achieved_reduction_percent = (shed_so_far / original_load) * 100

    return df, original_load, final_load, achieved_reduction_percent, enforcement_status


# =========================================================
# BILLING ENGINE WITH COMPATIBLE FAIRNESS CONDITIONS
# =========================================================

def billing_engine(
    baseline,
    original_usage,
    final_usage,
    mean_usage,
    grid_stress,
    new_company_growth_mode,
    refused_disconnect,
    achieved_reduction_percent,
    mandatory_reduction_percent,
    fair_conditions
):
    premium_usage = max(final_usage - baseline, 0)
    normal_usage = min(final_usage, baseline)

    bill = normal_usage * BASE_RATE

    bonus = 0
    penalty = 0
    premium_charge = 0
    discount = 0
    loyalty_discount = 0
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

    if refused_disconnect and grid_stress:
        if fair_conditions["customer_autonomy"]:
            status.append("User used manual override. Premium pricing or emergency enforcement may apply depending on physical grid stress.")
        else:
            status.append("User refused disconnection, but customer autonomy condition is disabled.")

    if not status:
        status.append("Normal billing condition.")

    return {
        "Normal Usage kWh": normal_usage,
        "Premium Usage kWh": premium_usage,
        "Premium Charge": premium_charge,
        "Penalty": penalty,
        "Penalty Waived": penalty_waived,
        "Bonus": bonus,
        "Discount": discount,
        "Loyalty Discount": loyalty_discount,
        "Final Bill": bill,
        "Status": " | ".join(status)
    }


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


def render_ac_plan_overlay(image_path, ac_states, positions):
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)

    symbol_font = get_font(96)
    label_font = get_font(28)

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
            outline_width=9
        )

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

    st.divider()

    if os.path.exists("Dr.jpg"):
        st.image("Dr.jpg")

    st.header("Dynamic Pricing Simulator")
    st.write("Supervised by: Dr. Alaa Hamam")


# =========================================================
# HOW TO USE PAGE
# =========================================================

if page == "How To Use":

    st.title("How To Use The Website")

    st.markdown("""
<div class="manual-box">

## 1. Website Purpose

This website is a training and research simulator for a smart electrical distribution system.

It combines:

- Historical consumption baseline calculation
- Dynamic pricing
- Peak event simulation
- Smart meter override
- User priority control
- Company emergency control
- Load shedding
- Real-life HVAC visual simulation
- Billing, penalty, premium, discount, and loyalty-discount calculation

The system is not a real utility SCADA system. It is a safe simulation that demonstrates how a company could manage load during high demand or physical line stress.

</div>

<div class="manual-box">

## 2. Fairness Conditions

The fairness conditions can work together.

You can activate:

- District-wide reduction
- Historical baseline protection
- Premium only above own baseline
- Low-baseline customer protection
- Customer autonomy
- Critical load protection
- Minimum service preservation
- Emergency enforcement
- Grid support discount
- Loyalty discount
- Progressive penalty
- Growth bonus

This makes the simulation closer to a real smart-grid policy engine.

</div>

<div class="manual-box">

## 3. Smart Meter Override Page

Use checkboxes instead of priority numbers.

- **Shed First** means disconnect early.
- **Comfort Load** means preserve if possible.
- **Luxury Load** means shed before comfort load.
- **Critical Load** means never disconnect.
- **Allow Company Emergency Control** means company can use this appliance during physical emergency.
- **Preserve Minimum Units** protects a minimum number of units.

</div>

<div class="manual-box">

## 4. Real Life Simulation

Place your HVAC image in the app folder with the exact name:

**ACs.png**

The page overlays:

- Green **O** for working AC
- Red **X** for disconnected AC

The number of working ACs is synchronized with the AC row in the smart meter table.

</div>

<div class="manual-box">

## 5. No Equipment Limits

The simulator does not limit lamps, ACs, washing machines, heavy machines, occupants, or house size.

Every number entered by the user is included in the calculation.

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

The dataset is synthetic and generated inside the program for training and demonstration.

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

## Random Forest Regressor

The model used here is a Random Forest Regressor.

A random forest is a group of many decision trees. Each tree gives a prediction, then the model averages the trees to produce a final result.

</div>

<div class="manual-box">

## Hybrid Baseline Logic

The simulator does not depend only on Random Forest.

Random Forest can be weak outside the training range, so this version uses:

- Random Forest prediction
- Engineering-style calculation

This guarantees that every entered unit is counted.

</div>

<div class="manual-box">

## MAE

MAE means Mean Absolute Error.

It tells the average prediction error in kWh. Lower MAE is better.

</div>

<div class="manual-box">

## R²

R² means Coefficient of Determination.

- Close to 1.00 means strong model fit
- Close to 0.00 means weak model fit
- Negative means worse than using the average

</div>

<div class="manual-box">

## Gaussian Randomness

Gaussian randomness means normal-distribution noise.

It is added because real electrical consumption is not perfectly fixed.

</div>

<div class="manual-box">

## Confidential Baseline Idea

In a real system, the customer-facing name should be:

**Historical Consumption Baseline**

or

**Customer Historical Consumption Profile**

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

    st.warning(
        "This page simulates a real HVAC plan. Put your plan image in the project folder with the exact name ACs.png."
    )

    st.markdown("""
<div class="scada-card">
<div class="big-status">HVAC Plan Live Overlay</div>
Use the controls below to turn each AC ON or OFF. The image updates immediately.
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

    st.divider()

    image_path = "ACs.png"

    if not os.path.exists(image_path):
        st.error(
            "ACs.png was not found. Add ACs.png to the same folder as this Streamlit file, then rerun the app."
        )
        st.stop()

    with st.expander("Edit X/O and AC Number Positions on the Plan"):
        st.info(
            "Default coordinates are adjusted for the uploaded image size 847 x 658 px. "
            "Adjust x/y and label_x/label_y if needed."
        )

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
                "label_x": st.column_config.NumberColumn("AC Number Label X", step=1),
                "label_y": st.column_config.NumberColumn("AC Number Label Y", step=1)
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
        positions=st.session_state.ac_overlay_positions
    )

    center_left, center_main, center_right = st.columns([0.03, 0.94, 0.03])

    with center_main:
        st.image(
            overlay_image,
            caption="Live HVAC SCADA Overlay",
            use_container_width=True
        )

    st.subheader("Real Life Simulation Status Table")

    status_df = pd.DataFrame([
        {
            "AC Unit": ac_name,
            "State": "Working" if state else "Disconnected",
            "Symbol": "O" if state else "X"
        }
        for ac_name, state in st.session_state.ac_unit_states.items()
    ])

    st.dataframe(status_df, use_container_width=True)

    st.success(
        "The AC status is synchronized with the ACs row in the Smart Meter Override Page."
    )

    st.stop()


# =========================================================
# SMART METER OVERRIDE PAGE
# =========================================================

if page == "Smart Meter Override Page":

    st.title("Smart Meter Override Page")

    st.warning(
        "This page lets the client manually override smart meter priority rules. "
        "The user's choices affect the SCADA Control Center simulation."
    )

    st.markdown("""
    ## Manual Control Philosophy

    You can decide:

    - Which appliance is connected or disconnected
    - Which appliance should shed first
    - Which appliance is comfort load
    - Which appliance is luxury load
    - Which appliance must never disconnect
    - Whether the company can use the appliance during emergency enforcement
    - Whether the user refuses all automatic disconnections
    """)

    st.divider()

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

    st.info(
        "The table uses checkboxes. Internal priority numbers are calculated automatically."
    )

    st.subheader("Edit Appliance Status and SCADA Conditions")

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
            "Shed First": st.column_config.CheckboxColumn("Shed First"),
            "Comfort Load": st.column_config.CheckboxColumn("Comfort Load"),
            "Luxury Load": st.column_config.CheckboxColumn("Luxury Load"),
            "Critical Load": st.column_config.CheckboxColumn("Critical / Never Disconnect"),
            "Allow Company Emergency Control": st.column_config.CheckboxColumn("Allow Company Emergency Control"),
            "Preserve Minimum Units": st.column_config.NumberColumn("Preserve Minimum Units", min_value=0, step=1),
            "Disconnectable": st.column_config.CheckboxColumn("Internal Disconnectable", disabled=True),
            "Critical": st.column_config.CheckboxColumn("Internal Critical", disabled=True),
            "User Priority": st.column_config.NumberColumn("Internal User Priority", disabled=True),
            "Company Priority": st.column_config.NumberColumn("Internal Company Priority", disabled=True)
        }
    )

    category_conflict_rows = []

    for _, row in edited_df.iterrows():
        active_categories = (
            int(bool(row["Shed First"])) +
            int(bool(row["Comfort Load"])) +
            int(bool(row["Luxury Load"])) +
            int(bool(row["Critical Load"]))
        )

        if active_categories > 1:
            category_conflict_rows.append(row["Appliance"])

    if category_conflict_rows:
        st.warning(
            "Some appliances have more than one category selected: "
            + ", ".join(category_conflict_rows)
            + ". Priority will be resolved in this order: Critical > Shed First > Luxury > Comfort."
        )

    st.session_state.appliance_config = apply_checkbox_priority_rules(edited_df)

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

    st.success(
        "Your smart meter override configuration has been saved in the session. "
        "Go to SCADA Control Center to see its effect."
    )

    st.stop()


# =========================================================
# SCADA CONTROL CENTER PAGE
# =========================================================

st.title("SCADA Dynamic Pricing & Smart Meter Control Center")

st.warning(
    "⚠ PEAK EVENT NOTIFICATION: High electrical demand is active. "
    "The smart meter may request load reduction. Premium Load Preservation Pricing may apply."
)

st.markdown("""
<div class="scada-card">
<div class="big-status">Integrated Operating Scenario</div>
This dashboard combines historical baseline calculation, dynamic tariffs, compatible fairness conditions,
manual smart meter override, priority-based load shedding, mandatory grid protection,
customer autonomy, and premium uninterrupted consumption pricing.
</div>
""", unsafe_allow_html=True)


# =========================================================
# FAIRNESS CONDITION ENGINE
# =========================================================

st.header("SCADA Fairness & Protection Conditions")

st.markdown("""
<div class="scada-card">
<div class="big-status">Compatible Fairness Conditions</div>
These conditions are not exclusive. You can activate several of them together to make the SCADA decision more fair and realistic.
</div>
""", unsafe_allow_html=True)

fc1, fc2, fc3 = st.columns(3)

with fc1:
    condition_district_reduction = st.checkbox(
        "District-wide load reduction",
        value=True,
        help="Utility requests a reduction percentage from all entities during high district load."
    )

    condition_historical_baseline = st.checkbox(
        "Historical baseline protection",
        value=True,
        help="Each customer is compared to their own historical baseline, not to neighbors."
    )

    condition_marginal_premium = st.checkbox(
        "Premium only above own baseline",
        value=True,
        help="Premium pricing applies only to usage above the customer's own baseline."
    )

    condition_low_baseline_protection = st.checkbox(
        "Low-baseline customer protection",
        value=True,
        help="Protects very low users from unfair shutdown or excessive penalty."
    )

with fc2:
    condition_customer_autonomy = st.checkbox(
        "Customer autonomy / manual override",
        value=True,
        help="Customer can choose appliance priorities and can refuse automatic shedding."
    )

    condition_critical_load_protection = st.checkbox(
        "Critical loads never disconnect",
        value=True,
        help="Critical loads such as lighting, refrigerators, or medical equipment are protected."
    )

    condition_minimum_service = st.checkbox(
        "Minimum service preservation",
        value=True,
        help="Preserves minimum units such as at least one AC in hot weather or one heater in cold weather."
    )

    condition_emergency_enforcement = st.checkbox(
        "Physical-grid emergency enforcement",
        value=True,
        help="If the line is physically stressed, the company can enforce minimum reduction after deadline."
    )

with fc3:
    condition_grid_support_discount = st.checkbox(
        "Grid support discount",
        value=True,
        help="Discount applies if the mandatory reduction is achieved during grid stress."
    )

    condition_loyalty_discount = st.checkbox(
        "Loyalty discount below baseline",
        value=True,
        help="Extra discount if the customer stays at or below their own baseline during peak stress."
    )

    condition_progressive_penalty = st.checkbox(
        "Progressive penalty by shortfall",
        value=True,
        help="Penalty depends on how far the customer is from the required reduction."
    )

    condition_growth_bonus = st.checkbox(
        "Company growth bonus when grid is stable",
        value=False,
        help="If the company wants demand growth and the grid is stable, high usage can receive a bonus."
    )

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

st.info(
    "Active fairness rules are combined together. For example, district-wide reduction, historical baseline protection, "
    "customer autonomy, low-baseline protection, and emergency enforcement can all work at the same time."
)


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

if grid_stress and peak_event:
    st.error(
        f"SCADA Alert: Line stress is real. User must reduce at least "
        f"{mandatory_reduction_percent}% unless fairness protections modify the effective requirement."
    )
else:
    st.success(
        "Grid is stable. Pricing and bonus modes can operate without mandatory physical protection."
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
        default_lamps=10,
        default_acs=4,
        default_washing=1,
        default_heavy=3,
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
# PERSON SELECTION
# =========================================================

st.divider()
st.header("Client Selection")

selected_person = st.radio(
    "Choose Client / Smart Meter",
    ["Person A", "Person B"],
    horizontal=True
)

selected_baseline = baseline_a if selected_person == "Person A" else baseline_b


# =========================================================
# USAGE COMMAND
# =========================================================

st.subheader("Client Usage During Peak Event")

requested_usage = st.number_input(
    "Requested / Original Usage During Peak Event kWh",
    min_value=0.0,
    value=float(selected_baseline + 2),
    step=0.1
)

st.info(
    "This value represents the user's intended usage before smart meter shedding or company enforcement."
)


# =========================================================
# EFFECTIVE FAIR REDUCTION LOGIC
# =========================================================

policy_mode = st.session_state.selected_user_policy
refuse_disconnect = st.session_state.refuse_disconnect
climate_mode = st.session_state.climate_mode

effective_voluntary_reduction_percent = voluntary_reduction_percent
effective_mandatory_reduction_percent = mandatory_reduction_percent

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

    st.warning(
        f"Low-baseline protection is active. Because this customer baseline is {selected_baseline:.2f} kWh, "
        f"the effective reduction request is capped at {LOW_BASELINE_MAX_REDUCTION_PERCENT}%."
    )

if not fair_conditions["district_reduction"]:
    effective_voluntary_reduction_percent = 0
    st.info(
        "District-wide load reduction is disabled. Voluntary reduction request is set to 0%, "
        "but emergency mandatory protection can still apply if enabled."
    )


# =========================================================
# SMART METER SIMULATION
# =========================================================

shed_df, original_load_kw, final_load_kw, achieved_reduction_percent, enforcement_status = smart_meter_shed_load(
    appliance_df=st.session_state.appliance_config,
    requested_reduction_percent=effective_voluntary_reduction_percent,
    policy_mode=policy_mode,
    refuse_disconnect=refuse_disconnect,
    climate_mode=climate_mode,
    mandatory_minimum_percent=effective_mandatory_reduction_percent,
    user_failed_to_respond=user_failed_to_respond,
    enforcement_enabled=(
        enforcement_enabled
        and grid_stress
        and peak_event
        and fair_conditions["emergency_enforcement"]
    ),
    minimum_service_enabled=fair_conditions["minimum_service"]
)

if original_load_kw > 0:
    usage_ratio = final_load_kw / original_load_kw
else:
    usage_ratio = 1

final_usage = requested_usage * usage_ratio

billing = billing_engine(
    baseline=selected_baseline,
    original_usage=requested_usage,
    final_usage=final_usage,
    mean_usage=mean_usage,
    grid_stress=grid_stress and peak_event,
    new_company_growth_mode=new_company_growth_mode,
    refused_disconnect=refuse_disconnect,
    achieved_reduction_percent=achieved_reduction_percent,
    mandatory_reduction_percent=effective_mandatory_reduction_percent,
    fair_conditions=fair_conditions
)


# =========================================================
# MAIN SCADA METRICS
# =========================================================

st.divider()
st.header("SCADA Live Status")

s1, s2, s3, s4, s5 = st.columns(5)

s1.metric("Original Connected Load", f"{original_load_kw:.2f} kW")
s2.metric("Final Connected Load", f"{final_load_kw:.2f} kW")
s3.metric("Achieved Reduction", f"{achieved_reduction_percent:.2f}%")
s4.metric("Final Usage", f"{final_usage:.2f} kWh")
s5.metric("Final Bill", f"{billing['Final Bill']:.2f} EGP")

e1, e2 = st.columns(2)
e1.metric("Effective Voluntary Reduction", f"{effective_voluntary_reduction_percent:.2f}%")
e2.metric("Effective Mandatory Reduction", f"{effective_mandatory_reduction_percent:.2f}%")

if achieved_reduction_percent < effective_mandatory_reduction_percent and grid_stress and peak_event:
    st.error("Grid Protection Warning: The effective mandatory reduction target was not achieved.")
else:
    st.success("Grid Protection Status: Effective reduction condition is acceptable.")

st.info(enforcement_status)


# =========================================================
# CONDITION EVALUATION
# =========================================================

st.divider()
st.header("SCADA Fairness Condition Evaluation")

condition_rows = []

condition_rows.append({
    "Condition": "District-wide load reduction",
    "Active": fair_conditions["district_reduction"],
    "Rule": f"Utility requests {effective_voluntary_reduction_percent}% effective reduction",
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
    "Condition": "Marginal premium above baseline",
    "Active": fair_conditions["marginal_premium"],
    "Rule": "Premium applies only above own baseline",
    "Result": f"Premium usage {billing['Premium Usage kWh']:.2f} kWh",
    "Status": "Premium applied" if billing["Premium Charge"] > 0 else "No premium"
})

condition_rows.append({
    "Condition": "Low-baseline protection",
    "Active": fair_conditions["low_baseline_protection"],
    "Rule": "Protect very low users from unfair shutdown or excessive penalty",
    "Result": f"Baseline {selected_baseline:.2f} kWh",
    "Status": "Protection applied" if selected_baseline <= LOW_BASELINE_THRESHOLD_KWH else "Not low-baseline case"
})

condition_rows.append({
    "Condition": "Customer autonomy",
    "Active": fair_conditions["customer_autonomy"],
    "Rule": "Customer can set priorities and refuse automatic shedding",
    "Result": "Refused automatic shedding" if refuse_disconnect else "Accepted smart meter control",
    "Status": "Override active" if refuse_disconnect else "Normal control"
})

condition_rows.append({
    "Condition": "Critical load protection",
    "Active": fair_conditions["critical_load_protection"],
    "Rule": "Critical loads are never disconnected",
    "Result": "Critical loads protected through smart meter table",
    "Status": "Enabled" if fair_conditions["critical_load_protection"] else "Disabled"
})

condition_rows.append({
    "Condition": "Minimum service preservation",
    "Active": fair_conditions["minimum_service"],
    "Rule": "Preserve minimum service such as AC in hot mode or heater in cold mode",
    "Result": climate_mode,
    "Status": "Enabled" if fair_conditions["minimum_service"] else "Disabled"
})

condition_rows.append({
    "Condition": "Emergency physical-grid enforcement",
    "Active": fair_conditions["emergency_enforcement"],
    "Rule": "Company can enforce minimum reduction if line stress is real",
    "Result": enforcement_status,
    "Status": "Available" if fair_conditions["emergency_enforcement"] else "Disabled"
})

condition_rows.append({
    "Condition": "Grid support discount",
    "Active": fair_conditions["grid_support_discount"],
    "Rule": f"Achieve at least {effective_mandatory_reduction_percent}% reduction",
    "Result": f"Achieved {achieved_reduction_percent:.2f}%",
    "Status": "Discount applied" if billing["Discount"] > 0 else "No discount"
})

condition_rows.append({
    "Condition": "Loyalty discount",
    "Active": fair_conditions["loyalty_discount"],
    "Rule": "Stay at or below own historical baseline during peak stress",
    "Result": f"Loyalty discount {billing['Loyalty Discount']:.2f} EGP",
    "Status": "Applied" if billing["Loyalty Discount"] > 0 else "Not applied"
})

condition_rows.append({
    "Condition": "Progressive penalty",
    "Active": fair_conditions["progressive_penalty"],
    "Rule": "Penalty increases according to reduction shortfall",
    "Result": f"Penalty {billing['Penalty']:.2f} EGP",
    "Status": "Applied" if billing["Penalty"] > 0 else "No penalty"
})

condition_rows.append({
    "Condition": "Penalty waiver",
    "Active": fair_conditions["low_baseline_protection"],
    "Rule": "Penalty can be waived for low-baseline customers staying within baseline",
    "Result": f"Penalty waived {billing['Penalty Waived']:.2f} EGP",
    "Status": "Waived" if billing["Penalty Waived"] > 0 else "Not waived"
})

condition_df = pd.DataFrame(condition_rows)
st.dataframe(condition_df, use_container_width=True)


# =========================================================
# PREMIUM PRICING MESSAGE
# =========================================================

st.subheader("Intro Message Catalogue")

if grid_stress and peak_event:
    st.warning(
        "Peak Event Active: The system requested load reduction because the issue is physical line stress, "
        "not only electricity price. Premium payment may preserve comfort, but mandatory reduction can still be enforced."
    )

if refuse_disconnect:
    st.error(
        "Client Policy: The user selected no disconnection. The system will apply premium convenience pricing. "
        "If the deadline expires during real stress, emergency enforcement may override the refusal."
    )

if new_company_growth_mode and not grid_stress:
    st.success(
        "Growth Mode Active: The company wants to increase average consumption. "
        "Users above the average may receive a bonus if growth bonus condition is active."
    )

if climate_mode == "Hot Summer - Cooling Priority":
    st.info("Climate Mode: Hot summer cooling priority. AC protection is applied when minimum service preservation is active.")
elif climate_mode == "Cold Winter - Heating Priority":
    st.info("Climate Mode: Cold winter heating priority. Water heater protection is applied when minimum service preservation is active.")
else:
    st.info("Climate Mode: Normal operation. No automatic seasonal equipment protection.")

st.markdown("""
### Tariff Name

**Premium Load Preservation Pricing under Dynamic Tariffs:  
An Uninterrupted Consumption Pay-for-Convenience Model for Peak Load Retention**
""")


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
    "Penalty Waived EGP": billing["Penalty Waived"],
    "Bonus EGP": billing["Bonus"],
    "Discount EGP": billing["Discount"],
    "Loyalty Discount EGP": billing["Loyalty Discount"],
    "Final Bill EGP": billing["Final Bill"],
    "Condition Status": billing["Status"]
}])

st.dataframe(billing_df, use_container_width=True)


# =========================================================
# SMART METER LOAD TABLE
# =========================================================

st.divider()
st.header("Smart Meter Load Shedding Result")

st.dataframe(shed_df, use_container_width=True)


# =========================================================
# GRAPHS
# =========================================================

st.divider()
st.header("SCADA Visual Analytics")

tab1, tab2, tab3, tab4 = st.tabs([
    "Load Shedding Graph",
    "Usage & Billing Graph",
    "Baseline Bell Curve",
    "Priority Comparison"
])


with tab1:
    fig_load = go.Figure()

    fig_load.add_trace(go.Bar(
        x=shed_df["Appliance"],
        y=shed_df["Connected Load kW"],
        name="Original Connected Load",
        marker_color="deepskyblue",
        text=shed_df["Connected Load kW"].round(2),
        textposition="auto"
    ))

    fig_load.add_trace(go.Bar(
        x=shed_df["Appliance"],
        y=shed_df["Shed kW"],
        name="Disconnected / Shed Load",
        marker_color="red",
        text=shed_df["Shed kW"].round(2),
        textposition="auto"
    ))

    fig_load.update_layout(
        title="Original Load vs Disconnected Load",
        xaxis_title="Appliance",
        yaxis_title="kW",
        barmode="group",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=26),
        height=650
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
            "Penalty Waived",
            "Bonus",
            "Discount",
            "Loyalty Discount",
            "Final Bill"
        ],
        "EGP": [
            billing["Premium Charge"],
            billing["Penalty"],
            billing["Penalty Waived"],
            billing["Bonus"],
            billing["Discount"],
            billing["Loyalty Discount"],
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

    if z_score < -1.5:
        st.success("Consumption is very low compared to the simulated population.")
    elif z_score < -0.5:
        st.info("Consumption is below average.")
    elif z_score <= 0.5:
        st.warning("Consumption is close to average.")
    elif z_score <= 1.5:
        st.warning("Consumption is above average.")
    else:
        st.error("Consumption is extremely high compared to most simulated homes.")


with tab4:
    priority_df = st.session_state.appliance_config.copy()

    fig_priority = go.Figure()

    fig_priority.add_trace(go.Bar(
        x=priority_df["Appliance"],
        y=priority_df["User Priority"],
        name="Internal User Priority",
        marker_color="lime",
        text=priority_df["User Priority"],
        textposition="auto"
    ))

    fig_priority.add_trace(go.Bar(
        x=priority_df["Appliance"],
        y=priority_df["Company Priority"],
        name="Internal Company Priority",
        marker_color="orange",
        text=priority_df["Company Priority"],
        textposition="auto"
    ))

    fig_priority.update_layout(
        title="Internal User Priority vs Company Priority",
        xaxis_title="Appliance",
        yaxis_title="Priority Number: Lower Means Disconnect First",
        barmode="group",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=26),
        height=650
    )

    st.plotly_chart(fig_priority, use_container_width=True)

    st.info(
        "The user controls priorities using checkboxes. The app converts those checkboxes into internal SCADA priority numbers automatically."
    )


# =========================================================
# FINAL SYSTEM SUMMARY
# =========================================================

st.divider()
st.header("Final SCADA Decision Summary")

if grid_stress and peak_event and achieved_reduction_percent < effective_mandatory_reduction_percent:
    st.error(
        "Final Decision: The user did not satisfy the effective minimum physical grid protection requirement. "
        "The company may apply enforcement, restriction, or blocking logic in this simulation."
    )
elif refuse_disconnect and grid_stress and peak_event:
    st.warning(
        "Final Decision: User preserved comfort and refused disconnection. "
        "Premium pricing is applied, but grid protection may still override if stress continues."
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
