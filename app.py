import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI FEMM EV Motor Simulator", layout="wide")

st.title("AI FEMM EV Motor Simulator")
st.write("Change FEMM-style motor variables and see how the electric car performs.")

data = pd.read_csv("bldc_data.csv")

X = data[["air_gap", "magnet_thickness", "turns", "current"]]
y = data["torque"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

st.sidebar.header("FEMM Motor Variables")

air_gap = st.sidebar.slider("Air Gap (mm)", 0.3, 1.0, 0.4)
magnet_thickness = st.sidebar.slider("Magnet Thickness (mm)", 2.0, 4.0, 2.5)
turns = st.sidebar.slider("Coil Turns", 60, 140, 90)
current = st.sidebar.slider("Current (A)", 1, 10, 5)

input_data = pd.DataFrame([{
    "air_gap": air_gap,
    "magnet_thickness": magnet_thickness,
    "turns": turns,
    "current": current
}])

torque = model.predict(input_data)[0]

efficiency = max(45, min(96, 82 - air_gap * 12 + magnet_thickness * 3 - current * 1.2))
heat_risk = max(5, min(100, current * 8 + turns * 0.12 + magnet_thickness * 2))
battery_drain = max(10, min(100, current * 7 + heat_risk * 0.25))
acceleration_time = max(2.8, min(15, 12 - torque * 10 + air_gap * 2))
top_speed = max(60, min(260, 90 + torque * 180 + current * 3))
ev_range = max(80, min(600, efficiency * 4.8 - current * 12))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Motor Performance")

    st.metric("Predicted Torque", f"{torque:.2f} Nm")
    st.metric("Efficiency", f"{efficiency:.0f}%")
    st.metric("Heat Risk", f"{heat_risk:.0f}%")

    st.write("Torque")
    st.progress(int(min(100, torque * 140)))

    st.write("Efficiency")
    st.progress(int(efficiency))

    st.write("Heat Risk")
    st.progress(int(heat_risk))

with col2:
    st.subheader("Electric Car in Action")

    st.metric("0–100 km/h Estimate", f"{acceleration_time:.1f} sec")
    st.metric("Estimated Top Speed", f"{top_speed:.0f} km/h")
    st.metric("Estimated EV Range", f"{ev_range:.0f} km")

    st.write("Acceleration Performance")
    st.progress(int(max(0, min(100, 100 - acceleration_time * 6))))

    st.write("Top Speed Potential")
    st.progress(int(top_speed / 260 * 100))

    st.write("EV Range")
    st.progress(int(ev_range / 600 * 100))

    st.write("Battery Drain")
    st.progress(int(battery_drain))

st.divider()

st.subheader("EV Simulation")

if acceleration_time < 5:
    st.success("🏎️ High-performance EV motor")
    car_line = "🚗💨💨💨💨💨"
    speed_label = "The car launches very fast."
elif acceleration_time < 8:
    st.info("🚙 Balanced EV motor")
    car_line = "🚗💨💨💨"
    speed_label = "The car has good acceleration and reasonable efficiency."
else:
    st.warning("🐢 Weak EV motor setup")
    car_line = "🚗💨"
    speed_label = "The car accelerates slowly."

st.markdown(f"## {car_line}")
st.write(speed_label)

st.write("### What changed in the car?")

if air_gap > 0.7:
    st.write("Large air gap → weaker magnetic coupling → lower torque → slower acceleration.")
elif air_gap <= 0.4:
    st.write("Small air gap → stronger magnetic coupling → higher torque → faster acceleration.")
else:
    st.write("Medium air gap → balanced magnetic coupling → stable EV performance.")

if current > 8:
    st.write("High current → more power, but higher heat and battery drain.")
elif current < 4:
    st.write("Low current → safer and cooler, but lower power.")
else:
    st.write("Moderate current → good balance between power and heat.")

if magnet_thickness > 3.2:
    st.write("Thicker magnets → stronger magnetic field, but more weight/cost.")
elif magnet_thickness < 2.4:
    st.write("Thin magnets → weaker field, lower torque.")
else:
    st.write("Balanced magnet thickness → stable torque and efficiency.")

st.divider()

st.subheader("Before vs After Example")

before, after = st.columns(2)

with before:
    st.markdown("### Standard Motor")
    st.write("Air gap: 0.8 mm")
    st.write("Torque: lower")
    st.write("EV effect: slower acceleration")
    st.progress(40)
    st.markdown("🚗💨")

with after:
    st.markdown("### Optimized Motor")
    st.write("Air gap: 0.4 mm")
    st.write("Torque: higher")
    st.write("EV effect: faster acceleration")
    st.progress(90)
    st.markdown("🚗💨💨💨💨")

st.divider()

st.subheader("AI Engineering Explanation")

if torque > 0.55 and heat_risk < 70:
    st.success("This motor setup would make the EV accelerate faster without extreme overheating.")
elif torque > 0.55 and heat_risk >= 70:
    st.warning("This motor gives strong acceleration, but heat risk is high. A cooling system would be needed.")
elif efficiency > 88:
    st.success("This setup is better for EV range because it wastes less energy.")
elif battery_drain > 75:
    st.error("This setup drains the battery quickly. It may reduce EV range.")
else:
    st.info("This is a moderate EV motor setup: safe, but not fully optimized for speed or range.")

st.divider()

st.subheader("Training Dataset")
st.dataframe(data)