from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "BLDC_ML_dataset.csv"
FEATURES = ["gap_mm", "magnet_thickness_mm", "current_A"]
STATOR_OD_MM = 38.50
ROTOR_YOKE_RADIAL_MM = 2.80


st.set_page_config(
    page_title="RotorOpt — оптимизация BLDC 12S14P",
    page_icon="⚙️",
    layout="centered",
)


@st.cache_resource
def load_model():
    data = pd.read_csv(DATA_PATH)
    model = Pipeline(
        [
            ("poly", PolynomialFeatures(3, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1e-5)),
        ]
    )
    model.fit(data[FEATURES], data["torque_mNm"])
    return model, len(data)


def recommend(model, current_A, minimum_safe_gap_mm, maximum_magnet_thickness_mm):
    gaps = np.arange(minimum_safe_gap_mm, 1.5001, 0.005)
    thicknesses = np.arange(1.50, maximum_magnet_thickness_mm + 0.0001, 0.01)
    search = pd.MultiIndex.from_product(
        [gaps, thicknesses, [current_A]], names=FEATURES
    ).to_frame(index=False)
    search["torque_mNm"] = model.predict(search[FEATURES])
    best = search.loc[search["torque_mNm"].idxmax()]

    magnet_inner_d = STATOR_OD_MM + 2 * best.gap_mm
    magnet_seat_d = magnet_inner_d + 2 * best.magnet_thickness_mm
    rotor_od = magnet_seat_d + 2 * ROTOR_YOKE_RADIAL_MM

    return {
        "gap_mm": float(best.gap_mm),
        "magnet_thickness_mm": float(best.magnet_thickness_mm),
        "current_A": float(best.current_A),
        "magnet_inner_diameter_mm": float(magnet_inner_d),
        "magnet_seat_diameter_mm": float(magnet_seat_d),
        "rotor_outer_diameter_mm": float(rotor_od),
        "torque_mNm": float(best.torque_mNm),
    }


model, dataset_size = load_model()

st.title("RotorOpt")
st.subheader("ML-проектирование 3D-печатного BLDC 12S14P")
st.caption(
    f"Подбор размеров внешнего ротора для статора Ø38,5 мм. "
    f"Модель обучена на {dataset_size} FEMM-расчётах."
)

with st.form("rotoropt_form"):
    current_A = st.slider(
        "Рабочий ток, А", min_value=1.0, max_value=3.0, value=2.0, step=0.1
    )
    minimum_safe_gap_mm = st.slider(
        "Минимальный безопасный воздушный зазор, мм",
        min_value=0.50,
        max_value=1.50,
        value=0.95,
        step=0.05,
    )
    maximum_magnet_thickness_mm = st.slider(
        "Максимальная доступная толщина магнита, мм",
        min_value=1.50,
        max_value=3.00,
        value=2.00,
        step=0.10,
    )
    submitted = st.form_submit_button(
        "Подобрать параметры", type="primary", use_container_width=True
    )

if submitted:
    result = recommend(
        model,
        current_A,
        minimum_safe_gap_mm,
        maximum_magnet_thickness_mm,
    )

    st.success("Оптимальный допустимый вариант найден")
    st.markdown("### Рекомендуемые параметры")
    col1, col2, col3 = st.columns(3)
    col1.metric("Воздушный зазор", f"{result['gap_mm']:.3f} мм")
    col2.metric(
        "Толщина магнита", f"{result['magnet_thickness_mm']:.3f} мм"
    )
    col3.metric("Рабочий ток", f"{result['current_A']:.2f} А")

    st.markdown("### Размеры ротора")
    st.write(
        f"**Внутренний диаметр по магнитам:** "
        f"{result['magnet_inner_diameter_mm']:.3f} мм"
    )
    st.write(
        f"**Диаметр посадки магнитов:** "
        f"{result['magnet_seat_diameter_mm']:.3f} мм"
    )
    st.write(
        f"**Наружный диаметр ротора:** "
        f"{result['rotor_outer_diameter_mm']:.3f} мм"
    )

    st.metric(
        "Прогнозируемый статический электромагнитный момент",
        f"{result['torque_mNm']:.3f} мН·м",
    )

st.divider()
st.caption(
    "Прогноз является результатом суррогатной ML-модели и не заменяет "
    "тепловую, механическую и экспериментальную проверку конструкции."
)
