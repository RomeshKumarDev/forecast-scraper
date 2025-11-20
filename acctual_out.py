import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("🌦️ Weather Actual Max/Min Updater")

# Upload CSV
uploaded_file = st.file_uploader("Upload Weather CSV", type=["csv"])

if uploaded_file:
    # Load CSV fresh every time
    df = pd.read_csv(uploaded_file)

    st.subheader("Select Date & Enter Actual Values")

    # Date dropdown
    selected_date = st.selectbox("Select Date", df["full_date"].unique())

    # Inputs for actual max/min
    col1, col2 = st.columns(2)
    with col1:
        actual_max = st.text_input("Actual Max Temperature (e.g., 32°C)")
    with col2:
        actual_min = st.text_input("Actual Min Temperature (e.g., 18°C)")

    # Add button
    if st.button("➕ Add Actual Max/Min"):
        # Update all rows of that date
        df.loc[df["full_date"] == selected_date, "actual_max"] = actual_max
        df.loc[df["full_date"] == selected_date, "actual_min"] = actual_min

        st.success(f"Actual values added for {selected_date}")

        # Show updated dataframe
        st.subheader("Updated Weather Data")
        st.dataframe(df, use_container_width=True)

        # Allow user to download updated CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Updated CSV",
            data=csv,
            file_name="updated_weather.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a weather CSV file to begin.")
