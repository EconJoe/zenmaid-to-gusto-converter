import streamlit as st
import pandas as pd
import io
import googlemaps
import os

st.set_page_config(page_title="ZenMaid to Gusto Converter", layout="centered")
st.title("ZenMaid to Gusto Converter")

st.markdown("""
This tool processes ZenMaid appointment and hours data into a format for uploading into Gusto.
Follow the steps below:
""")

from datetime import datetime, timedelta

# --- Pay Period Selection ---
current_year = datetime.now().year
pay_periods = []
for month in range(1, 13):
    start_1 = datetime(current_year, month, 1)
    end_1 = datetime(current_year, month, 15)
    start_2 = datetime(current_year, month, 16)
    if month == 12:
        end_2 = datetime(current_year, 12, 31)
    else:
        end_2 = datetime(current_year, month + 1, 1) - timedelta(days=1)
    pay_periods.append((f"Pay Period {2 * month - 1}: {start_1.strftime('%b %d')}–{end_1.strftime('%d')}", (start_1, end_1)))
    pay_periods.append((f"Pay Period {2 * month}: {start_2.strftime('%b %d')}–{end_2.strftime('%d')}", (start_2, end_2)))

pay_period_options = [label for label, _ in pay_periods]

# Determine default selection based on current date
now = datetime.now()
default_index = next(
    (i for i in reversed(range(len(pay_periods))) if pay_periods[i][1][1] < now),
    0
)

selected_label = st.selectbox("Select Pay Period (Required)", options=pay_period_options, index=default_index)
selected_period = dict(pay_periods)[selected_label]

# --- File upload ---
uploaded_appt_file = st.file_uploader("Upload ZenMaid_AppointmentDataExport CSV", type="csv")
if uploaded_appt_file and "ZenMaid_AppointmentDataExport" not in uploaded_appt_file.name:
    st.error("The appointments file must include 'ZenMaid_AppointmentDataExport' in its filename.")
elif uploaded_appt_file:
    try:
        uploaded_appt_file.seek(0)
        first_row_note = pd.read_csv(uploaded_appt_file, nrows=1).columns[0]
        import re
        match = re.search(r'from the beginning of (\d{2}-\d{2}-\d{4}) to the end of (\d{2}-\d{2}-\d{4})', first_row_note)
        if match:
            file_start_date = datetime.strptime(match.group(1), '%m-%d-%Y').date()
            file_end_date = datetime.strptime(match.group(2), '%m-%d-%Y').date()
            pay_period_start, pay_period_end = selected_period[0].date(), selected_period[1].date()
            if file_start_date != pay_period_start or file_end_date != pay_period_end:
                st.error(
                    f"The date range in the appointments file ({file_start_date} to {file_end_date}) "
                    f"does not match the selected pay period ({pay_period_start} to {pay_period_end})."
                )
    except Exception as e:
        st.warning(f"Could not validate date range in appointments file: {e}")
uploaded_hours_file = st.file_uploader("Upload ZENMAID_clocked_hours_export CSV", type="csv")

if uploaded_hours_file and "ZENMAID_clocked_hours" not in uploaded_hours_file.name:
    st.error("The hours file must include 'ZENMAID_clocked_hours' in its filename.")
elif uploaded_hours_file:
    try:
        uploaded_hours_file.seek(0)
        first_row_note = pd.read_csv(uploaded_hours_file, nrows=1).columns[0]
        import re
        match = re.search(r'from the beginning of (\d{4}-\d{2}-\d{2}) to the end of (\d{4}-\d{2}-\d{2})', first_row_note)
        if match:
            file_start_date = datetime.strptime(match.group(1), '%Y-%m-%d').date()
            file_end_date = datetime.strptime(match.group(2), '%Y-%m-%d').date()
            pay_period_start, pay_period_end = selected_period[0].date(), selected_period[1].date()
            if file_start_date != pay_period_start or file_end_date != pay_period_end:
                st.error(
                    f"The date range in the hours file ({file_start_date} to {file_end_date}) "
                    f"does not match the selected pay period ({pay_period_start} to {pay_period_end})."
                )
    except Exception as e:
        st.warning(f"Could not validate date range in hours file: {e}")

tips_input = st.number_input("Enter Total Tips for Pay Period", min_value=0.0, format="%.2f")
mileage_rate = st.number_input("Enter Mileage Reimbursement Rate ($/mile)", min_value=0.0, value=0.67, step=0.01, format="%.2f")
calculate_drive = st.checkbox("Include Drive Time and Mileage (Google Maps API required)", value=True)

# --- Button to begin processing ---
if st.button("Process Files"):
    # Prevent processing if date mismatch errors were already shown
    if uploaded_appt_file:
        uploaded_appt_file.seek(0)
        try:
            first_row_note = pd.read_csv(uploaded_appt_file, nrows=1).columns[0]
            import re
            match = re.search(r'from the beginning of (\d{2}-\d{2}-\d{4}) to the end of (\d{2}-\d{2}-\d{4})', first_row_note)
            if match:
                file_start_date = datetime.strptime(match.group(1), '%m-%d-%Y').date()
                file_end_date = datetime.strptime(match.group(2), '%m-%d-%Y').date()
                pay_period_start, pay_period_end = selected_period[0].date(), selected_period[1].date()
                if file_start_date != pay_period_start or file_end_date != pay_period_end:
                    st.error("Appointments file date range does not match selected pay period.")
                    st.stop()
        except Exception as e:
            st.error(f"Could not validate appointments file again: {e}")
            st.stop()

    if uploaded_hours_file:
        uploaded_hours_file.seek(0)
        try:
            first_row_note = pd.read_csv(uploaded_hours_file, nrows=1).columns[0]
            match = re.search(r'from the beginning of (\d{4}-\d{2}-\d{2}) to the end of (\d{4}-\d{2}-\d{2})', first_row_note)
            if match:
                file_start_date = datetime.strptime(match.group(1), '%Y-%m-%d').date()
                file_end_date = datetime.strptime(match.group(2), '%Y-%m-%d').date()
                pay_period_start, pay_period_end = selected_period[0].date(), selected_period[1].date()
                if file_start_date != pay_period_start or file_end_date != pay_period_end:
                    st.error("Clocked hours file date range does not match selected pay period.")
                    st.stop()
        except Exception as e:
            st.error(f"Could not validate hours file again: {e}")
            st.stop()

    if not uploaded_appt_file or not uploaded_hours_file:
        st.error("Please upload both files to continue.")
    elif "ZenMaid_AppointmentDataExport" not in uploaded_appt_file.name:
        st.error("The appointments file must include 'ZenMaid_AppointmentDataExport' in its filename.")
        st.error("Please upload both files to continue.")
    else:
        try:
            # Read the CSVs
            uploaded_hours_file.seek(0)
            uploaded_appt_file.seek(0)
            df_appointments = pd.read_csv(uploaded_appt_file, skiprows=1)
            df_hours = pd.read_csv(uploaded_hours_file, skiprows=2, on_bad_lines='skip')

            # Ensure required columns exist in both dataframes
            hours_expected_cols = ["Appointment ID", "Cleaner", "Customer", "Clocked Duration (in hours)"]
            missing_hours_cols = [col for col in hours_expected_cols if col not in df_hours.columns]
            if missing_hours_cols:
                st.error(f"Missing required columns in Clocked Hours file: {', '.join(missing_hours_cols)}")
                st.stop()

            appt_expected_cols = [
                "Custom Field: Trainers",
                "Custom Field: Trainees",
                "Custom Field: Free Clean",
                "Appointment ID",
                "Customer",
                "Customer Full Name",
                "Appointment Status",
                "Cleaners",
                "Appointment Date",
                "Start Time",
                "End Time",
                "Address Line1",
                "Address Line2",
                "Address Postal Code",
                "Address City",
                "Address State"
            ]
            for col in appt_expected_cols:
                if col not in df_appointments.columns:
                    df_appointments[col] = pd.NA

            # Rename Customer in hours to preserve it after merge
            df_hours = df_hours.rename(columns={"Customer": "Customer_hours"})
            df_hours = df_hours[["Appointment ID", "Cleaner", "Customer_hours", "Clocked Duration (in hours)"]]

            # Merge and compute pay categories
            df_merge = pd.merge(df_hours, df_appointments, on="Appointment ID", how="left")

            # --- Drive Time and Mileage Calculation ---
            mileage_df = pd.DataFrame()
            if calculate_drive:
                df_drive = df_appointments[df_appointments['Appointment Status'] != 'Cancelled'].copy()
                address_cols = ['Address Line1', 'Address Line2', 'Address City', 'Address State', 'Address Postal Code']
                df_drive['Full Address'] = df_drive[address_cols].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)
                df_drive['Appointment Date'] = pd.to_datetime(df_drive['Appointment Date'])
                df_drive['Start Time'] = pd.to_datetime(df_drive['Start Time'], format='%I:%M %p', errors='coerce').dt.time
                df_drive['Start DateTime'] = pd.to_datetime(df_drive['Appointment Date'].astype(str) + ' ' + df_drive['Start Time'].astype(str))
                df_drive['Cleaners'] = df_drive['Cleaners'].str.split(';')
                df_drive = df_drive.explode('Cleaners').reset_index(drop=True)
                df_drive['Cleaners'] = df_drive['Cleaners'].str.strip()
                df_drive.sort_values(by=['Cleaners', 'Appointment Date', 'Start DateTime'], inplace=True)
                df_drive['Destination Address'] = df_drive.groupby(['Cleaners', 'Appointment Date'])['Full Address'].shift(-1)
                df_drive['Destination Customer'] = df_drive.groupby(['Cleaners', 'Appointment Date'])['Customer Full Name'].shift(-1)
                df_drive.dropna(subset=['Destination Address'], inplace=True)
                df_drive.rename(columns={'Full Address': 'Origin Address', 'Cleaners': 'Cleaner', 'Customer Full Name': 'Origin Customer'}, inplace=True)
                
                API_KEY = st.secrets["google_maps_api_key"]
                if not API_KEY:
                    st.warning("Google Maps API key not set. Skipping drive time calculation.")
                else:
                    gmaps = googlemaps.Client(key=API_KEY)
                    def get_google_maps_distance(origin, destination):
                        try:
                            result = gmaps.distance_matrix(origins=origin, destinations=destination, mode="driving")
                            if result['status'] == 'OK':
                                distance_miles = result['rows'][0]['elements'][0]['distance']['value'] / 1609.34
                                duration_hours = result['rows'][0]['elements'][0]['duration']['value'] / 3600
                                return distance_miles, duration_hours
                        except Exception as e:
                            st.warning(f"Google Maps API error: {e}")
                        return 0.0, 0.0

                    df_drive[['Distance (miles)', 'Drive Time (hours)']] = df_drive.apply(
                        lambda row: pd.Series(get_google_maps_distance(row['Origin Address'], row['Destination Address'])), axis=1
                    )
                    df_drive['Reimbursements - Mileage'] = (df_drive['Distance (miles)'] * mileage_rate).round(2)
                    drive_time_df = df_drive[['Cleaner', 'Drive Time (hours)']].groupby('Cleaner', as_index=False).sum()
                    drive_time_df.rename(columns={'Drive Time (hours)': 'clk_drivetime'}, inplace=True)
                    mileage_df = df_drive[['Cleaner', 'Reimbursements - Mileage']].groupby('Cleaner', as_index=False).sum()

            # === Resume full processing logic ===
            # Calculate Tips
            tips_df = df_merge.groupby('Cleaner', as_index=False)['Clocked Duration (in hours)'].sum()
            total_hours = tips_df['Clocked Duration (in hours)'].sum()
            tips_df['Tips Allocated'] = ((tips_df['Clocked Duration (in hours)'] / total_hours) * tips_input).round(2)

            # Determine Job Roles
            def determine_job(row):
                cleaner = str(row['Cleaner'])
                if pd.notna(row.get('Custom Field: Trainers')) and cleaner in str(row['Custom Field: Trainers']):
                    return 'Trainer'
                elif pd.notna(row.get('Custom Field: Trainees')) and cleaner in str(row['Custom Field: Trainees']):
                    return 'Trainee'
                else:
                    return 'Cleaner'

            df_merge['Job'] = df_merge.apply(determine_job, axis=1)
            df_merge['clk_cleaner'] = 0.0
            df_merge.loc[df_merge['Job'] == 'Cleaner', 'clk_cleaner'] = df_merge['Clocked Duration (in hours)']
            df_merge.loc[df_merge['Job'].isin(['Trainer', 'Trainee']), 'clk_cleaner'] = 0.5 * df_merge['Clocked Duration (in hours)']
            df_merge['clk_trainer'] = df_merge['clk_trainee'] = 0.0
            df_merge.loc[df_merge['Job'] == 'Trainer', 'clk_trainer'] = df_merge['Clocked Duration (in hours)'] - df_merge['clk_cleaner']
            df_merge.loc[df_merge['Job'] == 'Trainee', 'clk_trainee'] = df_merge['Clocked Duration (in hours)'] - df_merge['clk_cleaner']

            # Save original cleaner hours
            df_merge['pre_flags_cleaner'] = df_merge['clk_cleaner']

            # Free cleans
            df_merge['is_free_clean'] = df_merge['Custom Field: Free Clean'] == 'Yes'
            df_merge['clk_marketing'] = 0.0
            df_merge.loc[df_merge['is_free_clean'], 'clk_marketing'] = df_merge['clk_cleaner']
            df_merge.loc[df_merge['is_free_clean'], 'clk_cleaner'] = 0

            # Weekly/Quarterly meetings
            df_merge['clk_wklymeet'] = 0.0
            df_merge['clk_qreview'] = 0.0
            df_merge.loc[df_merge['Customer_hours'] == 'Weekly Team Meeting', 'clk_wklymeet'] = df_merge['pre_flags_cleaner']
            df_merge.loc[df_merge['Customer_hours'] == 'Weekly Team Meeting', 'clk_cleaner'] = 0
            df_merge.loc[df_merge['Customer_hours'] == 'Quarterly Performance Review', 'clk_qreview'] = df_merge['pre_flags_cleaner']
            df_merge.loc[df_merge['Customer_hours'] == 'Quarterly Performance Review', 'clk_cleaner'] = 0

            # Collapse
            df_collapsed = df_merge.groupby('Cleaner').agg(
                clk_cleaner=('clk_cleaner', 'sum'),
                clk_trainer=('clk_trainer', 'sum'),
                clk_trainee=('clk_trainee', 'sum'),
                clk_marketing=('clk_marketing', 'sum'),
                clk_wklymeet=('clk_wklymeet', 'sum'),
                clk_qreview=('clk_qreview', 'sum')
            ).reset_index()

            if calculate_drive and not mileage_df.empty:
                df_collapsed = pd.merge(df_collapsed, drive_time_df, on='Cleaner', how='left')
                df_collapsed['clk_drivetime'] = df_collapsed['clk_drivetime'].fillna(0.0)
            else:
                df_collapsed['clk_drivetime'] = 0.0

            df_collapsed = pd.merge(df_collapsed, tips_df[['Cleaner', 'Tips Allocated']], on='Cleaner', how='left')

            df_long = df_collapsed.melt(
                id_vars=['Cleaner', 'Tips Allocated'],
                value_vars=['clk_cleaner', 'clk_trainer', 'clk_trainee', 'clk_marketing', 'clk_wklymeet', 'clk_qreview', 'clk_drivetime'],
                var_name='Job', value_name='Hours'
            )

            job_mapping = {
                'clk_cleaner': 'Cleaner',
                'clk_trainer': 'Trainer',
                'clk_trainee': 'Trainee',
                'clk_marketing': 'Marketing',
                'clk_wklymeet': 'Weekly Meeting',
                'clk_qreview': 'Quarterly Review',
                'clk_drivetime': 'Drive Time'
            }
            df_long['Job'] = df_long['Job'].map(job_mapping)
            df_long['Paycheck Tips'] = df_long.groupby('Cleaner')['Tips Allocated'].transform(lambda x: x.where(x.index == x.index[0]))
            df_long = df_long.drop(columns=['Tips Allocated'])

            df_long['Hours'] = df_long['Hours'].round(2).apply(lambda x: f"{x:.2f}")
            df_long = df_long[df_long['Hours'] != '0.00']
            df_long = df_long.sort_values(by=['Cleaner', 'Job']).reset_index(drop=True)

            df_long = df_long.rename(columns={
                'Cleaner': 'full_name',
                'Job': 'title',
                'Hours': 'regular_hours',
                'Paycheck Tips': 'paycheck_tips'
            })

            # Add reimbursement column
            if calculate_drive and not mileage_df.empty:
                mileage_totals = mileage_df.rename(columns={'Cleaner': 'full_name', 'Reimbursements - Mileage': 'reimbursement'})
                df_long = pd.merge(df_long, mileage_totals, on='full_name', how='left')
                df_long['reimbursement'] = df_long.groupby('full_name')['reimbursement'].transform(lambda x: x.where(x.index == x.index[0]))
                df_long['reimbursement'] = df_long['reimbursement'].where(df_long['reimbursement'].notna(), None).round(2)
            else:
                df_long['reimbursement'] = None

            # Export
            st.success("Processing complete!")
            st.dataframe(df_long)
            csv = df_long.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Gusto Upload CSV",
                data=csv,
                file_name="gusto_upload.csv",
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
