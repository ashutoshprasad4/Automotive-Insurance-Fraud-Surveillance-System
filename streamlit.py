import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = pickle.load(open('model12.pkl', 'rb'))

# Assuming the StandardScaler is fitted on the training data
scaler = StandardScaler()

# Preprocessing mappings
map_month = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

map_day = {
    'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
    'Friday': 5, 'Saturday': 6, 'Sunday': 7
}

AccidentArea_map = {'Urban': 0, 'Rural': 1}

Manufacture_company_map = {
    'Pontiac': 0, 'Toyota': 1, 'Honda': 2, 'Mazda': 3, 'Chevrolet': 4,
    'Accura': 5, 'Ford': 6, 'VW': 7, 'Dodge': 8, 'Saab': 9, 'Mercury': 10,
    'Saturn': 11, 'Nisson': 12, 'BMW': 13, 'Jaguar': 14, 'Porche': 15,
    'Mecedes': 16, 'Ferrari': 17, 'Lexus': 18
}

gender_map = {'Male': 0, 'Female': 1}

maritalstatus_map = {'Married': 0, 'Single': 1, 'Divorced': 2, 'Widow': 3}

fault_map = {'Policy Holder': 0, 'Third Party': 1}

Price_map = {
    'less than 20000': 0, '20000 to 29000': 1, '30000 to 39000': 2,
    '40000 to 59000': 3, '60000 to 69000': 4, 'more than 69000': 5
}

# Load the DataFrame for mapping
df = pd.read_csv('fraud_oracle_cleaned.csv')

# Initialize dictionaries for mappings
policy_type_map = {category: i for i, category in enumerate(df['PolicyType'].value_counts().index)}
vehicle_category_map = {category: i for i, category in enumerate(df['VehicleCategory'].value_counts().index)}
days_policy_accident_map = {category: i for i, category in enumerate(df['Days_Policy_Accident'].value_counts().index)}
days_policy_claim_map = {category: i for i, category in enumerate(df['Days_Policy_Claim'].value_counts().index)}
past_number_of_claims_map = {category: i for i, category in enumerate(df['PastNumberOfClaims'].value_counts().index)}
age_of_vehicle_map = {category: i for i, category in enumerate(df['AgeOfVehicle'].value_counts().index)}
police_report_filed_map = {category: i for i, category in enumerate(df['PoliceReportFiled'].value_counts().index)}
witness_present_map = {category: i for i, category in enumerate(df['WitnessPresent'].value_counts().index)}
agent_type_map = {category: i for i, category in enumerate(df['AgentType'].value_counts().index)}
number_of_suppliments_map = {category: i for i, category in enumerate(df['NumberOfSuppliments'].value_counts().index)}
address_change_claim_map = {category: i for i, category in enumerate(df['AddressChange_Claim'].value_counts().index)}
number_of_cars_map = {category: i for i, category in enumerate(df['NumberOfCars'].value_counts().index)}

# Streamlit UI
st.set_page_config(layout="wide")
st.title('Automotive Insurance Fraud Interception')

# Sidebar navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Home', 'About'])

if page == 'Home':
    # Input fields layout
    col1, col2 = st.columns(2)

    # Left column inputs
    with col1:
        WeekOfMonth = st.selectbox('Week of Month', range(1, 6))
        DayOfWeek = st.selectbox('Day of Week', list(map_day.keys()))
        Manufacture_company = st.selectbox('Manufacture Company', list(Manufacture_company_map.keys()))
        AccidentArea = st.selectbox('Accident Area', list(AccidentArea_map.keys()))
        DayOfWeekClaimed = st.selectbox('Day of Week Claimed', list(map_day.keys()))

    with col2:
        MonthClaimed = st.selectbox('Month Claimed', list(map_month.keys()))
        WeekOfMonthClaimed = st.selectbox('Week of Month Claimed', range(1, 6))
        Sex = st.selectbox('Sex', list(gender_map.keys()))
        MaritalStatus = st.selectbox('Marital Status', list(maritalstatus_map.keys()))
        Age = st.slider('Age', 18, 100)

    # Input fields layout continued
    col3, col4 = st.columns(2)

    # Right column inputs
    with col3:
        Fault = st.selectbox('Fault', list(fault_map.keys()))
        PolicyType = st.selectbox('Policy Type', list(policy_type_map.keys()))
        VehicleCategory = st.selectbox('Vehicle Category', list(vehicle_category_map.keys()))
        VehiclePrice = st.selectbox('Vehicle Price', list(Price_map.keys()))
        RepNumber = st.number_input('Rep Number', min_value=0)
        Deductible = st.number_input('Deductible', min_value=0)

    with col4:
        DriverRating = st.number_input('Driver Rating', min_value=0, max_value=5)
        Days_Policy_Accident = st.selectbox('Days Policy Accident', list(days_policy_accident_map.keys()))
        Days_Policy_Claim = st.selectbox('Days Policy Claim', list(days_policy_claim_map.keys()))
        PastNumberOfClaims = st.selectbox('Past Number of Claims', list(past_number_of_claims_map.keys()))
        AgeOfVehicle = st.selectbox('Age of Vehicle', list(age_of_vehicle_map.keys()))

    # Input fields layout continued
    col5, col6 = st.columns(2)

    with col5:
        PoliceReportFiled = st.selectbox('Police Report Filed', list(police_report_filed_map.keys()))
        WitnessPresent = st.selectbox('Witness Present', list(witness_present_map.keys()))
        AgentType = st.selectbox('Agent Type', list(agent_type_map.keys()))
        NumberOfSuppliments = st.selectbox('Number of Supplements', list(number_of_suppliments_map.keys()))

    with col6:
        AddressChange_Claim = st.selectbox('Address Change Claim', list(address_change_claim_map.keys()))
        NumberOfCars = st.selectbox('Number of Cars', list(number_of_cars_map.keys()))
        Year = st.number_input('Year', min_value=1994, max_value=1996)

    # Prepare data for prediction
    data = {
        'WeekOfMonth': WeekOfMonth,
        'DayOfWeek': map_day[DayOfWeek],
        'Manufacture_company': Manufacture_company_map[Manufacture_company],
        'AccidentArea': AccidentArea_map[AccidentArea],
        'DayOfWeekClaimed': map_day[DayOfWeekClaimed],
        'MonthClaimed': map_month[MonthClaimed],
        'WeekOfMonthClaimed': WeekOfMonthClaimed,
        'Sex': gender_map[Sex],
        'MaritalStatus': maritalstatus_map[MaritalStatus],
        'Age': Age,
        'Fault': fault_map[Fault],
        'PolicyType': policy_type_map[PolicyType],
        'VehicleCategory': vehicle_category_map[VehicleCategory],
        'VehiclePrice': Price_map[VehiclePrice],
        'RepNumber': RepNumber,
        'Deductible': Deductible,
        'DriverRating': DriverRating,
        'Days_Policy_Accident': days_policy_accident_map[Days_Policy_Accident],
        'Days_Policy_Claim': days_policy_claim_map[Days_Policy_Claim],
        'PastNumberOfClaims': past_number_of_claims_map[PastNumberOfClaims],
        'AgeOfVehicle': age_of_vehicle_map[AgeOfVehicle],
        'PoliceReportFiled': police_report_filed_map[PoliceReportFiled],
        'WitnessPresent': witness_present_map[WitnessPresent],
        'AgentType': agent_type_map[AgentType],
        'NumberOfSuppliments': number_of_suppliments_map[NumberOfSuppliments],
        'AddressChange_Claim': address_change_claim_map[AddressChange_Claim],
        'NumberOfCars': number_of_cars_map[NumberOfCars],
        'Year': Year
    }

    # Convert data to DataFrame
    df1 = pd.DataFrame([data])

    # Apply scaling
    df_scaled = scaler.fit_transform(df1)
    data_array = df1.to_numpy()

    # Make prediction
    if st.button('Predict'):
        prediction = model.predict(data_array)
        if prediction[0] == 1:
            st.error('Fraud Found')
        else:
            st.success('No Fraud Found')

elif page == 'About':
    st.title('About')
    st.write('This is a Streamlit web application for predicting automotive insurance fraud.')

# Increase font size for entire app
st.markdown(
    """
    <style>
    .css-17eq0hr {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
