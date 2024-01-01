from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, or_
from flask_migrate import Migrate
from sqlalchemy.orm import relationship
from flask_bootstrap import Bootstrap
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import holidays
import joblib
from collections import defaultdict
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'  # Replace with your actual PostgreSQL connection details
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key'  # Change this to a secure random key in production

bootstrap = Bootstrap(app)  # Initialize Bootstrap
db = SQLAlchemy(app)
migrate = Migrate(app, db)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    country = db.Column(db.String(2), nullable=False)
    birth_day = db.Column(db.Date, nullable=False)
    is_locked = db.Column(db.Boolean, default=False)
    money_in_no = db.Column(db.Float, default=0)
    money_out_no = db.Column(db.Float, default=0)
    money_in_gbp = db.Column(db.Float, default=0)
    money_out_gbp = db.Column(db.Float, default=0)
    money_in_day_no = db.Column(db.Float, default=0)
    money_out_day_no = db.Column(db.Float, default=0)
    money_in_sum = db.Column(db.Float, default=0)
    money_out_sum = db.Column(db.Float, default=0)
    money_in_day_sum = db.Column(db.Float, default=0)
    money_out_day_sum = db.Column(db.Float, default=0)
    balance_no = db.Column(db.Float, default=0)
    balance_day_no = db.Column(db.Float, default=0)
    balance_sum = db.Column(db.Float, default=0)
    balance_day_sum = db.Column(db.Float, default=0)
    balance = db.Column(db.Float, default=0)
    # Add this relationship in the User model
    transactions = db.relationship('Transaction', backref='user', lazy=True)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(20), nullable=False)
    state = db.Column(db.String(20), nullable=False)
    created_date = db.Column(db.DateTime, default=db.func.current_timestamp())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount_gb = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(10))

@app.route('/')
def home():
    # Fetch user transactions for the home page
    if 'username' in session:
        user = User.query.filter_by(username=session['username']).first()

        # Check if user is locked and provide a sign-out option
        if user.is_locked:
            session.pop('username', None)  # Clear session on sign-out
            return redirect(url_for('signin'))

        # Calculate total amount received
        received_amount = db.session.query(func.sum(Transaction.amount_gb)).filter_by(user_id=user.id, type='TOPUP', state='COMPLETED').scalar() or 0.0

        # Calculate total amount spent
        spent_amount = db.session.query(func.sum(Transaction.amount_gb)).filter_by(user_id=user.id, state='COMPLETED').\
            filter(or_(Transaction.type == 'CARD_PAYMENT', Transaction.type == 'TRANSFER', Transaction.type == 'ATM', Transaction.type == 'FEE')).\
            scalar() or 0.0

        # Calculate balance
        balance = received_amount - spent_amount

        transactions = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.created_date.desc()).all()
        return render_template('home.html', user=user, transactions=transactions, balance=balance, received_amount=received_amount, spent_amount=spent_amount)
    else:
        return redirect(url_for('signin'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        country = request.form.get('country')
        birth_day_str = request.form.get('birth_day')

        # Convert birth_day_str to a Python date object
        birth_day = datetime.strptime(birth_day_str, '%Y-%m-%d').date()

        # Check if the username already exists
        if User.query.filter_by(username=username).first():
            return 'Username already exists'

        # Create a new user
        new_user = User(username=username, password=password, country=country, birth_day=birth_day, is_locked=False)
        db.session.add(new_user)
        db.session.commit()

        session['username'] = username
        return redirect(url_for('home'))

    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if the username and password match
        user = User.query.filter_by(username=username, password=password).first()

        if user:
            # Check if the user is locked
            if user.is_locked:
                return 'Your account is locked. Please contact support.'

            session['username'] = username
            return redirect(url_for('home'))
        else:
            return 'Invalid username or password'


    return render_template('signin.html')

def load_my_model():
    # Load your H5 model
    model = load_model('fraud_detect_revoult1.h5')
    return model

no_data = []
# Function to get holidays for country and year
def country_holidays(country, year):
    # Using URL get all holidays for country and specified year
    api_url = f"https://date.nager.at/api/v3/publicholidays/{year}/{country}"

    response = requests.get(api_url)
    # Check if the response is successful
    if response.status_code == 200:
        return response.json()
    else:
        # If the response is not successful, return an empty list
        no_data.append(country)
        return []
    
def is_bank_holiday(created_date_transaction, country):
    # Extract the year from the created_date_transaction
    year = created_date_transaction.year

    # Get the list of holidays for the specified country and year
    holidays_list = country_holidays(country, year)

    # Check if created_date_transaction is in the list of holidays
    return created_date_transaction.strftime('%Y-%m-%d') in holidays_list

def public_holiday_col(final_data):
    final_data['is_bank_holiday'] = 0  # Initialize the column with 0


    # Dictionary to map country codes to prov values
    country_prov_mapping = {
        'RE': 'RE',
        'GP': 'GP',
        'PF': 'PF',
        'MQ': 'MQ',
        'NC': 'NC',
        'YT': 'YT',
        'BL': 'BL',
        'MF': 'MF'
    }

    # Dictionary to store holidays for each country
    all_holidays = {}
    
    # Extract the year from the created_date_transaction and country
    year = final_data['created_date_transaction'].dt.year
    country = final_data['country']

    # Iterate over each row to check if each country is in country_prov_mapping
    for index, row in final_data.iterrows():
        current_country = row['country']
        current_year = row['created_date_transaction'].year

        if current_country in country_prov_mapping:
            # Fetch holidays for countries in country_prov_mapping
            holidays_for_country = holidays.FRA(years=current_year, prov=current_country)
            all_holidays[current_country] = list(holidays_for_country.keys())

            # Update is_bank_holiday column based on the provided holidays for the current country
            holidays_list = all_holidays[current_country]
            for holiday_date in holidays_list:
                # Extract month and day from the holiday date
                holiday_month, holiday_day = holiday_date.month, holiday_date.day

                # Set is_bank_holiday to 1 for matching rows
                final_data.loc[(final_data['country'] == current_country) &
                            (final_data['created_date_transaction'].dt.month == holiday_month) &
                            (final_data['created_date_transaction'].dt.day == holiday_day),
                            'is_bank_holiday'] = 1
        else:
            # For other countries, use logic from requests API
            final_data.loc[(final_data['country'] == current_country) & 
                        (final_data['created_date_transaction'].dt.year == current_year),
                        'is_bank_holiday'] = final_data[
                (final_data['country'] == current_country) & 
                (final_data['created_date_transaction'].dt.year == current_year)
            ].apply(lambda row: 1 if is_bank_holiday(row['created_date_transaction'], current_country) else 0, axis=1)

def transaction_stats_col(final_data):
    #Getting Transaction Date
    final_data['T_CREATED_DATE_DAY'] = final_data['created_date_transaction'].dt.day

    # Create features related to the number and sum of transactions for each user
    final_data['TRANSACTION_NO'] = final_data.groupby(['user_id'], as_index=False)['id_transaction'].cumcount() + 1
    final_data['TRANSACTION_DAY_NO'] = final_data.groupby(['user_id', 'T_CREATED_DATE_DAY'], as_index=False)['id_transaction'].cumcount() + 1
    final_data['TRANSACTION_SUM'] = final_data.groupby(['user_id'], as_index=False)['amount_gb'].cumsum()
    final_data['TRANSACTION_DAY_SUM'] = final_data.groupby(['user_id', 'T_CREATED_DATE_DAY'], as_index=False)['amount_gb'].cumsum()

    # Create features for other transaction types
    transaction_types = ['TOPUP', 'CARD_PAYMENT', 'FEE', 'EXCHANGE', 'TRANSFER', 'ATM']
    for transaction_type in transaction_types:
        final_data[f'{transaction_type}_NO'] = final_data.groupby(['user_id', 'type'], as_index=False)['id_transaction'].cumcount() + 1
        final_data[f'{transaction_type}_DAY_NO'] = final_data.groupby(['user_id', 'T_CREATED_DATE_DAY', 'type'], as_index=False)['id_transaction'].cumcount() + 1
        final_data[f'{transaction_type}_SUM'] = final_data.groupby(['user_id', 'type'], as_index=False)['amount_gb'].cumsum()
        final_data[f'{transaction_type}_DAY_SUM'] = final_data.groupby(['user_id', 'T_CREATED_DATE_DAY', 'type'], as_index=False)['amount_gb'].cumsum()

    # Create features for transaction states
    transaction_states = ['COMPLETED', 'REVERTED', 'FAILED', 'DECLINED']
    for transaction_state in transaction_states:
        final_data[f'{transaction_state}_NO'] = final_data.groupby(['user_id', 'state'], as_index=False)['id_transaction'].cumcount() + 1
        final_data[f'{transaction_state}_DAY_NO'] = final_data.groupby(['user_id', 'T_CREATED_DATE_DAY', 'state'], as_index=False)['id_transaction'].cumcount() + 1
        final_data[f'{transaction_state}_SUM'] = final_data.groupby(['user_id', 'state'], as_index=False)['amount_gb'].cumsum()
        final_data[f'{transaction_state}_DAY_SUM'] = final_data.groupby(['user_id', 'T_CREATED_DATE_DAY', 'state'], as_index=False)['amount_gb'].cumsum()
    
     # Create binary columns indicating whether the transaction type is present
    for transaction_type in transaction_types:
        final_data[f'IS_{transaction_type}'] = (final_data['type'] == transaction_type).astype(int)

    # Create binary columns indicating whether the transaction state is present
    for transaction_state in transaction_states:
        final_data[f'IS_{transaction_state}'] = (final_data['state'] == transaction_state).astype(int)
    
    # Create 'MONEY_IN_GBP' column
    final_data['MONEY_IN_GBP'] = ((final_data['state'] == 'COMPLETED') & (final_data['type'] == 'TOPUP')).astype(int) * final_data['amount_gb']

    # Create 'MONEY_OUT_GBP' column
    final_data['MONEY_OUT_GBP'] = ((final_data['state'] == 'COMPLETED') & (final_data['type'].isin(['CARD_PAYMENT', 'TRANSFER', 'ATM', 'FEE']))).astype(int) * final_data['amount_gb']

def get_attribute_value(user, attribute_name):
    return getattr(user, attribute_name, None)

def finance_col(final_data):
    # Find users in the database based on user_id
    user = User.query.filter(User.id.in_(final_data['user_id'])).first()

    # Calculate the transaction amount in logarithmic scale
    final_data['LOG_AMOUNT_GBP'] = final_data['amount_gb'].apply(lambda x: np.log1p(x) if x > 0 else 0)
    # Create 'MONEY_IN_ONE' column
    final_data['MONEY_IN_ONE'] = ((final_data['state'] == 'COMPLETED') & (final_data['type'] == 'TOPUP')).astype(int)

    # Create 'MONEY_OUT_ONE' column
    final_data['MONEY_OUT_ONE'] = ((final_data['state'] == 'COMPLETED') & (final_data['type'].apply(lambda x: x in ['CARD_PAYMENT', 'TRANSFER', 'ATM', 'FEE']))).astype(int)

    # Assigning respective values
    final_data['MONEY_IN_NO'] = user.money_in_no
    final_data['MONEY_IN_SUM'] = user.money_in_sum
    final_data['MONEY_IN_DAY_NO'] = user.money_in_day_no
    final_data['MONEY_IN_DAY_SUM'] = user.money_in_day_sum

    final_data['MONEY_OUT_NO'] = user.money_out_no
    final_data['MONEY_OUT_SUM'] = user.money_out_sum
    final_data['MONEY_OUT_DAY_NO'] = user.money_out_day_no
    final_data['MONEY_OUT_DAY_SUM'] = user.money_out_day_sum

    # Calculate balance columns for the subset
    final_data['BALANCE_NO'] = user.balance_no
    final_data['BALANCE_SUM'] = user.balance_sum
    final_data['BALANCE_DAY_NO'] = user.balance_day_no
    final_data['BALANCE_DAY_SUM'] = user.balance_day_sum

    # Calculating Balance after transaction for the subset
    final_data['BALANCE'] = user.balance

    
def encode_cat_col(final_data):
    # Load the LabelEncoder objects
    le_type = joblib.load('type_label_encoder.joblib')
    le_state = joblib.load('state_label_encoder.joblib')
    le_country = joblib.load('country_label_encoder.joblib')
    le_age_group = joblib.load('age_group_label_encoder.joblib')
    le_currency = joblib.load('currency_label_encoder.joblib')

    # Use the loaded LabelEncoder objects for inference
    final_data['type'] = le_type.transform(final_data['type'])
    final_data['state'] = le_state.transform(final_data['state'])
    final_data['country'] = le_country.transform(final_data['country'])
    final_data['U_AGE_GROUP'] = le_age_group.transform(final_data['U_AGE_GROUP'])
    final_data['currency'] = le_currency.transform(final_data['currency'])

def create_time_cols(final_data):
    #Check how many transactions user made in different currenices
    final_data['CURRENCIES'] = final_data.groupby('user_id')['currency'].apply(lambda x: (~pd.Series(x).duplicated()).cumsum())

    # Converting to UTC time for created_date_transaction
    final_data['T_CREATED_DATE_GB'] = final_data['created_date_transaction'].dt.tz_localize('UTC').dt.tz_convert('Europe/London')

    # Extract relevant information from datetime column
    final_data['year_GB'] = final_data['T_CREATED_DATE_GB'].dt.year
    final_data['month_GB'] = final_data['T_CREATED_DATE_GB'].dt.month
    final_data['day_GB'] = final_data['T_CREATED_DATE_GB'].dt.day
    final_data['hour_GB'] = final_data['T_CREATED_DATE_GB'].dt.hour

    # Extract the hour of the day and month according to server time for created_date_transaction
    final_data['hour_of_day_GB'] = final_data['T_CREATED_DATE_GB'].dt.hour
    final_data['month'] = final_data['created_date_transaction'].dt.month
    # Calculate the difference between the creation date of the transaction and the user's birthdate
    final_data['AGE_AT_TRANSACTION'] = (final_data['created_date_transaction'] - pd.to_datetime(final_data['birth_day'])).dt.days // 365

def preprocess_input_data(user_data, transaction_data):
    # Create DataFrames from dictionaries
    user_data = pd.DataFrame(user_data, index=[0])
    transaction_data = pd.DataFrame(transaction_data, index=[0])
    
    # Calculate user's age
    user_age = (datetime.now().date() - pd.to_datetime(user_data['birth_day']).dt.date.iloc[0]).days // 365

    # Define age groups
    labels = [f'{i}-{i+9}' for i in range(0, 100, 10)]

    # Create 'U_AGE_GROUP' column
    user_data['U_AGE_GROUP'] = pd.cut([user_age], bins=range(0, 101, 10), right=False, labels=False)
    user_data['U_AGE_GROUP'] = labels[int(user_data['U_AGE_GROUP'][0])]

    # Convert categorical features to numerical using LabelEncoder
    le = LabelEncoder()

    # Add your existing columns here
    user_data['COUNTRY'] = le.fit_transform([user_data['country']])[0]
    transaction_data['TYPE'] = le.fit_transform([transaction_data['type']])[0]
    transaction_data['STATE'] = le.fit_transform([transaction_data['state']])[0]
    transaction_data['CURRENCY'] = le.fit_transform([transaction_data['currency']])[0]

    # 'AMOUNT_GBP' in transaction_data
    transaction_data['AMOUNT_GBP'] = transaction_data['amount_gb']

    # 'USER_AGE' in user_data
    user_data['USER_AGE'] = user_age

    # Add user data to transactions where they are the same
    final_data = pd.merge(transaction_data, user_data, left_on='user_id', right_on='id', how='left', suffixes=('_transaction', '_user'))


    # Renaming columns to make sense of data
    final_data = final_data.rename(columns={
        'age': 'USER_AGE',
        'CREATED_DATE_transaction': 'T_CREATED_DATE',
        'CREATED_DATE_user': 'U_CREATED_DATE',
        'BIRTH_DATE': 'U_BIRTH_DATE',
        'AGE_GROUP': 'U_AGE_GROUP'
    })

    final_data = final_data[
        ['id_transaction', 'user_id', 'country', 'created_date_transaction', 'type', 'state', 'amount_gb', 'currency', 'created_date_user', 'birth_day', 'U_AGE_GROUP']
    ]

    final_data['USER_AGE'] = (datetime.now() - pd.to_datetime(final_data['birth_day'])).astype('<m8[Y]')

    public_holiday_col(final_data)

    create_time_cols(final_data)

    transaction_stats_col(final_data)

    finance_col(final_data)

    # Drop unnecessary columns
    final_data.drop(columns=['created_date_user', 'birth_day', 'created_date_transaction', 'T_CREATED_DATE_GB'], inplace=True)

    mapping = { 'True': 1, 'No': 0 }
    final_data['is_bank_holiday'] = final_data['is_bank_holiday'].map(mapping)

    encode_cat_col(final_data)
    return final_data

def predict_transaction_outcome(preprocessed_data):
    preprocessed_data = preprocessed_data.iloc[:, :9].join(preprocessed_data.iloc[:, -81:])
    print(preprocessed_data)
    # Load the model
    model = load_model('fraud_detect_revoult12.h5')  # Corrected the model name
    # Make a prediction
    prediction = model.predict(preprocessed_data)

    # Return the predicted outcome
    return prediction

def perform_post_transaction_actions(user, transaction): 
    # Extract necessary data from user and transaction
    user_data = {
        'id': user.id,
        'username': user.username,
        'country': user.country,
        'birth_day': user.birth_day,
        'created_date': user.created_date
    }

    transaction_data = {
        'id': transaction.id,
        'user_id': transaction.user_id,
        'created_date': transaction.created_date,
        'type': transaction.type,
        'state': transaction.state,
        'amount_gb': transaction.amount_gb,
        'currency': transaction.currency
    }
    #calculate finance columns
    if transaction.state == 'COMPLETED':
        if transaction.type == 'TOPUP':
           user.money_in_no += 1
           user.money_in_sum += transaction.amount_gb
        else:
            user.money_out_no += 1
            user.money_out_sum += transaction.amount_gb

    # Calculate MONEY_IN_DAY_NO and MONEY_IN_DAY_SUM
    money_in_per_day = defaultdict(float)
    for transaction in [t for t in user.transactions if t.type == 'TOPUP']:
        money_in_per_day[transaction.created_date.date()] += transaction.amount_gb

    user.money_in_day_no = len(money_in_per_day)
    user.money_in_day_sum = sum(money_in_per_day.values())

    # Calculate MONEY_OUT_DAY_NO and MONEY_OUT_DAY_SUM
    money_out_per_day = defaultdict(float)
    for transaction in [t for t in user.transactions if t.type in ['CARD_PAYMENT', 'TRANSFER', 'ATM', 'FEE']]:
        money_out_per_day[transaction.created_date.date()] += transaction.amount_gb

    user.money_out_day_no = len(money_out_per_day)
    user.money_out_day_sum = sum(money_out_per_day.values())

    #calculate balances 
    user.balance_no = user.money_in_no - user.money_out_no
    user.balance_sum = user.money_in_sum - user.money_out_sum
    user.balance_day_no = user.money_in_day_no - user.money_out_day_no
    user.balance_day_sum = user.money_in_day_sum - user.money_out_day_sum
    user.balance = user.money_in_sum - user.money_out_sum

    # Commit the changes to the database
    db.session.commit()
    
    # Preprocess the transaction data
    preprocessed_data = preprocess_input_data(user_data, transaction_data)

    # Call your H5 model prediction function
    prediction = predict_transaction_outcome(preprocessed_data)

    print(f'The prediction performed on transaction is: {prediction}')
    
    # 0.6 is threshold means normal transaction, do nothing
    if prediction.astype(float) < 0.6:
        return

    # else lock the user
    user.is_locked = True
    db.session.commit()

@app.route('/perform_transaction', methods=['GET', 'POST'])
def perform_transaction():
    if 'username' in session:
        user = User.query.filter_by(username=session['username']).first()
        currency_options = ['PLN', 'RON', 'GBP', 'EUR', 'HUF', 'BGN', 'CZK', 'SEK', 'USD','SGD', 'CHF', 'HRK', 'NOK', 'JPY', 'CAD', 'UAH', 'DKK', 'TRY','BTC', 'PEN', 'AED', 'AUD', 'BYN', 'MYR', 'RSD', 'SCR', 'IDR','HKD', 'PHP', 'MAD', 'RUB', 'ETH', 'NZD', 'ISK', 'ZAR', 'INR','MXN', 'XPF', 'SAR', 'COP', 'THB', 'ALL', 'TWD', 'ILS', 'GEL','MKD', 'XRP', 'BRL', 'ARS', 'KRW', 'EGP', 'CRC', 'CLP', 'VND','TZS', 'BAM', 'QAR', 'BBD', 'BCH', 'KZT', 'CNY', 'LTC', 'AWG','XCD', 'MUR', 'SRD', 'MDL', 'ZMW', 'BSD', 'BOB', 'GHS', 'DOP','LKR', 'TND', 'MMK', 'KES', 'ETB', 'JOD', 'XOF', 'MOP', 'MGA','MNT', 'OMR', 'CVE', 'GMD', 'LBP', 'MWK', 'BHD', 'NAD', 'BDT','GTQ', 'PKR', 'AMD', 'NGN', 'HNL', 'JMD', 'LAK', 'BWP', 'XAF','MZN', 'AZN', 'FJD', 'BND', 'BZD', 'VES', 'DZD', 'UZS', 'NPR','KGS', 'MVR']

        if request.method == 'POST':
            transaction_type = request.form.get('transaction_type')
            amount_gb = float(request.form.get('amount_gb'))
            currency = request.form.get('currency')
            state = request.form.get('transaction_state')

            # Calculate total amount received
            received_amount = db.session.query(func.sum(Transaction.amount_gb)).filter_by(user_id=user.id, type='TOPUP', state='COMPLETED').scalar() or 0.0

            # Calculate total amount spent
            spent_amount = db.session.query(func.sum(Transaction.amount_gb)).filter_by(user_id=user.id, state='COMPLETED').filter(or_(Transaction.type == 'CARD_PAYMENT', Transaction.type == 'TRANSFER', Transaction.type == 'ATM', Transaction.type == 'FEE')).scalar() or 0.0

            # Calculate balance
            balance = received_amount - spent_amount

            # Check if the new transaction will result in a negative balance
            if transaction_type in ['CARD_PAYMENT', 'TRANSFER', 'ATM', 'FEE'] and balance - amount_gb < 0:
                return 'Transaction not allowed. Insufficient balance.'

            # Create a new transaction
            new_transaction = Transaction(
                type=transaction_type,
                state=state,
                user_id=user.id,
                amount_gb=amount_gb,
                currency=currency
            )
            db.session.add(new_transaction)
            db.session.commit()

            # Perform post-transaction actions
            perform_post_transaction_actions(user, new_transaction)    

            return redirect(url_for('home'))

        return render_template('perform_transaction.html', user=user, currency_options=currency_options)
    else:
        return redirect(url_for('signin'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the database tables
    app.run(debug=True)
