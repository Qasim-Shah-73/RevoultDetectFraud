from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, or_
from flask_migrate import Migrate  # Import the Migrate class
from flask_bootstrap import Bootstrap  # Import Bootstrap
from datetime import datetime
#from keras.models import load_model
import numpy as np

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

#def load_my_model():
    # Load your H5 model
 #   model = load_model('path/to/your/my_model.h5')
  #  return model

def predict_transaction_outcome(user_data, transaction_data):
    # Load the model
   # model = load_my_model()

    # Convert user and transaction data to a format suitable for your model
    # Example: you might need to preprocess the data, scale it, etc.

    #final_data = np.array([])
    # Make a prediction
#    prediction = model.predict(final_data)

    # Return the predicted outcome
    return 0

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

     # Call your H5 model prediction function
    prediction = predict_transaction_outcome(user_data, transaction_data)

    # 0 means normal transaction, do nothing
    if prediction == 0:
        return

    # 1 means potential issue, lock the user
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
