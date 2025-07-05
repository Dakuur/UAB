from flask import Flask, render_template, request, redirect, url_for, flash
import firebase_admin
from firebase_admin import credentials, auth, firestore

# Inicialitzar la app de Firebase
cred = credentials.Certificate("tax.json")  # Assegura't de tenir el fitxer .json de Firebase
firebase_admin.initialize_app(cred)

# Inicialitzar la app de Flask
app = Flask(__name__, template_folder='../front-end/templates', static_folder='../front-end/static')
app.secret_key = 'supersecretkey'  # Per a les sessions i missatges flash

# Obtenir la referència a Firestore
db = firestore.client()

# Ruta per a la pàgina principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta per a la pàgina de login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        try:
            # Intentem obtenir l'usuari per correu electrònic
            user = auth.get_user_by_email(email)
            
            # Aquí hauries d'afegir la lògica per verificar la contrasenya (Firebase Auth o el teu sistema)
            # Firebase Auth no permet verificar la contrasenya directament, per tant, aquest pas només verifica l'email
            flash(f'Benvingut, {user.email}', 'success')  # Missatge de benvinguda
            return redirect(url_for('index'))

        except firebase_admin.auth.UserNotFoundError:
            flash('Usuari no trobat', 'error')  # Missatge d'error si l'usuari no existeix
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')  # Missatge d'error general

    return render_template('login.html')


# Ruta per al registre d'usuaris
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Obtenir les dades del formulari
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')
        phone = request.form.get('phone')
        account_type = request.form.get('account_type')  # Per si és per si mateix o un familiar
        gender = request.form.get('gender')
        birth_date = request.form.get('birth_date')
        known_diseases = request.form.get('known_diseases')
        medication = request.form.get('medication')
        medical_history = request.form.get('medical_history')

        try:
            # Crear un nou usuari a Firebase Authentication
            user = auth.create_user(
                email=email,
                password=password
            )

            # Crear el document a Firestore amb la informació addicional
            user_data = {
                'account_type': account_type,
                'name': name,
                'phone': phone,
                'email': email,
                'gender': gender,
                'birth_date': birth_date,
                'known_diseases': known_diseases,
                'medication': medication,
                'medical_history': medical_history,
                'uid': user.uid
            }

            # Guardar les dades de l'usuari a Firestore sota la col·lecció "users"
            db.collection('users').document(user.uid).set(user_data)

            flash(f'Usuari {user.email} registrat correctament', 'success')
            return redirect(url_for('index'))
        except firebase_admin.auth.EmailAlreadyExistsError:
            flash('El correu ja està registrat', 'error')
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')

    return render_template('register.html')

# Iniciar l'aplicació Flask
if __name__ == '__main__':
    app.run(debug=True)
