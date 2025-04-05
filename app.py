from minio import Minio
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
from flask import Flask, render_template, jsonify, request, redirect, session, url_for, send_file, make_response, send_from_directory, url_for, flash
import base64
import zipfile
import io
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'J5e-t4s-0t4-PuyE-TrWQ'

# Подключение к базе данных PostgreSQL
def get_db_connection():
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="app-db",
        user="mlflow",
        password="password"
    )
    return conn

# MinIO конфигурация
MINIO_ACCESS_KEY = 'mlflow'
MINIO_SECRET_KEY = 'password'
MINIO_ENDPOINT = 's3:9000'
MINIO_BUCKET_NAME = 'myappbucket'

# Проверка переменных окружения
if not all([MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_ENDPOINT, MINIO_BUCKET_NAME]):
    raise ValueError("Необходимо задать переменные окружения MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_ENDPOINT и MINIO_BUCKET_NAME")

# Подключение к MinIO
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False  # True для HTTPS
)

# Проверка существования бакета и его создание, если необходимо
try:
    if not minio_client.bucket_exists(MINIO_BUCKET_NAME):
        minio_client.make_bucket(MINIO_BUCKET_NAME)
except Exception as e:
    print(f"Ошибка при работе с бакетом: {e}")

app.config['UPLOAD_FOLDER'] = 'uploads'  # Папка для временного хранения загруженных файлов
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

cl = ["cat", "chicken", "cow", "dog", "fox", "goat", "horse", "person", "racoon", "skunk"]
cl_dict = {i: c for i, c in enumerate(cl)}
print(cl_dict)
model = YOLO("best.pt")

# Кэш для результатов YOLO (словарь: имя файла -> список найденных объектов)
yolo_results_cache = {}

# Форма регистрации
class RegistrationForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[DataRequired(), Length(min=2, max=20)])
    password = PasswordField('Пароль', validators=[DataRequired()])
    confirm_password = PasswordField('Подтвердите пароль', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Зарегистрироваться')


# Форма входа
class LoginForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[DataRequired()])
    password = PasswordField('Пароль', validators=[DataRequired()])
    submit = SubmitField('Войти')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        hashed_password = generate_password_hash(password)  # Хэширование пароля

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (%s, %s) RETURNING id", (username, hashed_password))
        user_id = cur.fetchone()[0]  # Получаем ID нового пользователя
        conn.commit()
        cur.close()
        conn.close()

        # Создание бакета для пользователя
        user_bucket_name = f"{MINIO_BUCKET_NAME}-{user_id}"  # Имя бакета, основанное на ID пользователя
        try:
            if not minio_client.bucket_exists(user_bucket_name):
                minio_client.make_bucket(user_bucket_name)
        except Exception as e:
            print(f"Ошибка при создании бакета {user_bucket_name}: {e}")

        flash('Регистрация прошла успешно! Теперь вы можете войти в систему.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    message = None  # Переменная для хранения сообщения
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user and check_password_hash(user[2], password):  # user[2] - это хэш пароля
            session['username'] = username  # Сохранение имени пользователя в сессии
            message = 'Вы вошли в систему!'  # Устанавливаем сообщение
            return redirect(url_for('index2', username=username))  # Переход на главную страницу с именем пользователя
        else:
            message = 'Неправильное имя пользователя или пароль'  # Устанавливаем сообщение

    return render_template('login.html', form=form, message=message)

@app.route('/index/<username>', methods=['GET', 'POST'])
def index2(username):
    confidence_threshold = None  # Установите значение по умолчанию
    if request.method == "POST":
        selected_class = request.form.get("class")
        confidence_threshold = request.form.get("confidence")  # Получаем значение из формы
        if selected_class and confidence_threshold:
            try:
                confidence_threshold = float(confidence_threshold)  # Приводим к float
                return redirect(url_for("show_results", selected_class=selected_class, confidence_threshold=confidence_threshold))
            except ValueError:
                flash('Пожалуйста, введите допустимое значение для порога уверенности.', 'danger')
                return redirect(url_for('index2', username=username))
    return render_template("index.html", classes=cl, username=username, confidence_threshold=confidence_threshold)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'username' not in session:
        return jsonify({'error': 'Необходимо войти в систему'}), 403

    username = session['username']

    # Получаем ID пользователя из базы данных
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    user_id = cur.fetchone()[0]
    cur.close()
    conn.close()

    user_bucket_name = f"{MINIO_BUCKET_NAME}-{user_id}"  # Имя бакета пользователя
    files = request.files.getlist('files[]')

    if not files:
        return jsonify({'error': 'Нет файлов'}), 400

    filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                minio_client.fput_object(user_bucket_name, filename, filepath)
                os.remove(filepath)
                filenames.append(filename)
            except Exception as e:
                return jsonify({'error': f'Ошибка при загрузке файла {filename} в MinIO: {e}'}), 500
        else:
            return jsonify({'error': f'Недопустимый тип файла: {file.filename}'}), 400

    return jsonify({'message': 'Файлы успешно загружены', 'filenames': filenames}), 200

def non_max_suppression(boxes, scores, threshold):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[np.where(iou <= threshold)[0] + 1]

    return keep

def process_image(image_np, selected_class, confidence_threshold):
    height, width, _ = image_np.shape
    new_width = 480
    new_height = int(height * (new_width / width))
    resized_image = cv2.resize(image_np, (new_width, new_height))

    results = model.predict(resized_image)
    image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

    boxes = []
    scores = []
    labels = []

    for box in results[0].boxes.data:
        x1, y1, x2, y2, score, label = box
        if cl_dict[int(label)] == selected_class and score >= confidence_threshold:
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            labels.append(int(label))

    # Применяем Non-Maximum Suppression
    boxes = np.array(boxes)
    scores = np.array(scores)
    indices = non_max_suppression(boxes, scores, threshold=0.4)  # Порог IoU (перекрытие рамок)

    for i in indices:
        box = boxes[i]
        x1, y1, x2, y2 = box
        score = scores[i]
        text = f"{cl_dict[labels[i]]} {score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        if y1 - text_height - 10 >= 0:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, text, (int(x1 + 5), int(y1 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    _, image_buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(image_buffer).decode('utf-8')
    return image_base64

def find_images(selected_class, confidence_threshold):
    if 'username' not in session:
        return []

    username = session['username']

    # Получаем ID пользователя из базы данных
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    user_id = cur.fetchone()[0]
    cur.close()
    conn.close()

    user_bucket_name = f"{MINIO_BUCKET_NAME}-{user_id}"  # Имя бакета пользователя
    print(user_bucket_name)
    found_images = []

    objects = minio_client.list_objects(user_bucket_name, prefix="")
    for obj in objects:
        if obj.object_name.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                response = minio_client.get_object(user_bucket_name, obj.object_name)
                image_bytes = response.read()
                image_np = np.asarray(Image.open(BytesIO(image_bytes)))
                results = model.predict(image_np)

                for box in results[0].boxes.data:
                    x1, y1, x2, y2, score, label = box
                    if cl_dict[int(label)] == selected_class and score >= confidence_threshold:
                        found_images.append(obj.object_name)
                        break
            except Exception as e:
                print(f"Ошибка обработки изображения {obj.object_name}: {e}")
                continue
    return found_images


@app.route("/results/<selected_class>/<float:confidence_threshold>")
def show_results(selected_class, confidence_threshold):
    images = []
    cache_key = (selected_class, confidence_threshold)

    if cache_key not in yolo_results_cache:
        found_images = find_images(selected_class, confidence_threshold)  # Получаем список файлов
        yolo_results_cache[cache_key] = found_images
    else:
        found_images = yolo_results_cache[cache_key]


    for filename in found_images:
        try:
            response = minio_client.get_object(MINIO_BUCKET_NAME, filename)
            image_bytes = response.read()
            image_np = np.asarray(Image.open(BytesIO(image_bytes)))
            image_base64 = process_image(image_np, selected_class, confidence_threshold)
            images.append(image_base64)
        except Exception as e:
            print(f"Ошибка обработки изображения {filename}: {e}")
            continue

    return render_template("results.html", images=images, selected_class=selected_class, confidence_threshold=confidence_threshold,
                           download_url=url_for('download_results', selected_class=selected_class, confidence_threshold=confidence_threshold),
                           username=session['username'])


@app.route("/download/<selected_class>/<float:confidence_threshold>")
def download_results(selected_class, confidence_threshold):
    cache_key = (selected_class, confidence_threshold)
    found_images = yolo_results_cache.get(cache_key, []) # Получаем из кэша

    if not found_images:
        return "Нет изображений, соответствующих критериям поиска.", 404

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for filename in found_images:
            try:
                response = minio_client.get_object(MINIO_BUCKET_NAME, filename)
                zf.writestr(filename, response.read())
            except Exception as e:
                print(f"Ошибка загрузки изображения {filename} в архив: {e}")
                continue

    memory_file.seek(0)
    response = make_response(memory_file.read())
    response.headers["Content-Disposition"] = "attachment; filename=found_images.zip"
    response.headers["Content-Type"] = "application/zip"
    return response

@app.route('/logout')
def logout():
    session.pop('username', None)  # Удаляем имя пользователя из сессии
    return redirect(url_for('login'))  # Перенаправляем на страницу входа


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)  # host='0.0.0.0' позволяет обращаться к серверу извне