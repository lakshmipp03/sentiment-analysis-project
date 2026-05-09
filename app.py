from flask import Flask, render_template, request, redirect, session
from flask_session import Session

import pickle
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import nltk

from preprocess import clean_text
from wordcloud import WordCloud
from langdetect import detect

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Flask app
app = Flask(__name__)

# Secret key
app.secret_key = "sentimentproject"

# Session configuration
app.config["SESSION_TYPE"] = "filesystem"

Session(app)

# Load trained model
model = pickle.load(open("sentiment_model.pkl", "rb"))

# Load vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# ---------------- LOGIN PAGE ---------------- #

@app.route("/")
def login():

    return render_template("login.html")


# ---------------- LOGIN CHECK ---------------- #

@app.route("/login", methods=["POST"])
def check_login():

    username = request.form["username"]

    password = request.form["password"]

    if username == "admin" and password == "admin123":

        session["user"] = username

        return redirect("/home")

    return "Invalid Username or Password"


# ---------------- HOME PAGE ---------------- #

@app.route("/home")
def home():

    if "user" not in session:

        return redirect("/")

    return render_template("index.html")


# ---------------- SENTIMENT ANALYSIS ---------------- #

@app.route("/predict", methods=["POST"])
def predict():

    if "user" not in session:

        return redirect("/")

    # User review
    text = request.form["text"]

    # Clean review
    cleaned_text = clean_text(text)

    # Convert to vector
    vector = vectorizer.transform([cleaned_text])

    # Prediction
    prediction = model.predict(vector)[0]

    # Emotion detection
    if prediction == "positive":

        emotion = "😊 Happy"

    elif prediction == "negative":

        emotion = "😡 Angry"

    else:

        emotion = "😐 Neutral"

    # ---------------- FAKE REVIEW DETECTION ---------------- #

    fake_keywords = [

        "buy now",
        "click here",
        "free offer",
        "winner",
        "cash prize",
        "limited offer",
        "best best",
        "100% free",
        "subscribe"
    ]

    fake_review = "Genuine Review ✅"

    for word in fake_keywords:

        if word in text.lower():

            fake_review = "Fake / Spam Review ❌"

            break

    # ---------------- LANGUAGE DETECTION ---------------- #

    try:

        # Small words default to English
        if len(text.split()) <= 2:

            language = "English"

        else:

            language_code = detect(text)

            languages = {

                "en": "English",
                "ml": "Malayalam",
                "hi": "Hindi",
                "ta": "Tamil",
                "te": "Telugu",
                "kn": "Kannada",
                "fr": "French",
                "es": "Spanish",
                "so": "Somali",
                "fi": "Finnish"
            }

            language = languages.get(
                language_code,
                language_code
            )

    except:

        language = "Unknown"

    # ---------------- PIE CHART ---------------- #

    labels = [

        "Positive",
        "Negative",
        "Neutral"
    ]

    if prediction == "positive":

        values = [1, 0, 0]

    elif prediction == "negative":

        values = [0, 1, 0]

    else:

        values = [0, 0, 1]

    plt.figure(figsize=(4,4))

    plt.pie(

        values,

        labels=labels,

        autopct="%1.1f%%"
    )

    plt.title("Sentiment Result")

    plt.savefig("static/chart.png")

    plt.close()

    # ---------------- WORD CLOUD ---------------- #

    wordcloud = WordCloud(

        width=800,

        height=400,

        background_color="white"

    ).generate(cleaned_text)

    wordcloud.to_file("static/wordcloud.png")

    # ---------------- DATABASE ---------------- #

    conn = sqlite3.connect("history.db")

    cursor = conn.cursor()

    cursor.execute(

        """
        INSERT INTO history
        (
            review,
            prediction,
            emotion
        )

        VALUES (?, ?, ?)
        """,

        (
            text,
            prediction,
            emotion
        )
    )

    conn.commit()

    conn.close()

    # ---------------- RETURN RESULT ---------------- #

    return render_template(

        "index.html",

        prediction=prediction,

        emotion=emotion,

        fake_review=fake_review,

        language=language,

        chart="static/chart.png"
    )


# ---------------- CSV UPLOAD ---------------- #

@app.route("/upload", methods=["POST"])
def upload():

    if "user" not in session:

        return redirect("/")

    file = request.files["file"]

    df = pd.read_csv(file)

    results = []

    for review in df["review"]:

        cleaned = clean_text(review)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        results.append({

            "review": review,

            "prediction": prediction
        })

    return render_template(

        "index.html",

        bulk_results=results
    )


# ---------------- DASHBOARD ---------------- #

@app.route("/dashboard")
def dashboard():

    if "user" not in session:

        return redirect("/")

    conn = sqlite3.connect("history.db")

    cursor = conn.cursor()

    # Total reviews
    cursor.execute("SELECT COUNT(*) FROM history")

    total = cursor.fetchone()[0]

    # Positive reviews
    cursor.execute(

        "SELECT COUNT(*) FROM history WHERE prediction='positive'"
    )

    positive = cursor.fetchone()[0]

    # Negative reviews
    cursor.execute(

        "SELECT COUNT(*) FROM history WHERE prediction='negative'"
    )

    negative = cursor.fetchone()[0]

    # Neutral reviews
    cursor.execute(

        "SELECT COUNT(*) FROM history WHERE prediction='neutral'"
    )

    neutral = cursor.fetchone()[0]

    conn.close()

    # Dashboard graph
    labels = [

        "Positive",
        "Negative",
        "Neutral"
    ]

    values = [

        positive,
        negative,
        neutral
    ]

    plt.figure(figsize=(6,4))

    plt.bar(labels, values)

    plt.title("Sentiment Analytics")

    plt.savefig("static/dashboard_graph.png")

    plt.close()

    return render_template(

        "dashboard.html",

        total=total,

        positive=positive,

        negative=negative,

        neutral=neutral,

        graph="static/dashboard_graph.png"
    )


# ---------------- HISTORY ---------------- #

@app.route("/history")
def history():

    if "user" not in session:

        return redirect("/")

    conn = sqlite3.connect("history.db")

    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history")

    data = cursor.fetchall()

    conn.close()

    return render_template(

        "history.html",

        history=data
    )


# ---------------- ABOUT ---------------- #

@app.route("/about")
def about():

    if "user" not in session:

        return redirect("/")

    return render_template("about.html")


# ---------------- LOGOUT ---------------- #

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")


# ---------------- RUN APP ---------------- #

if __name__ == "__main__":

    app.run(debug=True)