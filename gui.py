import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load TPOT pipeline
@st.cache_resource
def load_model_bundle():
    try:
        bundle = joblib.load("model_bundle.joblib")
        return bundle
    except FileNotFoundError:
        st.error("Brak pliku 'model_bundle.joblib'. Najpierw wytrenuj model.")
        return None

bundle = load_model_bundle()
if bundle is None:
    st.stop()

# Unpack everything
model_pipeline = bundle["pipeline"]
imputer_cat = bundle["imputer_cat"]
imputer_num = bundle["imputer_num"]
label_encoders = bundle["label_encoders"]
scaler = bundle["scaler"]
categorical_cols_model = bundle["categorical_cols"]
numeric_cols_model = bundle["numeric_cols"]

bundle = load_model_bundle()
if bundle is None:
    st.stop()



def clean_experience(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if x == ">20": return 21
    if x == "<1": return 0
    if x.replace(".", "", 1).isdigit():
        return float(x)
    return np.nan


def clean_company_size(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if "-" in x:
        a, b = x.split("-")
        return (float(a) + float(b)) / 2
    if x == "10/49":
        return (10 + 49) / 2
    if x == "10000+":
        return 10000
    return np.nan


def clean_last_new_job(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if x == ">4": return 5
    if x == "never": return 0
    if x.isdigit(): return int(x)
    return np.nan


POLISH_MAPPINGS = {
    "gender": {
        "Male": "Mężczyzna",
        "Female": "Kobieta",
        "Other": "Inna/Inny"
    },
    "relevent_experience": {
        "Has relevent experience": "Posiada odpowiednie doświadczenie",
        "No relevent experience": "Nie posiada odpowiedniego doświadczenia"
    },
    "enrolled_university": {
        "no_enrollment": "Brak zapisu",
        "Full time course": "Studia dzienne",
        "Part time course": "Studia zaoczne"
    },
    "education_level": {
        "Masters": "Magister",
        "Graduate": "Absolwent",
        "High School": "Liceum/Technikum",
        "Phd": "Doktorat",
        "Primary School": "Szkoła podstawowa"
    },
    "major_discipline": {
        "STEM": "STEM (Nauki ścisłe)",
        "Humanities": "Nauki humanistyczne",
        "Other": "Inna dyscyplina",
        "Business Degree": "Studia biznesowe",
        "Arts": "Sztuka",
        "No Major": "Brak specjalizacji"
    },
    "company_type": {
        "Pvt Ltd": "Firma prywatna",
        "Funded Startup": "Finansowany Startup",
        "Public Sector": "Sektor publiczny",
        "Early Stage Startup": "Startup wczesnej fazy",
        "Other": "Inny typ",
        "NGO": "Organizacja pozarządowa (NGO)"
    }
}




st.set_page_config(
    page_title="Projekt Streamlit",
    page_icon="📊",
    layout="wide"
)

if "items" not in st.session_state:
    st.session_state["items"] = []

st.sidebar.title("📌 Nawigacja")
page = st.sidebar.radio(
    "Wybierz stronę:",
    ["🏠 Strona główna", "📂 Moduł danych", "📈 Wizualizacje", "🧠 Moduł Predykcyjny", "📝 Session State", "ℹ️ O projekcie"]
)

if page == "🏠 Strona główna":
    st.title("📘 Aplikacja Streamlit — projekt rozszerzony")

    st.write("""
    Witaj w poprawionej, estetycznej wersji aplikacji stworzonej w **Streamlit**.

    Aplikacja demonstruje:
    - ✔ pracę z danymi (wgrywanie, filtrowanie, statystyki),
    - ✔ interaktywne wizualizacje,
    - ✔ **Moduł Predykcyjny** z wytrenowanym modelem Gradient Boosting Classifier,
    - ✔ obsługę session_state,
    - ✔ wielostronicowy układ aplikacji,
    - ✔ nowoczesny wygląd i intuicyjny interfejs.
    """)

    st.image(
        "https://static.streamlit.io/examples/dice.jpg",
        width=300,
        caption="Grafika demo"
    )

    with st.expander("📦 Funkcje aplikacji"):
        st.write("""
        - Wgrywanie plików CSV  
        - Profil danych + statystyki  
        - Filtrowanie tabel  
        - Wybór kolumn do wizualizacji  
        - Wykresy: line, histogram, scatter, heatmap  
        - **Model Predykcyjny dla kandydata (target: chęć zmiany pracy)** - Dodawanie elementów przez użytkownika  
        """)

elif page == "📂 Moduł danych":

    st.title("📂 Moduł pracy z danymi")

    uploaded = st.file_uploader("Wgraj plik CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        st.success("📁 Dane poprawnie wczytane!")

        st.subheader("Podgląd danych")
        st.dataframe(df, use_container_width=True)

        with st.expander("📊 Podstawowe statystyki"):
            st.write(df.describe(include="all"))

        st.subheader("🔍 Filtrowanie danych")
        col = st.selectbox("Wybierz kolumnę do filtrowania:", df.columns)

        if df[col].dtype == "object":
            unique_vals = df[col].unique().tolist()
            selected = st.multiselect("Wybierz wartości:", unique_vals)
            if selected:
                df = df[df[col].isin(selected)]
        else:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            range_vals = st.slider("Zakres:", min_val, max_val, (min_val, max_val))
            df = df[df[col].between(range_vals[0], range_vals[1])]

        st.subheader("📄 Wynik filtrowania")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Pobierz przefiltrowany plik CSV", csv, "filtered.csv", "text/csv")

    else:
        st.info("Wgraj plik, aby rozpocząć analizę.")

elif page == "📈 Wizualizacje":

    st.title("📈 Interaktywne wizualizacje")

    uploaded = st.file_uploader("Wgraj plik CSV", type=["csv"], key="upload2")

    if not uploaded:
        st.info("Aby stworzyć wykres — wgraj dane.")
    else:
        df = pd.read_csv(uploaded)

        st.subheader("Wybór kolumn")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) == 0:
            st.error("Brak kolumn numerycznych — nie da się stworzyć wykresów.")
            st.stop()

        col_x = st.selectbox("Kolumna X:", numeric_cols)
        col_y = st.selectbox("Kolumna Y:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

        chart_type = st.radio("Rodzaj wykresu:", ["Line", "Histogram", "Scatter", "Heatmap"])

        if chart_type == "Line":
            st.line_chart(df[[col_x, col_y]])

        elif chart_type == "Histogram":
            fig, ax = plt.subplots()
            ax.hist(df[col_x].dropna(), bins=25)
            ax.set_title(f"Histogram: {col_x}")
            st.pyplot(fig)

        elif chart_type == "Scatter":
            fig, ax = plt.subplots()
            ax.scatter(df[col_x], df[col_y])
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            st.pyplot(fig)

        elif chart_type == "Heatmap":
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df.corr(), cmap="coolwarm", annot=True, ax=ax)
            ax.set_title("Macierz korelacji")
            st.pyplot(fig)




# In the prediction page:
elif page == "🧠 Moduł Predykcyjny":

    st.title("🧠 Predykcja chęci zmiany pracy (ML)")

    st.write("Wprowadź dane kandydata, aby sprawdzić predykcję modelu TPOT.")

    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender_pl = st.selectbox("Płeć:", list(POLISH_MAPPINGS["gender"].values()))
            relevent_experience_pl = st.selectbox("Odpowiednie doświadczenie:", list(POLISH_MAPPINGS["relevent_experience"].values()))
            enrolled_university_pl = st.selectbox("Zapisany na uniwersytet:", list(POLISH_MAPPINGS["enrolled_university"].values()))
            education_level_pl = st.selectbox("Poziom edukacji:", list(POLISH_MAPPINGS["education_level"].values()))
            major_discipline_pl = st.selectbox("Główna dyscyplina:", list(POLISH_MAPPINGS["major_discipline"].values()))
            company_type_pl = st.selectbox("Typ firmy:", list(POLISH_MAPPINGS["company_type"].values()))

        with col2:
            experience = st.slider("Lata Doświadczenia (0 to <1, 21 to >20):", 0, 21, 5)
            company_size_raw = st.selectbox("Rozmiar firmy (wybór z surowych danych):", ["<10", "10/49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"] )
            last_new_job_raw = st.selectbox("Ostatnia nowa praca (lata, 0 to never, 5 to >4):", ["never", "1", "2", "3", "4", ">4"])
            training_hours = st.number_input("Godziny szkoleń:", min_value=1, max_value=400, value=50)

        submit_pred = st.form_submit_button("Wykonaj Predykcję")

    if submit_pred:

        def get_raw_value(mapping_dict, selected_pl_value):
            reverse_map = {v: k for k, v in mapping_dict.items()}
            return reverse_map.get(selected_pl_value, selected_pl_value)

        print(get_raw_value(POLISH_MAPPINGS["major_discipline"], experience))
        # Step 1: Prepare input row
        input_data = pd.DataFrame([{
            "gender": get_raw_value(POLISH_MAPPINGS["gender"], gender_pl),
            "relevent_experience": get_raw_value(POLISH_MAPPINGS["relevent_experience"], relevent_experience_pl),
            "enrolled_university": get_raw_value(POLISH_MAPPINGS["enrolled_university"], enrolled_university_pl),
            "education_level": get_raw_value(POLISH_MAPPINGS["education_level"], education_level_pl),
            "major_discipline": get_raw_value(POLISH_MAPPINGS["major_discipline"], major_discipline_pl),
            "experience": clean_experience(str(get_raw_value(POLISH_MAPPINGS["major_discipline"], experience))),
            "company_size": clean_company_size(company_size_raw),
            "company_type": get_raw_value(POLISH_MAPPINGS["company_type"], company_type_pl),
            "last_new_job": clean_last_new_job(last_new_job_raw),
            "training_hours": training_hours
        }])

        # Step 2: Impute missing numeric values
        input_data[numeric_cols_model] = imputer_num.transform(input_data[numeric_cols_model])

        # Step 3: Impute missing categorical values
        input_data[categorical_cols_model] = imputer_cat.transform(input_data[categorical_cols_model])

        # Step 4: Encode categorical columns using fitted LabelEncoders
        for col in categorical_cols_model:
            input_data[col] = label_encoders[col].transform(input_data[col].astype(str))

        # Step 5: Scale numeric columns using fitted scaler
        input_data[numeric_cols_model] = scaler.transform(input_data[numeric_cols_model])

        # Step 6: Reorder columns exactly as in training
        input_data = input_data[
            ["gender", "relevent_experience", "enrolled_university", "education_level",
             "major_discipline", "experience", "company_size", "company_type",
             "last_new_job", "training_hours"]
        ]

        # Step 7: Predict
        prediction = model_pipeline.predict(input_data)
        prediction_proba = model_pipeline.predict_proba(input_data)[:, 1]

        st.subheader("Wynik Predykcji")
        if prediction[0] == 1:
            st.error(f"⚠️ **Wysokie ryzyko utraty kandydata!** (Prawdopodobieństwo: {prediction_proba[0]:.2f})")
            st.write("Model przewiduje, że kandydat **poszukuje** zmiany pracy (target = 1).")
        else:
            st.success(f"✅ **Kandydat stabilny.** (Prawdopodobieństwo: {prediction_proba[0]:.2f})")
            st.write("Model przewiduje, że kandydat **nie poszukuje** zmiany pracy (target = 0).")


elif page == "📝 Session State":

    st.title("📝 Lista elementów użytkownika")

    with st.form(key="add_form"):
        item = st.text_input("Wprowadź element:")
        submit = st.form_submit_button("Dodaj")

        if submit and item.strip():
            st.session_state["items"].append(item.strip())
            st.success(f"Dodano: **{item}**")

    st.subheader("Twoje elementy:")

    if st.session_state["items"]:
        for i, el in enumerate(st.session_state["items"], 1):
            st.write(f"{i}. {el}")
    else:
        st.info("Brak dodanych elementów.")

elif page == "ℹ️ O projekcie":
    st.title("ℹ️ Projekt Streamlit")
    st.write("""
    Rozszerzona wersja aplikacji stworzona jako baza pod projekt końcowy.  
    Możesz teraz:
    - analizować dane,
    - tworzyć wykresy,
    - filtrować tabelę,
    - pobierać wyniki,
    - **wykonywać predykcje ML**,
    - dodawać dynamiczne elementy.
    """)

    st.caption("Autorzy: Mikołaj Górecki, Michał Lutomirski, Jan Matłosz — ")