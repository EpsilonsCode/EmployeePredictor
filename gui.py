import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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
    ["🏠 Strona główna", "📂 Moduł danych", "🧠 Moduł Predykcyjny", "ℹ️ O projekcie"]
)

if page == "🏠 Strona główna":
    st.title("EmployeePredictor")

    st.write("""
        Tematem projektu jest stworzenie modelu, który służy do przewidywania czy pracownik szuka innej pracy, 
        w związku z czym zamierza się zwolnić. Modelu można użyć w celu ograniczenia strat.
    """)


    with st.expander("📦 Funkcjonalnść projektu"):
        st.write("""
        Projekt:
        
        Rozwiązuje problem klasyfikacji binarnej (HR Churn): Przewiduje zmienną target (czy pracownik szuka nowej pracy) na podstawie cech demograficznych i zawodowych zawartych w pliku aug_train.csv.

        Realizuje Custom Data Cleaning: Implementuje dedykowane funkcje parsujące "brudne" dane tekstowe do formatu numerycznego, obsługując przypadki brzegowe dla kolumn experience (np. ">20" → 21), company_size (parsowanie zakresów "10/49") oraz last_new_job .
        
        Przygotowuje Feature Pipeline: Przetwarza dane wejściowe poprzez usunięcie kolumn geograficznych (city), uzupełnienie braków danych (imputacja modą dla kategorii i średnią dla liczb), LabelEncoding zmiennych kategorycznych oraz standaryzację (StandardScaler) .
        
        Wykorzystuje AutoML (TPOT): Zamiast ręcznie dobieranego modelu (jak błędnie sugeruje dokumentacja mówiąca o Gradient Boosting ), aplikacja uruchamia TPOTClassifier – algorytm genetyczny, który przez 6 generacji automatycznie szuka i optymalizuje najlepszy potok klasyfikacyjny.
        
        Generuje artefakty wdrożeniowe: Po treningu serializuje kompletny stan "środowiska" (model, imputery, enkodery, skaler) do pliku model_bundle.joblib oraz eksportuje kod najlepszego znalezionego pipeline'u do best_model_pipeline.py, co pozwala na natychmiastowe użycie modelu na nowych danych
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

        input_data[numeric_cols_model] = imputer_num.transform(input_data[numeric_cols_model])

        input_data[categorical_cols_model] = imputer_cat.transform(input_data[categorical_cols_model])

        for col in categorical_cols_model:
            input_data[col] = label_encoders[col].transform(input_data[col].astype(str))

        input_data[numeric_cols_model] = scaler.transform(input_data[numeric_cols_model])

        input_data = input_data[
            ["gender", "relevent_experience", "enrolled_university", "education_level",
             "major_discipline", "experience", "company_size", "company_type",
             "last_new_job", "training_hours"]
        ]

        prediction = model_pipeline.predict(input_data)
        prediction_proba = model_pipeline.predict_proba(input_data)[:, 1]

        st.subheader("Wynik Predykcji")
        if prediction[0] == 1:
            st.write("Model przewiduje, że kandydat **poszukuje** zmiany pracy (target = 1).")
        else:
            st.write("Model przewiduje, że kandydat **nie poszukuje** zmiany pracy (target = 0).")


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