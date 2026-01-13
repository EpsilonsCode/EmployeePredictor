from sklearn.utils import resample
import seaborn as sns
from tpot import TPOTClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt


def plot_data_insights(data):
    numeric_cols = [
        "experience", "company_size", "last_new_job", "training_hours"
    ]
    categorical_cols = [
        "gender", "relevent_experience", "enrolled_university",
        "education_level", "major_discipline", "company_type"
    ]

    # 1️⃣ Target class composition
    plt.figure(figsize=(6,4))
    data['target'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Target Class Distribution')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.xticks([0,1], ['Not Looking','Looking'], rotation=0)
    plt.show()

    # 2️⃣ Correlation matrix
    plt.figure(figsize=(10,8))
    corr = data[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numeric Features')
    plt.show()

    # 3️⃣ Numeric feature distributions
    data[numeric_cols].hist(bins=15, figsize=(12, 8), color='lightgreen', edgecolor='black')
    plt.suptitle('Distribution of Numeric Features')
    plt.show()

    # 4️⃣ Boxplots of numeric features by target
    plt.figure(figsize=(12,8))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2,2,i)
        sns.boxplot(x='target', y=col, data=data, palette=['skyblue', 'salmon'])
        plt.title(f'{col} by Target')
    plt.tight_layout()
    plt.show()

    # 5️⃣ Numeric feature comparison by target (side-by-side KDE plots)
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 2, i)
        sns.kdeplot(data=data[data['target'] == 0], x=col, fill=True, label='Target 0',
                    color='skyblue')
        sns.kdeplot(data=data[data['target'] == 1], x=col, fill=True, label='Target 1',
                    color='salmon')
        plt.title(f'{col} Distribution by Target')
        plt.legend()
    plt.tight_layout()
    plt.show()

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

def main():
    data = pd.read_csv("aug_train.csv")
    print("Loaded rows:", len(data))
    print(data['target'].value_counts())
    data["experience"] = data["experience"].apply(clean_experience)
    data["company_size"] = data["company_size"].apply(clean_company_size)
    data["last_new_job"] = data["last_new_job"].apply(clean_last_new_job)
    df_majority = data[data.target == 0]
    df_minority = data[data.target == 1]

    df_majority_downsampled = resample(
        df_majority,
        replace=False,                  # sample without replacement
        n_samples=len(df_minority),     # match minority class size
        random_state=42
    )

    data_balanced = pd.concat([df_majority_downsampled, df_minority])
    data_balanced = data_balanced.sample(frac=1, random_state=42)  # shuffle
    print("Balanced class distribution:")
    print(data_balanced['target'].value_counts())

    categorical_cols = [
        "gender", "relevent_experience", "enrolled_university",
        "education_level", "major_discipline", "company_type"
    ]

    numeric_cols = [
        "experience",  "company_size", "last_new_job", "training_hours"
    ]

    X = data_balanced.drop(columns=["target", "enrollee_id"])

    # remove city information because of lack of city names in training dataset
    X = X.drop(columns=["city", "city_development_index"])

    y = data_balanced["target"]
    imputer_cat = SimpleImputer(strategy="most_frequent")
    imputer_num = SimpleImputer(strategy="mean")

    X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])
    X[numeric_cols] = imputer_num.fit_transform(X[numeric_cols])

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    tpot = TPOTClassifier(
        generations=6,
        population_size=20,
        verbosity=2,
        random_state=42,
    )
    tpot.fit(X_train, y_train)
    print("Validation Score:", tpot.score(X_test, y_train))

    # ----------------------------
    # Save pipeline bundle
    # ----------------------------
    model_bundle = {
        "pipeline": tpot.fitted_pipeline_,
        "imputer_cat": imputer_cat,
        "imputer_num": imputer_num,
        "label_encoders": label_encoders,
        "scaler": scaler,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols
    }
    joblib.dump(model_bundle, "model_bundle.joblib")
    tpot.export('best_model_pipeline.py')


if __name__ == "__main__":
    main()