import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lifelines import KaplanMeierFitter
import os

# --- Configuration ---
DATA_FILE_PATH = "../data/SR386_labels.csv"
RESULTS_DIR = "../results/"

# Ensure results directory exists
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def save_plot(fig, filename):
    """Helper function to save plots."""
    fig.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close(fig) # Close the plot to free memory
    print(f"Plot saved to {os.path.join(RESULTS_DIR, filename)}")

# --- 1. Data Loading and Initial Inspection ---
def load_and_inspect_data(file_path):
    """Loads data and performs initial inspection."""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        print("Please ensure 'SR386_labels.csv' is in the '../data/' directory.")
        return None

    print("\n--- Data Head ---")
    print(df.head())
    print("\n--- Data Info ---")
    df.info()
    print("\n--- Data Description ---")
    print(df.describe(include='all'))
    print("\n--- Missing Values (Initial) ---")
    print(df.isnull().sum())
    print("\n--- Unique values in 'died_within_5_years' ---")
    print(df['died_within_5_years'].unique())
    print("\n--- Unique values in 'days_till_death' ---")
    print(df['days_till_death'].unique())
    return df

# --- 2. Data Preprocessing and EDA ---
def preprocess_and_eda(df):
    """Handles preprocessing and performs EDA."""
    if df is None: return None
    print("\n--- Starting Preprocessing and EDA ---")

    # Replace 'NULL' strings with NaN and 'Alive' with a specific marker for days_till_death
    df.replace('NULL', np.nan, inplace=True)
    df.replace('FAIL', np.nan, inplace=True) # Assuming 'FAIL' in molecular data means missing

    # Target variable: 'died_within_5_years'
    # Convert to numeric, coercing errors. '1' for died, '0' for alive, NaN for others.
    df['died_within_5_years'] = pd.to_numeric(df['died_within_5_years'], errors='coerce')
    df.dropna(subset=['died_within_5_years'], inplace=True) # Remove rows where target is NaN after coercion
    df['died_within_5_years'] = df['died_within_5_years'].astype(int)

    # 'days_till_death': Convert to numeric. 'Alive' will become NaN after coercion.
    df['days_till_death_numeric'] = pd.to_numeric(df['days_till_death'], errors='coerce')

    # Convert 'age_at_diagnosis' to numeric
    df['age_at_diagnosis'] = pd.to_numeric(df['age_at_diagnosis'], errors='coerce')

    print("\n--- Missing Values (After 'NULL'/'FAIL' replacement and target cleaning) ---")
    print(df.isnull().sum())

    # EDA Visualizations
    # 1. Distribution of Age
    fig_age, ax_age = plt.subplots()
    sns.histplot(df['age_at_diagnosis'].dropna(), kde=True, ax=ax_age)
    ax_age.set_title('Distribution of Age at Diagnosis')
    save_plot(fig_age, 'age_distribution.png')

    # 2. Survival Status (died_within_5_years)
    fig_survival, ax_survival = plt.subplots()
    sns.countplot(x='died_within_5_years', data=df, ax=ax_survival)
    ax_survival.set_title('Survival Status (0=Alive, 1=Died within 5 years)')
    save_plot(fig_survival, 'survival_status_countplot.png')

    # 3. Days till death for those who died
    fig_days_death, ax_days_death = plt.subplots()
    sns.histplot(df[df['died_within_5_years'] == 1]['days_till_death_numeric'].dropna(), kde=True, ax=ax_days_death)
    ax_days_death.set_title('Distribution of Days till Death (for deceased patients)')
    save_plot(fig_days_death, 'days_till_death_distribution.png')

    # 4. Kaplan-Meier Survival Curve
    kmf = KaplanMeierFitter()
    # For Kaplan-Meier, 'event_observed' is 1 if death occurred, 0 if censored (alive)
    # 'duration' is 'days_till_death_numeric' for deceased, or max follow-up for alive (approximate with a large number or actual follow-up if available)
    # For simplicity, we'll use 'days_till_death_numeric' and 'died_within_5_years'.
    # We need a duration for ALL patients. If 'days_till_death_numeric' is NaN (patient alive), we need their observation time.
    # The problem states 'SR386_labels.csv — includes survival labels.' and 'SR1482_labels.csv — includes patient metadata.'
    # For a proper KM curve, we'd ideally merge with SR1482 to get follow-up times for censored patients.
    # As a proxy, if 'days_till_death' was 'Alive', we know they survived at least 5*365.25 days for the 'died_within_5_years' context.
    # However, 'days_till_death_numeric' will be NaN for them. 
    # Let's use a simplified approach based on available columns for now.
    # We need a duration column that is numeric for all, and an event column.
    
    # Create duration and event columns for KM plot
    # If patient died, duration is days_till_death_numeric, event is 1
    # If patient alive (died_within_5_years == 0), duration is max observed time for them (e.g. 5*365.25 if that's the study cutoff), event is 0
    # For this dataset, 'died_within_5_years' is the primary outcome. 'days_till_death' is only for those who died.
    # A more robust KM would use 'days_to_last_followup_or_death' and an 'event_status' column.
    # Given the current structure, let's plot KM for patients with 'days_till_death_numeric' available.
    
    df_km = df.copy()
    df_km['duration_km'] = df_km['days_till_death_numeric']
    # For KM, event is 1 if died, 0 if censored. 'died_within_5_years' is already this.
    df_km['event_km'] = df_km['died_within_5_years']

    # We need to handle cases where 'died_within_5_years' is 0 (alive) but 'days_till_death_numeric' is NaN.
    # These are censored observations. Their duration is at least up to the point they were known to be alive.
    # If 'died_within_5_years' is 0, they survived at least 5 years. So duration can be set to 5*365.25
    df_km.loc[df_km['died_within_5_years'] == 0, 'duration_km'] = 5 * 365.25 
    df_km.dropna(subset=['duration_km', 'event_km'], inplace=True) # Ensure no NaNs in these critical KM columns

    if not df_km.empty:
        fig_km, ax_km = plt.subplots()
        kmf.fit(durations=df_km['duration_km'], event_observed=df_km['event_km'])
        kmf.plot_survival_function(ax=ax_km)
        ax_km.set_title('Kaplan-Meier Survival Curve (Overall Survival)')
        ax_km.set_xlabel('Days')
        ax_km.set_ylabel('Survival Probability')
        save_plot(fig_km, 'kaplan_meier_overall_survival.png')
    else:
        print("Skipping Kaplan-Meier plot due to insufficient data after processing.")

    return df

# --- 3. Binary Classification: Predicting 5-Year Survival ---
def perform_binary_classification(df):
    """Performs binary classification for 5-year survival."""
    if df is None: return
    print("\n--- Starting Binary Classification (5-Year Survival) ---")

    # Features: select relevant clinical/pathological features. Avoid data leakage.
    # Example features (need to check availability and handle NaNs/categoricals):
    # 'age_at_diagnosis', 'sex', 'site_of_tumour_grouping', 'stage', 'kras_ex_2', 'braf_mutant_status', 'mmr_loss_binary'
    features = ['age_at_diagnosis', 'sex', 'site_of_tumour_grouping', 'stage', 
                'kras_ex_2', 'nras_ex_2', 'braf_mutant_status', 'mmr_loss_binary', 
                'primary_metastatic', 'pT', 'pN', 'differentiation']
    target = 'died_within_5_years'

    # Filter out columns not present or mostly NaN to avoid issues
    available_features = [f for f in features if f in df.columns]
    df_model = df[available_features + [target]].copy()
    df_model.dropna(subset=[target], inplace=True) # Ensure target is not NaN

    # Identify numerical and categorical features
    numerical_features = df_model[available_features].select_dtypes(include=np.number).columns.tolist()
    categorical_features = df_model[available_features].select_dtypes(exclude=np.number).columns.tolist()

    # Create preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_features),
        ('categorical', categorical_pipeline, categorical_features)
    ], remainder='passthrough') # Use 'passthrough' if some features don't need processing, or 'drop'

    X = df_model[available_features]
    y = df_model[target]

    if X.empty or y.empty:
        print("Not enough data for binary classification after preprocessing.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # --- Logistic Regression ---
    print("\nTraining Logistic Regression...")
    log_reg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')) # Added class_weight
    ])
    log_reg_pipeline.fit(X_train, y_train)
    y_pred_log_reg = log_reg_pipeline.predict(X_test)
    
    print("Logistic Regression Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
    print(classification_report(y_test, y_pred_log_reg))
    
    cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
    fig_cm_log_reg, ax_cm_log_reg = plt.subplots()
    sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Blues', ax=ax_cm_log_reg,
                xticklabels=['Survived', 'Died'], yticklabels=['Survived', 'Died'])
    ax_cm_log_reg.set_title('Logistic Regression Confusion Matrix')
    ax_cm_log_reg.set_xlabel('Predicted')
    ax_cm_log_reg.set_ylabel('Actual')
    save_plot(fig_cm_log_reg, 'log_reg_confusion_matrix.png')

    # --- Random Forest Classifier with GridSearchCV ---
    print("\nTraining Random Forest Classifier with GridSearchCV...")
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Define a smaller grid for faster initial testing
    param_grid_rf = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__class_weight': ['balanced', 'balanced_subsample'] 
    }

    grid_search_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search_rf.fit(X_train, y_train)

    print("\nBest Parameters for Random Forest Classifier:")
    print(grid_search_rf.best_params_)
    
    best_rf_model = grid_search_rf.best_estimator_
    y_pred_rf = best_rf_model.predict(X_test)

    print("\nTuned Random Forest Classifier Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(classification_report(y_test, y_pred_rf))

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    fig_cm_rf, ax_cm_rf = plt.subplots()
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax_cm_rf,
                xticklabels=['Survived', 'Died'], yticklabels=['Survived', 'Died'])
    ax_cm_rf.set_title('Tuned Random Forest Classifier Confusion Matrix')
    ax_cm_rf.set_xlabel('Predicted')
    ax_cm_rf.set_ylabel('Actual')
    save_plot(fig_cm_rf, 'rf_clf_tuned_confusion_matrix.png')

    # Feature importances from the tuned Random Forest model
    try:
        preprocessor_fitted = best_rf_model.named_steps['preprocessor']
        categorical_features_fitted = preprocessor_fitted.transformers_[1][2] # Get the actual cat feature names used by onehot
        onehot_cols = preprocessor_fitted.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features_fitted)
        
        numerical_features_fitted = []
        # Check if numerical pipeline was active and get features
        if len(preprocessor_fitted.transformers_[0][2]) > 0 and preprocessor_fitted.transformers_[0][1] != 'drop':
             numerical_features_fitted = preprocessor_fitted.transformers_[0][2]

        feature_names_transformed = list(numerical_features_fitted) + list(onehot_cols)
        
        importances = best_rf_model.named_steps['classifier'].feature_importances_
        feature_importances = pd.Series(importances, index=feature_names_transformed).sort_values(ascending=False)

        if not feature_importances.empty:
            fig_fi_rf, ax_fi_rf = plt.subplots(figsize=(10, 8))
            feature_importances.head(15).plot(kind='barh', ax=ax_fi_rf)
            ax_fi_rf.set_title('Tuned Random Forest Classifier Feature Importances')
            ax_fi_rf.set_xlabel('Importance')
            plt.tight_layout()
            save_plot(fig_fi_rf, 'rf_clf_tuned_feature_importances.png')
        else:
            print("No feature importances to plot for tuned Random Forest.")
            
    except Exception as e:
        print(f"Could not plot feature importances for tuned Random Forest: {e}")

    return df

# --- 4. Regression: Predicting Days Till Death ---
def perform_regression_analysis(df):
    """Performs regression analysis for days till death."""
    if df is None: return
    print("\n--- Starting Regression Analysis (Days Till Death) ---")

    # Filter for patients who died and have 'days_till_death_numeric'
    df_reg = df[df['died_within_5_years'] == 1].copy()
    df_reg.dropna(subset=['days_till_death_numeric'], inplace=True)
    
    if df_reg.shape[0] < 10: # Arbitrary threshold for minimum samples
        print("Not enough data for regression analysis after filtering for deceased patients.")
        return

    features = ['age_at_diagnosis', 'sex', 'site_of_tumour_grouping', 'stage', 
                'kras_ex_2', 'nras_ex_2', 'braf_mutant_status', 'mmr_loss_binary',
                'primary_metastatic', 'pT', 'pN', 'differentiation']
    target_reg = 'days_till_death_numeric'

    available_features_reg = [f for f in features if f in df_reg.columns]
    df_reg_model = df_reg[available_features_reg + [target_reg]].copy()

    numerical_features_reg = df_reg_model[available_features_reg].select_dtypes(include=np.number).columns.tolist()
    categorical_features_reg = df_reg_model[available_features_reg].select_dtypes(exclude=np.number).columns.tolist()

    preprocessor_reg = ColumnTransformer([
        ('numerical', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_features_reg),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features_reg)
    ], remainder='passthrough')

    X_reg = df_reg_model[available_features_reg]
    y_reg = df_reg_model[target_reg]

    if X_reg.empty or y_reg.empty:
        print("Not enough data for regression after preprocessing.")
        return

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

    # --- Linear Regression ---
    print("\nTraining Linear Regression...")
    lin_reg_pipeline = Pipeline([
        ('preprocessor', preprocessor_reg),
        ('regressor', LinearRegression())
    ])
    lin_reg_pipeline.fit(X_train_reg, y_train_reg)
    y_pred_lin_reg = lin_reg_pipeline.predict(X_test_reg)

    print("Linear Regression Results:")
    print(f"Mean Squared Error: {mean_squared_error(y_test_reg, y_pred_lin_reg):.2f}")
    print(f"R-squared: {r2_score(y_test_reg, y_pred_lin_reg):.4f}")

    # Scatter plot of actual vs. predicted for Linear Regression
    fig_scatter_lin_reg, ax_scatter_lin_reg = plt.subplots()
    ax_scatter_lin_reg.scatter(y_test_reg, y_pred_lin_reg, alpha=0.5)
    ax_scatter_lin_reg.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'k--', lw=2) # y = x line
    ax_scatter_lin_reg.set_xlabel("Actual Days Till Death")
    ax_scatter_lin_reg.set_ylabel("Predicted Days Till Death")
    ax_scatter_lin_reg.set_title("Linear Regression: Actual vs. Predicted")
    save_plot(fig_scatter_lin_reg, "lin_reg_actual_vs_predicted.png")
    print("Linear Regression Actual vs. Predicted plot saved to ../results/lin_reg_actual_vs_predicted.png")

    # --- Random Forest Regressor ---
    print("\nTraining Random Forest Regressor...")
    rf_reg_pipeline = Pipeline([
        ('preprocessor', preprocessor_reg),
        ('regressor', RandomForestRegressor(random_state=42, n_estimators=100))
    ])
    rf_reg_pipeline.fit(X_train_reg, y_train_reg)
    y_pred_rf_reg = rf_reg_pipeline.predict(X_test_reg)

    print("Random Forest Regressor Results:")
    print(f"Mean Squared Error: {mean_squared_error(y_test_reg, y_pred_rf_reg):.2f}")
    print(f"R-squared: {r2_score(y_test_reg, y_pred_rf_reg):.4f}")

    # Scatter plot of actual vs. predicted for Random Forest Regressor
    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter(y_test_reg, y_pred_rf_reg, alpha=0.5)
    ax_scatter.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'k--', lw=2)
    ax_scatter.set_xlabel('Actual Days Till Death')
    ax_scatter.set_ylabel('Predicted Days Till Death')
    ax_scatter.set_title('Actual vs. Predicted Days Till Death (Random Forest Regressor)')
    save_plot(fig_scatter, 'rf_reg_actual_vs_predicted.png')

    # Feature Importances for Random Forest Regressor
    try:
        ohe_feature_names_reg = rf_reg_pipeline.named_steps['preprocessor']\
            .named_transformers_['categorical']\
            .named_steps['onehot']\
            .get_feature_names_out(categorical_features_reg)
        
        all_feature_names_reg = numerical_features_reg + list(ohe_feature_names_reg)
        importances_reg = rf_reg_pipeline.named_steps['regressor'].feature_importances_
        feature_importance_df_reg = pd.DataFrame({'feature': all_feature_names_reg, 'importance': importances_reg})
        feature_importance_df_reg = feature_importance_df_reg.sort_values(by='importance', ascending=False).head(15)

        fig_fi_reg, ax_fi_reg = plt.subplots(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df_reg, ax=ax_fi_reg)
        ax_fi_reg.set_title('Top 15 Feature Importances (Random Forest Regressor)')
        save_plot(fig_fi_reg, 'rf_reg_feature_importances.png')
    except Exception as e:
        print(f"Could not plot feature importances for Random Forest Regressor: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    print("Starting SurGen Dataset Survival Analysis Script...")
    # 1. Load Data
    raw_df = load_and_inspect_data(DATA_FILE_PATH)
    
    if raw_df is not None:
        # 2. Preprocess and EDA
        processed_df = preprocess_and_eda(raw_df.copy()) # Use a copy to avoid modifying original raw_df
        
        if processed_df is not None:
            # 3. Binary Classification
            perform_binary_classification(processed_df.copy())
            
            # 4. Regression Analysis
            perform_regression_analysis(processed_df.copy())
        else:
            print("Halting script due to errors in preprocessing/EDA.")
    else:
        print("Halting script due to errors in data loading.")
    
    print("\nAnalysis complete. Check the '../results/' directory for plots.")
