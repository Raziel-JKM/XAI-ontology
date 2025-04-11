import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import shap
from owlready2 import *
import logging
import matplotlib.pyplot as plt
import os
import traceback
import seaborn as sns
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier
import json
import dice_ml
import types
from owlready2 import sync_reasoner

def normalize_feature_name(name):
    """Helper function to normalize feature names consistently."""
    return name.replace('.', '_')

# Define feature_mapping globally AFTER normalize_feature_name is defined
feature_mapping = {
    normalize_feature_name('Process.1'): 'Velocity_1',
    normalize_feature_name('Process.13'): 'Pressure_Rise_Time',
    normalize_feature_name('Process.7'): 'Cylinder_Pressure'
    # Add more mappings if necessary
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler('/Users/raziel/Desktop/XAI/xai_process.log'),
                             logging.StreamHandler()])

plt.rcParams['font.family'] = 'AppleGothic'  # macOS의 경우
plt.rcParams['axes.unicode_minus'] = False

def preprocess_data(data_path):
    try:
        # 데이터 로드 (다중 헤더 처리)
        data = pd.read_csv(data_path, header=[0, 1])
        # 컬럼 이름 병합: ('Level1', 'Level2') -> 'Level1_Level2'
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
        logging.info(f"Data loaded from {data_path}, shape: {data.shape}")
        logging.info(f"Original Columns: {list(data.columns)}")

        # 식별자 및 타겟 컬럼 분리
        id_col = 'Process_id' # 예시, 실제 컬럼명 확인 필요
        product_type_col = 'Process_Product_Type' # 예시
        defect_cols = [col for col in data.columns if col.startswith('Defects_')]
        process_cols = [col for col in data.columns if col.startswith('Process_') and col not in [id_col, product_type_col]]
        sensor_cols = [col for col in data.columns if col.startswith('Sensor_')]

        numeric_features = process_cols + sensor_cols
        categorical_features = [product_type_col] # 필요 시 추가

        logging.info(f"Identified ID column: {id_col}")
        logging.info(f"Identified Categorical features: {categorical_features}")
        logging.info(f"Identified Numeric features (Process & Sensor): {len(numeric_features)}")
        logging.info(f"Identified Defect columns: {defect_cols}")

        # 수치형 데이터 변환 및 결측치 처리 (Process & Sensor)
        for col in numeric_features:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col].fillna(data[col].median(), inplace=True) # 중앙값으로 변경

        # 범주형 데이터 결측치 처리
        for col in categorical_features:
            if data[col].isna().any():
                most_frequent = data[col].mode()[0]
                data[col].fillna(most_frequent, inplace=True)

        # 이상치 제거 (IQR 방식) - 수치형 피처에만 적용
        original_shape = data.shape
        for col in numeric_features:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        logging.info(f"Data shape after outlier removal: {data.shape} (Removed {original_shape[0] - data.shape[0]} rows)")


        # 최종 타겟 변수 'AnyDefect' 생성 (어떤 불량이든 하나라도 있으면 1)
        data['AnyDefect'] = data[defect_cols].any(axis=1).astype(int)
        logging.info(f"Target variable 'AnyDefect' created. Distribution:\n{data['AnyDefect'].value_counts(normalize=True)}")

        # 사용할 피처 선택 (식별자, 개별 불량 컬럼 제외)
        features_to_use = numeric_features + categorical_features
        final_data = data[features_to_use + ['AnyDefect']].copy()


        # 범주형 데이터 인코딩 (One-Hot Encoding 추천)
        if categorical_features:
            final_data = pd.get_dummies(final_data, columns=categorical_features, drop_first=True)
            logging.info(f"Applied One-Hot Encoding. New shape: {final_data.shape}")


        # 수치형 데이터 정규화 (AnyDefect 제외한 모든 수치형 피처)
        scaler = StandardScaler()
        numeric_cols_to_scale = final_data.select_dtypes(include=np.number).columns.drop('AnyDefect')
        final_data[numeric_cols_to_scale] = scaler.fit_transform(final_data[numeric_cols_to_scale])
        logging.info("Applied StandardScaler to numeric features.")

        return final_data # 정규화된 최종 데이터 반환

    except Exception as e:
        logging.error(f"Error in data preprocessing: {str(e)}")
        logging.error(traceback.format_exc()) # Detailed traceback
        raise


def train_model(data):
    try:
        # 독립 변수(X)와 종속 변수(y) 설정
        target_col = 'AnyDefect' # 새로 생성한 타겟 변수 사용
        if target_col not in data.columns:
             logging.error(f"Target variable '{target_col}' not found in the processed data.")
             raise ValueError(f"Target variable '{target_col}' not found.")

        X = data.drop(columns=[target_col])
        y = data[target_col]

        logging.info(f"X shape: {X.shape}, y shape: {y.shape}")
        logging.info(f"Target variable distribution:\n{y.value_counts(normalize=True)}")

        # 클래스 불균형 확인
        class_counts = y.value_counts()
        unique_classes = len(class_counts)
        logging.info(f"Unique classes: {unique_classes}")
        if unique_classes < 2:
            logging.error("Less than 2 classes found in target variable. Cannot train model.")
            raise ValueError("Less than 2 classes found in target variable.")
        if 1 not in class_counts or class_counts[1] == 0:
             logging.warning("No defect samples (class 1) found in the dataset after preprocessing.")
             # Optionally handle this case, e.g., skip SMOTE or raise error

        # 학습/테스트 데이터 분리 (80:20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        logging.info(f"Train set size before SMOTE: {X_train.shape}, Test set size: {X_test.shape}")
        logging.info(f"Train target distribution before SMOTE:\n{y_train.value_counts(normalize=True)}")

        # SMOTE 적용 (훈련 데이터에만)
        if 1 in y_train.value_counts() and y_train.value_counts()[1] > 1: # 불량 샘플이 2개 이상 있을 때만 적용
             try:
                 # 논문 기준: 불량 비율 10%로 조정 (sampling_strategy=minority_samples / majority_samples)
                 n_majority = y_train.value_counts()[0]
                 n_minority_target = int(n_majority * (10 / 90)) # 10% 불량 비율 = 1/9 소수/다수 비율
                 if y_train.value_counts()[1] < n_minority_target: # 현재 불량이 목표보다 적을 때만
                     smote_strategy = {1: n_minority_target}
                     smote = SMOTE(sampling_strategy=smote_strategy, random_state=42, k_neighbors=min(5, y_train.value_counts()[1]-1) if y_train.value_counts()[1]>1 else 1)
                     X_train, y_train = smote.fit_resample(X_train, y_train)
                     logging.info(f"SMOTE applied to training data. New shape: {X_train.shape}")
                     logging.info(f"Train target distribution after SMOTE:\n{y_train.value_counts(normalize=True)}")
                 else:
                     logging.info("SMOTE not applied as current minority class ratio is already >= 10%.")

             except ValueError as smote_err:
                 logging.warning(f"SMOTE failed: {smote_err}. Proceeding without SMOTE.")
             except Exception as smote_ex:
                 logging.error(f"Unexpected error during SMOTE: {smote_ex}. Proceeding without SMOTE.")
        else:
             logging.warning("SMOTE not applied due to insufficient minority samples in the training set.")


        # Reset index after SMOTE and split
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        # --- 이후 모델 평가 및 학습 로직은 기존과 유사하게 진행 ---
        # 다양한 알고리즘 평가 (XGBoost, RandomForest 등)
        models = {
            # 논문 기준 XGBoost 하이퍼파라미터 적용 시도
            'XGBoost': xgb.XGBClassifier(
                 max_depth=5, learning_rate=0.1, n_estimators=200,
                 subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                 eval_metric='logloss', random_state=42, use_label_encoder=False # use_label_encoder=False 권장
                 ),
            'RandomForest': RandomForestClassifier(random_state=42),
            # 'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000) # 필요 시 주석 해제
        }
        # ... (기존 모델 평가 로직) ...
        # 각 알고리즘 평가
        best_model = None
        best_score = 0
        cv_results = {}

        for name, model in models.items():
            if model is None:
                continue

            logging.info(f"Evaluating {name}...")

            try:
                # 교차 검증 수행 (F1-score 기준)
                # SMOTE 적용된 훈련 데이터로 CV 수행
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1) # n_jobs=-1 추가하여 병렬 처리
                avg_cv_score = np.mean(cv_scores)
                cv_results[name] = avg_cv_score

                logging.info(f"{name} 5-fold CV F1-score: {avg_cv_score:.4f}")

                if avg_cv_score > best_score:
                    best_score = avg_cv_score
                    best_model_name = name
                    # Clone the best model estimator
                    from sklearn.base import clone
                    best_model = clone(model)

            except Exception as e:
                logging.error(f"Error evaluating {name}: {str(e)}")
                logging.error(traceback.format_exc())


        if best_model is None:
            logging.warning("No model evaluation succeeded. Falling back to basic XGBoost.")
            best_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            best_model_name = 'XGBoost'
        else:
             logging.info(f"Selected best model based on CV F1-score: {best_model_name} ({best_score:.4f})")


        # 최종 모델 학습 (SMOTE 적용된 전체 훈련 데이터 사용)
        logging.info(f"Training final model ({best_model_name})...")
        best_model.fit(X_train, y_train)

        # 테스트 세트로 성능 평가
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] # 불량(1) 클래스 확률

        # 다양한 성능 지표 계산
        try:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0) # 불량 클래스(1) 기준 F1

            logging.info(f"--- Test Set Performance ({best_model_name}) ---")
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"F1-score (Positive Class 1): {f1:.4f}")

            try:
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                logging.info(f"Precision (Positive Class 1): {precision:.4f}")
                logging.info(f"Recall (Positive Class 1): {recall:.4f}")
            except Exception as e:
                logging.warning(f"Error calculating precision/recall: {str(e)}")

            # 혼동 행렬
            cm = confusion_matrix(y_test, y_pred)
            logging.info(f"Confusion Matrix:\n{cm}")
            # Confusion Matrix 시각화
            try:
                 plt.figure(figsize=(6, 4))
                 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
                 plt.xlabel('Predicted Label')
                 plt.ylabel('True Label')
                 plt.title(f'Confusion Matrix ({best_model_name})')
                 cm_path = f"{output_dir}/confusion_matrix.png" # output_dir 정의 필요
                 # plt.savefig(cm_path) # 주석 처리 - output_dir 정의 후 사용
                 plt.close()
                 # logging.info(f"Confusion matrix plot saved to {cm_path}")
            except Exception as plot_e:
                 logging.warning(f"Could not save confusion matrix plot: {plot_e}")


            # 분류 보고서
            try:
                target_names = ['Class 0 (Good)', 'Class 1 (Defect)']
                report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
                logging.info(f"Classification Report:\n{report}")
            except Exception as e:
                logging.warning(f"Error generating classification report: {str(e)}")

        except Exception as e:
            logging.error(f"Error evaluating final model: {str(e)}")

        # 반환값 수정: 타겟 컬럼 이름 반환
        return best_model, X_test, y_test, target_col

    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def apply_xai(model, X_test, y_test, output_dir, defect_col):
    try:
        # SHAP 값 계산
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # SHAP 값 요약 플롯 생성
        try:
            plt.figure(figsize=(10, 8))
            if isinstance(shap_values, list):
                logging.info("Binary classification model detected, selecting positive class SHAP values")
                shap_values_processed = shap_values[1] # 양성 클래스(1)의 SHAP 값 선택
                # shap==0.41 버전에서는 matplotlib 인수가 없음
                shap.summary_plot(shap_values_processed, X_test, show=False)
            else:
                logging.info("Using single array of SHAP values")
                shap_values_processed = shap_values
                # shap==0.41 버전에서는 matplotlib 인수가 없음
                shap.summary_plot(shap_values_processed, X_test, show=False)

            plot_path = f'{output_dir}/shap_summary_plot.png'
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"SHAP summary plot saved to {plot_path}")
        except Exception as plot_e:
            logging.error(f"Error generating or saving SHAP summary plot: {plot_e}")
            plt.close() # 오류 발생 시 플롯 닫기

        # 특성 중요도 저장
        feature_importance = None
        try:
            if isinstance(shap_values, list):
                shap_mean_abs = np.abs(shap_values[1]).mean(axis=0)
            else:
                shap_mean_abs = np.abs(shap_values).mean(axis=0)

            if shap_mean_abs.ndim == 0: # Ensure it's iterable
                 shap_mean_abs = np.array([shap_mean_abs])

            if len(shap_mean_abs) == len(X_test.columns):
                 feature_importance = pd.DataFrame({
                     'Feature': X_test.columns,
                     'SHAP_Value': shap_mean_abs
                 }).sort_values('SHAP_Value', ascending=False)
                 feature_importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
                 logging.info("SHAP values calculated and feature importance saved.")
            else:
                 logging.error(f"Mismatch in SHAP values ({len(shap_mean_abs)}) and columns ({len(X_test.columns)}).")
                 raise ValueError("SHAP values and column length mismatch.")

        except Exception as e:
            logging.error(f"Error calculating or saving feature importance from SHAP: {str(e)}")
            logging.info("Falling back to model's built-in feature_importances_")
            try:
                feature_importance = pd.DataFrame({
                    'Feature': X_test.columns,
                    'SHAP_Value': model.feature_importances_
                }).sort_values('SHAP_Value', ascending=False)
                feature_importance.to_csv(f'{output_dir}/feature_importance_fallback.csv', index=False)
                logging.info("Fallback feature importance saved.")
            except Exception as fallback_e:
                 logging.error(f"Error using fallback feature_importances_: {fallback_e}")
                 # 최후의 수단: 빈 DataFrame 또는 기본값
                 feature_importance = pd.DataFrame({'Feature': X_test.columns, 'SHAP_Value': np.zeros(len(X_test.columns))})


        # --- Counterfactual Explanations ---
        # (기존 Counterfactual 코드 유지 - 필요시 dice-ml 관련 로직 추가 수정)
        try:
            logging.info("Attempting to generate Counterfactual Explanations...")
            continuous_features = X_test.select_dtypes(include=np.number).columns.tolist()
            dice_data_df = X_test.copy()
            dice_data_df[defect_col] = y_test
            d = dice_ml.Data(dataframe=dice_data_df, continuous_features=continuous_features, outcome_name=defect_col)
            m = dice_ml.Model(model=model, backend="sklearn")
            exp = dice_ml.Dice(d, m, method="random")
            y_pred_test = model.predict(X_test)
            X_defect_instances = X_test[y_pred_test == 1]

            if not X_defect_instances.empty:
                query_instances = X_defect_instances.head(3)
                logging.info(f"Generating counterfactuals for {len(query_instances)} defect instance(s)...")
                dice_exp = exp.generate_counterfactuals(query_instances, total_CFs=5, desired_class="opposite")
                if dice_exp.cf_examples_list and dice_exp.cf_examples_list[0].final_cfs_df is not None:
                     cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                     logging.info(f"Counterfactuals for first defect instance (showing changes only):\n{cf_df}")
                     cf_json_path = f"{output_dir}/counterfactuals.json"
                     cf_df.to_json(cf_json_path, orient='records', indent=4)
                     logging.info(f"Counterfactual explanations saved to {cf_json_path}")
                else:
                     logging.warning("Could not generate counterfactuals for the selected instances.")
            else:
                logging.info("No defect instances found in the test set to generate counterfactuals for.")
        except ImportError:
            logging.warning("dice-ml library not found. Skipping Counterfactual Explanations.")
        except Exception as cf_e:
            logging.error(f"Error generating Counterfactual Explanations: {str(cf_e)}")
            logging.error(traceback.format_exc())
        # --- Counterfactual Explanations 끝 ---

        return feature_importance # 정상 처리 또는 Fallback 결과 반환

    except Exception as e:
        logging.error(f"Error in apply_xai function: {str(e)}")
        logging.error(traceback.format_exc())
        # 최후의 수단: 빈 DataFrame 또는 기본값 반환
        if 'X_test' in locals() and hasattr(X_test, 'columns'):
             return pd.DataFrame({'Feature': X_test.columns, 'SHAP_Value': np.zeros(len(X_test.columns))})
        else:
             return pd.DataFrame() # 데이터 로드 실패 시

def construct_ontology(ontology_path, config_path=None):
    try:
        # Check if ontology file exists
        if os.path.exists(ontology_path):
            onto = get_ontology(f"file://{os.path.abspath(ontology_path)}").load()
            logging.info(f"Loaded existing ontology from: {ontology_path}")
        else:
            # Create a new ontology if file doesn't exist
            ontology_iri = f"file://{os.path.abspath(ontology_path)}"
            onto = get_ontology(ontology_iri)
            logging.info(f"Creating new ontology with IRI: {ontology_iri}")

        # Check if config file exists and load it
        if config_path and os.path.exists(config_path):
            logging.info(f"Loading ontology structure from config: {config_path}")
            with open(config_path, 'r') as file:
                config = json.load(file)

            with onto:
                # Create classes from config
                for cls_config in config['classes']:
                    parent_name = cls_config.get('parent', 'owl.Thing') # Default to owl.Thing
                    # Correctly reference owl.Thing and handle other potential base classes
                    if parent_name == 'owl.Thing':
                        parent_class = owl.Thing
                    else:
                        parent_class = onto[parent_name] # Find parent within the ontology
                    
                    if not parent_class:
                        # Try finding in owl namespace if not in onto (e.g., owl.FunctionalProperty)
                        try:
                            parent_class = getattr(owl, parent_name.split('.')[-1])
                        except AttributeError:
                             logging.warning(f"Parent class or entity '{parent_name}' not found for class '{cls_config['name']}'. Skipping.")
                             continue

                    # Check if class already exists
                    existing_class = onto[cls_config['name']]
                    if existing_class:
                        # Optionally update properties or just skip creation
                        logging.debug(f"Class '{cls_config['name']}' already exists. Skipping creation.")
                    else:
                        try:
                            NewClass = types.new_class(cls_config['name'], (parent_class,))
                            logging.info(f"Created class: {cls_config['name']} (Parent: {parent_class.name})")
                            for prop, value in cls_config.get('properties', {}).items():
                                if hasattr(NewClass, prop):
                                    setattr(NewClass, prop, value)
                                else:
                                    logging.warning(f"Property '{prop}' not standard for class annotations. Adding anyway.")
                                    # For annotations like comment, directly add
                                    if prop == 'comment':
                                         NewClass.comment.append(value)
                                    # Add other custom props if needed, might require defining annotation properties
                        except Exception as class_e:
                             logging.error(f"Error creating class '{cls_config['name']}': {class_e}")

                # Create properties from config
                for prop_config in config.get('properties', []):
                    prop_name = prop_config['name']
                    prop_type = prop_config.get('type', 'object') # object or data
                    domain_name = prop_config.get('domain')
                    range_name = prop_config.get('range')

                    # Check if property exists
                    existing_prop = onto[prop_name]
                    if existing_prop:
                        logging.debug(f"Property '{prop_name}' already exists. Skipping creation.")
                        continue

                    domain_class = onto[domain_name] if domain_name else None
                    range_class = None
                    if range_name:
                         if prop_type == 'object':
                             range_class = onto[range_name]
                         elif range_name.startswith('xsd:'):
                              # Handle XSD datatypes
                              try:
                                   range_class = getattr(owl, range_name.split(':')[-1]) 
                              except AttributeError:
                                   logging.warning(f"XSD Datatype '{range_name}' not found. Using default.")
                                   range_class = owl.xsd_string # Default or handle appropriately
                         else:
                              # Assume custom datatype or potentially an object property range misspelled
                              range_class = onto[range_name] # Try finding as class first
                              if not range_class:
                                   logging.warning(f"Range '{range_name}' not found as class or known datatype for '{prop_name}'.")
                                   # Decide on a default or skip

                    try:
                        if prop_type == 'object':
                            NewProp = types.new_class(prop_name, (ObjectProperty,))
                            if domain_class: NewProp.domain = [domain_class]
                            if range_class: NewProp.range = [range_class]
                            logging.info(f"Created Object Property: {prop_name} (Domain: {domain_name}, Range: {range_name})")
                        elif prop_type == 'data':
                            NewProp = types.new_class(prop_name, (DataProperty,))
                            if domain_class: NewProp.domain = [domain_class]
                            if range_class: NewProp.range = [range_class] # e.g., xsd:float
                            logging.info(f"Created Data Property: {prop_name} (Domain: {domain_name}, Range: {range_name})")
                        else:
                             logging.warning(f"Unknown property type '{prop_type}' for '{prop_name}'. Skipping.")
                             continue
                             
                        # Add annotations like comments
                        for anno_prop, value in prop_config.get('properties', {}).items():
                             if anno_prop == 'comment':
                                 NewProp.comment.append(value)
                    except Exception as prop_e:
                        logging.error(f"Error creating property '{prop_name}': {prop_e}")
        else:
            logging.warning(f"Ontology configuration file not found at {config_path}. Ontology might be incomplete or using defaults.")
            # Optionally, create some default structure if config is missing
            with onto:
                if not onto['ManufacturingProcess']:
                     types.new_class('ManufacturingProcess', (owl.Thing,))
                if not onto['Parameter']:
                     types.new_class('Parameter', (owl.Thing,))
                # Add other minimal default classes/props if needed

        # Save the ontology
        onto.save(file=ontology_path, format="rdfxml") # Use rdfxml format for better compatibility
        logging.info(f"Ontology saved to: {ontology_path}")
        return onto

    except Exception as e:
        logging.error(f"Error in ontology construction: {str(e)}")
        logging.error(traceback.format_exc())
        # Return None or raise the exception depending on desired behavior
        return None

def integrate_xai_with_ontology(onto, top_features, X_test, defect_col):
    try:
        logging.info(f"Integrating XAI insights for {defect_col} into ontology")
        
        # Global feature_mapping is used here implicitly or explicitly if passed
        # No need to redefine it here

        # 불량 유형 매핑 (Keep as is for now)
        defect_mappings = {
            'Short_Shot_1': 'ShortShotDefect',
            'Bubble_1': 'BubbleDefect',
            'Blow_Hole_1': 'BlowHoleDefect',
            'Exfoliation_1': 'ExfoliationDefect',
            'Stain_1': 'StainDefect',
            'Dent_1': 'DentDefect',
            'Deformation_1': 'DeformationDefect',
            'Crack_1': 'CrackDefect',
            'Scratch_1': 'ScratchDefect',
            'Defects': 'DefectGeneral',
            'Defects.1': 'BubbleDefect',
            'Defects.2': 'ExfoliationDefect',
            'Defects.3': 'BlowHoleDefect'
        }
        
        with onto:
            # 일반적인 불량 추가 (원본 데이터 열 이름이 매핑에 없는 경우)
            defect_name = defect_mappings.get(defect_col, 'DefectGeneral')
            
            # 결함 유형 확인 또는 생성
            DefectType = None
            for cls in onto.classes():
                if cls.name == 'DefectType':
                    DefectType = cls
                    break
            
            if not DefectType:
                logging.error("DefectType class not found in ontology")
                raise ValueError("DefectType class not found")
                
            # 결함 인스턴스 찾기 또는 생성
            defect_instance = None
            for defect in list(DefectType.subclasses()):
                if defect.name == defect_name.replace('Defect', ''):
                    for instance in defect.instances():
                        defect_instance = instance
                        break
                    if not defect_instance and defect.name == 'ShortShot':
                        defect_instance = defect('ShortShotDefect')
                    elif not defect_instance:
                        defect_instance = defect(f"{defect.name}Defect")
                    break
            
            if not defect_instance:
                # 기본적으로 ShortShot 타입으로 설정
                for cls in onto.classes():
                    if cls.name == 'ShortShot':
                        defect_instance = cls('ShortShotDefect')
                        break
            
            if not defect_instance:
                logging.error("Failed to find or create defect instance")
                raise ValueError("Failed to find or create defect instance")
            
            logging.info(f"Using defect instance: {defect_instance.name}")
            
            # SHAP 결과를 기반으로 온톨로지 업데이트
            processed_features_set = set() # Keep track of processed normalized names
            for idx, row in top_features.iterrows():
                original_feature_name = row['Feature']
                shap_value = row['SHAP_Value']

                # Normalize the original feature name first
                normalized_feature = normalize_feature_name(original_feature_name)

                # Apply specific mapping if available, otherwise use the normalized name
                ontology_param_name = feature_mapping.get(normalized_feature, normalized_feature)
                processed_features_set.add(ontology_param_name) # Add the final name used in ontology

                logging.info(f"Processing feature: {original_feature_name} -> {ontology_param_name} (SHAP: {shap_value})")

                # 매개변수 클래스 및 하위 클래스 탐색
                param_cls = None
                for cls in onto.classes():
                    if cls.name == 'Parameter':
                        param_cls = cls
                        break
                
                if not param_cls:
                    logging.error("Parameter class not found in ontology")
                    raise ValueError("Parameter class not found")
                
                # 적절한 매개변수 클래스 찾기
                appropriate_cls = None
                for subcls in param_cls.subclasses():
                    if 'Velocity' in normalized_feature and subcls.name == 'Velocity':
                        appropriate_cls = subcls
                        break
                    elif ('Pressure' in normalized_feature or 'pressure' in normalized_feature) and subcls.name == 'Pressure':
                        appropriate_cls = subcls
                        break
                    elif ('Temp' in normalized_feature or 'temp' in normalized_feature) and subcls.name == 'Temperature':
                        appropriate_cls = subcls
                        break
                    elif ('Time' in normalized_feature or 'time' in normalized_feature) and subcls.name == 'Time':
                        appropriate_cls = subcls
                        break
                
                if not appropriate_cls:
                    # 기본적으로 ProcessParameter 사용
                    for cls in onto.classes():
                        if cls.name == 'ProcessParameter':
                            appropriate_cls = cls
                            break
                
                if not appropriate_cls:
                    logging.error("Could not find appropriate parameter class")
                    continue
                
                # 매개변수 인스턴스 찾기 또는 생성
                param_instance = None
                for instance in appropriate_cls.instances():
                    if instance.name == ontology_param_name:
                        param_instance = instance
                        break
                
                if not param_instance:
                    param_instance = appropriate_cls(ontology_param_name)
                    logging.info(f"Created new parameter instance: {ontology_param_name}")
                
                # 속성 및 관계 찾기
                causesDefect = None
                hasValue = None
                hasImportance = None
                
                for prop in onto.properties():
                    if prop.name == 'causesDefect':
                        causesDefect = prop
                    elif prop.name == 'hasValue':
                        hasValue = prop
                    elif prop.name == 'hasImportance':
                        hasImportance = prop
                
                if not causesDefect or not hasValue:
                    logging.error("Required properties not found in ontology")
                    raise ValueError("Required properties not found")
                
                # 관계 및 속성 설정
                # SHAP 값에 기반한 중요도 설정 (절대값 사용)
                importance = abs(shap_value)
                if importance > 0:
                    if hasImportance:
                        hasImportance[param_instance] = [importance]
                    
                    # SHAP 값의 부호에 따라 인과 관계 방향 결정
                    if shap_value > 0:  # 양의 관계: 변수값 증가 -> 결함 발생 가능성 증가
                        causesDefect[param_instance] = [defect_instance]
                        logging.info(f"Added positive causal relationship: {param_instance.name} causes {defect_instance.name}")
                    else:  # 음의 관계: 변수값 증가 -> 결함 발생 가능성 감소
                        for prop in onto.properties():
                            if prop.name == 'preventsDefect':
                                prop[param_instance] = [defect_instance]
                                logging.info(f"Added negative causal relationship: {param_instance.name} prevents {defect_instance.name}")
                                break
                
                # 변수의 현재 값 설정
                try:
                    if original_feature_name in X_test.columns:
                        param_value = float(X_test[original_feature_name].mean())
                        hasValue[param_instance] = [param_value]
                        logging.info(f"Set value of {param_instance.name} to {param_value}")
                except Exception as e:
                    logging.warning(f"Could not set value for {param_instance.name}: {str(e)}")
                
            # 공정 인스턴스 찾기
            process_instance = None
            for cls in onto.classes():
                if cls.name == 'DieCastingProcess':
                    if list(cls.instances()):
                        process_instance = list(cls.instances())[0]
                        break
            
            # 공정에 매개변수 연결
            if process_instance:
                hasParameter = None
                for prop in onto.properties():
                    if prop.name == 'hasParameter':
                        hasParameter = prop
                        break
                
                if hasParameter:
                    # 모든 처리된 매개변수를 공정에 연결
                    for cls in param_cls.subclasses():
                        for instance in cls.instances():
                            if instance not in hasParameter[process_instance]:
                                hasParameter[process_instance].append(instance)
            
            logging.info(f"XAI insights successfully integrated into ontology")
        return onto, processed_features_set
    except Exception as e:
        logging.error(f"Error in integrating XAI with ontology: {str(e)}")
        logging.error(traceback.format_exc())
        return None, set()

def verify_ontology_comprehensive(onto, top_features, processed_ontology_features, output_dir):
    """
    온톨로지의 일관성을 검증하고 종합적인 보고서를 생성합니다.
    Accepts a set of feature names actually processed and added/used in the ontology.
    """
    try:
        # 출력 디렉토리 확인 및 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 온톨로지 일관성 검사
        with onto:
            try:
                # 이곳에서 도메인 전문가가 할 수 있는 검증을 시뮬레이션
                # 온톨로지 클래스 및 인스턴스 요약
                class_count = len(list(onto.classes()))
                instance_count = len(list(onto.individuals()))
                property_count = len(list(onto.properties()))
                
                # XAI에서 중요하다고 판단된 특성들이 온톨로지에 잘 표현되어 있는지 확인
                xai_important_normalized = {normalize_feature_name(f) for f in top_features['Feature']}
                missing_in_ontology = xai_important_normalized - processed_ontology_features

                # Log missing features
                for feature_name in missing_in_ontology:
                    # Check if it was mapped using the global mapping
                    mapped_name = feature_mapping.get(feature_name, feature_name)
                    if mapped_name not in processed_ontology_features:
                         logging.warning(f"XAI important feature '{feature_name}' (used as '{mapped_name}' in ontology logic) not found/processed in ontology.")

                # 불량 유형 및 원인 관계 분석
                defect_causes = {}
                prevention_measures = {}
                
                for cls in onto.classes():
                    if cls.name == 'DefectType':
                        for defect_cls in cls.subclasses():
                            for defect in defect_cls.instances():
                                # 해당 불량을 유발하는 매개변수 찾기
                                causes = []
                                preventions = []
                                
                                for param in onto.individuals():
                                    # causesDefect 속성 확인
                                    for prop in onto.properties():
                                        if prop.name == 'causesDefect' and prop[param] and defect in prop[param]:
                                            causes.append(param.name)
                                        elif prop.name == 'preventsDefect' and prop[param] and defect in prop[param]:
                                            preventions.append(param.name)
                                
                                defect_causes[defect.name] = causes
                                prevention_measures[defect.name] = preventions
                
                # 결과 저장
                ontology_report = f'{output_dir}/ontology_verification_report.txt'
                with open(ontology_report, 'w') as f:
                    f.write("온톨로지 검증 보고서\n")
                    f.write("===================\n\n")
                    
                    f.write(f"1. 온톨로지 요약\n")
                    f.write(f"   - 클래스 수: {class_count}\n")
                    f.write(f"   - 인스턴스 수: {instance_count}\n")
                    f.write(f"   - 속성 수: {property_count}\n\n")
                    
                    f.write(f"2. XAI 결과와의 통합\n")
                    f.write(f"   - XAI에서 식별된 중요 특성 수: {len(top_features)}\n")
                    f.write(f"   - 온톨로지에 통합된 특성 수: {len(processed_ontology_features)}\n")
                    f.write(f"   - 통합된 특성 (온톨로지 이름 기준): {', '.join(sorted(list(processed_ontology_features)))}\n\n")
                    
                    if missing_in_ontology:
                         f.write("   - 온톨로지 처리에서 누락된 중요 특성 (정규화된 이름):\n")
                         for missing_name in sorted(list(missing_in_ontology)):
                              f.write(f"     * {missing_name}\n")
                         f.write("     (원인: 매핑 부족 또는 통합 로직 오류 가능성)\n\n")
                    
                    f.write(f"3. 불량 유형 및 원인 분석\n")
                    for defect, causes in defect_causes.items():
                        f.write(f"   - {defect}:\n")
                        if causes:
                            f.write(f"     * 원인: {', '.join(causes)}\n")
                        else:
                            f.write(f"     * 원인: 명시된 원인 없음\n")
                        
                        preventions = prevention_measures.get(defect, [])
                        if preventions:
                            f.write(f"     * 방지 매개변수: {', '.join(preventions)}\n")
                        else:
                            f.write(f"     * 방지 매개변수: 명시된 방지책 없음\n")
                    
                    f.write("\n4. 결론 및 권장사항\n")
                    if len(processed_ontology_features) < len(top_features):
                        f.write("   - 일부 중요 특성이 온톨로지에 통합되지 않았습니다. 도메인 지식 보강이 필요합니다.\n")
                    
                    for defect, causes in defect_causes.items():
                        if not causes:
                            f.write(f"   - {defect}의 원인이 명시되지 않았습니다. 도메인 전문가와 함께 원인을 파악해야 합니다.\n")
                
                logging.info(f"온톨로지 검증 보고서가 {ontology_report}에 저장되었습니다.")
                
                # 온톨로지 시각화 (간단한 텍스트 형태)
                ontology_viz = f'{output_dir}/ontology_visualization.txt'
                with open(ontology_viz, 'w') as f:
                    f.write("온톨로지 구조 시각화\n")
                    f.write("===================\n\n")
                    
                    # 클래스 계층 구조
                    f.write("클래스 계층 구조:\n")
                    for cls in onto.classes():
                        if not cls.is_a:  # 최상위 클래스
                            f.write(f"- {cls.name}\n")
                            for subcls in cls.subclasses():
                                if subcls != cls:  # 자기 자신 제외
                                    f.write(f"  - {subcls.name}\n")
                                    for subsubcls in subcls.subclasses():
                                        if subsubcls != subcls:  # 자기 자신 제외
                                            f.write(f"    - {subsubcls.name}\n")
                    
                    # 주요 관계
                    f.write("\n주요 관계:\n")
                    for defect, causes in defect_causes.items():
                        if causes:
                            for cause in causes:
                                f.write(f"- {cause} --causesDefect--> {defect}\n")
                    
                    for defect, preventions in prevention_measures.items():
                        if preventions:
                            for prevention in preventions:
                                f.write(f"- {prevention} --preventsDefect--> {defect}\n")
                
                logging.info(f"온톨로지 시각화가 {ontology_viz}에 저장되었습니다.")
                return True
            except Exception as e:
                logging.error(f"온톨로지 검증 중 오류: {str(e)}")
                return False
    except Exception as e:
        logging.error(f"온톨로지 검증 시작 실패: {str(e)}")
        return False

def verify_ontology(onto, important_param):
    try:
        # Normalize the input parameter name like in integration
        normalized_param_name = normalize_feature_name(important_param)
        # Apply mapping if exists, otherwise use normalized name (using global mapping)
        ontology_param_name = feature_mapping.get(normalized_param_name, normalized_param_name)
        
        logging.info(f"Verifying ontology for parameter: {important_param} (using name: {ontology_param_name})")

        with onto:
            # Search for the instance using the final ontology_param_name
            target_param = onto[ontology_param_name]
            
            if target_param:
                 logging.info(f"Found parameter instance: {target_param.name}")
                 # ... (rest of the verification logic using target_param) ...
            else:
                logging.warning(f"Parameter instance '{ontology_param_name}' not found in ontology during verification.")
                print(f"\n매개변수 {ontology_param_name}에 대한 온톨로지 정보를 찾을 수 없습니다.")

    except Exception as e:
        logging.error(f"Error in ontology verification: {str(e)}")
        print(f"온톨로지 검증 중 오류 발생: {str(e)}")

def analyze_xai_results(top_features, model, X_test, output_dir):
    """
    XAI 결과를 분석하고 시각화합니다.
    """
    try:
        # 출력 디렉토리 확인 및 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 특성 중요도 요약
        logging.info("XAI 결과 분석:")
        for idx, row in top_features.iterrows():
            feature = row['Feature']
            shap_value = row['SHAP_Value']
            logging.info(f"특성: {feature}, SHAP 값: {shap_value:.6f}")
            
        # 결과의 타당성 분석
        non_zero_features = top_features[abs(top_features['SHAP_Value']) > 0.001]
        if len(non_zero_features) == 0:
            logging.warning("모든 특성의 SHAP 값이 0에 가깝습니다. 모델이 특성을 제대로 활용하지 못하고 있을 수 있습니다.")
            
            # 모델 특성 중요도 직접 추출 시도
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance = pd.DataFrame(
                        {'Feature': X_test.columns, 'Importance': importances}
                    ).sort_values('Importance', ascending=False)
                    
                    logging.info("모델의 자체 특성 중요도(feature_importances_):")
                    for idx, row in feature_importance.head(10).iterrows():
                        feature = row['Feature']
                        importance = row['Importance']
                        logging.info(f"특성: {feature}, 중요도: {importance:.6f}")
                        
                        # 시각화
                        plt.figure(figsize=(10, 6))
                        plt.barh(feature_importance['Feature'].head(10), feature_importance['Importance'].head(10))
                        plt.xlabel('특성 중요도')
                        plt.ylabel('특성')
                        plt.title('모델 자체 특성 중요도')
                        plt.tight_layout()
                        plt.savefig(f'{output_dir}/model_feature_importance.png')
                        plt.close()
            except Exception as e:
                logging.warning(f"모델 특성 중요도 추출 실패: {str(e)}")
                
        # 결과 저장
        summary_file = f'{output_dir}/xai_analysis_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("XAI 분석 결과 요약\n")
            f.write("===================\n\n")
            
            f.write("특성 중요도 (SHAP):\n")
            for idx, row in top_features.iterrows():
                f.write(f"- {row['Feature']}: {row['SHAP_Value']:.6f}\n")
                
            if len(non_zero_features) == 0:
                f.write("\n주의: 모든 특성의 SHAP 값이 매우 작습니다. 모델이 특성을 제대로 활용하지 못하고 있을 수 있습니다.\n")
                f.write("가능한 원인:\n")
                f.write("1. 타겟 변수의 불균형 (클래스 불균형)\n")
                f.write("2. 모델이 과적합 또는 과소적합됨\n")
                f.write("3. 사용된 특성들이 타겟 변수와 관련이 적음\n")
                
        logging.info(f"XAI 분석 결과가 {summary_file}에 저장되었습니다.")
        return True
    except Exception as e:
        logging.error(f"XAI 결과 분석 중 오류: {str(e)}")
        return False

def explore_data(data, output_dir):
    """
    데이터 탐색 및 시각화를 수행합니다.
    """
    try:
        # 출력 디렉토리 확인 및 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
            
        # 기본 통계 정보
        logging.info(f"데이터셋 형태: {data.shape}")
        
        # 결측치 확인
        missing_values = data.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if not missing_values.empty:
            logging.info(f"결측치가 있는 열:\n{missing_values}")
            
            # 결측치 시각화
            plt.figure(figsize=(12, 6))
            missing_values.plot(kind='bar')
            plt.title('결측치 개수')
            plt.xlabel('열')
            plt.ylabel('결측치 개수')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/missing_values.png')
            plt.close()
            
        # 데이터 분포 시각화 (수치형 열)
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        
        # 히스토그램
        for i in range(0, len(numeric_columns), 4):  # 4개씩 묶어서 처리
            cols = numeric_columns[i:i+4]
            if len(cols) > 0:
                plt.figure(figsize=(15, 10))
                for j, col in enumerate(cols, 1):
                    if j <= len(cols):  # 실제 열 수보다 많은 subplot을 만들지 않도록
                        plt.subplot(2, 2, j)
                        sns.histplot(data[col], kde=True)
                        plt.title(f'{col} 분포')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/distribution_hist_{i}.png')
                plt.close()
                
        # 상관관계 분석
        if len(numeric_columns) > 0:
            # 상관관계 행렬 계산
            correlation_matrix = data[numeric_columns].corr()
            
            # 상관관계 히트맵 생성
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
            plt.title('수치형 변수 간 상관관계')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/correlation_heatmap.png')
            plt.close()
            
            # 강한 상관관계(절대값 0.7 이상)를 가진 변수 쌍 찾기
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i):
                    if abs(correlation_matrix.iloc[i, j]) >= 0.7:
                        strong_correlations.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))
            
            if strong_correlations:
                logging.info("강한 상관관계를 가진 변수 쌍:")
                for var1, var2, corr in strong_correlations:
                    logging.info(f"{var1} - {var2}: {corr:.4f}")
        
        # 타겟 변수 분포 확인 (데이터에 'Defects' 또는 다른 불량 관련 열이 있다고 가정)
        defect_columns = [col for col in data.columns if 'Defect' in col or 'defect' in col.lower()]
        if defect_columns:
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(defect_columns[:8], 1):  # 최대 8개까지만 표시
                plt.subplot(2, 4, i)
                value_counts = data[col].value_counts()
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.title(f'{col} 분포')
                plt.xlabel('값')
                plt.ylabel('빈도')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/defect_distribution.png')
            plt.close()
            
            # 각 불량 유형의 빈도 로깅
            for col in defect_columns:
                value_counts = data[col].value_counts()
                logging.info(f"{col} 빈도:\n{value_counts}")
        
        logging.info("데이터 탐색 및 시각화 완료")
        return True
    except Exception as e:
        logging.error(f"데이터 탐색 중 오류 발생: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def compare_ontology_xai(onto, top_features, output_dir):
    comparison_report = f'{output_dir}/ontology_xai_comparison_report.txt'
    with open(comparison_report, 'w') as f:
        f.write("온톨로지 추론 결과와 XAI 분석 결과 비교 보고서\n")
        f.write("=========================================\n\n")

        for idx, row in top_features.iterrows():
            feature = normalize_feature_name(row['Feature'])
            shap_value = row['SHAP_Value']
            ontology_feature = feature_mapping.get(feature, feature)

            if onto[ontology_feature]:
                f.write(f"특성 '{ontology_feature}'는 온톨로지에 존재하며, SHAP 값은 {shap_value:.4f}입니다.\n")
            else:
                f.write(f"특성 '{ontology_feature}'는 온톨로지에 존재하지 않으며, SHAP 값은 {shap_value:.4f}입니다.\n")

    logging.info(f"Ontology and XAI comparison report saved to {comparison_report}")

def main():
    data_path = '/Users/raziel/Desktop/XAI/Dataset/data/DieCasting_Quality_Raw_Data.csv'
    
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(data_path):
            logging.error(f"데이터 파일을 찾을 수 없습니다: {data_path}")
            return
            
        logging.info(f"XAI와 온톨로지 통합 프로세스 시작 (데이터: {data_path})")
        
        # 주요 출력 디렉토리 설정
        output_dir = '/Users/raziel/Desktop/XAI/output'
        ontology_dir = '/Users/raziel/Desktop/XAI/ontology'
        
        for directory in [output_dir, ontology_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"디렉토리 생성: {directory}")
            
        # 1단계: 데이터 준비 및 전처리
        data = preprocess_data(data_path)
        logging.info("데이터 전처리 완료")
        
        # 1.1단계: 데이터 탐색 및 시각화
        explore_data(data, output_dir)
        logging.info("데이터 탐색 및 시각화 완료")

        # 2단계: AI 모델 학습
        model, X_test, y_test, defect_col = train_model(data)
        logging.info("모델 학습 완료")

        # 3단계: XAI 기법 적용
        top_features = apply_xai(model, X_test, y_test, output_dir, defect_col)
        logging.info("XAI 분석 완료")
        
        # 3.1단계: XAI 결과 분석
        analyze_xai_results(top_features, model, X_test, output_dir)
        logging.info("XAI 결과 분석 완료")

        # 4단계: 온톨로지 구축
        ontology_path = f"{ontology_dir}/diecasting_ontology.owl"
        config_path = '/Users/raziel/Desktop/XAI/ontology/diecasting_ontology_config.json'
        updated_ontology_path = f"{ontology_dir}/diecasting_ontology_updated.owl"
        inferred_ontology_path = f"{ontology_dir}/diecasting_ontology_inferred.owl"
        onto = construct_ontology(ontology_path, config_path)
        if not onto:
            logging.error("Ontology construction failed. Exiting.")
            return False # Stop execution if ontology failed
        logging.info(f"Ontology construction/loading completed: {ontology_path}")

        # 5단계: XAI와 온톨로지 접목
        onto_integrated, processed_features_names = integrate_xai_with_ontology(onto, top_features, X_test, defect_col)
        if not onto_integrated:
             logging.error("XAI integration with ontology failed.")
        else:
             onto = onto_integrated # Update onto object
             onto.save(file=updated_ontology_path, format="rdfxml")
             logging.info(f"XAI results integrated and saved to: {updated_ontology_path}")
             ontology_path = updated_ontology_path

        # 6단계: 온톨로지 검증
        # Pass the set of processed feature names to comprehensive verification
        verify_ontology_comprehensive(onto, top_features, processed_features_names, output_dir)
        logging.info("Comprehensive ontology verification completed.")
        
        # 주요 특성 선택 (Use original feature name from top_features for verification call)
        important_param_original = top_features.iloc[0]['Feature'] if not top_features.empty else "N/A"
        verify_ontology(onto, important_param_original) # Pass the original name
        
        # 7단계: 결과 요약 및 보고서 생성
        summary_report = f"{output_dir}/xai_ontology_integration_report.txt"
        with open(summary_report, 'w') as f:
            f.write("XAI와 온톨로지 통합 보고서\n")
            f.write("==========================\n\n")
            
            f.write("1. 프로젝트 요약\n")
            f.write(f"   - 데이터셋: {data_path}\n")
            f.write(f"   - 레코드 수: {data.shape[0]}\n")
            f.write(f"   - 변수 수: {data.shape[1]}\n")
            f.write(f"   - 타겟 변수: {defect_col}\n\n")
            
            f.write("2. 모델 성능\n")
            f.write(f"   - 사용된 알고리즘: {type(model).__name__}\n")
            f.write(f"   - 모델 파라미터: {model.get_params()}\n\n")
            
            f.write("3. XAI 결과\n")
            f.write(f"   - 주요 특성:\n")
            for idx, row in top_features.head(5).iterrows():
                f.write(f"     * {row['Feature']}: {row['SHAP_Value']:.6f}\n")
            
            f.write("\n4. 온톨로지 통합\n")
            f.write(f"   - 온톨로지 파일: {updated_ontology_path}\n")
            
            f.write("\n5. 후속 조치 및 권장사항\n")
            if top_features['SHAP_Value'].abs().max() < 0.01:
                f.write("   - 모델의 특성 중요도가 낮습니다. 다음과 같은 개선이 필요합니다:\n")
                f.write("     * 더 많은 데이터 수집\n")
                f.write("     * 타겟 변수 재검토 (불균형 문제 해결)\n")
                f.write("     * 특성 엔지니어링 강화\n")
            else:
                f.write("   - 주요 특성과 불량 간의 관계를 심층 분석하여 불량 예방책 수립\n")
                f.write("   - 온톨로지를 활용한 지식 기반 의사결정 지원 시스템 구축\n")
        
        logging.info(f"통합 보고서가 {summary_report}에 저장되었습니다.")
        
        # 결과 출력
        print("\n======= XAI와 온톨로지 통합 프로세스 결과 =======")
        print(f"1. 데이터셋: {data_path}")
        print(f"2. 타겟 변수: {defect_col}")
        print(f"3. 상위 중요 변수: {', '.join(top_features['Feature'].values[:3])}")
        print(f"4. 온톨로지 저장 위치: {updated_ontology_path}")
        if not top_features['SHAP_Value'].abs().max() < 0.01:
            important_feature = top_features.iloc[0]['Feature']
            print(f"5. 주요 온톨로지 관계: {important_feature} → {defect_col}")
        print(f"6. 결과 보고서: {summary_report}")
        print("===============================================\n")
        
        # 온톨로지 추론 (메인 함수 내에서 실행)
        logging.info("Starting ontology reasoning...")
        try:
            sync_reasoner([onto]) # Pass the ontology object in a list
            logging.info("Ontology reasoning completed.")
            # 추론 결과 저장
            onto.save(file=inferred_ontology_path, format="rdfxml")
            logging.info(f"Inferred ontology saved to {inferred_ontology_path}")
        except Exception as reasoner_e:
             logging.error(f"Error during ontology reasoning or saving inferred ontology: {reasoner_e}")
             logging.error(traceback.format_exc())

        # XAI-온톨로지 비교 (추론된 온톨로지 사용 가능하면 사용)
        compare_ontology_xai(onto, top_features, output_dir) # 비교 함수는 수정 필요 없을 수 있음

        logging.info("XAI와 온톨로지 통합 프로세스 성공적으로 완료!")
        return True
    except Exception as e:
        logging.error(f"프로세스 실행 중 오류 발생: {str(e)}")
        logging.error(traceback.format_exc())
        
        print("\n======= 오류 발생 =======")
        print(f"프로세스 실행 중 오류가 발생했습니다: {str(e)}")
        print("자세한 내용은 로그 파일을 확인하세요.")
        print("=========================\n")
        return False

if __name__ == "__main__":
    main()
