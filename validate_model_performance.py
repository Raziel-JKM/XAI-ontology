import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('model_validation.log'),
                              logging.StreamHandler()])

# 출력 디렉토리 생성
output_dir = '/Users/raziel/Desktop/XAI/output/model_validation'
os.makedirs(output_dir, exist_ok=True)

def preprocess_data(data_path='/Users/raziel/Desktop/XAI/Dataset/data/DieCasting_Quality_Raw_Data.csv'):
    """
    데이터 전처리 함수
    """
    # 데이터 로드
    data = pd.read_csv(data_path)
    logging.info(f"원본 데이터 크기: {data.shape}")
    
    # 수치형 컬럼 추출
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # 이상치 제거 (IQR 방법)
    clean_data = data.copy()
    for col in numeric_columns:
        if clean_data[col].nunique() > 1:
            Q1 = clean_data[col].quantile(0.25)
            Q3 = clean_data[col].quantile(0.75)
            IQR = Q3 - Q1
            clean_data = clean_data[(clean_data[col] >= (Q1 - 1.5 * IQR)) & 
                                   (clean_data[col] <= (Q3 + 1.5 * IQR))]
    
    logging.info(f"이상치 제거 후 데이터 크기: {clean_data.shape}")
    
    # 타겟 변수 설정 (불량 컬럼 찾기)
    defect_cols = [col for col in clean_data.columns if 'Defect' in col or 'defect' in col.lower()]
    if not defect_cols:
        # 타겟 변수를 찾을 수 없는 경우, 'Defects' 컬럼 생성
        logging.info("불량 컬럼을 찾을 수 없어 'Defects' 컬럼을 생성합니다.")
        if 'Process.1' in clean_data.columns:
            threshold = clean_data['Process.1'].quantile(0.95)  # 상위 5%를 불량으로 간주
            clean_data['Defects'] = (clean_data['Process.1'] > threshold).astype(int)
        else:
            # 임의의 불량 데이터 생성 (약 5%)
            clean_data['Defects'] = 0
            defect_indices = np.random.choice(
                clean_data.index, 
                size=int(len(clean_data) * 0.05), 
                replace=False
            )
            clean_data.loc[defect_indices, 'Defects'] = 1
        
        defect_col = 'Defects'
    else:
        defect_col = defect_cols[0]
    
    # 독립변수와 타겟 변수 분리
    X = clean_data.drop(columns=[defect_col])
    y = clean_data[defect_col]
    
    # 수치형 데이터만 선택
    X_numeric = X.select_dtypes(include=[np.number])
    
    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_numeric.columns)
    
    # SMOTE 적용
    if len(np.unique(y)) > 1:
        try:
            smote = SMOTE(sampling_strategy=0.2, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_scaled_df, y)
            logging.info(f"SMOTE 적용 후 데이터 크기: X={X_resampled.shape}, y={y_resampled.shape}")
            defect_ratio = np.mean(y_resampled)
            logging.info(f"SMOTE 적용 후 불량 비율: {defect_ratio:.2%}")
        except Exception as e:
            logging.error(f"SMOTE 적용 중 오류 발생: {str(e)}")
            X_resampled, y_resampled = X_scaled_df, y
    else:
        X_resampled, y_resampled = X_scaled_df, y
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    logging.info(f"훈련 세트 크기: X_train={X_train.shape}, y_train={y_train.shape}")
    logging.info(f"테스트 세트 크기: X_test={X_test.shape}, y_test={y_test.shape}")
    
    return X_train, X_test, y_train, y_test, X_numeric.columns

def train_and_evaluate_models():
    """
    각 예측 모델의 성능을 훈련하고 평가합니다.
    1. XGBoost
    2. RandomForest
    3. LogisticRegression
    """
    # 데이터 전처리
    X_train, X_test, y_train, y_test, feature_names = preprocess_data()
    
    # 모델 목록
    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1, 
            objective='binary:logistic',
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=6, 
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            solver='liblinear', 
            random_state=42
        )
    }
    
    # 결과 저장
    results = {}
    
    # 혼동 행렬 시각화를 위한 그림 설정
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
    
    # 각 모델 훈련 및 평가
    for i, (model_name, model) in enumerate(models.items()):
        logging.info(f"{model_name} 모델 훈련 시작...")
        model.fit(X_train, y_train)
        
        # 모델 저장
        joblib.dump(model, f'{output_dir}/{model_name.lower()}_model.pkl')
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 성능 평가
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # 혼동 행렬
        cm = confusion_matrix(y_test, y_pred)
        
        # 결과 저장
        results[model_name] = {
            'model': model,
            'metrics': metrics,
            'confusion_matrix': cm
        }
        
        # 로깅
        logging.info(f"{model_name} 성능 평가 결과:")
        for metric_name, value in metrics.items():
            logging.info(f"  {metric_name}: {value:.4f}")
        
        # 혼동 행렬 시각화
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i])
        axes[i].set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]:.4f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # 그래프 저장
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png')
    
    # 성능 비교 그래프
    plt.figure(figsize=(10, 6))
    
    # 모델별 성능 지표
    model_names = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    
    # 성능 지표별 그래프
    for metric_name in metrics_names:
        values = [results[model_name]['metrics'][metric_name] for model_name in model_names]
        plt.bar([f"{model} ({metric_name})" for model in model_names], values)
    
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.title('모델별 성능 비교')
    plt.ylabel('점수')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_performance_comparison.png')
    
    # 결과 요약
    summary = {model_name: results[model_name]['metrics'] for model_name in model_names}
    
    # 결과 테이블 생성
    summary_df = pd.DataFrame(summary).transpose()
    summary_df.to_csv(f'{output_dir}/model_performance_summary.csv')
    
    logging.info(f"모델 성능 요약:\n{summary_df}")
    
    # 모델 하이퍼파라미터 최적화 (XGBoost 모델)
    logging.info("XGBoost 모델 하이퍼파라미터 최적화 시작...")
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }
    
    grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(objective='binary:logistic', random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # 최적 파라미터 및 성능
    logging.info(f"최적 파라미터: {grid_search.best_params_}")
    logging.info(f"최적 F1 점수: {grid_search.best_score_:.4f}")
    
    # 최적화된 모델 성능 평가
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    
    best_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_best),
        'precision': precision_score(y_test, y_pred_best, zero_division=0),
        'recall': recall_score(y_test, y_pred_best, zero_division=0),
        'f1': f1_score(y_test, y_pred_best, zero_division=0)
    }
    
    logging.info("최적화된 XGBoost 모델 성능:")
    for metric_name, value in best_metrics.items():
        logging.info(f"  {metric_name}: {value:.4f}")
    
    # 최적화된 모델 저장
    joblib.dump(best_model, f'{output_dir}/xgboost_optimized_model.pkl')
    
    # 논문과의 비교
    paper_metrics = {
        'XGBoost': {'accuracy': 0.9992, 'precision': 1.0000, 'recall': 0.9912, 'f1': 0.9955},
        'RandomForest': {'accuracy': 0.9983, 'precision': 0.9911, 'recall': 0.9823, 'f1': 0.9866},
        'LogisticRegression': {'accuracy': 0.5000, 'precision': 0.0000, 'recall': 0.0000, 'f1': 0.0000}
    }
    
    # 논문 결과와 비교
    comparison_results = {}
    for model_name in model_names:
        comparison = {}
        for metric in metrics_names:
            actual = results[model_name]['metrics'][metric]
            expected = paper_metrics[model_name][metric]
            diff = actual - expected
            comparison[metric] = {
                'actual': actual,
                'expected': expected,
                'difference': diff
            }
        comparison_results[model_name] = comparison
    
    # 비교 결과 로깅
    logging.info("논문 결과와의 비교:")
    for model_name, comparison in comparison_results.items():
        logging.info(f"  {model_name}:")
        for metric, values in comparison.items():
            logging.info(f"    {metric}: 실제={values['actual']:.4f}, 논문={values['expected']:.4f}, 차이={values['difference']:.4f}")
    
    return results, comparison_results, best_model

if __name__ == "__main__":
    results, comparison, best_model = train_and_evaluate_models()
    
    # 결과 요약 출력
    logging.info("검증 완료: 모델 성능 평가 및 비교")
    
    # 성능이 가장 좋은 모델 찾기
    best_f1 = 0
    best_model_name = ""
    
    for model_name, result in results.items():
        f1 = result['metrics']['f1']
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_name
    
    logging.info(f"가장 성능이 좋은 모델: {best_model_name} (F1 점수: {best_f1:.4f})") 