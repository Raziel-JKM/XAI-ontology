import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import joblib
import shap
from dice_ml import DiCE
from dice_ml.utils import helpers
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('xai_validation.log'),
                              logging.StreamHandler()])

# 출력 디렉토리 생성
output_dir = '/Users/raziel/Desktop/XAI/output/xai_validation'
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
    
    return X_train, X_test, y_train, y_test, X_numeric.columns, scaler

def train_xgboost_model(X_train, y_train):
    """
    XGBoost 모델 훈련
    """
    # 모델 로드 시도 (이미 훈련된 모델이 있는 경우)
    model_path = '/Users/raziel/Desktop/XAI/output/model_validation/xgboost_model.pkl'
    
    if os.path.exists(model_path):
        logging.info(f"훈련된 모델을 로드합니다: {model_path}")
        model = joblib.load(model_path)
    else:
        logging.info("새 모델을 훈련합니다.")
        model = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1, 
            objective='binary:logistic',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # 모델 저장
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
    
    return model

def validate_shap_analysis(model, X_test, feature_names):
    """
    SHAP 값을 계산하고 시각화하여 모델의 예측을 설명
    """
    logging.info("SHAP 분석 시작...")
    
    # 분석 표본 선택 (최대 100개)
    sample_size = min(100, X_test.shape[0])
    X_sample = X_test.iloc[:sample_size]
    
    # SHAP 설명자 초기화
    explainer = shap.Explainer(model)
    
    try:
        # SHAP 값 계산
        shap_values = explainer(X_sample)
        
        # 전체 요약 그래프 (전체 특성의 중요도)
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values, max_display=10, show=False)
        plt.title("SHAP 특성 중요도")
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_feature_importance.png')
        
        # 상위 10개 특성의 중요도 값 저장
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values.values).mean(0)
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
        feature_importance.to_csv(f'{output_dir}/shap_top10_features.csv', index=False)
        
        # 상위 10개 특성의 중요도 출력
        logging.info("상위 10개 특성 중요도:")
        for idx, row in feature_importance.iterrows():
            logging.info(f"  {row['feature']}: {row['importance']:.6f}")
        
        # SHAP 요약 그래프 (특성 값과 SHAP 값 간의 관계)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values.values, X_sample, feature_names=feature_names, show=False)
        plt.title("SHAP 요약 그래프")
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_summary_plot.png')
        
        # 샘플 사례에 대한 SHAP 값 계산
        instance_idx = 0  # 첫 번째 샘플 사례
        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(shap_values[instance_idx], max_display=10, show=False)
        plt.title(f"샘플 사례 설명 (ID: {instance_idx})")
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_instance_explanation.png')
        
        # 특성 의존성 그래프
        if len(feature_importance) > 0:
            top_feature = feature_importance.iloc[0]['feature']
            plt.figure(figsize=(10, 6))
            shap.plots.scatter(shap_values[:, top_feature], show=False)
            plt.title(f"SHAP 의존성 그래프: {top_feature}")
            plt.tight_layout()
            plt.savefig(f'{output_dir}/shap_dependency_{top_feature}.png')
        
        return {
            'top_features': feature_importance['feature'].tolist(),
            'shap_values_mean': np.abs(shap_values.values).mean(0).tolist()
        }
        
    except Exception as e:
        logging.error(f"SHAP 분석 중 오류 발생: {str(e)}")
        return None

def validate_counterfactual_analysis(model, X_train, X_test, y_train, feature_names, scaler):
    """
    Counterfactual 설명 생성 및 검증
    """
    logging.info("Counterfactual 분석 시작...")
    
    try:
        # 데이터 준비 (DiCE는 원본 형식의 데이터와 모델을 필요로 함)
        d = X_train.copy()
        d['target'] = y_train
        
        # DiCE 데이터 인터페이스 생성
        dice_data = helpers.load_custom_dataset(dataframe=d, continuous_features=feature_names.tolist(), outcome_name='target')
        
        # DiCE 모델 인터페이스 생성
        dice_model = DiCE(dice_data, model)
        
        # 불량으로 예측된 사례 선택
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        y_pred = model.predict(X_test)
        defect_indices = np.where(y_pred == 1)[0]
        
        if len(defect_indices) == 0:
            logging.warning("테스트 세트에서 불량으로 예측된 사례가 없습니다.")
            # 임의의 샘플을 선택
            instance_for_cf = X_test_df.iloc[0].copy()
        else:
            # 불량으로 예측된 첫 번째 사례 선택
            defect_idx = defect_indices[0]
            instance_for_cf = X_test_df.iloc[defect_idx].copy()
        
        logging.info(f"Counterfactual 생성을 위한 샘플: {instance_for_cf.name}")
        
        # Counterfactual 생성
        cf_examples = dice_model.generate_counterfactuals(
            instance_for_cf, 
            total_CFs=3, 
            desired_class=0  # 정상 클래스로 변경
        )
        
        # 결과 시각화 및 저장
        plt.figure(figsize=(12, 8))
        cf_examples.visualize_as_dataframe(show_only_changes=True)
        plt.title("Counterfactual 예시")
        plt.tight_layout()
        plt.savefig(f'{output_dir}/counterfactual_examples.png')
        
        # Counterfactual 결과 분석
        cf_df = cf_examples.cf_examples_list[0].final_cfs_df
        
        if cf_df.empty:
            logging.warning("생성된 Counterfactual이 없습니다.")
            return {
                'success': False,
                'error': "Counterfactual을 생성할 수 없습니다."
            }
        
        # 변경된 특성 확인
        original_instance = pd.DataFrame([instance_for_cf], columns=feature_names)
        
        # 각 Counterfactual에 대한 변경 사항 분석
        cf_changes = []
        
        for i in range(cf_df.shape[0]):
            cf_instance = cf_df.iloc[i].drop('target')  # 타겟 열 제외
            
            # 원본과 Counterfactual 간의 차이 계산
            changes = {}
            for feature in feature_names:
                if feature in cf_instance.index:
                    diff = cf_instance[feature] - original_instance[feature].values[0]
                    # 중요한 변경사항만 포함 (값의 변화가 있는 경우)
                    if abs(diff) > 1e-5:  # 작은 값 필터링
                        changes[feature] = {
                            'original': original_instance[feature].values[0],
                            'counterfactual': cf_instance[feature],
                            'difference': diff
                        }
            
            cf_changes.append({
                'cf_index': i,
                'changes': changes
            })
        
        # 결과 저장
        with open(f'{output_dir}/counterfactual_changes.txt', 'w') as f:
            f.write(f"원본 인스턴스 (예측 클래스: 불량):\n")
            for feature in feature_names:
                f.write(f"  {feature}: {original_instance[feature].values[0]:.4f}\n")
            
            f.write("\nCounterfactual 변경 사항 (예측 클래스: 정상):\n")
            for cf in cf_changes:
                f.write(f"\nCounterfactual #{cf['cf_index'] + 1}:\n")
                
                if not cf['changes']:
                    f.write("  변경 사항 없음\n")
                else:
                    # 변경 크기순으로 정렬
                    sorted_changes = sorted(
                        cf['changes'].items(), 
                        key=lambda x: abs(x[1]['difference']), 
                        reverse=True
                    )
                    
                    for feature, change in sorted_changes:
                        f.write(f"  {feature}: {change['original']:.4f} -> {change['counterfactual']:.4f} (변화량: {change['difference']:.4f})\n")
        
        # 주요 변경사항 로깅
        logging.info("Counterfactual 분석 결과:")
        for cf in cf_changes:
            logging.info(f"  Counterfactual #{cf['cf_index'] + 1}:")
            
            if not cf['changes']:
                logging.info("    변경 사항 없음")
            else:
                # 변경 크기순으로 정렬
                sorted_changes = sorted(
                    cf['changes'].items(), 
                    key=lambda x: abs(x[1]['difference']), 
                    reverse=True
                )
                
                for feature, change in sorted_changes[:5]:  # 상위 5개만 로깅
                    logging.info(f"    {feature}: {change['original']:.4f} -> {change['counterfactual']:.4f} (변화량: {change['difference']:.4f})")
        
        return {
            'success': True,
            'counterfactuals_count': cf_df.shape[0],
            'changes': cf_changes
        }
        
    except Exception as e:
        logging.error(f"Counterfactual 분석 중 오류 발생: {str(e)}")
        
        # 오류 메시지 파일에 저장
        with open(f'{output_dir}/counterfactual_error.txt', 'w') as f:
            f.write(f"Counterfactual 분석 중 오류 발생: {str(e)}")
        
        return {
            'success': False,
            'error': str(e)
        }

def compare_with_paper_results(shap_results):
    """
     주요 SHAP 특성들과 비교합니다.
    """
    if shap_results is None:
        logging.error("SHAP 결과가 없어 논문 비교를 수행할 수 없습니다.")
        return
    
    #  상위 특성들
    paper_top_features = [
        'Process.1', 'Process.2', 'Process.3', 'Process.4', 'Process.5'
    ]
    
    # 논문에 있는 중요한 특성들이 현재 분석에서도 상위에 랭크되는지 확인
    found_features = []
    for paper_feature in paper_top_features:
        if paper_feature in shap_results['top_features']:
            rank = shap_results['top_features'].index(paper_feature) + 1
            found_features.append((paper_feature, rank))
    
    # 결과 로깅
    logging.info("논문 결과와 SHAP 분석 비교:")
    
    if not found_features:
        logging.info("   중요 특성이 현재 분석의 상위 특성에 포함되지 않았습니다.")
    else:
        for feature, rank in found_features:
            logging.info(f"  {feature}: 현재 분석 순위 #{rank}")
    
    # 논문에 언급된 특성 중 현재 분석의 상위 10개에 포함된 특성 비율
    paper_features_in_top10 = len(found_features) / len(paper_top_features)
    logging.info(f"  논문 특성 중 현재 분석 상위 10개에 포함된 비율: {paper_features_in_top10:.2%}")
    
    return {
        'paper_features_found': found_features,
        'paper_features_in_top10_ratio': paper_features_in_top10
    }

if __name__ == "__main__":
    # 데이터 전처리
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data()
    
    # XGBoost 모델 훈련
    model = train_xgboost_model(X_train, y_train)
    
    # SHAP 분석 검증
    shap_results = validate_shap_analysis(model, X_test, feature_names)
    
    # 논문 결과와 비교
    if shap_results:
        compare_results = compare_with_paper_results(shap_results)
    
    # Counterfactual 분석 검증
    cf_results = validate_counterfactual_analysis(model, X_train, X_test, y_train, feature_names, scaler)
    
    # 로깅
    logging.info("XAI 분석 검증 완료")
    
    # 결과 요약
    with open(f'{output_dir}/xai_validation_summary.txt', 'w') as f:
        f.write("XAI 분석 검증 요약\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. SHAP 분석 결과:\n")
        if shap_results:
            f.write(f"  - 상위 5개 중요 특성: {', '.join(shap_results['top_features'][:5])}\n")
        else:
            f.write("  - SHAP 분석 실패\n")
        
        f.write("\n2. 논문 결과와 비교:\n")
        if 'compare_results' in locals() and compare_results:
            f.write(f"  - 논문 특성 중 현재 분석 상위 10개에 포함된 비율: {compare_results['paper_features_in_top10_ratio']:.2%}\n")
            f.write("  - 논문 특성 현재 순위:\n")
            for feature, rank in compare_results['paper_features_found']:
                f.write(f"    * {feature}: #{rank}\n")
        else:
            f.write("  - 비교 분석 수행 불가\n")
        
        f.write("\n3. Counterfactual 분석 결과:\n")
        if cf_results['success']:
            f.write(f"  - 생성된 Counterfactual 수: {cf_results['counterfactuals_count']}\n")
            if cf_results['counterfactuals_count'] > 0:
                f.write("  - 주요 변경 특성:\n")
                for cf in cf_results['changes']:
                    f.write(f"    * Counterfactual #{cf['cf_index'] + 1}:\n")
                    if not cf['changes']:
                        f.write("      - 변경 사항 없음\n")
                    else:
                        sorted_changes = sorted(
                            cf['changes'].items(), 
                            key=lambda x: abs(x[1]['difference']), 
                            reverse=True
                        )
                        for feature, change in sorted_changes[:3]:  # 상위 3개만 표시
                            f.write(f"      - {feature}: {change['original']:.4f} -> {change['counterfactual']:.4f}\n")
        else:
            f.write(f"  - Counterfactual 분석 실패: {cf_results['error']}\n")
        
        f.write("\n4. 결론:\n")
        if shap_results and cf_results['success']:
            f.write("  - XAI 분석이 성공적으로 수행되었습니다.\n")
            if 'compare_results' in locals() and compare_results and compare_results['paper_features_in_top10_ratio'] > 0.5:
                f.write("  - 논문 결과와 유사한 특성 중요도가 도출되었습니다.\n")
            else:
                f.write("  - 논문 결과와 현재 분석 결과에 차이가 있습니다.\n")
        else:
            f.write("  - 일부 XAI 분석에 실패하였습니다.\n") 
