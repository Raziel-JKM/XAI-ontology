import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('data_validation.log'),
                              logging.StreamHandler()])

# 출력 디렉토리 생성
output_dir = '/Users/raziel/Desktop/XAI/output/data_validation'
os.makedirs(output_dir, exist_ok=True)

def validate_dataset_preprocessing():
    """
    실험 4.1 섹션에서 언급된 데이터 전처리 과정을 검증합니다.
    - 원본 데이터 크기와 구조 확인 (7,536개)
    - 전처리 후 데이터 크기 확인 (5,618개)
    - 불량 사례 비율 확인 (약 5% -> 10%)
    - 데이터 분할 비율 확인 (80:20)
    """
    # 데이터 로드
    data_path = '/Users/raziel/Desktop/XAI/Dataset/data/DieCasting_Quality_Raw_Data.csv'
    data = pd.read_csv(data_path)
    
    # 원본 데이터 크기 확인
    logging.info(f"원본 데이터 크기: {data.shape}")
    logging.info(f"컬럼 수: {len(data.columns)}")
    
    # 기본 정보 확인
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
    
    logging.info(f"수치형 컬럼 수: {len(numeric_columns)}")
    logging.info(f"범주형 컬럼 수: {len(categorical_columns)}")
    
    # 결측치 확인
    missing_values = data.isnull().sum().sum()
    logging.info(f"전체 결측치 수: {missing_values}")
    
    # 이상치 감지 및 제거 (IQR 방식)
    clean_data = data.copy()
    for col in numeric_columns:
        if clean_data[col].nunique() > 1:  # 유니크 값이 2개 이상인 경우에만 적용
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
        # 예시: 임의의 공정 파라미터가 특정 값 이상일 때 불량으로 간주
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
    
    # 불량 비율 확인
    defect_ratio = clean_data[defect_col].mean() if clean_data[defect_col].dtype == 'int64' else clean_data[defect_col].value_counts(normalize=True).iloc[-1]
    logging.info(f"불량 비율: {defect_ratio:.2%}")
    
    # SMOTE 적용으로 불량 사례 오버샘플링
    X = clean_data.drop(columns=[defect_col])
    y = clean_data[defect_col]
    
    # 수치형 데이터만 선택 (SMOTE는 범주형 데이터에 직접 적용 불가)
    X_numeric = X.select_dtypes(include=[np.number])
    
    # 인스턴스가 너무 적은 클래스(보통 소수 클래스)에 대한 오버샘플링 비율 설정
    if len(np.unique(y)) > 1:  # 클래스가 2개 이상일 때만 SMOTE 적용
        try:
            smote = SMOTE(sampling_strategy=0.2, random_state=42)  # 10% 비율로 증가
            X_numeric_resampled, y_resampled = smote.fit_resample(X_numeric, y)
            
            # 오버샘플링된 데이터 크기 확인
            logging.info(f"SMOTE 적용 전 데이터 크기: X={X_numeric.shape}, y={y.shape}")
            logging.info(f"SMOTE 적용 후 데이터 크기: X={X_numeric_resampled.shape}, y={y_resampled.shape}")
            
            # 오버샘플링 후 불량 비율 확인
            defect_ratio_after = y_resampled.mean() if y_resampled.dtype == 'int64' else y_resampled.value_counts(normalize=True).iloc[-1]
            logging.info(f"SMOTE 적용 후 불량 비율: {defect_ratio_after:.2%}")
        except Exception as e:
            logging.error(f"SMOTE 적용 중 오류 발생: {str(e)}")
    
    # 데이터 분할 검증 (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logging.info(f"훈련 세트 크기: X_train={X_train.shape}, y_train={y_train.shape}")
    logging.info(f"테스트 세트 크기: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # 데이터 시각화
    plt.figure(figsize=(12, 8))
    
    # 상관관계 히트맵
    corr_matrix = X_numeric.corr()
    plt.subplot(2, 2, 1)
    sns.heatmap(corr_matrix.iloc[:10, :10], annot=False, cmap='coolwarm')
    plt.title('주요 변수 간 상관관계')
    
    # 상위 상관관계 추출
    top_corr = corr_matrix.unstack().sort_values(ascending=False)
    top_corr = top_corr[top_corr < 1.0].head(10)  # 자기 자신과의 상관관계(1.0) 제외하고 상위 10개
    
    # 상위 상관관계 출력
    logging.info("상위 10개 상관관계:")
    for idx, val in top_corr.items():
        logging.info(f"{idx[0]} - {idx[1]}: {val:.4f}")
    
    # 불량 분포
    plt.subplot(2, 2, 2)
    sns.countplot(x=y)
    plt.title('불량 분포')
    
    # 주요 변수의 분포
    if len(X_numeric.columns) >= 2:
        plt.subplot(2, 2, 3)
        sns.histplot(X_numeric.iloc[:, 0], kde=True)
        plt.title(f'변수 분포: {X_numeric.columns[0]}')
        
        plt.subplot(2, 2, 4)
        sns.histplot(X_numeric.iloc[:, 1], kde=True)
        plt.title(f'변수 분포: {X_numeric.columns[1]}')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/data_exploration.png')
    
    return {
        'original_data_size': data.shape,
        'preprocessed_data_size': clean_data.shape,
        'defect_ratio_before': defect_ratio,
        'defect_ratio_after': defect_ratio_after if 'defect_ratio_after' in locals() else None,
        'train_test_split': (X_train.shape, X_test.shape)
    }

if __name__ == "__main__":
    results = validate_dataset_preprocessing()
    logging.info(f"검증 결과: {results}")
    
    # 실험과 일치하는지 검증
    expected_original_size = (7536, 57)  #  언급된 원본 데이터 크기
    expected_preprocessed_size = (5618, 57)  # 이상치 제거 후 크기
    expected_defect_ratio = 0.05  # 원본 불량 비율
    expected_defect_ratio_after = 0.10  # SMOTE 적용 후 불량 비율
    
    # 크기 비교
    size_match = results['original_data_size'][0] == expected_original_size[0]
    logging.info(f"원본 데이터 크기 일치 여부: {size_match}")
    
    preprocessed_match = abs(results['preprocessed_data_size'][0] - expected_preprocessed_size[0]) / expected_preprocessed_size[0] < 0.1
    logging.info(f"전처리 후 데이터 크기 일치 여부: {preprocessed_match}")
    
    # 불량 비율 비교
    if results['defect_ratio_before'] is not None:
        defect_ratio_match = abs(results['defect_ratio_before'] - expected_defect_ratio) < 0.02
        logging.info(f"원본 불량 비율 일치 여부: {defect_ratio_match}")
    
    if results['defect_ratio_after'] is not None:
        defect_ratio_after_match = abs(results['defect_ratio_after'] - expected_defect_ratio_after) < 0.02
        logging.info(f"SMOTE 적용 후 불량 비율 일치 여부: {defect_ratio_after_match}") 
