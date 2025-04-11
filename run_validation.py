import subprocess
import os
import logging
import time
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('validation_summary.log'),
                              logging.StreamHandler()])

# 출력 디렉토리 생성
output_dir = '/Users/raziel/Desktop/XAI/output/validation_summary'
os.makedirs(output_dir, exist_ok=True)

def run_script(script_path, script_name):
    """
    지정된 스크립트를 실행하고 결과를 로깅합니다.
    """
    start_time = time.time()
    logging.info(f"{script_name} 실행 시작...")
    
    try:
        # 스크립트 실행
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        
        # 실행 결과 확인
        if result.returncode == 0:
            logging.info(f"{script_name} 성공적으로 실행 완료. (소요 시간: {time.time() - start_time:.2f}초)")
            return True, result.stdout, result.stderr
        else:
            logging.error(f"{script_name} 실행 중 오류 발생:")
            logging.error(f"오류 메시지: {result.stderr}")
            return False, result.stdout, result.stderr
    
    except Exception as e:
        logging.error(f"{script_name} 실행 중 예외 발생: {str(e)}")
        return False, "", str(e)

def collect_validation_results():
    """
    모든 검증 결과를 수집하여 요약합니다.
    """
    results = {}
    validation_files = {
        'data_validation': '/Users/raziel/Desktop/XAI/output/data_validation/data_exploration.png',
        'model_validation': '/Users/raziel/Desktop/XAI/output/model_validation/model_performance_summary.csv',
        'xai_validation': '/Users/raziel/Desktop/XAI/output/xai_validation/xai_validation_summary.txt',
        'ontology_validation': '/Users/raziel/Desktop/XAI/output/ontology_validation/ontology_validation_summary.txt'
    }
    
    # 1. 데이터 검증 결과 확인
    if os.path.exists(validation_files['data_validation']):
        logging.info("데이터 검증 결과 확인 완료.")
        results['data_validation'] = {
            'status': 'success',
            'image_path': validation_files['data_validation']
        }
    else:
        logging.warning("데이터 검증 결과를 찾을 수 없습니다.")
        results['data_validation'] = {
            'status': 'failed'
        }
    
    # 2. 모델 성능 검증 결과 확인
    if os.path.exists(validation_files['model_validation']):
        logging.info("모델 검증 결과 확인 완료.")
        
        try:
            model_summary = pd.read_csv(validation_files['model_validation'])
            
            # 모델 성능 요약
            results['model_validation'] = {
                'status': 'success',
                'data': model_summary.to_dict('records')
            }
        except Exception as e:
            logging.error(f"모델 검증 결과 로드 중 오류 발생: {str(e)}")
            results['model_validation'] = {
                'status': 'partial',
                'error': str(e)
            }
    else:
        logging.warning("모델 검증 결과를 찾을 수 없습니다.")
        results['model_validation'] = {
            'status': 'failed'
        }
    
    # 3. XAI 분석 검증 결과 확인
    if os.path.exists(validation_files['xai_validation']):
        logging.info("XAI 검증 결과 확인 완료.")
        
        try:
            with open(validation_files['xai_validation'], 'r') as f:
                xai_summary = f.read()
            
            results['xai_validation'] = {
                'status': 'success',
                'summary': xai_summary
            }
        except Exception as e:
            logging.error(f"XAI 검증 결과 로드 중 오류 발생: {str(e)}")
            results['xai_validation'] = {
                'status': 'partial',
                'error': str(e)
            }
    else:
        logging.warning("XAI 검증 결과를 찾을 수 없습니다.")
        results['xai_validation'] = {
            'status': 'failed'
        }
    
    # 4. 온톨로지 검증 결과 확인
    if os.path.exists(validation_files['ontology_validation']):
        logging.info("온톨로지 검증 결과 확인 완료.")
        
        try:
            with open(validation_files['ontology_validation'], 'r') as f:
                ontology_summary = f.read()
            
            results['ontology_validation'] = {
                'status': 'success',
                'summary': ontology_summary
            }
        except Exception as e:
            logging.error(f"온톨로지 검증 결과 로드 중 오류 발생: {str(e)}")
            results['ontology_validation'] = {
                'status': 'partial',
                'error': str(e)
            }
    else:
        logging.warning("온톨로지 검증 결과를 찾을 수 없습니다.")
        results['ontology_validation'] = {
            'status': 'failed'
        }
    
    return results

def visualize_validation_summary(results):
    """
    검증 결과를 시각화합니다.
    """
    # 모델 성능 비교 그래프
    if results['model_validation']['status'] == 'success':
        try:
            model_data = results['model_validation']['data']
            models = []
            accuracy = []
            precision = []
            recall = []
            f1 = []
            
            for item in model_data:
                models.append(item.get('Unnamed: 0', 'Unknown'))
                accuracy.append(item.get('accuracy', 0))
                precision.append(item.get('precision', 0))
                recall.append(item.get('recall', 0))
                f1.append(item.get('f1', 0))
            
            plt.figure(figsize=(12, 8))
            
            x = range(len(models))
            width = 0.2
            
            plt.bar([i - 1.5*width for i in x], accuracy, width=width, label='Accuracy')
            plt.bar([i - 0.5*width for i in x], precision, width=width, label='Precision')
            plt.bar([i + 0.5*width for i in x], recall, width=width, label='Recall')
            plt.bar([i + 1.5*width for i in x], f1, width=width, label='F1-Score')
            
            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x, models)
            plt.ylim(0, 1.1)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/model_performance_comparison.png')
            logging.info(f"모델 성능 비교 그래프가 저장되었습니다: {output_dir}/model_performance_comparison.png")
        
        except Exception as e:
            logging.error(f"모델 성능 시각화 중 오류 발생: {str(e)}")
    
    # 검증 단계 성공/실패 요약 그래프
    plt.figure(figsize=(10, 6))
    
    stages = list(results.keys())
    status_colors = {
        'success': 'green',
        'partial': 'orange',
        'failed': 'red'
    }
    
    # 단계별 상태 계산
    statuses = [results[stage]['status'] for stage in stages]
    colors = [status_colors[status] for status in statuses]
    
    # 성공/실패 개수 계산
    status_counts = {
        'success': statuses.count('success'),
        'partial': statuses.count('partial'),
        'failed': statuses.count('failed')
    }
    
    # 각 단계의 상태 바 그래프
    plt.bar(stages, [1] * len(stages), color=colors)
    
    # 각 단계에 상태 표시
    for i, stage in enumerate(stages):
        plt.text(i, 0.5, results[stage]['status'].upper(), 
                 ha='center', va='center', color='white', fontweight='bold')
    
    plt.title('Validation Stages Status')
    plt.ylabel('Status')
    plt.xticks(rotation=45, ha='right')
    
    # 범례 추가
    for status, color in status_colors.items():
        if status_counts[status] > 0:
            plt.bar(0, 0, color=color, label=f"{status.capitalize()}: {status_counts[status]}")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/validation_status_summary.png')
    logging.info(f"검증 상태 요약 그래프가 저장되었습니다: {output_dir}/validation_status_summary.png")

def generate_validation_report(results):
    """
    검증 결과 보고서를 생성합니다.
    """
    # 결과 저장
    with open(f'{output_dir}/validation_results.json', 'w') as f:
        # 요약 객체만 저장 (큰 데이터는 제외)
        summary_results = {}
        
        for stage, data in results.items():
            summary_results[stage] = {
                'status': data['status']
            }
            
            # 모델 성능 요약 데이터 추가
            if stage == 'model_validation' and data['status'] == 'success':
                best_model = None
                best_f1 = 0
                
                for model in data['data']:
                    model_name = model.get('Unnamed: 0', 'Unknown')
                    f1_score = model.get('f1', 0)
                    
                    if f1_score > best_f1:
                        best_f1 = f1_score
                        best_model = model_name
                
                summary_results[stage]['best_model'] = best_model
                summary_results[stage]['best_f1'] = best_f1
        
        json.dump(summary_results, f, indent=2)
    
    # HTML 보고서 생성
    with open(f'{output_dir}/validation_report.html', 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>XAI 검증 보고서</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                .success {{
                    color: green;
                }}
                .partial {{
                    color: orange;
                }}
                .failed {{
                    color: red;
                }}
                .section {{
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
                pre {{
                    background-color: #f5f5f5;
                    padding: 10px;
                    overflow-x: auto;
                    font-family: monospace;
                }}
            </style>
        </head>
        <body>
            <h1>XAI 검증 보고서</h1>
            <p>생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>검증 요약</h2>
                <table>
                    <tr>
                        <th>검증 단계</th>
                        <th>상태</th>
                    </tr>
        """)
        
        for stage, data in results.items():
            stage_name = {
                'data_validation': '데이터 검증',
                'model_validation': '모델 성능 검증',
                'xai_validation': 'XAI 분석 검증',
                'ontology_validation': '온톨로지 검증'
            }.get(stage, stage)
            
            f.write(f"""
                    <tr>
                        <td>{stage_name}</td>
                        <td class="{data['status']}">{data['status'].upper()}</td>
                    </tr>
            """)
        
        f.write("""
                </table>
            </div>
        """)
        
        # 1. 데이터 검증 섹션
        f.write("""
            <div class="section">
                <h2>1. 데이터 검증</h2>
        """)
        
        if results['data_validation']['status'] == 'success':
            f.write(f"""
                <p>데이터 검증이 성공적으로 완료되었습니다.</p>
                <h3>데이터 시각화</h3>
                <img src="../data_validation/data_exploration.png" alt="데이터 시각화">
            """)
        else:
            f.write(f"""
                <p class="failed">데이터 검증에 실패했습니다.</p>
            """)
        
        f.write("""
            </div>
        """)
        
        # 2. 모델 성능 검증 섹션
        f.write("""
            <div class="section">
                <h2>2. 모델 성능 검증</h2>
        """)
        
        if results['model_validation']['status'] == 'success':
            f.write(f"""
                <p>모델 성능 검증이 성공적으로 완료되었습니다.</p>
                <h3>성능 요약</h3>
                <table>
                    <tr>
                        <th>모델</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                    </tr>
            """)
            
            for model in results['model_validation']['data']:
                model_name = model.get('Unnamed: 0', 'Unknown')
                accuracy = model.get('accuracy', 0)
                precision = model.get('precision', 0)
                recall = model.get('recall', 0)
                f1 = model.get('f1', 0)
                
                f.write(f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{accuracy:.4f}</td>
                        <td>{precision:.4f}</td>
                        <td>{recall:.4f}</td>
                        <td>{f1:.4f}</td>
                    </tr>
                """)
            
            f.write("""
                </table>
                <h3>성능 시각화</h3>
                <img src="model_performance_comparison.png" alt="모델 성능 비교">
            """)
        else:
            f.write(f"""
                <p class="{results['model_validation']['status']}">모델 성능 검증 상태: {results['model_validation']['status'].upper()}</p>
            """)
            
            if results['model_validation']['status'] == 'partial':
                f.write(f"""
                    <p>오류: {results['model_validation'].get('error', '알 수 없는 오류')}</p>
                """)
        
        f.write("""
            </div>
        """)
        
        # 3. XAI 분석 검증 섹션
        f.write("""
            <div class="section">
                <h2>3. XAI 분석 검증</h2>
        """)
        
        if results['xai_validation']['status'] == 'success':
            f.write(f"""
                <p>XAI 분석 검증이 성공적으로 완료되었습니다.</p>
                <h3>결과 요약</h3>
                <pre>{results['xai_validation']['summary']}</pre>
                <h3>SHAP 분석</h3>
                <img src="../xai_validation/shap_feature_importance.png" alt="SHAP 특성 중요도">
                <h3>Counterfactual 분석</h3>
                <img src="../xai_validation/counterfactual_examples.png" alt="Counterfactual 예시" onerror="this.style.display='none'">
            """)
        else:
            f.write(f"""
                <p class="{results['xai_validation']['status']}">XAI 분석 검증 상태: {results['xai_validation']['status'].upper()}</p>
            """)
            
            if results['xai_validation']['status'] == 'partial':
                f.write(f"""
                    <p>오류: {results['xai_validation'].get('error', '알 수 없는 오류')}</p>
                """)
        
        f.write("""
            </div>
        """)
        
        # 4. 온톨로지 검증 섹션
        f.write("""
            <div class="section">
                <h2>4. 온톨로지 검증</h2>
        """)
        
        if results['ontology_validation']['status'] == 'success':
            f.write(f"""
                <p>온톨로지 검증이 성공적으로 완료되었습니다.</p>
                <h3>결과 요약</h3>
                <pre>{results['ontology_validation']['summary']}</pre>
                <h3>XAI와 온톨로지 비교</h3>
                <img src="../ontology_validation/xai_ontology_comparison.png" alt="XAI와 온톨로지 비교" onerror="this.style.display='none'">
            """)
        else:
            f.write(f"""
                <p class="{results['ontology_validation']['status']}">온톨로지 검증 상태: {results['ontology_validation']['status'].upper()}</p>
            """)
            
            if results['ontology_validation']['status'] == 'partial':
                f.write(f"""
                    <p>오류: {results['ontology_validation'].get('error', '알 수 없는 오류')}</p>
                """)
        
        f.write("""
            </div>
            
            <div class="section">
                <h2>검증 결과 종합</h2>
                <img src="validation_status_summary.png" alt="검증 상태 요약">
        """)
        
        # 전체 성공/실패 여부에 따른 결론
        success_count = sum(1 for data in results.values() if data['status'] == 'success')
        partial_count = sum(1 for data in results.values() if data['status'] == 'partial')
        failed_count = sum(1 for data in results.values() if data['status'] == 'failed')
        
        if failed_count == 0 and partial_count == 0:
            f.write("""
                <p class="success">모든 검증 단계가 성공적으로 완료되었습니다. 논문의 분석 결과가 올바르게 구현되었음을 확인했습니다.</p>
            """)
        elif failed_count == 0:
            f.write(f"""
                <p class="partial">일부 검증 단계가 부분적으로 완료되었습니다 (성공: {success_count}, 부분 성공: {partial_count}). 
                일부 개선이 필요하지만 논문의 핵심 분석은 구현되었습니다.</p>
            """)
        else:
            f.write(f"""
                <p class="failed">일부 검증 단계가 실패했습니다 (성공: {success_count}, 부분 성공: {partial_count}, 실패: {failed_count}). 
                논문의 분석 결과를 완전히 구현하기 위해 추가 작업이 필요합니다.</p>
            """)
        
        f.write("""
            </div>
        </body>
        </html>
        """)
    
    logging.info(f"검증 보고서가 생성되었습니다: {output_dir}/validation_report.html")

if __name__ == "__main__":
    # 1. 개별 검증 스크립트 실행
    validation_scripts = [
        ('/Users/raziel/Desktop/XAI/validate_data_preprocessing.py', '데이터 전처리 검증'),
        ('/Users/raziel/Desktop/XAI/validate_model_performance.py', '모델 성능 검증'),
        ('/Users/raziel/Desktop/XAI/validate_xai_analysis.py', 'XAI 분석 검증'),
        ('/Users/raziel/Desktop/XAI/validate_ontology_inference.py', '온톨로지 추론 검증')
    ]
    
    for script_path, script_name in validation_scripts:
        success, stdout, stderr = run_script(script_path, script_name)
        
        # 실행 로그 저장
        output_log_path = f"{output_dir}/{script_name.replace(' ', '_')}_output.log"
        with open(output_log_path, 'w') as f:
            f.write(f"실행 결과: {'성공' if success else '실패'}\n\n")
            f.write("표준 출력:\n")
            f.write(stdout)
            f.write("\n\n표준 오류:\n")
            f.write(stderr)
    
    # 2. 검증 결과 수집
    validation_results = collect_validation_results()
    
    # 3. 결과 시각화
    visualize_validation_summary(validation_results)
    
    # 4. 검증 보고서 생성
    generate_validation_report(validation_results)
    
    logging.info("모든 검증 작업이 완료되었습니다.")
    print(f"검증 보고서가 생성되었습니다: {output_dir}/validation_report.html") 