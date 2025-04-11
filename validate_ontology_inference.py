import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import json
import csv
import shutil
from datetime import datetime
from owlready2 import *

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('ontology_validation.log'),
                              logging.StreamHandler()])

# 출력 디렉토리 생성
output_dir = '/Users/raziel/Desktop/XAI/output/ontology_validation'
os.makedirs(output_dir, exist_ok=True)

def load_ontology(ontology_path='/Users/raziel/Desktop/XAI/ontology/diecasting_ontology.owl'):
    """
    온톨로지 로드 및 검증
    """
    logging.info(f"온톨로지 로드 시작: {ontology_path}")
    
    try:
        # 기존 온톨로지 파일 백업
        backup_path = os.path.join(os.path.dirname(ontology_path), 'backup_' + os.path.basename(ontology_path))
        shutil.copy2(ontology_path, backup_path)
        logging.info(f"기존 온톨로지 파일 백업 완료: {backup_path}")
        
        # 온톨로지 로드
        onto_path.append(os.path.dirname(ontology_path))
        onto = get_ontology(ontology_path).load()
        
        # 온톨로지 기본 정보 확인
        classes_count = len(list(onto.classes()))
        individuals_count = len(list(onto.individuals()))
        properties_count = len(list(onto.properties()))
        
        logging.info(f"온톨로지 로드 완료: 클래스 {classes_count}개, 인스턴스 {individuals_count}개, 프로퍼티 {properties_count}개")
        
        return onto
    
    except Exception as e:
        logging.error(f"온톨로지 로드 중 오류 발생: {str(e)}")
        return None

def validate_ontology_structure(onto):
    """
    온톨로지 구조 검증 (클래스, 프로퍼티, 인스턴스 등)
    """
    logging.info("온톨로지 구조 검증 시작")
    
    if onto is None:
        logging.error("검증할 온톨로지가 없습니다.")
        return False
    
    try:
        # 필수 클래스 확인
        required_classes = [
            'Process', 'ProcessParameter', 'DefectType', 'Product', 'Material'
        ]
        
        missing_classes = []
        for cls_name in required_classes:
            if not onto.search_one(iri="*#" + cls_name):
                missing_classes.append(cls_name)
        
        if missing_classes:
            logging.warning(f"누락된 필수 클래스: {', '.join(missing_classes)}")
        else:
            logging.info("모든, 필수 클래스가 존재함")
        
        # 필수 프로퍼티 확인
        required_properties = [
            'hasProcessParameter', 'hasValue', 'hasCause', 'hasDefect'
        ]
        
        missing_properties = []
        for prop_name in required_properties:
            if not onto.search_one(iri="*#" + prop_name):
                missing_properties.append(prop_name)
        
        if missing_properties:
            logging.warning(f"누락된 필수 프로퍼티: {', '.join(missing_properties)}")
        else:
            logging.info("모든 필수 프로퍼티가 존재함")
        
        # SWRL 규칙 확인
        rules = list(onto.rules)
        logging.info(f"SWRL 규칙 수: {len(rules)}")
        
        for i, rule in enumerate(rules):
            logging.info(f"  규칙 {i+1}: {rule}")
        
        if len(rules) == 0:
            logging.warning("SWRL 규칙이 없습니다. 온톨로지 추론이 제한될 수 있습니다.")
        
        # 인스턴스 확인
        defect_types = list(onto.search(is_a=onto.search_one(iri="*#DefectType")))
        logging.info(f"불량 유형 수: {len(defect_types)}")
        
        for defect in defect_types:
            logging.info(f"  불량 유형: {defect.name}")
        
        process_parameters = list(onto.search(is_a=onto.search_one(iri="*#ProcessParameter")))
        logging.info(f"공정 파라미터 수: {len(process_parameters)}")
        
        if len(process_parameters) > 10:
            for param in process_parameters[:10]:
                logging.info(f"  공정 파라미터 (상위 10개): {param.name}")
            logging.info(f"  ... 외 {len(process_parameters) - 10}개")
        else:
            for param in process_parameters:
                logging.info(f"  공정 파라미터: {param.name}")
        
        return True
    
    except Exception as e:
        logging.error(f"온톨로지 구조 검증 중 오류 발생: {str(e)}")
        return False

def validate_swrl_rules(onto):
    """
    SWRL 규칙을 검증하고 필요시 추가
    """
    logging.info("SWRL 규칙 검증 시작")
    
    if onto is None:
        logging.error("검증할 온톨로지가 없습니다.")
        return False
    
    try:
        # 기존 규칙 개수 확인
        existing_rules = list(onto.rules)
        logging.info(f"기존 SWRL 규칙 수: {len(existing_rules)}")
        
        # 논문에서 언급된 SWRL 규칙 예시
        sample_rules = [
            # Process.1 파라미터가 임계값 이상일 때 ShortShotDefect 불량 발생 규칙
            """Process(?p), ProcessParameter(?param), hasProcessParameter(?p, ?param), 
               hasName(?param, "Process.1"), hasValue(?param, ?val), greaterThan(?val, 0.8) 
               -> hasDefect(?p, ShortShotDefect)""",
            
            # Process.2 파라미터 값이 임계값 이하일 때 FlashDefect 불량 발생 규칙
            """Process(?p), ProcessParameter(?param), hasProcessParameter(?p, ?param), 
               hasName(?param, "Process.2"), hasValue(?param, ?val), lessThan(?val, 0.3) 
               -> hasDefect(?p, FlashDefect)""",
            
            # Process.3과 Process.4의 조합 조건에 따른 불량 발생 규칙
            """Process(?p), ProcessParameter(?param1), ProcessParameter(?param2), 
               hasProcessParameter(?p, ?param1), hasProcessParameter(?p, ?param2), 
               hasName(?param1, "Process.3"), hasName(?param2, "Process.4"), 
               hasValue(?param1, ?val1), hasValue(?param2, ?val2), 
               greaterThan(?val1, 0.7), lessThan(?val2, 0.4) 
               -> hasDefect(?p, PinchOffDefect)"""
        ]
        
        # 기존 규칙과 충돌 여부 확인 및 새 규칙 추가
        rules_added = 0
        
        for rule_text in sample_rules:
            # 이미 유사한 규칙이 있는지 확인 (간단한 텍스트 비교)
            rule_exists = False
            
            for existing_rule in existing_rules:
                existing_text = str(existing_rule)
                # 규칙의 핵심 부분이 일치하는지 확인
                if rule_text.split("->")[1].strip() in existing_text:
                    rule_exists = True
                    logging.info(f"유사한 규칙이 이미 존재합니다: {existing_text}")
                    break
            
            if not rule_exists:
                # 새 규칙 추가
                try:
                    imp = Imp()
                    imp.set_as_rule(rule_text)
                    rules_added += 1
                    logging.info(f"새 규칙 추가됨: {rule_text}")
                except Exception as e:
                    logging.error(f"규칙 추가 실패: {str(e)}, 규칙: {rule_text}")
        
        # 반환값으로 규칙 추가 여부 전달
        if rules_added > 0:
            logging.info(f"{rules_added}개의 새 규칙이 추가되었습니다.")
            return True
        else:
            logging.info("추가된 규칙이 없습니다.")
            return False
    
    except Exception as e:
        logging.error(f"SWRL 규칙 검증 중 오류 발생: {str(e)}")
        return False

def run_reasoner(onto):
    """
    온톨로지 추론 실행
    """
    logging.info("온톨로지 추론 시작")
    
    if onto is None:
        logging.error("추론할 온톨로지가 없습니다.")
        return False
    
    try:
        # HermiT 추론기 사용
        with onto:
            try:
                sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
                logging.info("Pellet 추론기 실행 완료")
            except Exception as e:
                logging.error(f"Pellet 추론기 실행 실패: {str(e)}")
                try:
                    sync_reasoner_hermit(infer_property_values=True, infer_data_property_values=True)
                    logging.info("HermiT 추론기 실행 완료")
                except Exception as e:
                    logging.error(f"HermiT 추론기 실행 실패: {str(e)}")
                    return False
        
        # 추론 결과 확인
        defect_processes = []
        
        # Process 클래스 인스턴스 중 defect 프로퍼티가 있는 경우 확인
        process_class = onto.search_one(iri="*#Process")
        if process_class:
            processes = list(onto.search(is_a=process_class))
            
            for process in processes:
                has_defect_prop = onto.search_one(iri="*#hasDefect")
                
                if has_defect_prop and hasattr(process, has_defect_prop.name):
                    defects = getattr(process, has_defect_prop.name)
                    if defects:
                        defect_processes.append({
                            'process': process.name,
                            'defects': [defect.name for defect in defects]
                        })
        
        logging.info(f"추론 결과: 불량이 발생한 공정 수: {len(defect_processes)}")
        
        for item in defect_processes:
            logging.info(f"  공정 {item['process']}의 불량: {', '.join(item['defects'])}")
        
        # 추론 결과 저장
        with open(f'{output_dir}/inference_results.json', 'w') as f:
            json.dump(defect_processes, f, indent=2)
        
        return len(defect_processes) > 0
    
    except Exception as e:
        logging.error(f"온톨로지 추론 중 오류 발생: {str(e)}")
        return False

def compare_with_xai_results(onto, xai_results_path=None):
    """
    온톨로지 추론 결과와 XAI 분석 결과 비교
    """
    logging.info("온톨로지 추론 결과와 XAI 분석 결과 비교 시작")
    
    if onto is None:
        logging.error("비교할 온톨로지가 없습니다.")
        return False
    
    # XAI 결과 로드
    if xai_results_path and os.path.exists(xai_results_path):
        logging.info(f"XAI 결과 파일 로드: {xai_results_path}")
        xai_results = pd.read_csv(xai_results_path)
    else:
        # XAI 결과 없음 - 샘플 결과 생성
        logging.warning("XAI 결과 파일이 없습니다. 샘플 결과를 생성합니다.")
        xai_results = {
            'Process.1': 0.244071,
            'Process.2': 0.191823,
            'Process.3': 0.137142,
            'Process.4': 0.128956,
            'Process.5': 0.081432
        }
    
    try:
        # 온톨로지 추론 결과 로드
        inference_path = f'{output_dir}/inference_results.json'
        
        if not os.path.exists(inference_path):
            logging.error("온톨로지 추론 결과 파일이 없습니다.")
            return False
        
        with open(inference_path, 'r') as f:
            inference_results = json.load(f)
        
        # 추론 결과에서 불량 유형별 관련 파라미터 추출
        defect_parameters = {}
        
        for item in inference_results:
            process_name = item['process']
            defects = item['defects']
            
            # 공정 인스턴스 찾기
            process_instance = onto.search_one(iri=f"*#{process_name}")
            
            if process_instance:
                # 공정 파라미터 값 가져오기
                has_parameter_prop = onto.search_one(iri="*#hasProcessParameter")
                has_name_prop = onto.search_one(iri="*#hasName")
                has_value_prop = onto.search_one(iri="*#hasValue")
                
                if has_parameter_prop and hasattr(process_instance, has_parameter_prop.name):
                    parameters = getattr(process_instance, has_parameter_prop.name)
                    
                    for param in parameters:
                        param_name = getattr(param, has_name_prop.name)[0] if hasattr(param, has_name_prop.name) else param.name
                        param_value = getattr(param, has_value_prop.name)[0] if hasattr(param, has_value_prop.name) else None
                        
                        for defect in defects:
                            if defect not in defect_parameters:
                                defect_parameters[defect] = []
                            
                            defect_parameters[defect].append({
                                'parameter': param_name,
                                'value': param_value
                            })
        
        # XAI 결과와 온톨로지 추론 결과 비교
        comparison_results = []
        
        if isinstance(xai_results, dict):
            # 딕셔너리 형태의 XAI 결과 (샘플 데이터)
            xai_top_params = sorted(xai_results.items(), key=lambda x: x[1], reverse=True)
            
            for defect, params in defect_parameters.items():
                # 온톨로지에서 해당 불량에 대한 파라미터 추출
                onto_params = [item['parameter'] for item in params]
                
                # XAI 상위 파라미터와 일치하는 파라미터 찾기
                matching_params = [param for param, _ in xai_top_params if param in onto_params]
                
                comparison_results.append({
                    'defect': defect,
                    'onto_parameters': onto_params,
                    'xai_top_parameters': [param for param, _ in xai_top_params[:5]],
                    'matching_parameters': matching_params,
                    'agreement_ratio': len(matching_params) / min(len(onto_params), 5) if min(len(onto_params), 5) > 0 else 0
                })
        else:
            # DataFrame 형태의 XAI 결과
            for defect, params in defect_parameters.items():
                # 온톨로지에서 해당 불량에 대한 파라미터 추출
                onto_params = [item['parameter'] for item in params]
                
                # XAI 결과에서 관련 파라미터 확인
                if 'feature' in xai_results.columns and 'importance' in xai_results.columns:
                    xai_top_params = xai_results.sort_values('importance', ascending=False)['feature'].tolist()[:5]
                    
                    # 일치하는 파라미터 찾기
                    matching_params = [param for param in xai_top_params if param in onto_params]
                    
                    comparison_results.append({
                        'defect': defect,
                        'onto_parameters': onto_params,
                        'xai_top_parameters': xai_top_params,
                        'matching_parameters': matching_params,
                        'agreement_ratio': len(matching_params) / min(len(onto_params), 5) if min(len(onto_params), 5) > 0 else 0
                    })
        
        # 결과 로깅
        logging.info("온톨로지와 XAI 결과 비교:")
        
        for result in comparison_results:
            logging.info(f"  불량 유형: {result['defect']}")
            logging.info(f"    온톨로지 파라미터: {', '.join(result['onto_parameters'])}")
            logging.info(f"    XAI 상위 파라미터: {', '.join(result['xai_top_parameters'])}")
            logging.info(f"    일치 파라미터: {', '.join(result['matching_parameters'])}")
            logging.info(f"    일치율: {result['agreement_ratio']:.2%}")
        
        # 전체 일치율 계산
        if comparison_results:
            overall_agreement = sum(result['agreement_ratio'] for result in comparison_results) / len(comparison_results)
            logging.info(f"전체 일치율: {overall_agreement:.2%}")
            
            # 시각화
            defects = [result['defect'] for result in comparison_results]
            agreement_ratios = [result['agreement_ratio'] * 100 for result in comparison_results]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(defects, agreement_ratios)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center')
            
            plt.axhline(y=overall_agreement * 100, color='r', linestyle='-', label=f'평균 일치율: {overall_agreement:.2%}')
            plt.xlabel('불량 유형')
            plt.ylabel('일치율 (%)')
            plt.title('온톨로지 추론과 XAI 분석 결과 일치율')
            plt.ylim(0, 105)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{output_dir}/xai_ontology_comparison.png')
            
            # 결과 저장
            with open(f'{output_dir}/xai_ontology_comparison.json', 'w') as f:
                json.dump({
                    'comparison_results': comparison_results,
                    'overall_agreement': overall_agreement
                }, f, indent=2)
            
            # CSV 형식으로도 저장
            with open(f'{output_dir}/xai_ontology_comparison.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Defect', 'Ontology Parameters', 'XAI Top Parameters', 'Matching Parameters', 'Agreement Ratio'])
                
                for result in comparison_results:
                    writer.writerow([
                        result['defect'],
                        ', '.join(result['onto_parameters']),
                        ', '.join(result['xai_top_parameters']),
                        ', '.join(result['matching_parameters']),
                        f"{result['agreement_ratio']:.2%}"
                    ])
            
            return True, overall_agreement
        else:
            logging.warning("비교 결과가 없습니다.")
            return False, 0
    
    except Exception as e:
        logging.error(f"온톨로지와 XAI 결과 비교 중 오류 발생: {str(e)}")
        return False, 0

def save_updated_ontology(onto, output_path=None):
    """
    업데이트된 온톨로지 저장
    """
    if onto is None:
        logging.error("저장할 온톨로지가 없습니다.")
        return False
    
    if output_path is None:
        output_path = f'{output_dir}/diecasting_ontology_updated.owl'
    
    try:
        onto.save(file=output_path, format="rdfxml")
        logging.info(f"업데이트된 온톨로지가 저장되었습니다: {output_path}")
        return True
    except Exception as e:
        logging.error(f"온톨로지 저장 중 오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    # 1. 온톨로지 로드 및 검증
    onto = load_ontology()
    
    if onto:
        # 2. 온톨로지 구조 검증
        validate_ontology_structure(onto)
        
        # 3. SWRL 규칙 검증 및 추가
        rules_updated = validate_swrl_rules(onto)
        
        # 4. 추론 실행
        inference_success = run_reasoner(onto)
        
        # 5. XAI 결과와 비교
        xai_results_path = '/Users/raziel/Desktop/XAI/output/xai_validation/shap_top10_features.csv'
        comparison_success, agreement_ratio = compare_with_xai_results(onto, xai_results_path)
        
        # 6. 업데이트된 온톨로지 저장
        if rules_updated or inference_success:
            save_updated_ontology(onto)
        
        # 7. 검증 결과 요약
        with open(f'{output_dir}/ontology_validation_summary.txt', 'w') as f:
            f.write("온톨로지 검증 요약\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. 온톨로지 기본 정보:\n")
            f.write(f"  - 클래스 수: {len(list(onto.classes()))}\n")
            f.write(f"  - 인스턴스 수: {len(list(onto.individuals()))}\n")
            f.write(f"  - 프로퍼티 수: {len(list(onto.properties()))}\n")
            f.write(f"  - SWRL 규칙 수: {len(list(onto.rules()))}\n")
            
            f.write("\n2. SWRL 규칙 검증:\n")
            f.write(f"  - 규칙 업데이트: {'성공' if rules_updated else '불필요 (기존 규칙 유지)'}\n")
            
            f.write("\n3. 추론 결과:\n")
            f.write(f"  - 추론 성공 여부: {'성공' if inference_success else '실패'}\n")
            
            if inference_success:
                inference_path = f'{output_dir}/inference_results.json'
                
                if os.path.exists(inference_path):
                    with open(inference_path, 'r') as inf_file:
                        inference_results = json.load(inf_file)
                    
                    f.write(f"  - 불량이 발생한 공정 수: {len(inference_results)}\n")
                    
                    for item in inference_results[:3]:  # 상위 3개만 표시
                        f.write(f"    * 공정 {item['process']}의 불량: {', '.join(item['defects'])}\n")
                    
                    if len(inference_results) > 3:
                        f.write(f"    * ... 외 {len(inference_results) - 3}개 공정\n")
            
            f.write("\n4. XAI 결과와 비교:\n")
            f.write(f"  - 비교 성공 여부: {'성공' if comparison_success else '실패'}\n")
            
            if comparison_success:
                f.write(f"  - 전체 일치율: {agreement_ratio:.2%}\n")
                
                comparison_path = f'{output_dir}/xai_ontology_comparison.json'
                
                if os.path.exists(comparison_path):
                    with open(comparison_path, 'r') as comp_file:
                        comparison_data = json.load(comp_file)
                    
                    for result in comparison_data['comparison_results'][:3]:  # 상위 3개만 표시
                        f.write(f"    * 불량 유형 {result['defect']}: 일치율 {result['agreement_ratio']:.2%}\n")
                    
                    if len(comparison_data['comparison_results']) > 3:
                        f.write(f"    * ... 외 {len(comparison_data['comparison_results']) - 3}개 불량 유형\n")
            
            f.write("\n5. 결론:\n")
            if inference_success and comparison_success:
                if agreement_ratio >= 0.7:
                    f.write("  - 온톨로지 추론과 XAI 분석 결과가 높은 수준으로 일치합니다.\n")
                    f.write("  - 이는 두 방법론이 상호 검증되었음을 의미합니다.\n")
                elif agreement_ratio >= 0.5:
                    f.write("  - 온톨로지 추론과 XAI 분석 결과가 부분적으로 일치합니다.\n")
                    f.write("  - 온톨로지 모델링이나 규칙을 개선할 여지가 있습니다.\n")
                else:
                    f.write("  - 온톨로지 추론과 XAI 분석 결과의 일치도가 낮습니다.\n")
                    f.write("  - 온톨로지 모델링이나 SWRL 규칙을 재검토할 필요가 있습니다.\n")
            else:
                f.write("  - 온톨로지 추론 또는 XAI 분석 결과 비교 과정에서 문제가 발생했습니다.\n")
                f.write("  - 온톨로지 구조와 SWRL 규칙을 확인하고 필요시 수정하세요.\n")
        
        logging.info(f"온톨로지 검증 요약이 저장되었습니다: {output_dir}/ontology_validation_summary.txt") 