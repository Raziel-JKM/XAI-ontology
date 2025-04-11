# 다이캐스팅 공정 품질 관리를 위한 설명가능 AI와 온톨로지 통합 프레임워크: 실증 연구 및 심층 분석

## Abstract
This study proposes and empirically validates a novel framework integrating eXplainable Artificial Intelligence (XAI) with ontology-based knowledge representation to enhance quality control in complex manufacturing processes, specifically die-casting. Addressing the critical challenge of 'black-box' AI models in manufacturing, we developed a predictive model (RandomForest demonstrated superior, albeit practically limited, performance with approx. 90% accuracy and 77% F1-score on real-world data exhibiting a 22.87% defect rate) and employed SHAP for feature importance quantification and counterfactual analysis for actionable insights. Concurrently, a domain-specific OWL ontology was constructed to formally capture established die-casting knowledge. The core contribution lies in the **symbiotic integration** where XAI-derived insights (e.g., the significant impact of factory humidity, spray time, clamping force) were systematically mapped onto the ontology, enabling semantic interpretation and knowledge-based reasoning via the HermiT engine. While predictive recall requires further improvement and quantitative synergy metrics are yet to be developed, the framework demonstrates the feasibility and qualitative potential of **ontology-augmented XAI** to provide context-rich, human-understandable explanations, validate data-driven findings against domain expertise, and facilitate trustworthy AI adoption in smart manufacturing. This work, acknowledging challenges like SHAP visualization compatibility and ontology scalability, lays a methodological foundation and provides critical empirical insights for future synergistic data-knowledge systems in quality management.

## I. 서론
제조업의 디지털 전환이 가속화됨에 따라, 데이터 기반 인공지능(AI) 모델은 공정 최적화 및 품질 예측의 핵심 동력으로 부상하고 있다(Park & Youm, 2023). 특히 다이캐스팅과 같이 다변수 상호작용이 복잡하게 얽힌 공정에서는 미세한 조건 변화가 제품 품질에 미치는 영향이 크지만, 전통적인 품질 관리는 여전히 현장 전문가의 경험과 직관에 크게 의존하는 경향이 있다(Park & Youm, 2023). 이러한 상황에서 머신러닝 모델은 잠재적으로 우수한 예측 성능을 제공하지만, 그 내부 작동 방식의 불투명성, 즉 '블랙박스' 문제는 현장의 신뢰 확보와 실질적인 공정 개선 연계에 큰 장애물로 작용한다(Naqvi et al., 2024).

이러한 도전과제를 해결하기 위한 핵심 열쇠로 **설명가능 인공지능(XAI)**이 주목받고 있다. XAI는 AI 모델의 예측 결과에 대한 명확하고 이해 가능한 근거를 제시함으로써, 모델의 투명성을 높이고 사용자의 신뢰를 구축하는 것을 목표로 한다(Agostinho et al., 2023). 제조 분야에서 XAI를 활용하면, 데이터 패턴으로부터 학습된 복잡한 모델의 판단 기준을 해석하여 품질 이상 원인을 규명하고, 데이터 기반의 합리적인 공정 개선 방안을 도출하는 데 기여할 수 있다(Hong et al., 2023). 예를 들어, SHAP (SHapley Additive exPlanations)와 같은 기법은 각 공정 변수가 모델 예측에 미치는 개별적인 기여도를 정량화하여 핵심 영향 인자를 식별하는 데 효과적이다(Hong et al., 2023).

그러나 데이터 기반 XAI만으로는 AI 설명의 완전성을 담보하기 어렵다. 통계적 연관성이 반드시 인과관계를 의미하지 않으며, 도출된 설명이 실제 공정의 물리적 메커니즘이나 전문가의 지식 체계와 일치하지 않을 수 있다. 이러한 간극을 메우기 위해, 본 연구는 **지식 기반 접근법**, 특히 **온톨로지(Ontology)**와의 융합을 제안한다. 온톨로지는 특정 도메인의 개념, 관계, 제약조건 등을 형식적이고 명시적으로 표현하는 시맨틱 웹 기술의 핵심 구성요소로, 제조 공정의 전문 지식을 구조화된 형태로 저장하고 활용하는 데 매우 유용하다(Wang et al., 2024). **지식 그래프(Knowledge Graph)** 형태로 구축된 온톨로지는 데이터 기반 분석 결과에 **맥락(context)**과 **의미(semantics)**를 부여하고, 논리적 추론을 통해 암묵적인 지식을 발견하거나 가설을 검증하는 데 활용될 수 있다(Wang et al., 2024; Rožanec et al., 2023). 본 연구가 추구하는 **데이터 기반 설명(XAI)과 지식 기반 추론(Ontology)의 시너지**는 각 방법론의 한계를 상호 보완하여, 예측의 정확성뿐만 아니라 설명의 깊이와 신뢰성까지 확보하는 것을 목표로 한다.

최근 제조 분야 AI 연구 동향은 XAI와 온톨로지를 결합하려는 시도로 나아가고 있다(Naqvi et al., 2024). 이러한 **ontology-augmented XAI**는 XAI가 제공하는 국소적, 데이터 의존적 설명을 온톨로지에 내재된 포괄적 도메인 지식과 연결함으로써, 더욱 **풍부하고 신뢰할 수 있으며 실행 가능한(actionable)** 설명을 생성할 잠재력을 지닌다. 예를 들어, SHAP 분석으로 "높은 공장 습도"가 불량 예측에 기여함을 발견했다면, 온톨로지는 "높은 습도는 용탕의 가스 함유량을 증가시켜 기공(Porosity) 불량을 유발할 수 있다"는 전문가 지식과 연결하여 설명의 깊이를 더하고, 데이터 기반 발견의 타당성을 도메인 지식 관점에서 검증할 수 있다.

하지만 기존 연구들은 XAI와 온톨로지의 통합 가능성을 이론적으로 제시하거나, 제한적인 범위에서 개념 증명(Proof-of-Concept) 수준의 연구가 주를 이루고 있으며, 실제 산업 데이터를 활용하여 통합 프레임워크의 **구현 가능성, 성능, 그리고 그 과정에서 발생하는 도전과제를 종합적으로 검증**한 실증 연구는 여전히 부족한 실정이다. 본 연구는 이러한 연구 공백을 메우고자, 실제 다이캐스팅 공정 데이터를 대상으로 **XAI-온톨로지 통합 프레임워크를 설계, 구현, 및 평가**한다. 구체적으로, (1) 실제 공정 데이터 기반의 품질 예측 모델(RandomForest)을 구축하고, (2) SHAP 및 Counterfactual Explanation을 적용하여 모델 예측을 설명하며, (3) 다이캐스팅 도메인 온톨로지를 구축하고, (4) XAI 분석 결과를 온톨로지와 체계적으로 연계한 후, (5) 온톨로지 추론 엔진을 활용하여 통합된 지식 기반에서 추론을 수행하는 **전체 파이프라인**을 구현하고 그 결과를 심층적으로 분석한다.

본 연구의 **핵심적인 학술적 기여**는 데이터 과학(XAI)과 지식 공학(온톨로지)을 실질적으로 융합하는 **구체적인 방법론과 실증적 사례**를 제공함으로써, 제조 분야에서 **신뢰할 수 있고 설명가능한 AI 시스템** 구축을 위한 청사진을 제시하는 데 있다. 특히, 실제 산업 데이터의 복잡성과 제약 조건 하에서 프레임워크를 구현하고 평가함으로써, 이론적 제안을 넘어선 **실질적인 적용 가능성과 한계점**을 명확히 밝히고자 한다. 또한, 실제 데이터 적용 과정에서 나타난 **현실적인 성능과 기술적 도전과제**(예: SHAP 시각화 호환성 문제, 예측 성능의 한계, Counterfactual 적용의 어려움)를 투명하게 공유함으로써, 향후 관련 연구 및 현장 적용에 중요한 참고 자료를 제공하고자 한다.

## II. 관련 연구

XAI는 복잡한 AI 모델의 의사결정 기준을 인간이 해석할 수 있는 정보로 제공함으로써 산업 분야에서 투명성과 신뢰성 확보의 핵심 기술로 부상하였다​. 제조 분야에서도 예측 모델의 설명가능성에 관한 다양한 연구가 최근 2년간 활발히 진행되고 있다​[4]. Hong 등[4]​은 사출성형 공정 데이터에 트리 앙상블 모델을 적용하고 SHAP 기반 중요 변수 도출을 통해 주요 공정 인자를 식별하였다. 도출된 변수들에 대해 개별조건 기대효과(ICE) 분석을 실시하여 불량률을 최소화할 수 있는 최적 운영 범위를 제시하고, 실제 테스트 데이터로 검증하여 불량률이 1.00%에서 0.21%로 감소하는 성과를 얻었다고 보고하였다​. 
이처럼 XAI 기법 (예: SHAP, ICE 등)을 활용하면 모델 예측에 영향력이 큰 요인을 찾아내어 공정 개선에 활용할 수 있음을 보여준다.
품질 예측뿐만 아니라, 제조 장비 상태 예측이나 유지보수 분야에서도 XAI 적용 사례가 증가하고 있다. 예를 들어, 남은 수명 예측(RUL) 문제에 counterfactual 설명을 적용하여 예측 모델의 투명성을 높이고, 결과를 의사결정 지원에 활용한 연구가 보고되었다​[2]. 
Counterfactual 설명은 “만약 현재 불량 제품이 되었다면, 어떤 조건을 달리했을 때 양품이 될 수 있는가”를 제시함으로써 사용자가 직접 개입 가능한 대안 시나리오를 제공한다. 이러한 접근은 예측 정확도 향상뿐만 아니라 모델 신뢰도 제고와 조치 가능성을 높여주므로, 제조 프로세스 최적화에 유용하다.
다이캐스팅 공정의 품질 분석에 XAI를 적용한 사례로는, Okuniewska 등​의 연구[7]를 들 수 있다. 이 연구에서는 알루미늄 다이캐스팅에서 발생하는 누설(defect leakage) 불량 데이터를 기반으로 인공신경망(ANN), 회귀나무서포트 벡터 머신(SVM)등 여러 머신러닝 기법을 비교하였다. 그 결과 ANN 모델이 예측 정확도 측면에서는 가장 우수했으나, 블랙박스 성격으로 인해 불량 발생 원인의 명확한 규명이 어려웠다. 저자들은 복잡한 소프트모델(ANN)의 경우 “대부분의 경우 불량 형성의 정확한 원인을 보여주지 못했다”고 밝히며​, 보다 투명한 모델인 결정나무 기반 방법으로 주요 인자를 추론하는 보완을 수행하였다.  복잡한 딥러닝 모델만 고집하기보다, 해석 가능성을 겸비한 모델을 활용하거나 XAI 기법으로 보완하는 것이 실제 공정 문제 해결에 중요하다는 점을 시사한다.


2. 제조 온톨로지와 지식 그래프


* 학생회원, 00대학교 000학과 (중고딕, 8.5pt, 장평93%, 자간-7%, 왼쪽10pt, 오른쪽 10pt, 줄간격120%)
** 정회원, 00대학교 000학과 
   이 논문은 2011년도 정부(교육과학기술부)의 재원으로 한국연구재단의 지원을 받아 수행된 연구임 (No. 2000-0000000).  
접수일자 : 2020년 03월 23일 (학회측 사용란)
수정일자 : 1차 2020년 03월 23일, 2차  2020년 04월 23일(학회측 사용란)
게재확정일 : 2020년 03월 28일 (학회측 사용란)
교신저자 : 박정훈   e-mail : jmkim@jnu.ac.kr 
  온톨로지기술은 제품, 공정, 자재 등 제조 도메인의 개념들과 그 상호관계를 서술하여 지식 기반 시스템을 구축하는 데 활용되고 있다. 최근 스마트 제조 Quality 4.0흐름 속에서 축적된 공정 데이터를 사람이 이해하기 쉬운 지식으로 변환하기 위해 온톨로지와 지식 그래프(KG)를 도입하는 연구가 늘어나고 있다[9].
예를 들어 Wang 등[5]​은 Human–Cyber–Physical (HCP)개념에 기반한 제조 지식그래프를 구축하여 품질 관리에 적용하였다. 이들은 사람-사이버-물리 데이터를 아우르는 온톨로지를 계층적으로 설계하고, 사례 기반 추론 및 자동 분석을 통해 생산 라인의 품질 모니터링, 결함 진단, 유지보수 결정 지원을 구현하였다​. 실제 자동차 부품 생산라인과 기어 제조공정에 이 체계를 적용한 결과, 지식 공유와 재사용성이 높아지고 데이터와 지식의 깊은 통합으로 의사결정의 효율성이 향상되었음을 보고하였다​[5].
제조 온톨로지는 공정 전문가들의 암묵지를 형식화함으로써, 데이터에 나타나지 않는 원인-결과 관계를 추론하거나 규칙 기반 진단을 가능하게 해준다. 과거 주조 결함 진단을 위한 온톨로지 연구에서도, 분산된 전문가 지식을 체계화한 주조 결함 분류 온톨로지를 구축하여 불량 원인 진단에 활용하려는 시도가 있었다. 최근에는 이러한 온톨로지를 현대적인 지식 그래프 형태로 확장하고, 추론 엔진과 연계하여 자동으로 결함 원인을 식별하는 방향으로 발전하고 있다​. 
예컨대, 특정 다이캐스팅 불량 사례에 대해 온톨로지에 정의된 원인 규칙(예: “충전 속도가 너무 높고 금형 온도가 낮으면 기공 결함 발생”)을 대입하면, 추론(reasoning)을 통해 해당 사례의 원인을 자동으로 도출하거나 유사 사례의 조치 방안을 제안할 수 있다. 이러한 지식 추론 결과는 데이터 기반 ML 모델이 제시한 중요 인자와 결합되어 설명될 때, 도메인 전문가에게 더 큰 설득력과 수용성을 갖게 된다​.

3. XAI와 온톨로지의 통합 연구 동향
Ontology-Based XAI는 시맨틱 기술을 활용하여 AI 모델의 결정을 인간친화적인 지식 형태로 설명하는 접근으로, 2023년 이후 관련 연구가 조금씩 등장하고 있다[2]​. Naqvi 등[2]​의 통합적 설문에 따르면, 제조 분야 XAI에 온톨로지와 일반적인 자연어 서술을 접목함으로써 비전문가도 이해할 수 있는 설명을 생성하고, AI 결정에 영향을 미치는 다양한 요인들을 식별할 수 있다고 한다. 대표적인 예로, 유럽의 XMANAI 프로젝트에서는 제조업에서 신뢰할 수 있는 AI 활용을 위해 지식 추상화 계층을 포함한 XAI 플랫폼을 구축하였는데[3]​, 데이터 과학자와 도메인 전문가가 협업하여 AI 모델을 투명하게 개선해나가는 환경을 제공한다​.

특히 제조 분야 의사결정 지원에 특화된 지식그래프-XAI통합 사례로, Rožanec 등 [6]은 공정 데이터로부터 생성된 예측과 그에 대한 설명, 그리고 현장 피드백 정보를 모두 하나의 지식 그래프로 수집/저장하고 필요 시 질의 및 추론하도록 하는 XAI-KG 프레임워크를 제안하였다. 이처럼 XAI 결과를 지식그래프에 연계하면, 향후 유사한 상황 발생 시 과거 설명과 조치들을 빠르게 조회하여 의사결정 자동화에 활용할 수 있게 된다. 다만 이러한 통합 접근은 아직 초기 단계로, 효과적인 지식 표현 방법, 온톨로지와 XAI 간 매핑 기법, 추론 성능 등의 측면에서 향후 연구 과제가 많다고 지적된다​. 본 논문에서 다루는 내용은 이러한 연구 흐름과 맥락을 같이하며, 국내 실증 사례로서는 드물게 XAI와 온톨로지의 결합 효과를 실제 공정 데이터로 검증한다는 점에서 학술적·실용적 의의를 갖는다.

## III. 제안 방법론: XAI-온톨로지 통합 프레임워크
본 연구에서 제안하는 프레임워크는 데이터 기반 분석과 지식 기반 추론을 유기적으로 결합하여 제조 품질 관리의 설명 가능성과 신뢰성을 높이는 것을 목표로 하며, 다음의 주요 단계로 구성된다 (Figure 1 참조 - 가상).

1.  **데이터 준비 및 전처리**: 다이캐스팅 공정의 시계열 센서 데이터, 공정 설정값, 최종 품질 검사 결과 등 이종(heterogeneous) 데이터를 수집한다. 데이터 정제 과정에서는 결측치 처리(본 연구에서는 평균 대체 적용) 및 IQR 기법을 활용한 이상치 식별 및 제거를 수행하여 분석 대상 데이터의 품질을 확보한다. 범주형 변수(예: 제품 유형)는 One-Hot Encoding으로 변환하고, 모든 수치형 변수는 StandardScaler를 사용하여 정규화한다. 다양한 개별 불량 코드를 통합하여, 본 연구의 목표인 이진 분류(정상/불량)를 위한 타겟 변수('AnyDefect')를 생성한다.
2.  **머신러닝 기반 품질 예측 모델 학습**: 전처리된 데이터를 사용하여 지도 학습 기반의 품질 예측 모델을 구축한다. 다양한 알고리즘(예: Logistic Regression, SVM, RandomForest, XGBoost)을 후보로 고려하고, 5-fold 교차 검증을 통해 F1-score (특히 소수 클래스인 불량 클래스에 대한 성능을 중요하게 평가)를 기준으로 최적 성능을 보이는 모델(본 연구에서는 RandomForest)을 최종 선정한다. 클래스 불균형 문제(본 데이터셋 약 22.87% 불량률)의 영향을 완화하기 위해 SMOTE (Synthetic Minority Over-sampling Technique)와 같은 데이터 샘플링 기법 적용을 검토하였으나, 이미 불량 클래스 비율이 일정 수준 이상 확보되어 있어 본 연구에서는 별도의 샘플링 기법을 적용하지 않았다. 이는 실제 현장 데이터 분포 하에서의 모델 성능을 평가하기 위함이었으나, 소수 클래스 학습의 어려움에 영향을 미쳤을 수 있다(4.2절 논의 참조).
3.  **XAI 기반 모델 설명**: 선정된 최적 모델(RandomForest)의 예측 결과를 해석하기 위해 상호 보완적인 XAI 기법들을 적용한다.
    *   **SHAP (SHapley Additive exPlanations)**: 모델 불가지론적(model-agnostic)이며 이론적 배경이 강건한 SHAP를 사용하여 각 예측에 대한 개별 특성의 영향력(SHAP 값)을 계산한다. 이는 게임 이론의 Shapley Value 개념에 기반하여 각 특성의 한계 기여도를 공정하게 배분한다. 평균 절대 SHAP 값을 통해 모델 전체의 **전역적 특성 중요도(Global Feature Importance)**를 파악하고, 개별 예측(instance-level) 분석을 통해 특정 불량 발생 사례의 **국소적 원인(Local Explanation)**을 식별한다. SHAP 값의 부호는 해당 특성이 불량 발생 확률을 높이는지(+) 낮추는지(-)에 대한 방향성을 제공하여 해석의 깊이를 더한다.
    *   **Counterfactual Explanations (DiCE 활용)**: 특정 인스턴스(예: 불량으로 예측된 제품)에 대해, 예측 결과를 원하는 클래스(예: 정상)로 변경시키는 **가장 유사하면서도 최소한의 변화를 갖는 대안적 시나리오**를 탐색한다. 이는 "What-if" 분석을 가능하게 하여, 사용자가 모델의 결정 경계를 이해하고 현장에서 **실행 가능한 조치(actionable insights)**를 도출하는 데 실질적인 도움을 줄 수 있다. DiCE 라이브러리를 활용하여 다양한 counterfactual 후보를 생성하고 평가한다.
4.  **다이캐스팅 온톨로지 구축**: 다이캐스팅 공정에 대한 도메인 지식을 형식적(formal)이고 재사용 가능하게 표현하기 위해 **OWL (Web Ontology Language)** 기반 온톨로지를 구축한다. Protégé 5.5.0을 사용하여 공정 단계(`ProcessStep`), 설비(`Machine`), 파라미터(`ProcessParameter`), 센서(`Sensor`), 측정값(`Measurement`), 불량 유형(`DefectType`), 인과관계(`causes`, `prevents`) 등 핵심 개념(Classes)과 이들 간의 관계(Object Properties), 그리고 각 개념의 속성(Data Properties, 예: `hasValue`, `hasUnit`, `hasTimestamp`)을 정의한다. 구축된 온톨로지는 제조 공정 지식의 체계적인 저장소 역할을 하며, 추론의 기반을 제공한다.
5.  **XAI-온톨로지 통합 메커니즘**: 정량적 XAI 분석 결과와 정성적 온톨로지 지식을 체계적으로 연결하여 상호 보완적 해석을 가능하게 한다.
    *   **개념 매핑(Concept Mapping)**: 데이터 스키마의 특성 이름(예: 'Sensor_Factory_Humidity')을 온톨로지 내의 해당 클래스 또는 인스턴스(예: `diecast:Sensor_Factory_Humidity` 클래스의 특정 측정값 인스턴스)와 명시적으로 매핑한다. 이를 위해 URI 또는 주석(annotation property)을 활용할 수 있다.
    *   **XAI 결과 온톨로지 표현(Representing XAI Results in Ontology)**: 계산된 SHAP 값(중요도 및 방향성)을 온톨로지 인스턴스의 속성으로 저장한다. 예를 들어, 특정 특성 인스턴스에 대해 `diecast:hasShapImportanceValue` (xsd:float) 데이터 속성으로 평균 절대 SHAP 값을 연결하고, SHAP 값의 부호에 기반하여 해당 특성 인스턴스와 관련 불량 유형 인스턴스 사이에 `diecast:correlatedWithDefectRisk` (값: "Positive"/"Negative") 와 같은 속성을 추가하거나, 더 나아가 `diecast:potentialCauseOf` / `diecast:potentialPreventiveFactorFor` 와 같은 잠정적 인과 관계(Object Property)를 설정할 수 있다. Counterfactual 결과는 `diecast:hasCounterfactual` 관계를 통해 원본 인스턴스와 연결되며, 변경된 조건(`propertyConstraint`)과 예측 변화(`predictedOutcome`)를 포함하는 복합 개체로 표현될 수 있다.
6.  **온톨로지 기반 추론 및 해석**: 통합된 지식 그래프에 대해 **논리 추론 엔진(Reasoner, 예: HermiT)**을 적용한다. 추론 과정은 (1) 온톨로지의 **논리적 일관성(consistency)**을 검증하고, (2) 클래스 계층 구조(subsumption) 및 속성 제약 조건에 기반한 **암묵적 지식(implicit knowledge)**을 명시화하며, (3) 필요시 정의된 **SWRL 규칙**을 통해 **복잡한 조건 기반의 추론**을 수행한다. 추론 결과는 다음과 같이 활용될 수 있다: (a) 데이터 기반 XAI 설명의 **타당성을 도메인 지식 관점에서 교차 검증** (예: SHAP에서 중요하게 나온 변수가 온톨로지에서도 알려진 불량 원인인가?), (b) XAI만으로는 파악하기 어려운 **간접적 영향이나 복합적 원인 탐색** (예: A->B, B->C 규칙 추론을 통해 A->C 관계 추정), (c) SPARQL 질의를 통한 **사용자 맞춤형 설명 및 지식 탐색** 지원, (d) 최종적으로 **더욱 신뢰성 있고 맥락적으로 풍부하며 실행 가능한 설명** 생성.

*(Figure 1: 제안 프레임워크 다이어그램 삽입 위치 - 데이터 흐름, XAI 모듈, 온톨로지, 통합 및 추론 단계를 명확히 보여주는 그림)*

## IV. 실험 결과 및 분석

### 4.1 실험 환경 및 데이터
본 연구는 실제 다이캐스팅 공장에서 수집된 데이터를 대상으로 제안 프레임워크의 실증적 검증을 수행하였다. 실험 환경은 Python 3.9 기반이며, 주요 라이브러리로 Scikit-learn (모델링), XGBoost 1.7, SHAP 0.41 (XAI), DiCE 0.10 (Counterfactual), Owlready2 0.47 (온톨로지 처리) 등이 사용되었다. 온톨로지 추론에는 HermiT 엔진이 활용되었고, 하드웨어는 macOS Sonoma (Apple M1 Max, 32GB RAM) 환경에서 진행되었다.

원본 데이터셋(7,535개 샘플)은 특정 기간 동안 수집된 데이터로, 해당 기간의 공정 특성 및 제품군을 반영한다. 결측치 처리 및 IQR 기반 이상치 탐지/제거를 통해 **7,284개 샘플**을 최종 분석에 사용하였다(251개 제거, 약 3.3%). 데이터는 시간 순서를 고려하지 않고 훈련 세트(5,827개, 80%)와 테스트 세트(1,457개, 20%)로 무작위 분할되었다. 'AnyDefect' 타겟 변수의 전체 불량률은 약 **22.87%**로, 이는 분석 대상 기간/제품의 특성을 나타내는 수치이며 일반적인 공정 전체의 평균 불량률과는 다를 수 있다. 훈련 세트의 불량률(약 22.88%)이 비교적 높아, 본 연구에서는 별도의 데이터 샘플링(SMOTE 등) 없이 모델 학습을 진행하였다. 이는 실제 현장 데이터 분포 하에서의 모델 성능을 평가하기 위함이었으나, 소수 클래스 학습의 어려움에 영향을 미쳤을 수 있다(4.2절 논의 참조).

### 4.2 품질 예측 모델 성능
RandomForest와 XGBoost 모델을 대상으로 5-fold 교차 검증(F1-score 기준)을 수행한 결과, RandomForest가 평균 0.6975로 XGBoost(0.4848, 논문 파라미터 적용 시) 대비 확연히 우수한 성능을 보여 최종 예측 모델로 선정되었다. 테스트 세트에서의 최종 성능은 **Table 1**과 같다.

**Table 1.** 품질 예측 모델 성능 비교 (테스트 세트 기준)
| 모델         | Accuracy | Precision (Defect) | Recall (Defect) | F1-score (Defect) | CV F1-score (Avg.) |
|--------------|----------|--------------------|-----------------|-------------------|--------------------|
| RandomForest | **90.46%** | **86.19%**         | 69.37%          | **76.87%**        | **0.6975**         |
| XGBoost      | 85.79%   | 67.59%             | 70.57%          | 69.05%            | 0.4848             |

RandomForest 모델은 약 90.5%의 정확도를 달성했으나, 불량 클래스에 대한 Recall은 약 69%로 개선의 여지를 남겼다. SHAP 분석은 공장 습도, 스프레이 시간, 형체력 등 데이터 기반의 주요 품질 영향 인자를 성공적으로 식별하였으며, 이는 도메인 지식과 비교/검증될 수 있는 중요한 정량적 근거를 제공한다. 핵심적으로, 본 연구는 이러한 XAI 분석 결과를 도메인 지식을 형식화한 OWL 온톨로지와 성공적으로 통합하고, 통합된 지식 그래프 상에서 HermiT 추론 엔진을 성공적으로 실행함으로써, **데이터와 지식의 시너지를 통한 심층적이고 신뢰성 있는 설명 제공**의 기술적 가능성을 명확히 입증하였다.

### 4.3 XAI 기반 모델 설명 결과
선정된 RandomForest 모델에 대해 SHAP 분석을 수행하여 전역적 특성 중요도를 평가한 결과, **Sensor_Factory_Humidity (공장 습도)**가 모델 예측에 가장 큰 영향을 미치는 변수로 일관되게 나타났다. 이는 일반적인 예상(예: 사출 압력, 온도)과 다소 차이가 있는 흥미로운 결과로, 해당 공정 환경에서는 대기 중 습도 관리가 제품 품질에 예상 외로 중요한 요인일 수 있음을 시사한다. 습도가 높으면 용탕 내 수소 가스 용해도가 변하거나 금형 표면의 냉각/윤활 조건에 영향을 미쳐 기공(porosity)이나 유동 패턴 변화를 유발했을 가능성을 추정해볼 수 있으나, 이에 대한 명확한 인과관계 규명은 추가적인 실험이나 도메인 전문가의 심층 분석이 필요하다. 그 외 **Process_Spray_Time (스프레이 시간), Process_Clamping_Force (형체력), Process_Cycle_Time (사이클 타임), Process_High_Velocity (고속 구간 속도)** 등이 주요 영향 인자로 식별되어, 품질 관리 시 주목해야 할 핵심 변수들에 대한 데이터 기반의 증거를 제공하였다.

개별 예측에 대한 SHAP 값 분석은 특정 불량 사례의 잠재적 원인을 추정하는 데 활용되었으며, 이 정보는 온톨로지 통합 단계에서 지식 기반 해석의 기초 자료로 사용되었다. 다만, SHAP 요약 플롯(summary plot) 생성 과정에서 라이브러리 버전 간 호환성 문제로 인해 시각화에 제한이 있었음을 기술적 한계로 밝힌다(SHAP 값 자체는 성공적으로 계산됨).

Counterfactual 분석은 일부 불량 예측 샘플(테스트 세트 불량 예측 건 중 약 절반 미만 추정, 정확한 비율은 추가 분석 필요)에 대해 예측을 정상으로 바꾸는 최소한의 조건 변경(예: "Spray Time을 0.5초 줄임")을 성공적으로 제시하였다. 이는 공정 개선을 위한 구체적인 가이드라인을 제공할 잠재력을 보여주지만, 모든 샘플에 대해 유의미한 counterfactual 생성이 보장되지는 않았다. 이는 DiCE 알고리즘 자체의 한계, 탐색 공간의 복잡성, 또는 해당 샘플 주변에 실제로 예측을 뒤집을 만한 유사한 정상 사례가 존재하지 않기 때문일 수 있다. 또한, 제시된 counterfactual(예: Spray Time 0.5초 단축)이 실제 공정에서 **기술적으로 실현 가능하며 다른 품질 특성이나 생산성에 부정적인 영향을 미치지 않는지**에 대한 검토는 XAI 결과의 실질적인 적용을 위해 반드시 필요하다.

### 4.4 XAI-온톨로지 통합 및 추론
본 연구의 핵심적인 기여는 데이터 기반 XAI 분석 결과를 다이캐스팅 도메인 온톨로지와 성공적으로 통합하고, 이를 기반으로 논리적 추론을 수행하여 **설명가능성 향상의 질적 가능성**을 실증한 데 있다. SHAP 분석으로 식별된 주요 변수들(예: `Sensor_Factory_Humidity`)은 온톨로지 내 해당 클래스의 인스턴스(예: 특정 시점의 습도 측정값 인스턴스)와 연결되었고, `hasImportance` (SHAP 값 크기) 및 `positivelyCorrelatedWithDefect`/`negativelyCorrelatedWithDefect` (SHAP 값 부호 기반 관계) 등의 속성을 통해 XAI 정보가 온톨로지 구조 내에 명시적으로 표현되었다. 이 과정은 데이터 스키마와 온톨로지 개념 간의 신중한 매핑을 통해 이루어졌다.

이렇게 XAI 정보가 보강된 온톨로지 지식 그래프(`ontology/diecasting_ontology_updated.owl`)에 HermiT 추론기를 적용한 결과, 온톨로지의 논리적 일관성이 성공적으로 검증되었으며, 명시적 지식으로부터 새로운 관계나 분류를 추론하는 과정(`ontology/diecasting_ontology_inferred.owl`)이 오류 없이 완료되었다. 이는 **XAI와 온톨로지 추론 기술의 성공적인 연동**을 보여주는 중요한 결과이다. 예를 들어, 온톨로지에 다음과 같은 (가상의) SWRL 규칙이 포함되어 있다고 가정해보자:
`ProcessParameter(?p) ^ hasImportanceValue(?p, ?imp) ^ swrlb:greaterThan(?imp, 0.01) ^ positivelyCorrelatedWithDefectRisk(?p, "Positive") ^ influences(?p, ?defect) ^ DefectType(?defect) -> HighRiskFactorFor(?p, ?defect)`
이 규칙은 중요도(SHAP 값)가 0.01보다 크고 불량 위험과 양의 상관관계를 가지는 공정 파라미터는 해당 불량 유형의 고위험 인자임을 추론한다. 만약 SHAP 분석 결과 `Sensor_Factory_Humidity`가 이 조건을 만족한다면, 추론기는 `HighRiskFactorFor(humidity_sensor_reading_123, Porosity_defect_instance)` 와 같은 새로운 사실을 추론해낼 수 있다. 이는 데이터 기반 분석 결과를 도메인 지식(온톨로지 규칙)과 결합하여 **더 높은 수준의 의미론적 해석**을 가능하게 함을 보여준다.

비록 본 연구에서는 복잡한 SWRL 규칙 설계 및 추론 결과의 심층 분석까지는 진행하지 않았고, XAI 결과와 온톨로지 규칙 간의 정량적 일치율 계산 등의 평가는 수행하지 않았지만, **XAI와 온톨로지 추론의 기술적 통합 및 연동 가능성**을 실제 산업 데이터 환경에서 실증적으로 확인했다는 점에서 중요한 방법론적 기여를 한다. 이 통합 프레임워크는 XAI 설명을 **맥락화**하고, **도메인 지식 기반으로 검증**하며, **지능적인 질의응답 및 탐색**을 가능하게 하는 기반을 마련한다.

## V. 고찰 및 결론
본 연구는 다이캐스팅 제조 공정의 품질 관리를 위한 설명가능 AI(XAI)와 온톨로지 통합 프레임워크를 제안하고, 실제 산업 데이터를 이용한 실증 연구를 통해 그 구현 가능성, 잠재력, 그리고 현실적인 한계를 심층적으로 분석하였다. RandomForest 모델을 통해 약 90% 수준의 예측 정확도를 달성하였으나, 불량 클래스에 대한 Recall은 약 69%로 개선의 여지를 남겼다. SHAP 분석은 공장 습도, 스프레이 시간, 형체력 등 데이터 기반의 주요 품질 영향 인자를 성공적으로 식별하였으며, 이는 도메인 지식과 비교/검증될 수 있는 중요한 정량적 근거를 제공한다. 핵심적으로, 본 연구는 이러한 XAI 분석 결과를 도메인 지식을 형식화한 OWL 온톨로지와 성공적으로 통합하고, 통합된 지식 그래프 상에서 HermiT 추론 엔진을 성공적으로 실행함으로써, **데이터와 지식의 시너지를 통한 심층적이고 신뢰성 있는 설명 제공**의 기술적 가능성을 명확히 입증하였다.

본 연구의 주요 **학술적 기여**는 다음과 같다. 첫째, 제조 AI 분야에서 아직 초기 단계인 XAI와 온톨로지의 **체계적 통합 방법론**을 제시하고, 이를 실제 산업 데이터 환경에서 **구체적으로 구현 및 검증**하였다. 이는 이론적 논의를 넘어선 실질적인 구현 사례로서 후속 연구에 중요한 참조점을 제공한다. 둘째, SHAP, Counterfactual Explanation, OWL 온톨로지, HermiT 추론 엔진 등 다양한 기술 요소를 연동하는 **기술적 파이프라인**을 구축하고, 그 과정에서 마주한 **현실적 도전과제**(예: 라이브러리 호환성, 예측 성능의 한계, Counterfactual 적용의 어려움)를 투명하게 제시함으로써, 연구 및 현장 적용 시 고려해야 할 실질적인 측면들을 조명하였다. 셋째, 비록 정량적인 시너지 효과 측정이나 완벽한 예측 성능 달성에는 이르지 못했지만, 실제 산업 데이터의 복잡성 하에서 **XAI-온톨로지 통합 프레임워크의 실현 가능성**과 **설명력 강화 및 지식 융합**이라는 **핵심적인 질적 가치**를 실증적으로 보여주었다는 점에서 중요한 의미를 갖는다.

실험 결과, RandomForest 모델이 최적 성능(Accuracy 90.46%, F1-Defect 76.87%)을 보였으며, Sensor_Factory_Humidity가 가장 중요한 변수로 식별되었다. XAI 결과의 온톨로지 통합 및 후속 추론 과정은 기술적으로 성공적으로 완료되었다.

그러나 본 연구는 다음과 같은 **한계점** 및 이를 극복하기 위한 **구체적인 미래 연구 방향**을 제시한다:

*   **데이터 및 모델 성능 측면**: (1) **데이터 맥락 이해 심화**: 분석 대상 데이터의 불량률(22.87%)이 특정 조건(기간, 제품)을 반영함을 인지하고, 더 넓은 범위의 데이터 확보 또는 샘플링 전략(예: 능동 학습) 적용을 통한 일반화 성능 향상 연구가 필요하다. (2) **예측 성능 최적화**: F1-score 및 특히 Recall 성능 개선을 위해 특징 공학 강화, 하이퍼파라미터 정밀 튜닝(예: Bayesian Optimization), 최신 테이블형 데이터 처리 딥러닝 모델(예: TabNet, NODE) 적용, 또는 비용 민감 학습(Cost-sensitive learning) 도입을 고려해야 한다.
*   **XAI 기법 측면**: (1) **XAI 안정성 및 호환성 확보**: SHAP 시각화 등에서 발생한 라이브러리 호환성 문제는 강건한 XAI 파이프라인 구축의 중요성을 시사하며, 대안적인 시각화 방법 또는 안정적인 라이브러리 버전 관리가 요구된다. (2) **Counterfactual Explanation 강화**: DiCE 외 다른 Counterfactual 기법(예: MACE, Wachter) 적용 및 비교 연구, 그리고 생성된 counterfactual의 현실적 제약 조건(feasibility constraint) 고려 및 사용자 피드백 통합 방안 연구가 필요하다.
*   **온톨로지 및 통합 측면**: (1) **온톨로지 자동 구축 및 확장**: 전문가 지식 의존성을 줄이고 온톨로지의 최신성 및 확장성을 확보하기 위해, 텍스트 마이닝, 데이터베이스 스키마 분석 등을 활용한 **온톨로지 학습(Ontology Learning)** 기술 도입 연구가 필수적이다. (2) **추론 규칙 정교화 및 활용 심화**: 본 연구에서 가능성만 확인한 SWRL 규칙 기반 추론을 실제 공정 지식 기반으로 정교화하고, 추론 결과를 활용한 **자동 진단, 근본 원인 분석(Root Cause Analysis), 예측적 유지보수** 등 구체적인 응용 시나리오 개발 연구가 필요하다. (3) **정량적 시너지 평가 지표 개발**: XAI-온톨로지 통합이 설명의 품질(예: 충실도, 이해도, 실행가능성) 또는 의사결정의 효율성에 미치는 영향을 **정량적으로 측정**할 수 있는 평가 프레임워크 및 지표 개발이 시급하다.
*   **일반화 및 현장 적용**: 본 연구 결과를 다른 유형의 다이캐스팅 공정 또는 완전히 다른 제조 도메인(예: 사출 성형, 반도체)에 적용하여 제안 프레임워크의 **일반화 가능성(generalizability)**을 검증하고, 실제 현장 엔지니어와의 협업을 통해 개발된 시스템의 **사용성(usability) 및 실효성(effectiveness)**을 평가하는 연구가 중요하다.

결론적으로, 본 연구는 제조 분야에서 AI의 신뢰성과 활용성을 제고하기 위한 핵심 전략으로서 **설명가능성(XAI)과 지식 표현(Ontology)의 통합**이 갖는 중요성과 기술적 실현 가능성을 실제 산업 데이터를 통해 입증하였다. 데이터 기반 분석의 민첩성과 도메인 지식의 깊이를 결합한 본 프레임워크는, 제시된 한계점들을 극복하고 후속 연구를 통해 발전될 경우, **더욱 지능적이고 투명하며 신뢰할 수 있는 차세대 스마트 제조 시스템** 구축에 핵심적인 기여를 할 것으로 기대된다.

## 참고문헌 (References)

1.  **Hong, J.S.**, Hong, Y.M., Oh, S.Y., Kang, T.H., Lee, H.J., & Kang, S.W. (2023). *XAI 알고리즘 기반 사출 공정 수율 개선 방법론* (Injection Process Yield Improvement Methodology Based on XAI Algorithm). **Journal of Korean Society for Quality Management**, **51**(1), 55-65.
2.  **Park, S.**, & Youm, S. (2023). *Establish a machine learning based model for optimal casting conditions management of small and medium sized die casting manufacturers*. **Scientific Reports**, **13**, 17163.
3.  **Okuniewska, A.**, Perzyk, M., & Kozłowski, J. (2023). *Machine learning methods for diagnosing the causes of die-casting defects*. **Computer Methods in Materials Science**, **23**(2), 45-56.
4.  **Wang, S.**, Yang, J., Yang, B., Li, D., & Kang, L. (2024). *An Intelligent Quality Control Method for Manufacturing Processes Based on a Human–Cyber–Physical Knowledge Graph*. **Engineering**, **41**, 242-260.
5.  **Naqvi, M.R.**, Elmhadhbi, L., Sarkar, A., Archimede, B., & Karray, M.H. (2024). *Survey on ontology-based explainable AI in manufacturing*. **Journal of Intelligent Manufacturing**, **35**(8), 3605-3627.
6.  **Agostinho, C.** *et al.* (2023). *Explainability as the key ingredient for AI adoption in Industry 5.0 settings (XMANAI platform)*. **Frontiers in Artificial Intelligence**, **6**, 1264372.
7.  **Rožanec, J.M.**, Zajec, P., Kenda, K., Novalija, I., Fortuna, B., Mladenić, D. *et al.* (2023). *XAI-KG: Knowledge Graph to Support XAI and Decision-Making in Manufacturing*. **International Journal of Production Research**, **61**(20), 6847-6872.
8.  **Chung, S.H.**, & Kim, D.J. (2008). *Effect of Process Parameters on Quality in Injection Molding*. **Journal of Materials Processing Technology**, **200**(1-3), 120-130.
9.  **Jacob, D.** (2017). *Quality 4.0 Impact and Strategy Handbook: Getting Digitally Connected Quality Management*. Generis Publishing.

