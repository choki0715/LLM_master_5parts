# LLM 최신 기술 마스터 과정: sLLM 기반 에이전트 서비스 만들기

> AI 원내교육 고급과정 | 강사: 김의중 (아이덴티파이 대표)

## 교육 목표

- **LLM 핵심 원리 이해**: Transformer 아키텍처, Attention, Position Encoding의 수학적 원리
- **모델 파인튜닝 직접 구현**: HuggingFace Hub 기반 SFT, LoRA, DPO, GRPO 코드 구현
- **RAG 시스템 설계/구축**: Vector RAG, Graph RAG, 온톨로지 RAG 구축 (Apache AGE 기반)
- **AI 에이전트 개발**: LangGraph, AG2, MCP, Function Calling 활용 멀티 에이전트 시스템
- **지식 증류 & 최신 연구**: DeepSeek-R1의 CoT 증류, GRPO 강화학습 구조 분석
- **Vibe Coding 실무 활용**: Claude Code, Windsurf 등 AI 코딩 도구 활용
- **클라우드 배포 & 모니터링**: AWS/GCP 기반 에이전트 배포, GitHub 형상 관리, LangSmith 모니터링

## 선수 지식

- Python 프로그래밍 기초
- 선형대수 및 확률 통계 기초
- 딥러닝 개념
- PyTorch 기초
- VSCode 등 IDE 사용 경험

## 실습 환경

- **GPU**: NVIDIA RTX 4060+ (8GB VRAM 이상)
- **Python**: 3.10+
- **CUDA**: 12.1+
- **디스크**: 50GB 이상 여유 공간
- **사용 SW**: HuggingFace Hub, PostgreSQL, ChromaDB, Neo4j

## 환경 설정

```bash
# 1. 저장소 클론
git clone https://github.com/choki0715/LLM_master_5parts.git
cd LLM_master_5parts

# 2. 자동 환경 설정 (Ubuntu)
bash setup.sh

# 또는 수동 설정:
# 2-1. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2-2. 패키지 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
cp .env.example .env
# .env 파일에 API 키 입력

# 4. 환경 점검
jupyter notebook setup_check.ipynb
```

---

## 커리큘럼

### 1파트: LLM 기초와 생태계

#### [01] 딥러닝 기초, 자연어 처리, LLM 기본
| 노트북 | 내용 |
|--------|------|
| `part1/01_deep_learning_basics.ipynb` | 딥러닝 개념과 언어모델의 역사 |
| `part1/02_nlp_encoding_tokenization.ipynb` | 자연어 처리: 인코딩, 토큰화, 임베딩(Word2Vec) |
| `part1/03_llm_overview_sllm.ipynb` | LLM 현황 및 sLLM 기본 실습 |

#### [02] 트랜스포머, BERT, GPT, HuggingFace 생태계, LangChain
| 노트북 | 내용 |
|--------|------|
| `part1/04_transformer_bert_gpt.ipynb` | 트랜스포머: Attention, BERT, GPT 구조 비교 |
| `part1/05_huggingface_ecosystem.ipynb` | HuggingFace 생태계 및 활용 방법 |
| `part1/06_langchain_practice.ipynb` | LangChain 실습 |

---

### 2파트: 지식 증류와 파인튜닝

#### [03] 지식증류와 MoE 아키텍처
| 노트북 | 내용 |
|--------|------|
| `part2/07_knowledge_distillation.ipynb` | 지식증류: 하드 증류, 소프트 증류, Temperature 작동 원리 |
| `part2/08_scaling_law.ipynb` | Scaling Law: 최적의 데이터 규모와 모델 크기 결정 |
| `part2/09_moe_deepseek.ipynb` | MoE: DeepSeek의 차별화, 학습 및 추론 속도의 우수성 |

#### [04] 파인튜닝: SFT, PEFT(LoRA), 고품질 데이터셋 만들기
| 노트북 | 내용 |
|--------|------|
| `part2/10_sft_huggingface_trl.ipynb` | 전체 미세조정: HuggingFace Trainer API vs SFTTrainer (TRL) |
| `part2/11_lora_peft_theory.ipynb` | 부분 미세조정 (PEFT): LoRA 이론 |
| `part2/11b_lora_peft_practice.ipynb` | LoRA vs FFT 실전 비교 실습 |
| `part2/12_continuous_learning.ipynb` | 지속 학습 (Continuous Pretraining) |
| `part2/12b_instruction_tuning.ipynb` | Instruction 학습 |

---

### 3파트: 정렬, 강화학습, 양자화

#### [05] LLM 정렬 및 추론 강화
| 노트북 | 내용 |
|--------|------|
| `part3/13_rl_basics_alignment.ipynb` | 강화학습 기초, 정렬 및 강화학습 |
| `part3/14_deepseek_r1_analysis.ipynb` | DeepSeek-R1 사례 분석 |
| `part3/15a_preference_data.ipynb` | Preference 데이터 수집/생성 |
| `part3/15b_dpo_practice.ipynb` | DPO 학습 실습 |
| `part3/15c_grpo_practice.ipynb` | GRPO 이론 및 실습 |

#### [06] 모델 경량화: 양자화 기법
| 노트북 | 내용 |
|--------|------|
| `part3/16_quantization_concepts.ipynb` | 양자화 개념 및 기법 비교분석 |
| `part3/17_gptq_awq_qlora.ipynb` | GPTQ, AWQ, QLoRA 비교 |
| `part3/18_quantization_practice.ipynb` | 양자화 실습 |

---

### 4파트: RAG 시스템

#### [07] 지식 증강 및 RAG 기초
| 노트북 | 내용 |
|--------|------|
| `part4/19_rag_fundamentals.ipynb` | RAG 기본개념과 파이프라인 |
| `part4/20_vector_db_comparison.ipynb` | 벡터 DB 심층 비교 분석 및 실습 |
| `part4/21_rag_practice.ipynb` | LangChain RAG 어플리케이션 구현 |

#### [08] 그래프 RAG와 온톨로지 RAG 구현
| 노트북 | 내용 |
|--------|------|
| `part4/22_advanced_rag_base.ipynb` | 벡터 RAG의 한계 및 Advanced RAG |
| `part4/23_graph_rag.ipynb` | 그래프 RAG: Neo4j 기반 지식 그래프 |
| `part4/24_ontology_rag.ipynb` | 온톨로지 RAG: Apache AGE 기반 추론 시스템 |

---

### 5파트: AI 에이전트와 프로젝트

#### [09] 바이브 코딩을 이용한 AI 에이전트 구현 실습
| 노트북 | 내용 |
|--------|------|
| `part5/25_vibe_coding_intro.ipynb` | 바이브 코딩이란? |
| `part5/26_claude_code_agent.ipynb` | Claude Code를 이용한 AI Agent 구현 실습 |
| `part5/27_tool_calling_function.ipynb` | Tool Calling Function 만들기 |

#### [10] 프로젝트 실습: 특정 도메인에 최적화된 sLLM 파인튜닝
| 노트북 | 내용 |
|--------|------|
| `part5/28_agent_tech_stack_langgraph.ipynb` | Agent AI 기술 스택, LangGraph 기반 워크플로우 |
| `part5/29_data_pipeline_training.ipynb` | 데이터 수집 → 정제 파이프라인 |
| `part5/29b_project_training.ipynb` | 학습 수행 |
| `part5/30a_evaluation.ipynb` | 성능 평가 및 반복 개선 |
| `part5/30b_deployment.ipynb` | 배포 |

---

## 수강 추천

- LLM의 내재화 또는 전략적 활용 방법을 수립하고자 하는 분
- 효과적인 RAG를 구축하고자 하는 분
- 데이터 보안을 위해 자체 sLLM 구축을 계획하시는 분
- LLM과 GPT에 대한 이해를 넓히고자 하는 분
- 자체적인 인공지능 에이전트를 구현하고자 하는 분

## 교재

- 딥러닝 개념과 활용 (김의중, 미리어드스페이스)

---

**© Copyright AIDENTIFY. All rights reserved.**
