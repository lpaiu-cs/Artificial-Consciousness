# Changelog

이 문서는 현재 공개 `git` 히스토리를 바탕으로 주요 변경을 재구성한 것입니다.
저신호 커밋 메시지(`mf` 등)는 가능한 범위에서 주변 변경 맥락에 맞춰 묶었습니다.

## 2026-03-22

### Tests

- 공개 smoke suite에 legacy `ltm_graph.json` / delta log의 startup scrub migration 회귀를 추가했습니다. 이미 유출된 boundary payload가 `MemoryGraph` 초기화 시 sanitize + rewrite 되는 경로를 재현 가능하게 검증합니다. (`0b7f1f1`)

## 2026-03-21

### Security

- boundary 민감 payload가 canonical SQLite 외 저장물에 평문으로 남지 않도록, graph snapshot / delta에는 public projection만 남기고 full payload는 canonical encrypted blob으로 분리했습니다. (`10dc464`)
- canonical boundary payload에 at-rest 암호화를 적용했습니다. (`b5aa660`)
- stored boundary 집행, boundary semantics, user-visible response repair, scope tightening을 통해 경계 관련 누수와 과잉 보상 경로를 정리했습니다. (`766de35`, `509d87c`, `0af54c5`, `351e0e2`, `7811196`)

### Validation And Ops

- pinned model 설정과 model eval gate를 추가하고, 공개 저장소에 재현 가능한 smoke suite를 포함시켰습니다. (`b62894a`)
- boundary semantic fallback에 candidate budget과 cache를 도입해 비용과 지연을 제한했습니다. (`b62894a`)

### Memory And Retrieval

- referent cache와 role alias 기반 target resolution을 추가하고, fulfillment-aware reliability를 relation update에 연결했습니다. (`1764f37`)

## 2026-03-20

### Memory Model

- 기존 `Insight` 중심 구조를 `Entity - Claim - Episode + Note` 중심 구조로 재편했습니다. 새로운 write path는 `Claim`과 `Note`를 우선 저장하고, canonical state와 서사 맥락을 분리합니다. (`e942d20`)
- active claim, open loop, interaction policy, relation state를 위한 canonical SQLite store와 query planning 기반 read path를 도입했습니다. (`4f24c92`)

### Stability

- P0 / P1 / P2 단계의 메모리 안정화 작업을 반영했습니다.
  - empty-delta 처리, assistant self-memory embedding, claim provenance persistence, scope handling (`28ea211`)
  - open loop lifecycle, multi-active / interval merge, entity-aware planner (`e973f1c`)
  - relation signal 분리와 user-scoped session mood (`2d13fac`)

### Privacy And Access Control

- durable cursor의 평문성, legacy `shared` scope, participant ACL, memory-safe boundary handling을 순차적으로 정리했습니다. (`847acd9`, `d5f0d2f`)
- `.env`를 추적 대상에서 제거하고 예제 파일만 남겨 운영 보안을 강화했습니다. (`b3a663a`)

## 2026-01-27 to 2026-01-31

### Early Architecture Work

- logger 개선, ego/controller 정리, nickname abuse 방지, identity 관련 기초 구조 작업을 진행했습니다. (`56f7b0a`, `42f815c`, `8326c89`)
- 아키텍처 다이어그램과 README 자산을 갱신했습니다. (`8632d86`, `9f6f574`)
- 저신호 유지보수성 커밋이 몇 차례 포함되어 있습니다. (`e571aa7`, `ae9c7df`)
