# step_diff — STEP B-Rep 기반 설계 변경 객체 대응 및 변경 유형 판정

설계 변경 **전(OLD)** STEP 파일과 **후(NEW)** STEP 파일을 입력받아, 각 파일 속 개별 CAD 객체 사이의
**전역 1:1 대응 관계**, **변환 정보(이동·회전·스케일·대칭)**, **설계 변경 유형**(STAY / MODIFIED /
INSERTION / DELETION)을 출력합니다.

- 입력 데이터: STEP에 저장된 **B-Rep 정보만 직접 사용** (면·모서리 유형, 해석적 곡면 파라미터,
  치수·부피·면적, 면-모서리-정점 연결 관계)
- **사용하지 않는 방법**: 머신러닝, RANSAC, ICP, PCA(관성 주축 정렬 포함), 점군 샘플링, 테셀레이션
- 언어/환경: Python 3.10+, VS Code (실행·디버그·테스트 구성 포함)

---

## 1. 설치

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # cadquery-ocp(OpenCascade), numpy, scipy, pytest
```

`cadquery-ocp`는 OpenCascade 커널의 Python 바인딩(OCP)입니다. pip로 설치되며 conda가 필요 없습니다.

### VS Code
- 폴더를 열면 `.vscode/settings.json`이 `.venv` 인터프리터와 pytest를 자동 설정합니다.
- **실행 및 디버그(F5)** 구성:
  - `step_diff: compare example OLD/NEW` – 예제 파일 비교, `out/report.json`, `out/report.csv` 생성
  - `step_diff: compare chosen STEP files` – 실행 시 OLD/NEW 경로를 입력받아 비교
  - `examples: regenerate demo STEP files` – 예제 STEP 재생성
  - `pytest: all tests`
- 테스트 탐색기에서 `tests/`의 테스트를 바로 실행할 수 있습니다.

## 2. 사용법

```bash
python -m step_diff OLD.step NEW.step -o report.json --csv report.csv
```

```python
from step_diff import compare_step_files, Config

report = compare_step_files("old.step", "new.step", Config(rel_pos_tol=1e-4))
for r in report["results"]:
    print(r["old"], r["new"], r["status"], r["change_types"])
```

예제(`examples/old.step`, `examples/new.step`, 정답은 `examples/expected.json`) 실행 결과:

```
OLD                      NEW                      RESULT
----------------------------------------------------------------------------------------------------
[0] plate                [0] plate                STAY
[1] bracket              [1] bracket              MODIFIED: TRANSLATION(dx=+5.0000, dy=-3.0000, dz=+2.0000)
[2] block                [2] block                MODIFIED: ROTATION(30.0000 deg about [+0.0000, +0.0000, +1.0000] +Z)
[3] pin                  [3] pin                  MODIFIED: UNIFORM_SCALING(s=1.500000)
[4] housing              [4] housing              MODIFIED: TRANSLATION(dx=+3.0000, dy=+4.0000, dz=+5.0000) + ROTATION(45.0000 deg about [+0.5774, +0.5774, +0.5774] oblique) + UNIFORM_SCALING(s=1.200000)
[5] rib                  [5] rib                  MODIFIED: NON_UNIFORM_SCALING([2.000000, 1.000000, 1.000000])
[6] boss                 [6] boss                 MODIFIED: TRANSLATION(dx=+2.0000, dy=+0.0000, dz=+0.0000) + SHAPE_CHANGE(vol x0.9127, faces 6->7)
[7] shaft                [7] shaft                MODIFIED: ROTATION(20.0000 deg about [+1.0000, +0.0000, +0.0000] +X, axisymmetric)
[9] cube                 [8] cube                 STAY
[10] chiral              [9] chiral               MODIFIED: MIRROR
[8] old_only             -                        DELETION
-                        [10] new_only            INSERTION
-                        [11] plate_copy          INSERTION
```

- `cube`는 Z축 90° 회전했지만 정육면체의 대칭으로 형상이 완전히 동일하므로 **STAY**입니다
  (24개의 등가 자세 중 최소 회전 = 0°).
- `plate_copy`는 `plate`와 형상이 완전히 같지만 AABB가 겹치지 않으므로 **다른 객체(INSERTION)** 입니다.

---

## 3. 전체 파이프라인

| 단계 | 내용 | 코드 |
|---|---|---|
| 1 | OLD/NEW STEP 읽기 및 객체 분리 (Solid 단위, 표면 모델은 Shell 단위). 어셈블리 배치(Location) 적용, 제품/인스턴스 이름 보존 | `step_io.py` |
| 2 | 각 객체에 IDX 부여 (OLD 0..N-1, NEW 0..M-1) | `step_io.py` |
| 3 | B-Rep 특징 추출 | `features.py` |
| 4 | OLD 0부터 AABB 생성 → AABB가 겹치는 NEW 객체만 후보로 선정 | `pipeline.py`, `compare.py` |
| 5 | 후보 쌍마다 변환 추정 + 유사도 계산 | `transform.py`, `compare.py` |
| 6 | 전역 1:1 매칭 (헝가리안 알고리즘, 총 유사도 최대화) | `pipeline.py` |
| 7 | 매칭 쌍: 1차 STAY/MODIFIED → 2차 이동/회전/스케일 → 3차 독립 성분 분리. 미매칭 OLD → DELETION, 미매칭 NEW → INSERTION | `compare.py`, `pipeline.py` |

### 3.1 B-Rep 특징 (`features.py`)
모두 B-Rep의 정확한 기하로부터 계산합니다 (샘플링·메쉬 없음).

| 수준 | 특징 |
|---|---|
| 객체 | 부피, 표면적, 부피 중심(가우스 적분), 정밀 AABB(`BRepBndLib::AddOptimal`, 테셀레이션 미사용), 면/모서리/정점 수, 오일러 특성 |
| 면(Face) | 곡면 유형(PLANE, CYLINDER, CONE, SPHERE, TORUS, BSPLINE …), 면적, 면 중심, 평면의 **방향 있는 법선**(면 orientation 반영), 원통/원뿔/토러스 축, 반지름, 원뿔 반각, 모서리·와이어 수 |
| 모서리(Edge) | 곡선 유형(LINE, CIRCLE, BSPLINE …), 길이 |
| 정점(Vertex) | 정확한 좌표 |
| 위상 | 면 인접 그래프(공유 모서리) + 면 라벨(유형·모서리 수·와이어 수)에 대한 **Weisfeiler-Lehman 해시** (위상 지문) |

### 3.2 AABB 후보 필터
`OLD_i`와 `NEW_j`의 AABB가 (수치 허용오차 + `aabb_margin`) 안에서 겹칠 때만 후보입니다.
겹치지 않으면 형상이 완전히 같아도 **설계 의도가 다른 객체**로 보고 비교하지 않습니다.

### 3.3 변환 추정 (`transform.py`) — 결정론적, 전수 검증
세 단계 추정기를 순서대로 적용합니다.

**(1) 유사 변환** `x' = c_new + s·Q·(x − c_old)` (Q = 회전, 또는 회전+대칭)

1. **스케일** `s = ∛(V_new / V_old)` — 정확한 부피비. 면적비 `s²`로 교차 검증.
2. **면 대응 후보**: 각 OLD 면에 대해 같은 곡면 유형, 같은 모서리/와이어 수, `면적·s²`, `반지름·s`,
   원뿔 반각, 그리고 **부피 중심으로부터 면 중심까지 거리**(회전 불변)가 일치하는 NEW 면만 후보로 남깁니다.
3. **회전 가설 열거 (RANSAC 아님, 전수 열거)**: 후보 수가 가장 적은 앵커 면 1~2개를 골라, 그 앵커의
   *모든* 후보 대응 각각에 대해 두 개의 독립 방향(평면 법선, 면 중심 오프셋, 축)으로 회전을
   **TRIAD 공식으로 정확히 계산**합니다. 유효한 변환이라면 앵커 면은 반드시 후보 중 하나로 가야 하므로
   이 열거는 빠짐없이 모든 해를 포함합니다.
4. **엄격한 검증**: 가설마다 OLD의 **모든 면**(중심·법선·축)과 **모든 정점**이 NEW에 1:1로
   허용오차 내 대응해야 채택합니다.
5. **정밀화**: 검증된 정확한 대응 쌍 전체에 대한 닫힌 형식 최소제곱 회전(SVD, 1회 계산).
   반복 최근접점 탐색(ICP)이 아니며, 대응은 이미 검증되어 고정된 상태입니다.
6. **대칭 처리**: 여러 회전이 모두 검증되면(대칭 형상) **가장 작은 회전**을 보고하고
   `equivalent_solutions`에 등가 해 개수를 기록합니다.
   - 축대칭 형상(원통·축 등): 축 자신에 대한 회전은 관측 불가 → **회전축 방향의 정확한 최소 회전**을
     보고하고 `determinacy = "axisymmetric"`으로 표시합니다 (정점은 원통 좌표로 검증하여 seam 위치 차이는 무시).
   - 방향 특징이 없는 형상(구): `determinacy = "none"`.
7. **대칭(MIRROR)**: 고유 회전으로 설명되지 않으면 반사를 포함해 같은 절차를 수행하고,
   결과를 "반사 평면 법선 + 그 법선에 대한 잔여 회전"으로 분해하여 보고합니다.

**(2) 비균일 스케일** `x' = c_new + R·E·D·Eᵀ·(x − c_old)`
위상(면·모서리·정점 수, WL 해시)이 같고 유사 변환이 실패할 때, 객체 자체의 평면 법선으로 만든 좌표계 E에서
축별 스케일 D를 **정밀 AABB 치수비**로 계산하고, 모든 정점 대응과 부피비 `∏D`로 검증합니다.
여러 해가 있으면 "늘어난 축 수가 가장 적고, 회전이 가장 작은" 해를 택합니다. (예: 길이만 바뀐 리브)

**(3) 부분 강체 정합 (형상 변경)** `x' = R·x + t`
형상이 바뀐 경우(구멍 추가 등), 면적·반지름이 같은 면 쌍에서 회전 가설을 만들고, 면 중심의 정확한
이동 벡터 **투표**로 변경되지 않은 부분을 가장 많이 겹치게 하는 (R, t)를 찾습니다. 변경되지 않은 면의
면적 비율, 제거/변경된 OLD 면 목록, 추가/변경된 NEW 면 목록을 함께 출력합니다.

### 3.4 유사도 점수와 전역 매칭
```
score = 0.5·shape + 0.3·spatial + 0.2·size
shape   = 1.0 (유사 변환 검증) | 0.9 (비균일 스케일 검증)
        | 0.5·topology + 0.25·geometry + 0.25·부분정합 면적비   (그 외)
spatial = 0.5·AABB IoU + 0.5·exp(−중심거리 / 평균 대각선)
size    = min(V_old, V_new) / max(V_old, V_new)
```
- topology: WL 해시 다중집합 Jaccard, 면/모서리/정점 수 비, 오일러 특성 일치
- geometry: 곡면 유형별 면적 분포, 곡선 유형별 길이 분포, 밀집도 `V/A^1.5` (모두 스케일 불변)

`shape ≥ 0.35`이고 `score ≥ 0.3`인 쌍만 매칭 가능하며, 비용 행렬 `−score`에 대해 헝가리안 알고리즘으로
**전역 1:1 최적 매칭**을 구합니다. 같은 형상의 객체가 여러 개 겹쳐 있어도 공간 점수로 올바르게 분리됩니다.

### 3.5 변경 유형 판정
| 단계 | 판정 |
|---|---|
| 1차 | 모든 성분이 임계값 이하이면 **STAY**, 하나라도 있으면 **MODIFIED** |
| 2차 | `TRANSLATION`: 부피 중심 이동량 `dx, dy, dz` (전역 X/Y/Z), `moved_axes` |
|     | `ROTATION`: 회전각(°), 회전축 단위벡터, 축 라벨(±X/±Y/±Z/oblique), XYZ 오일러각, 회전 중심 |
|     | `UNIFORM_SCALING`: 3축 동일 스케일일 때만, scale factor 출력 |
| 확장 | `NON_UNIFORM_SCALING` (축별 비율 — uniform 아님), `MIRROR` (반사 평면), `SHAPE_CHANGE` (형상 변형) |
| 3차 | 위 성분을 **각각 독립적으로** 판정하여 `change_types` 리스트로 출력 (복합 변형 = 여러 항목) |
| 미매칭 | OLD에만 있음 → **DELETION**, NEW에만 있음 → **INSERTION** |

변환 분해 기준: 회전·스케일은 **OLD 부피 중심** 기준이고, 이동은 부피 중심의 변위입니다.
다른 기준점이 필요하면 `matrix_4x4`(전역 좌표 4×4 동차 행렬)를 사용하세요.

---

## 4. 출력 형식 (JSON)

```jsonc
{
  "summary": { "old_objects": 11, "new_objects": 12, "matched": 10,
               "status_counts": {...}, "change_type_counts": {...} },
  "old_objects": [ { "idx": 0, "name": "plate", "volume": ..., "bbox_min": [...], "face_types": {...}, ... } ],
  "new_objects": [ ... ],
  "candidates": { "0": [ { "new_idx": 0, "score": 1.0, "alignment": "SIMILARITY", "eligible": true } ] },
  "results": [
    {
      "old": {"idx": 4, "name": "housing"}, "new": {"idx": 4, "name": "housing"},
      "status": "MODIFIED",
      "change_types": ["TRANSLATION", "ROTATION", "UNIFORM_SCALING"],
      "basis": "exact",                     // exact | partial_alignment | centroid_only
      "transformations": {
        "translation": {"detected": true, "dx": 3.0, "dy": 4.0, "dz": 5.0, "moved_axes": ["X","Y","Z"]},
        "rotation":    {"detected": true, "angle_deg": 45.0, "axis": [0.577, 0.577, 0.577],
                        "axis_label": "oblique", "euler_xyz_deg": [...], "center": [...],
                        "determinacy": "full", "equivalent_solutions": 1},
        "scaling":     {"detected": true, "type": "uniform", "factor": 1.2},
        "mirror":      {"detected": false}
      },
      "shape": { "changed": false, "volume_ratio": 1.728, "faces_old": 11, "faces_new": 11, ... },
      "matrix_4x4": [[...], [...], [...], [0, 0, 0, 1]],
      "scores": {"total": 0.77, "shape": 1.0, "spatial": 0.52, "size": 0.58}
    },
    { "old": {"idx": 8, "name": "old_only"}, "new": null, "status": "DELETION", ... },
    { "old": null, "new": {"idx": 10, "name": "new_only"}, "status": "INSERTION", ... }
  ]
}
```
`--csv`는 결과당 한 행(대응 IDX, 상태, 변경 유형, dx/dy/dz, 회전각/축, 스케일)의 표를 씁니다.

## 5. 허용오차 (`config.py`, CLI 옵션)

| 항목 | 기본값 | 의미 |
|---|---|---|
| `rel_pos_tol` | 1e-4 | 위치 비교 허용오차 / 객체 AABB 대각선 |
| `rel_value_tol` | 1e-4 | 면적·반지름·부피 상대 허용오차 |
| `angle_tol_deg` | 0.01 | 법선·축 방향 허용오차 |
| `translation_eps` | 1e-4 | 이 값 이하의 이동은 "이동 없음" (모델 단위) |
| `rotation_eps_deg` | 1e-3 | 이 값 이하의 회전은 "회전 없음" |
| `scale_eps` | 1e-5 | `|s−1|`이 이 값 이하이면 "스케일 없음" |
| `aabb_margin` | 0 | AABB 겹침 판정 추가 여유 |
| `detect_mirror`, `detect_nonuniform` | True | 확장 변환 탐지 on/off (`--no-mirror`, `--no-nonuniform`) |

STEP을 내보낸 CAD의 정밀도가 낮으면 `--rel-pos-tol 1e-3 --rel-value-tol 1e-3`처럼 완화하세요.

## 6. 테스트

```bash
pytest -q
```
- `test_demo.py`: 예제 STEP 쓰기 → 읽기 → 정답(`expected`)과 상태·유형·수치 비교
- `test_transforms.py`: 임의 축/각도 회전 정확도(1e-6°), 대칭 형상, 복합 변환, 미세 이동(0.001),
  축대칭, 구, 대칭+회전, 회전된 형상 변경, 회전+비균일 스케일, 4×4 행렬 검증
- `test_matching.py`: 동일 형상 이웃 객체의 전역 매칭, 먼 이동(→ DELETION + INSERTION), 무관한 형상
- `test_step_io.py`: 이름 보존, 어셈블리 인스턴스 배치, 이름 없는 STEP

## 7. 가정과 한계
- 각 STEP 파일의 객체는 의미 있는 단위(Solid/Shell)로 분리되어 있다고 가정합니다.
- 회전 대칭 형상은 대칭 등가 자세를 구분할 수 없으므로(형상적으로 동일) 최소 회전을 보고합니다.
  원통처럼 축대칭이면 회전축 방향(최소 회전)까지만 정확히 판정합니다.
- 비균일 스케일은 객체 자신의 평면 방향에 정렬된 축 기준 스케일(치수 변경)을 대상으로 합니다.
  원통이 타원기둥으로 바뀌는 등 곡면 유형이 바뀌는 경우는 `SHAPE_CHANGE`로 분류됩니다.
- 부분 정합(형상 변경 시 이동·회전)은 스케일 변화가 없다고 가정합니다.
- 내보낸 CAD에 따라 같은 형상이라도 면 분할(seam, 분할 면)이 달라지면 유사 변환 검증이 실패하고
  `SHAPE_CHANGE`로 분류될 수 있습니다.

## 8. 구조
```
step_diff/
  occ.py         OCP 버전 호환 계층 (컬렉션, 다운캐스트, 정밀 AABB, 질량 특성)
  step_io.py     STEP 읽기/쓰기, 어셈블리 전개, 객체 분리, 이름
  features.py    B-Rep 특징, 면 인접 그래프, WL 위상 해시
  transform.py   유사 변환 / 대칭 / 비균일 스케일 / 부분 강체 정합 추정
  compare.py     유사도, 쌍 분석, 변경 유형 분류
  pipeline.py    AABB 필터 → 쌍 분석 → 헝가리안 전역 매칭 → 결과
  report.py      텍스트/JSON/CSV 출력
  samples.py     정답이 알려진 예제 형상 생성기
  __main__.py    CLI
examples/        예제 OLD/NEW STEP, 정답, 보고서
tests/           pytest
```
