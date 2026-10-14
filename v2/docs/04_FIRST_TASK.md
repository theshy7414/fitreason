# 04 — 第一個可執行任務（M0）：Scaffold + 兩個 DB + tiny seed + `/health` + `/evidence`

目標：從空資料夾做到「`make up && make seed && make test` 全綠」。預估 **4–6 小時**。本文件只描述任務；**尚未實作**。

## 環境確認（2026-10-12 於本機實測）

| 需求 | 實測 | 對本任務的影響 |
|---|---|---|
| Docker 29.1.3 + Compose 2.40.3，daemon 正常，使用者在 docker 群組 | ✅ | 可直接 `docker compose up` |
| Python 3.10.12（系統）、uv 0.12.0、pip 26.2.1 | ✅ | 用 `uv` 建 venv：`uv venv --python 3.12`（uv 會自行下載 3.12；若離線就用 3.10，程式碼不要用 3.11+ 語法） |
| Node / npm | ❌ 未安裝 | **本任務不需要**；M4 前：`curl -fsSL https://fnm.vercel.app/install \| bash && fnm install --lts` |
| 連接埠 7474/7687/5432/8000 | 空閒 | 不需改 port |
| 記憶體 15 GB | ✅ | Neo4j heap 限 512m 即可 |

## Step 0 — 專案骨架（15 分鐘）

```bash
cd /home/chi/outfitmatch
mkdir -p api/app/stores api/tests db/neo4j/seed db/postgres scripts
cat > .gitignore <<'EOF'
.venv/
__pycache__/
.env
data/raw/
node_modules/
*.pt
.pytest_cache/
EOF
cat > .env.example <<'EOF'
NEO4J_PASSWORD=outfitmatch_dev_pw
POSTGRES_PASSWORD=outfitmatch_dev_pw
EOF
cp .env.example .env
uv venv --python 3.12 && source .venv/bin/activate
cat > pyproject.toml <<'EOF'
[project]
name = "outfitmatch-v2"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = ["fastapi>=0.115", "uvicorn[standard]>=0.30", "neo4j>=5.26", "psycopg[binary]>=3.2", "pydantic-settings>=2.4"]
[project.optional-dependencies]
dev = ["pytest>=8", "httpx>=0.27"]
[tool.pytest.ini_options]
testpaths = ["api/tests"]
EOF
uv pip install -e ".[dev]"
```

## Step 1 — `docker-compose.yml`（Neo4j + PostgreSQL，含 healthcheck）

```yaml
services:
  neo4j:
    image: neo4j:5.26-community
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_server_memory_heap_max__size: 512m
    ports: ["7474:7474", "7687:7687"]
    volumes: [neo4j_data:/data]
    healthcheck:
      test: ["CMD-SHELL", "cypher-shell -u neo4j -p ${NEO4J_PASSWORD} 'RETURN 1' || exit 1"]
      interval: 5s
      timeout: 5s
      retries: 20
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: outfit
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: outfitmatch
    ports: ["5432:5432"]
    volumes: [pg_data:/var/lib/postgresql/data]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U outfit -d outfitmatch"]
      interval: 5s
      timeout: 5s
      retries: 20
volumes: { neo4j_data: {}, pg_data: {} }
```

`Makefile`：

```make
up:    ; docker compose up -d --wait
down:  ; docker compose down            # 不加 -v，保留 volume
seed:  ; .venv/bin/python scripts/seed.py
api:   ; .venv/bin/uvicorn api.app.main:app --reload --port 8000
test:  ; .venv/bin/pytest -q
```

## Step 2 — tiny seed dataset（合成，無圖片、無授權疑慮）

`db/neo4j/seed/items.csv`（`image_uri` 先給 placeholder，`license=synthetic`）：

```csv
id,name,category,image_uri,source,license,attribution
top_01,White tee,top,placeholder://top_01,synthetic,CC0-synthetic,none
top_02,Navy shirt,top,placeholder://top_02,synthetic,CC0-synthetic,none
top_03,Grey hoodie,top,placeholder://top_03,synthetic,CC0-synthetic,none
bottom_01,Blue jeans,bottom,placeholder://bottom_01,synthetic,CC0-synthetic,none
bottom_02,Black chinos,bottom,placeholder://bottom_02,synthetic,CC0-synthetic,none
bottom_03,Grey joggers,bottom,placeholder://bottom_03,synthetic,CC0-synthetic,none
shoes_01,White sneakers,shoes,placeholder://shoes_01,synthetic,CC0-synthetic,none
shoes_02,Brown loafers,shoes,placeholder://shoes_02,synthetic,CC0-synthetic,none
shoes_03,Running shoes,shoes,placeholder://shoes_03,synthetic,CC0-synthetic,none
outer_01,Beige trench,outerwear,placeholder://outer_01,synthetic,CC0-synthetic,none
outer_02,Denim jacket,outerwear,placeholder://outer_02,synthetic,CC0-synthetic,none
bag_01,Black backpack,bag,placeholder://bag_01,synthetic,CC0-synthetic,none
```

`db/neo4j/seed/outfits.csv`：

```csv
outfit_id,item_id
o1,top_01
o1,bottom_01
o1,shoes_01
o2,top_01
o2,bottom_02
o2,shoes_02
o2,outer_01
o3,top_02
o3,bottom_01
o3,shoes_02
o4,top_03
o4,bottom_03
o4,shoes_03
o4,bag_01
```

## Step 3 — schema + `scripts/seed.py`

- `db/neo4j/schema.cypher`：`02_V2_ARCHITECTURE.md` §3 的 constraints（先不建 vector index，M2 再建）。
- `db/postgres/schema.sql`：§4 的 `category / item / outfit / outfit_item`（先不建 `embedding` 欄位與 `compat_score`）。
- `scripts/seed.py`（冪等）：讀兩份 CSV → Neo4j 用 `MERGE`（Item、Category、Outfit、`IN_CATEGORY`、`CONTAINS`）；PostgreSQL 用 `INSERT ... ON CONFLICT DO NOTHING`。重跑不得產生重複資料。

## Step 4 — FastAPI（兩個 endpoint）

`api/app/main.py` 行為規格：

- `GET /api/v1/health` → `200 {"status":"ok","neo4j":"ok","postgres":"ok"}`；任一 DB 連不上 → `503`，且 body 標明哪個壞了。
- `GET /api/v1/evidence?a=<item_id>&b=<item_id>&store=neo4j|pg`（預設 neo4j）→
  `200 {"a":..,"b":..,"store":..,"co_outfits":[..outfit_id..],"bridges":[{"item_id":..,"support":int}], "elapsed_ms": float}`。
  - `co_outfits` = `02_V2_ARCHITECTURE.md` E1；`bridges` = E2（兩 store 各自實作同一個查詢，排序：support desc、item_id asc，以保證兩邊結果可比）。
  - 找不到 item → `404 {"error":{"code":"item_not_found",...}}`。
- `api/app/stores/neo4j_store.py` 與 `pg_store.py` 實作同一個 `EvidenceStore` Protocol（`co_outfits(a,b)`、`bridges(a,b)`、`ping()`），之後 M3/M5 直接擴充。

## Step 5 — 驗收測試（`api/tests/test_evidence.py`，必須全過）

依 seed 資料手算的期望值（已用 E1/E2 逐步推導）：

| 測試 | 輸入 (a, b) | 期望 `co_outfits` | 期望 `bridges` |
|---|---|---|---|
| T1 直接證據 | `top_01`, `bottom_01` | `["o1"]` | （任意，不檢查） |
| T2 多件同 outfit | `top_01`, `shoes_02` | `["o2"]` | `[{"item_id":"bottom_01","support":1}]` |
| T3 只有間接證據 | `top_02`, `bottom_02` | `[]` | `[{"item_id":"shoes_02","support":1}]` |
| T4 完全無關 | `top_01`, `bag_01` | `[]` | `[]` |
| T5 兩 store 一致 | 上述 4 組 | neo4j 與 pg 的 `co_outfits`、`bridges` **完全相同** | 同左 |
| T6 health | — | `status=="ok"` 且 `neo4j=="ok"`、`postgres=="ok"` | — |
| T7 404 | `a=nope` | HTTP 404、`error.code=="item_not_found"` | — |
| T8 冪等 | 連跑 `make seed` 兩次 | Neo4j `MATCH (n) RETURN count(n)` 與 PG `select count(*) from outfit_item` 在兩次之後數值不變（節點 = 12 items + 5 categories + 4 outfits = 21；`outfit_item` = 14 列） | — |

（T3 說明：`top_02` 與 `bottom_02` 沒有共同 outfit；`top_02` 的夥伴 `shoes_02` 同時出現在 `o2`（含 `bottom_02`），所以 `shoes_02` 是橋接單品。）

## 完成定義（DoD）

```bash
cd /home/chi/outfitmatch
make up && make seed && make test          # 全綠
curl -s localhost:8000/api/v1/health       # {"status":"ok","neo4j":"ok","postgres":"ok"}
curl -s 'localhost:8000/api/v1/evidence?a=top_01&b=shoes_02'
git add -A && git commit -m "M0: scaffold, compose, seed, health+evidence"
git tag m0
```

並確認：`docker compose down && docker compose up -d --wait` 後資料仍在（volume 保留）；從乾淨 clone 重跑一次指令序列成功。

## 不在本任務內（刻意）

ML、CLIP、圖片上傳、React、vector index、bench、Fashionpedia。做完 M0 才進 M1。
