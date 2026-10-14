# 02 — V2 最小可行架構（MVA）

原則：一個 repo、一個 `docker-compose.yml`、一個 FastAPI 行程、模型在行程內。不拆微服務。

## 1. 元件

| 元件 | 技術 | 責任 |
|---|---|---|
| `api/` | FastAPI + Pydantic v2 + `neo4j` driver + `psycopg[binary]` | recommend / evidence / upload / health；`store=neo4j\|pg` 可切換 |
| `ml/` | PyTorch + `open_clip`（CLIP ViT-B/32）+ 小型 compat head | image → embedding；(items) → compatibility score；訓練腳本與 model card |
| `db/neo4j/` | Neo4j 5.26 community（vector index 需 ≥5.11） | schema、seed、evidence queries |
| `db/postgres/` | PostgreSQL 16 + `pgvector` | 對照 schema、同題 SQL |
| `web/` | React + TypeScript + Vite | Upload / Browse / Recommendations / Evidence / Licenses |
| `bench/` | Python 腳本 | 同一組查詢對 Neo4j / PG 量 p50/p95 |
| `data/` | gitignored；只放 manifest（committed）與圖（不 commit） | 見 §6 |

## 2. 資料與授權決策（先講結論）

> 授權是以 2026-10-12 的網頁條款為準；**無法離線驗證的部分都標明**。我不是律師；公開發佈前請再看一次原始條款。

### 2.1 候選資料來源評估

| 來源 | 授權/條款（查到的） | 能公開展示？ | 能拿來訓練？ | 判斷 |
|---|---|---|---|---|
| **自己拍的衣物照片** | 你擁有版權 | ✅ | ✅ | **公開 demo 的主要圖源** |
| **Fashionpedia**（~46–48k 張，segmentation + attributes） | 標註/ontology：**CC BY 4.0**；**圖片版權不屬於 Fashionpedia**，須遵守 Flickr / Unsplash / Burst / Freestocks / Kaboompics / Pexels 各自條款；使用者自負責任（官方 data_license 頁） | ⚠️ 只能轉載「單張授權允許」者；不建議 repo/demo 內散布 | ⚠️ 見 2.2 | **訓練 compat head 的主資料**（只在本機使用，不散布圖） |
| **Wikimedia Commons** 單件衣物圖 | **逐檔授權**（CC0 / PD / CC BY / CC BY-SA 等），無統一授權 | ✅ 若逐檔篩 CC0/PD/CC BY(-SA) 並附 attribution | ✅（CC0/PD/CC BY 無 ML 限制；BY-SA 需評估 share-alike 對模型的影響，建議只用 CC0/PD/BY） | **公開 demo 的補充圖源**；量少，要手動 curate（~50–150 張） |
| **Fashion-MNIST**（Zalando，28×28 灰階） | **MIT** | ✅ | ✅ | 解析度太低，**只用於 unit test / CI smoke test** |
| **Polyvore Outfits**（Vasileva et al.，68,306 outfits、261,058 items） | 標註標示 CC BY 4.0，但**作者不擁有圖片/metadata 版權**，僅供 research and educational use；HF 上為 **gated，需 institutional email，限 academic / non-profit research**；原 Polyvore 網址已失效 | ❌ | ❌ 不適合（作品集/產品不是明確的 academic non-profit research） | **不用**。若將來要做學術比較，需自己申請並遵守條款，與產品線分離 |
| **DeepFashion** | **僅限 non-commercial research**；禁止複製/散布圖片及衍生資料；需簽協議 | ❌ | ❌ | 不用 |
| **Fashion-Gen** | 查不到明確資料授權（論文僅說明供研究），**無法驗證** | ❌ | ❌ | 不用（無法驗證 = 不用） |
| **iMaterialist (Fashion)** | GitHub repo 為 MIT，但圖片由 Wish 提供，**圖片授權未說明** | ❌ | ❌ | 不用（Fashionpedia 即其 2019/2020 後繼，授權說明較清楚） |
| **H&M Kaggle** | 官方競賽頁沒有給出明確可再散布授權；受 Kaggle 競賽規則約束（我無法讀取規則頁全文：回傳 429） | ❌ | ❌ | 不用（無法驗證） |
| **Amazon Reviews 2023** | 維護者明說「無法指定授權」，僅為研究目的、須遵守倫理與法規；圖片是 Amazon 的 URL，未授權下載/再散布 | ❌ | ❌ | 不用 |
| **Unsplash** | 授權明文**禁止**用於 ML/AI 資料集；官方 Dataset 方案：Lite 可內部商用訓練但**不可散布**，Full 僅非商用 | ❌（不可當資料集散布） | ❌（自行爬取） | 不爬。只有 Fashionpedia 內來自 Unsplash 的圖要**排除** |
| **Pexels** | ToS 明文禁止為 ML 目的 scraping / 用 API 蒐集建資料集；禁止 standalone 散布 | ❌ | ❌ | 不爬。Fashionpedia 內來自 Pexels 的圖要**排除** |
| v1 的 QueenShop / Instagram 資料 | 爬取的第三方版權圖 | ❌ | ❌ | 不沿用 |

### 2.2 決策

1. **公開 demo 目錄（Item 圖）**：自己拍的照片（flat-lay，白底，~60–100 件，6 個 category）+ 逐檔確認的 Wikimedia Commons（CC0/PD/CC BY）。`items.license`、`items.source_url`、`items.attribution` 為**必填欄位**，缺值不能入庫；前端有 Licenses 頁列出全部 attribution。
2. **訓練資料（只在本機，不散布圖與衍生資料集）**：Fashionpedia。
   - 先用 `images[].license` 與 `original_url` 過濾：**排除** Unsplash / Pexels 來源（其條款禁止 ML 用途）；**保守版**只留 Flickr 上授權為 CC BY / CC BY-SA / CC0 / PD 的圖，排除 NC / ND。
   - 過濾後剩多少張**現在無法離線確認**（M2 第一步先算；若 < 8k 張，head 很小仍可訓練，並在 model card 註明）。
   - 發佈到 repo 的只有：manifest（image id、license id、attribution）、訓練腳本、**小型 head 權重**（~1–2 MB）、model card。CLIP 影像 embedding 不 commit。
   - 風險聲明（寫進 model card）：Flickr CC 授權對 ML 訓練的適用性有法律灰色地帶，本專案屬非商業作品集、不散布圖；若要商業化需重新評估。
3. **「Outfit」的定義**（因為沒有 stylist 策展資料）：Fashionpedia 街拍/紅毯/秀場照片中，**同一張圖裡同一個人穿著的單品**視為一套 outfit（以 segmentation 標註裁出單品）。這是「真實穿搭共現」而非「專家認定搭配良好」——**在 README 與 UI 要誠實寫明**。
   - 正例：同一 outfit 內的單品對 / 缺一件 fill-in-the-blank。
   - 負例：同 category 的其他圖單品（hard negative：同 category；easy：隨機）。
4. 排除 SHIFT15M（無原始圖片）、Polyvore、DeepFashion、Fashion-Gen、H&M、Amazon、Unsplash/Pexels 爬取。

### 2.3 模型整合

```
image ──► CLIP ViT-B/32 image encoder (frozen, 512-d, L2-norm) ──► e_i
                                                                   │
pair head: input = concat(e_a, e_b, e_a⊙e_b, onehot(cat_a), onehot(cat_b)) ──► MLP(→256→1, dropout 0.3)
FITB: score(query items, candidate c) = mean_i pair(i, c)  → softmax over K candidates (CE loss)
```

- 重用研究線的**架構與訓練配方**（hidden 256 / dropout 0.3 / AdamW lr 3e-4 wd 1e-2 / batch 256 / early stop on val NLL）；**不重用 SHIFT15M 特徵或權重**。
- 評估：同 image 不跨 split（避免洩漏）；FITB accuracy（K=4，隨機 25%）、pairwise AUC、對照基線（只用 category prior、CLIP cosine similarity）。**必須勝過 CLIP-cosine 與 category-prior 基線才算「真的有 compatibility」**。
- `confidence` = 單模型 MSP。對**使用者上傳圖**（與訓練分佈不同）回傳 `ood_warning`（與訓練 embedding 質心的 cosine 距離高於 val 的 95 百分位時為 true）。**不宣稱校準**，因研究線顯示分佈變動下可靠度下降。
- 推論成本：CLIP ViT-B/32 on RTX 3070 Ti 單張 ≈ 數 ms；CPU 亦可（~100–300 ms/張，未實測）。

## 3. Neo4j schema 草案

```cypher
// constraints
CREATE CONSTRAINT item_id    IF NOT EXISTS FOR (i:Item)     REQUIRE i.id IS UNIQUE;
CREATE CONSTRAINT outfit_id  IF NOT EXISTS FOR (o:Outfit)   REQUIRE o.id IS UNIQUE;
CREATE CONSTRAINT cat_name   IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT user_id    IF NOT EXISTS FOR (u:User)     REQUIRE u.id IS UNIQUE;

// vector index（512-d CLIP）
CREATE VECTOR INDEX item_embedding IF NOT EXISTS
FOR (i:Item) ON (i.embedding)
OPTIONS { indexConfig: { `vector.dimensions`: 512, `vector.similarity_function`: 'cosine' } };

// nodes
(:Item    {id, name, image_uri, embedding, source, source_url, license, attribution, origin})  // origin: 'seed'|'fashionpedia'|'upload'
(:Category{name, group})            // group: top|bottom|outerwear|shoes|bag|accessory
(:Outfit  {id, source, license})
(:User    {id, created_at})
(:Style   {name})                   // 可選輔助標籤

// relationships
(:Item)-[:IN_CATEGORY]->(:Category)
(:Outfit)-[:CONTAINS {slot}]->(:Item)
(:User)-[:UPLOADED]->(:Item)
(:Item)-[:HAS_STYLE {confidence}]->(:Style)                       // 可選
(:Item)-[:COMPATIBLE_WITH {score, model_version, rank}]->(:Item)  // 離線物化 top-K（每件 K=20）；線上以 model 現算也可
```

### Evidence queries（草案）

```cypher
// E1: 兩件單品共同出現的 outfit（直接證據）
MATCH (a:Item {id:$a})<-[:CONTAINS]-(o:Outfit)-[:CONTAINS]->(b:Item {id:$b})
RETURN o.id AS outfit, o.source AS source
ORDER BY outfit LIMIT 20;

// E2: 沒有直接共現時，用「共同鄰居」做間接證據（a 的 outfit 夥伴，與 b 的 outfit 夥伴重疊）
MATCH (a:Item {id:$a})<-[:CONTAINS]-(:Outfit)-[:CONTAINS]->(m:Item)<-[:CONTAINS]-(:Outfit)-[:CONTAINS]->(b:Item {id:$b})
WHERE m <> a AND m <> b
RETURN m.id AS bridge, count(*) AS support
ORDER BY support DESC LIMIT 10;

// E3: 針對 query item 在目標 category 的候選：模型分數 + 共現次數（混合排名的 evidence 欄位）
MATCH (a:Item {id:$a})-[r:COMPATIBLE_WITH]->(c:Item)-[:IN_CATEGORY]->(:Category {name:$cat})
OPTIONAL MATCH (a)<-[:CONTAINS]-(o:Outfit)-[:CONTAINS]->(c)
RETURN c.id AS candidate, r.score AS model_score, count(DISTINCT o) AS co_outfits
ORDER BY model_score DESC LIMIT $k;

// E4: 向量近鄰（相似單品，用來解釋「長得像 X，而 X 常和 b 一起出現」）
CALL db.index.vector.queryNodes('item_embedding', $k, $embedding) YIELD node, score
RETURN node.id, score;
```

## 4. PostgreSQL schema 草案（對照組）

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE category (name text PRIMARY KEY, grp text NOT NULL);
CREATE TABLE item (
  id text PRIMARY KEY, name text, image_uri text NOT NULL,
  category text NOT NULL REFERENCES category(name),
  embedding vector(512),
  origin text NOT NULL CHECK (origin IN ('seed','fashionpedia','upload')),
  source text NOT NULL, source_url text, license text NOT NULL, attribution text NOT NULL
);
CREATE TABLE outfit (id text PRIMARY KEY, source text NOT NULL, license text);
CREATE TABLE outfit_item (
  outfit_id text REFERENCES outfit(id), item_id text REFERENCES item(id), slot text,
  PRIMARY KEY (outfit_id, item_id)
);
CREATE INDEX outfit_item_item ON outfit_item(item_id);
CREATE TABLE compat_score (
  a text REFERENCES item(id), b text REFERENCES item(id),
  score real NOT NULL, model_version text NOT NULL, rank int,
  PRIMARY KEY (a, b)
);
CREATE INDEX item_embedding_hnsw ON item USING hnsw (embedding vector_cosine_ops);

-- E1 等價
SELECT oi1.outfit_id FROM outfit_item oi1 JOIN outfit_item oi2 USING (outfit_id)
WHERE oi1.item_id = $1 AND oi2.item_id = $2 LIMIT 20;

-- E2 等價（共同鄰居，2 跳）
SELECT m.item_id AS bridge, count(*) AS support
FROM outfit_item a JOIN outfit_item m ON m.outfit_id = a.outfit_id AND m.item_id <> a.item_id
JOIN outfit_item m2 ON m2.item_id = m.item_id AND m2.outfit_id <> m.outfit_id
JOIN outfit_item b ON b.outfit_id = m2.outfit_id AND b.item_id = $2
WHERE a.item_id = $1 AND m.item_id <> $2
GROUP BY m.item_id ORDER BY support DESC LIMIT 10;
```

**比較重點**（`bench/`）：E1（1 跳）、E2（2 跳）、E2 擴成 3 跳（Cypher 一行 vs SQL 再 join 一層 / recursive CTE）、向量 top-K（Neo4j vector index vs pgvector HNSW）。規模 1k / 10k / 100k outfits（用合成 outfit，和真實資料分開；報告需標明「合成」）。指標：p50 / p95 latency、查詢行數、寫法複雜度。**預期**：1 跳差不多，3 跳 Neo4j 較有優勢；若實測不是如此，如實寫。

## 5. API 清單（FastAPI，`/api/v1`）

| Method | Path | 說明 |
|---|---|---|
| GET | `/health` | `{status, neo4j, postgres, model_loaded, model_version}` |
| GET | `/categories` | category 清單 |
| GET | `/items?category=&origin=&limit=&offset=` | 瀏覽 |
| GET | `/items/{id}` | 單品 + license/attribution |
| POST | `/items` | multipart 上傳圖片 + `category`；回傳 item（含 embedding 已寫入） |
| POST | `/recommend` | body `{item_ids:[...], target_category, k=10, store:"neo4j"\|"pg", ensemble:false}` → `[{item, score, confidence, ood_warning, evidence:{co_outfits, bridges}}]` |
| POST | `/score` | body `{item_ids:[...]}` → 這組單品的 compatibility 分數 |
| GET | `/evidence?a=&b=&store=` | E1 + E2 結果（outfit 列表、bridge 單品） |
| GET | `/outfits/{id}` | outfit 內容 |
| GET | `/licenses` | 所有圖源與 attribution（前端 Licenses 頁） |
| GET | `/dev/bench/evidence?store=&hops=` | 僅 dev：單次查詢計時 |

錯誤格式：`{"error": {"code": "...", "message": "..."}}`。上傳限制：JPEG/PNG、≤ 5 MB、最長邊縮到 512 px、去 EXIF。

## 6. ML scoring service 介面（`ml/scoring.py`）

```python
from typing import Protocol, Sequence
import numpy as np

class Encoder(Protocol):
    dim: int                                   # 512
    def encode(self, images: Sequence["PIL.Image.Image"]) -> np.ndarray: ...   # (N, dim) L2-normalised

class CompatScorer(Protocol):
    model_version: str                         # e.g. "head-v1+clip-vit-b32"
    def score_pair(self, a_emb: np.ndarray, a_cat: str, b_emb: np.ndarray, b_cat: str) -> float: ...
    def rank(self, query: Sequence[tuple[np.ndarray, str]],
             candidates: Sequence[tuple[str, np.ndarray, str]], k: int) -> list["Scored"]: ...

class Scored(TypedDict):
    item_id: str
    score: float          # logit
    confidence: float     # MSP over the candidate set (single model)
    ood_warning: bool
```

- 單一模型；`ensemble=true` 載入 5 個 seed 的 head 取 mean-prob（可選，MVP 不做）。
- 離線 job `ml/materialize.py`：對所有 item 兩兩（同 category 排除）算分，寫 `COMPATIBLE_WITH` / `compat_score`（top-20/item）。

## 7. React 畫面

1. **Upload**：拖放圖片 → 選 category → 預覽 → 送出。
2. **Browse / Closet**：以 category 篩選的格狀清單；選取 1–N 件作為「query outfit」。
3. **Recommendations**：選目標 category（例：已選上衣+褲子 → 找鞋子）→ 卡片（score、confidence 條、OOD 警示 badge、「為什麼」按鈕）。
4. **Evidence drawer / graph**：顯示共同出現的 outfit（縮圖列）、bridge 單品、圖視覺化（`react-force-graph` 或 `cytoscape`，節點 ≤ 30）。
5. **Store toggle + latency**：頂部開關 Neo4j / PostgreSQL，顯示該次查詢 ms（對應 bench）。
6. **Licenses**：所有圖片的來源/授權/attribution 表；About 頁寫明「outfit = 同一張照片中穿在同一人身上的單品」。

## 8. 目錄結構（目標）

```
outfitmatch-v2/
  docker-compose.yml  .env.example  Makefile  pyproject.toml  .gitignore
  api/        app/{main.py, settings.py, deps.py, routers/, stores/{neo4j_store.py,pg_store.py}}  tests/
  ml/         encoder.py scoring.py train.py materialize.py  MODEL_CARD.md
  db/neo4j/   schema.cypher  seed/{items.csv,outfits.csv}  queries.cypher
  db/postgres/ schema.sql
  web/        (Vite React)
  bench/      run_bench.py  RESULTS.md
  data/       manifests/ (committed)   raw/ (gitignored)
  docs/
```
