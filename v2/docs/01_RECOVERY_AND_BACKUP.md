# 復原紀錄（歷史，不要照這份重建目錄）

2026-10-14 已整理：個人專案只留 `/home/chi/outfitmatch`（GitHub `theshy7414/outfitmatch`）。`/home/chi/code/` 與裡面的 v1 復原副本已刪除，那些都是 GitHub 或 Windows 上已有的複本。v1 帳密留在 Windows `Documents/NTU-DBMS-2025-Spring/OutfitMatch`，沒有上傳。下面是 2026-10-12 的原始紀錄。

# 01 — 復原與備份（Phase 1 findings）

稽核時間：2026-10-12 約 22:40（UTC+8）。全程唯讀（`ls`/`find`/`sha256sum -c`/`diff`/`cmp`/`gh` 讀取、`git clone` 到新目錄），沒有刪除、覆蓋、移動任何既有檔案，沒有訓練模型。

## 0. 先看這個：路徑已經變了

- `/home/chi/code/` **整個目錄現在不存在**（不只 `NTU-DBMS-2025-Spring`、`ai`、`ntu-push`）。
- 研究資料夾現在在 **`/home/chi/outfitmatch-research/`**（mtime 22:21），不在 `/home/chi/code/outfitmatch-research/`。`EXPERIMENT2_STATUS_REPORT.md` 內的相對路徑基準請改讀成這個位置。
- 為了放復原內容與 V2，我新建了 `/home/chi/code/`（空的新目錄）底下的 `outfitmatch-v1-recovered/` 與 `outfitmatch-v2/`。
- 我沒有追查是誰/什麼造成移動或刪除。

## 1. 研究資料（outfitmatch-research）狀態：完整

| 項目 | 結果 |
|---|---|
| 大小 / 檔案數 | 43 GB（`data` 42G、`pylibs` 410M、`runs` 177M、`manifests` 14M、`results` 14M…）；12,640 個檔案 |
| 資料 | `step0/data/features_v1`（39 GB：`features_fp16.npy`、`required_fp16.npy`、`item_ids.npy`、shards）、`data/outfits`（2.7 GB）、`data/catalog`、`data/year_labels` 都在 |
| Checkpoints | 49 個 `best.pt`，包含 `step0/runs/phase2/visual_seed0..4/best.pt`、`visual_cat_seed0..4` |
| Scores | `step0/runs/exp2/scores.npz`（4.8 MB）、`main_bootstrap.pkl`、`linear/`（35 個目錄）都在 |
| Scripts / docs / results | `scripts/` 48 檔、`results/` 26 檔、`docs/` 14 檔、`pkg/` 20 檔 |
| 報告 | `EXPERIMENT1_SPEC.md`、`EXPERIMENT2_PREREG.md`、`EXPERIMENT2_STATUS_REPORT.md`（22:21） |
| Manifest 驗證 | `cd step0/manifests && sha256sum -c MANIFEST_SHA256.txt` → **25/25 OK**（唯讀） |
| 關鍵檔 SHA256 | 備份目錄內的 `SHA256_critical.txt`（320 行，路徑相對 `step0/`）對現有研究資料夾逐檔 `sha256sum -c` → **320/320 OK** |

## 2. 備份 `/home/chi/backups/outfitmatch-research-20261012/`：與現有資料一致（`pylibs` 除外）

- 42 GB；`backup.log`：`RSYNC_DONE`、`VERIFY_DONE`。
- 檔案清單（相對路徑 + 位元組大小）比對：除 `pylibs/`（410 MB、11,935 檔，可由 pip 重建，**備份刻意未含**）之外完全相同；備份只多一個 `SHA256_critical.txt`。
- `diff -rq` 對 `runs`、`manifests`、`scripts`、`results`、`docs` → 無差異；`cmp`：`phase2/visual_seed0/best.pt`、`exp2/scores.npz` 相同。
- **未做**：對 39 GB 的 `data/features_v1` 逐位元組 diff（備份當時已做 SHA256 驗證，這次只比清單與大小）。
- **風險**：備份與原始資料在**同一顆磁碟**（`/dev/sdd`，WSL2 ext4 vhdx）。同一個 `rm -rf`、vhdx 損毀、`wsl --unregister` 都會一起消失。**目前等於還是 single point of failure。**

## 3. 建議的備份方案（只列指令，**我沒有執行**；請你核准後再跑）

目標：最小成本做到「3 份、2 種媒介、1 份異地」。研究資料夾我沒有動，所以下面都是提案。

### 3.1 程式 + 文件 + 小型結果 → git，private GitHub remote

`gh auth status`：已登入 `theshy7414`（https，token 有效）。需要進 git 的部分很小（`runs` 177M、`results` 14M、`manifests` 14M、`scripts`/`docs`/`pkg` < 1 MB；沒有單檔超過 20 MB，不需要 Git LFS）。

> **發現**：研究線的 scripts/manifests/results/logs 已經在 GitHub **public** repo `theshy7414/NTU-DBMS-2025-Spring`（commit `b382081`，2026-10-12 22:01）。那是公開的；若你不想公開，要改成 private（`gh repo edit theshy7414/NTU-DBMS-2025-Spring --visibility private --accept-visibility-change-consequences`）— 這是遠端變更，需你同意。

```bash
cd /home/chi/outfitmatch-research
git init -b main
cat > .gitignore <<'EOF'
step0/data/
step0/pylibs/
**/*.npy
**/__pycache__/
EOF
# 注意：checkpoint(.pt) 與 scores.npz 要進版控 → 不要忽略 *.pt；上面的 **/*.npy 不會擋 scores.npz
git add -A && git commit -m "snapshot: scripts, manifests, results, runs (checkpoints+scores)"
gh repo create outfitmatch-research --private --source=. --push      # ← 需要使用者核准
```

`data/features_v1`（39 GB）可由 SHIFT15M 官方來源重新下載，用 `manifests/*.jsonl.gz` 與 `step0/scripts/30_stream_extract_features.py` 重建；不進 git。

### 3.2 全量備份 → Windows 端磁碟（不同磁碟/不同檔案系統）

`/mnt/c`（NTFS，剩 402 GB）可用；`/mnt/d` 是空掛載點（不是可用的磁碟）。

```bash
mkdir -p /mnt/c/Users/szuni/Backups/outfitmatch
# 研究資料（不含 pylibs），用 --link-dest 做每日快照，不覆蓋舊快照
SNAP=/mnt/c/Users/szuni/Backups/outfitmatch/research-$(date +%Y%m%d-%H%M)
LAST=$(ls -d /mnt/c/Users/szuni/Backups/outfitmatch/research-* 2>/dev/null | tail -1)
rsync -a --info=progress2 --exclude 'step0/pylibs/' \
  ${LAST:+--link-dest="$LAST"} /home/chi/outfitmatch-research/ "$SNAP/"
# 驗證
cd "$SNAP/step0" && sha256sum -c /home/chi/backups/outfitmatch-research-20261012/SHA256_critical.txt | grep -vc ': OK'   # 期望 0
```

註：NTFS 經 9p 不支援 hardlink 時 `--link-dest` 會退化成完整複製（42 GB，磁碟夠）；退化也沒關係。

### 3.3 外接硬碟（真正的異地/離線）

插上外接碟（例如掛成 `/mnt/e`，需 `sudo mount -t drvfs E: /mnt/e`）後執行同樣的 `rsync -a --exclude 'step0/pylibs/' /home/chi/outfitmatch-research/ /mnt/e/outfitmatch-research/`。每個實驗里程碑做一次即可。

### 3.4 自動化（systemd 已啟用：`/etc/wsl.conf` 有 `systemd=true`，`systemctl is-system-running`=running）

```bash
mkdir -p ~/.config/systemd/user ~/bin
cat > ~/bin/backup-outfitmatch.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
B=/mnt/c/Users/szuni/Backups/outfitmatch
SNAP=$B/research-$(date +%Y%m%d-%H%M)
LAST=$(ls -d $B/research-* 2>/dev/null | tail -1 || true)
rsync -a --exclude 'step0/pylibs/' ${LAST:+--link-dest="$LAST"} /home/chi/outfitmatch-research/ "$SNAP/"
# 只留最近 7 份：只刪「備份目標」內的舊快照，不碰來源
ls -dt $B/research-* | tail -n +8 | xargs -r rm -rf
EOF
chmod +x ~/bin/backup-outfitmatch.sh
cat > ~/.config/systemd/user/backup-outfitmatch.service <<'EOF'
[Service]
Type=oneshot
ExecStart=%h/bin/backup-outfitmatch.sh
EOF
cat > ~/.config/systemd/user/backup-outfitmatch.timer <<'EOF'
[Timer]
OnCalendar=daily
Persistent=true
[Install]
WantedBy=timers.target
EOF
systemctl --user daemon-reload && systemctl --user enable --now backup-outfitmatch.timer
```

（WSL 關機時 timer 不會跑；`Persistent=true` 會在下次開機補跑。）

### 3.5 防誤刪（`rm -rf`）

```bash
# 1) 現有備份設為唯讀（只改權限，不改內容）
chmod -R a-w /home/chi/backups/outfitmatch-research-20261012
# 2) 關鍵小檔加 immutable（ext4；需 sudo，連 root 也得先 chattr -i 才能刪）
sudo chattr -R +i /home/chi/backups/outfitmatch-research-20261012/step0/runs \
                  /home/chi/backups/outfitmatch-research-20261012/step0/manifests \
                  /home/chi/backups/outfitmatch-research-20261012/SHA256_critical.txt
# 3) 以後清理一律先 dry-run：不要在 shell 設 rm -rf alias；用 trash-cli
sudo apt install -y trash-cli    # 之後用 trash-put 取代 rm
```

### 3.6 優先順序（做哪些就夠）

1. **今天**：3.1（git + private GitHub）— 5 分鐘，保住所有程式與 checkpoint/scores。
2. **今天**：3.2 一次全量 rsync 到 `/mnt/c/...`（約 42 GB）。
3. 之後：3.4 timer、3.5 唯讀/immutable。
4. 里程碑時：3.3 外接碟。
- 不建議為此引入 DVC（資料可重新下載、checkpoint 很小，git 即可）。

## 4. v1（NTU-DBMS-2025-Spring / OutfitMatch v1）復原結果：**完整可復原**

### 4.1 原本的本機路徑

| 路徑 | 狀態 |
|---|---|
| `/home/chi/code/NTU-DBMS-2025-Spring/` | 不存在 |
| `/home/chi/code/ntu-push/`（含 `.git`） | 不存在（整個 `/home/chi/code` 不存在，無法檢查 `.git` 的 packed objects / fsck） |
| `/home/chi/code/ai/` | 不存在；全檔案系統也找不到名稱相符的複本 |
| `~/.local/share/Trash` | 不存在 |
| `~/.cursor/projects/home-chi-code-NTU-DBMS-2025-Spring/` | 只有本次稽核對話；agent-transcripts 只含 22b1452d（本次） |
| `/mnt/c/Users/szuni/.cursor/projects/wsl-localhost-Ubuntu-22-04-home-chi-code-ntu-push/agent-transcripts/4153e349-….jsonl` | 729 KB、22:08。是 Cursor 在 ntu-push 開的**對話紀錄**（可能含當時的檔案內容片段），**未用到、未修改**；備註為最後手段 |
| `~/.cursor-server/data/User/History` | 沒有任何 entry 提到 NTU-DBMS / ntu-push |

### 4.2 實際找到的來源（三個，互相印證）

1. **GitHub `theshy7414/NTU-DBMS-2025-Spring`**（public，唯讀確認）。10 個 commits，最後推送 2026-10-12T14:01:44Z（= 22:01，在刪除之前）：
   - `fbec04ae`（2026-01-05）— **v1 最終版**（Neo4j-centric 架構、`ARCHITECTURE.md`、`OutfitMatch/` 後端、`ui/`）
   - `9f36948b`（2025-12-30）、`68c01f63`（Refactor: Migrate to Neo4j-centric…）、`10b7751b`（final demo (python)）、`57df41d`、`6a91692f`、`1e2ec8a`、`82e517c`、`4817ea3`
   - `b382081`（2026-10-12）— 把 v1 換成研究線檔案（v1 檔案在這個 commit 被刪除，但歷史完整保留）
2. **Windows 端工作副本** `/mnt/c/Users/szuni/Documents/NTU-DBMS-2025-Spring/OutfitMatch/`（含 `.git`，HEAD=`fbec04a`，working tree clean，remote=`theshy7414/DatabaseSystem_FinalProject`）。與 GitHub 的 `fbec04ae` 逐檔相同（忽略 CRLF），另有未追蹤的 `OutfitMatch/config/settings.py` 與 `.env`（含憑證）。
3. **GitHub `Eskartur/OutfitMatch`**（private，使用者有 push 權限；2025-06 的原始組員版，6 commits）。

### 4.3 已複製到新目錄 `/home/chi/code/outfitmatch-v1-recovered/`

- `ntu-dbms-github-clone/`（完整歷史）、`eskartur-outfitmatch-clone/`、`windows-documents-copy/`（排除 `.env`、`venv/`；`settings.py` 已 `chmod 600`）。說明見該目錄 `README_RECOVERY.md`。

### 4.4 Recoverable / Missing 對照

| 項目 | 狀態 | 位置 |
|---|---|---|
| v1 原始碼：`server.py`、`loader/*`（instagram_neo4j、shop_neo4j、shop_postgres）、`database/*`（init_neo4j_schema、build_relationships）、`query/*`、`config/settings.example.py`、`requirements.txt` | ✅ | `ntu-dbms-github-clone`@`fbec04ae`；`windows-documents-copy/OutfitMatch/` |
| `ARCHITECTURE.md`、`README.md` | ✅ | 同上 |
| React UI（`ui/`） | ✅ | 同上 |
| 資料 `queenshop_all_products(.csv / _with_style.csv)`（各約 1,641 筆） | ✅ | `OutfitMatch/data/` |
| 測試圖 `test/images/{cap,pants,top}.jpg` | ✅ | `fbec04ae` 的 tree；Windows 副本也在 |
| 完整 git 歷史（10 commits） | ✅ | GitHub clone |
| Neo4j 資料庫內容（Desktop 實例：`~/.Neo4jDesktop/relate-data/dbmss/dbms-*`） | 🟡 Windows 端有 3 個 DBMS 實例的設定檔，**未檢查/未複製**資料內容（v1 資料可由 CSV + loader 重建） | `/mnt/c/Users/szuni/.Neo4jDesktop/` |
| `.env`、`settings.py`（API keys、DB 密碼） | 🟡 只在 Windows 端；已複製 `settings.py`（600），`.env` 刻意未複製 | Windows `Documents/NTU-DBMS-2025-Spring/OutfitMatch/OutfitMatch/` |
| 刪除前**未推送**的 v1 修改 | ❓ 無法證明沒有。b382081 在 22:01 推送，之後到 22:17 若有改動則遺失；但研究檔（EXPERIMENT*）已搬到 `/home/chi/outfitmatch-research`，且比 GitHub 版新（STATUS_REPORT 22:21） | — |
| `/home/chi/code/ai/` 內容 | ❌ 找不到任何複本，也不知道內容；請你回想是否為重要資料 | — |
| ntu-push 的本機 `.git`（hooks、reflog、stash、未推送分支） | ❌ 已不存在；但 GitHub 上 main 是完整的 | — |
| 研究線在 GitHub 的快照 | ✅ 僅 scripts/manifests/results/logs/figures（無 data、無 `best.pt`、無 `scores.npz`） | GitHub b382081 |

**結論**：v1 程式、資料 CSV、文件、歷史全部可復原；真正缺失的只有 `/home/chi/code/ai/` 的內容與 22:01 之後可能存在的未推送修改。

## 5. v1 的授權/版權提醒（影響 V2 決策）

v1 的資料是爬來的 QueenShop 商品與 Instagram 貼文（`image_url` 指向 `photo.queenshop.com.tw`），**不應在公開作品集重新散布**。V2 不沿用這些圖片；僅沿用概念（Category / Style / Item graph）。v1 repo 本身（GitHub public）內含該 CSV，若在意，可考慮改成 private。

## 6. 環境檢查（讓第一個任務現實可行）

| 工具 | 結果 |
|---|---|
| Docker | `Docker version 29.1.3`；daemon 正常（`docker ps` 可用，使用者在 `docker` 群組） |
| Docker Compose | `v2.40.3` |
| Python | 系統 `3.10.12`；`~/miniconda3` `3.13.5` |
| uv | `0.12.0`（`~/.local/bin/uv`，可自行抓 Python 3.12） |
| pip | `26.2.1` |
| Node / npm | **未安裝**（`node: command not found`）— M4（React）前要裝，見 `04_FIRST_TASK.md` |
| GPU | RTX 3070 Ti（`nvidia-smi` 正常）；系統 python 有 `torch 2.8.0+cu128`、`cuda.is_available()=True` |
| 記憶體 | 15 GB（Neo4j + Postgres + API 同時跑沒問題） |
| 連接埠 | 7474 / 7687 / 5432 / 8000 / 5173 目前都空閒 |
