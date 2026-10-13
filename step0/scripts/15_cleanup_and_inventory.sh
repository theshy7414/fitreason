#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
rm -f docs/LICENSE   # 14-byte "404: Not Found" stub from a wrong URL (the real files are LICENSE.CC / LICENSE.MIT)
cat > .gitignore <<'EOF'
# Step 0 experiment workspace: never commit data, features, downloaded packages or local deps
data/
pylibs/
pkg/
*.npz
*.tar.gz
*.json
EOF
echo "--- disk usage ---"
du -sh data pylibs pkg results docs scripts 2>/dev/null
du -sh .
echo "--- tracked-by-nothing, files list (scripts/docs/results) ---"
find scripts docs results -type f | sort
echo "--- v1 repo state (must show only the untracked docs/ draft from the previous round) ---"
cd ~/code/NTU-DBMS-2025-Spring && git status --short && git log --oneline -1
