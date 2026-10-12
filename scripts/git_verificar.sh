#!/usr/bin/env bash
# git_verificar.sh — mostra o que entraria no git, SEM mexer na pasta do projeto.
# Uso, na raiz do projeto:
#   bash scripts/git_verificar.sh            (usa gitignore.txt ou, se não existir, .gitignore)
#   bash scripts/git_verificar.sh outro.txt  (testa outro ficheiro de regras)
# Só lê. O teste usa um repositório git temporário em /tmp, apagado no fim.
set -u
cd "${NOVA:-$PWD}" || exit 1

GI="${1:-}"
if [ -z "$GI" ]; then
  if [ -f gitignore.txt ]; then GI=gitignore.txt; else GI=.gitignore; fi
fi
[ -f "$GI" ] || { echo "ERRO: não encontro $GI em $PWD"; exit 1; }
echo "raiz: $PWD | regras: $GI"

# o mesmo critério do hook scripts/hooks/pre-commit
PERMITIDO='\.(py|sh|md|ya?ml|json|txt|toml|cfg|ini|html|js|css|tex|bib|cff)$|(^|/)(\.gitignore|\.gitattributes|pre-commit|LICENSE)$'
EXT_DADOS='\.(jpe?g|png|bmp|tiff?|gif|webp|mp4|avi|mov|mkv|csv|tsv|xlsx?|parquet|feather|db|sqlite3?|docx?|pdf|pptx|ipynb|npy|npz|pt|pth|ckpt|pkl|joblib|h5|hdf5|safetensors|onnx|bin|zip|tar|gz|7z|log)$'

echo "== 1. git atual =="
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "repositório: $(git rev-parse --show-toplevel)"
  echo "ramo: $(git symbolic-ref --short HEAD 2>/dev/null || echo '?') | commits: $(git rev-list --count HEAD 2>/dev/null || echo 0) | ficheiros já seguidos: $(git ls-files | wc -l)"
  em=$(git config user.email)
  if [[ $em =~ @users\.noreply\.github\.com$ ]]; then em="noreply do GitHub (ok)"
  elif [ -n "$em" ]; then em="NÃO é o noreply: o email apareceria nos commits"
  else em="(nenhum)"; fi
  echo "autor: $(git config user.name || echo '(sem nome)') | email: $em | hooks: $(git config core.hooksPath || echo '(não ativos)')"
  echo "já seguidos com extensão de dados: $(git ls-files | grep -Eic "$EXT_DADOS")"
  git ls-files | grep -Ei "$EXT_DADOS" | head -n 5 | sed 's/^/  /'
else
  echo "sem repositório git:"; git rev-parse --is-inside-work-tree 2>&1 | head -n 2 | sed 's/^/  /'
fi
if [ "$GI" != .gitignore ] && [ -f .gitignore ]; then
  echo "ATENÇÃO: já existe um .gitignore; o teste usa os dois juntos"
fi

echo "== 2. raiz do projeto (pastas acabam em /) =="
ls -Ap | paste -sd' ' | fold -s -w 110 | sed 's/^/  /'
find . -maxdepth 1 -type l -printf '  ligação: %p -> %l\n'

echo "== 3. o que ENTRARIA no git =="
T=$(mktemp -d) || exit 1
trap 'rm -rf "$T"' EXIT
git -c init.defaultBranch=main init -q "$T/r"
mkdir -p "$T/r/.git/info"
cp "$GI" "$T/r/.git/info/exclude"
G() { git -c core.quotePath=false --git-dir="$T/r/.git" --work-tree="$PWD" "$@"; }

G ls-files -o --exclude-standard > "$T/entra.txt"
bytes=$(G ls-files -o --exclude-standard -z | xargs -0 -r du -b | awk '{s+=$1} END {print s+0}')
echo "total: $(wc -l < "$T/entra.txt") ficheiros, $((bytes / 1024)) KB"
awk -F/ '{print (NF > 1 ? "  " $1 "/" : "  (raiz) " $1)}' "$T/entra.txt" | sort | uniq -c
echo "-- não é código nem texto (deve vir vazio) --"
grep -Eiv "$PERMITIDO" "$T/entra.txt" | head -n 20 | sed 's/^/  /'
echo "-- 5 maiores (bytes) --"
G ls-files -o --exclude-standard -z | xargs -0 -r du -b | sort -n | tail -n 5 | sed 's/^/  /'

echo "-- endereços de email nesses ficheiros (quantos, ficheiro, só o domínio) --"
G ls-files -o --exclude-standard -z \
  | xargs -0 -r grep -HIoE '[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}' 2>/dev/null \
  | awk -F: '{n = split($NF, e, "@"); print "  " $1 "  @" e[n]}' | sort | uniq -c

echo "== 4. dentro das pastas que entram, fica DE FORA (quantos, pasta, extensão) =="
mapfile -t PASTAS < <(grep -E '^!/[^*]+/$' "$GI" | sed 's#^!/##; s#/$##')
G ls-files -o -i --exclude-standard -- "${PASTAS[@]}" \
  | awk -F/ '{n = split($NF, a, "."); print "  " $1 "/ ." (n > 1 ? a[n] : "(sem extensão)")}' \
  | sort | uniq -c

echo "== 5. sets/CAL.json: só a estrutura (chaves e tipos, sem valores) =="
python - <<'PY'
import json, pathlib
p = pathlib.Path("sets/CAL.json")
if not p.exists():
    print("  (não existe)")
else:
    d = json.loads(p.read_text(encoding="utf-8"))
    def forma(x):
        if isinstance(x, dict):
            itens = list(x.items())
            s = ", ".join(f"{k}: {forma(v)}" for k, v in itens[:15])
            return "{" + s + (", …" if len(itens) > 15 else "") + "}"
        if isinstance(x, list):
            return f"[{len(x)} × {forma(x[0]) if x else 'vazia'}]"
        return type(x).__name__
    print(" ", forma(d))
PY
