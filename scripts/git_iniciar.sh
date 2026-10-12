#!/usr/bin/env bash
# git_iniciar.sh — cria o repositório git local e faz o primeiro commit (corre-se uma só vez).
# Uso, na raiz do projeto:
#   bash scripts/git_iniciar.sh <o teu email noreply do GitHub>
# O email noreply está no GitHub, em Settings → Emails → "Keep my email addresses private"
# (é do tipo 12345678+utilizador@users.noreply.github.com). Liga os commits à tua conta
# sem mostrar o email pessoal.
set -u
cd "${NOVA:-$PWD}" || exit 1
NOME="Paulo Freitas"
EMAIL="${1:-}"

# 0) verificações antes de mexer em alguma coisa
if [[ ! $EMAIL =~ ^([0-9]+\+)?[A-Za-z0-9-]+@users\.noreply\.github\.com$ ]] || [[ $EMAIL == 12345678+utilizador@* ]]; then
  echo "ERRO: falta o teu email noreply do GitHub, ou não tem o formato certo."
  echo "  Uso: bash scripts/git_iniciar.sh 12345678+utilizador@users.noreply.github.com  (com o teu número e utilizador)"
  echo "  Está no GitHub, em Settings → Emails → 'Keep my email addresses private'."
  exit 1
fi
for f in scripts/hooks/pre-commit scripts/git_verificar.sh; do
  [ -f "$f" ] || { echo "ERRO: falta $f (corre primeiro o bloco 1)."; exit 1; }
done
[ -f gitignore.txt ] || [ -f .gitignore ] || { echo "ERRO: falta o gitignore.txt na raiz do projeto."; exit 1; }
if [ -d .git ]; then
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERRO: o git não aceita esta pasta:"
    git rev-parse --is-inside-work-tree 2>&1 | head -n 2
    exit 1
  fi
  if git rev-parse -q --verify HEAD >/dev/null; then
    echo "ERRO: este repositório já tem $(git rev-list --count HEAD) commit(s). Não mexi em nada: cola esta saída no chat."
    exit 1
  fi
fi

# 1) ficheiros de regras com o nome final
[ -f gitignore.txt ] && mv -f gitignore.txt .gitignore
[ -f gitattributes.txt ] && mv -f gitattributes.txt .gitattributes
chmod +x scripts/hooks/pre-commit scripts/git_verificar.sh scripts/git_iniciar.sh

# 2) repositório, ramo main, autor (só neste repositório) e verificação automática
[ -d .git ] || git init -q
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERRO: o git recusa usar esta pasta:"
  git rev-parse --is-inside-work-tree 2>&1 | head -n 4 | sed 's/^/  /'
  echo "Se fala em 'dubious ownership', corre o comando 'git config --global --add safe.directory …' que a mensagem sugere"
  echo "e volta a correr este script. Noutro caso, cola esta saída no chat."
  exit 1
fi
git symbolic-ref HEAD refs/heads/main
git config user.name "$NOME"
git config user.email "$EMAIL"
git config core.hooksPath scripts/hooks

# a verificação automática tem de conseguir correr (o projeto está noutro disco, em /mnt)
scripts/hooks/pre-commit >/dev/null 2>&1
rc=$?
if [ "$rc" -ge 126 ]; then
  echo "ERRO: o sistema não deixa correr a verificação automática (scripts/hooks/pre-commit, código $rc)."
  echo "Sem ela, os commits não são verificados. Cola esta saída no chat."
  exit 1
fi

# 3) ambiente conda, para o recriar noutra máquina
if command -v conda >/dev/null 2>&1; then
  conda env export --no-builds | grep -v '^prefix:' > environment.yml
else
  echo "AVISO: não encontrei o conda; o environment.yml não foi criado (faltou o source …/inicio.sh?)."
fi

# 4) mostrar o que vai entrar e pedir confirmação
echo "== Vai entrar no primeiro commit (quantos, pasta) =="
git -c core.quotePath=false ls-files -o --exclude-standard \
  | awk -F/ '{print "  " (NF > 1 ? $1 "/" : "(raiz) " $1)}' | sort | uniq -c
read -r -p "Fazer o primeiro commit com estes ficheiros? [s/N] " resposta
if [[ ! $resposta =~ ^[sS]$ ]]; then
  echo "Cancelado: nada entrou no histórico. Podes voltar a correr este script."
  exit 1
fi

# 5) primeiro commit e tag do marco
git add -A
if git commit -q -m "Estado inicial: código da Etapa 0 (dados, conjuntos, anotação, avaliador v1.0)"; then
  git tag -a etapa0 -m "Etapa 0 concluída (5–10 out 2026)"
  git log --oneline --decorate -n 1
  echo "autor: $(git log -1 --format='%an <%ae>')"
else
  echo "O commit não foi feito (a razão está acima). Cola esta saída no chat."
  exit 1
fi
