rm -rf readme_history
mkdir -p readme_history

i=1
git log --reverse --format=%H -- readme.md README.md | while IFS= read -r h; do
  if git cat-file -e "${h}:README.md" 2>/dev/null; then
    git show "${h}:README.md" > "readme_history/readme${i}.md"
    i=$((i+1))
  elif git cat-file -e "${h}:readme.md" 2>/dev/null; then
    git show "${h}:readme.md" > "readme_history/readme${i}.md"
    i=$((i+1))
  else
    echo "SKIP (no README at) $h"
  fi
done
