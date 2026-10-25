#!/usr/bin/env bash
# Run on the authorized host after the local branch has been pushed.
set -euo pipefail
commit="${1:?exact pushed commit required}"
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || exit 2
repo=https://github.com/H-Dw/EvoMolSteer.git
branch=codex/uniguide-flowr-agent
destination=/workspace/EvoMolSteer
cache=/workspace/uniguide_git_cache
mkdir -p "$cache"
export GIT_TERMINAL_PROMPT=0
prefixes=('' 'https://ghfast.top/' 'https://gh-proxy.com/' 'https://ghproxy.net/' 'https://gh.llkk.cc/')
selected=''
for i in "${!prefixes[@]}"; do
    bare="$cache/${commit}_${i}.git"
    log="$cache/${commit}_${i}.log"
    if [[ -d "$bare" ]] && git --git-dir="$bare" cat-file -e "$commit^{commit}" 2>/dev/null; then
        selected="$bare"; break
    fi
    # Never reuse a partial failed clone or forward a credential to a mirror.
    if [[ -e "$bare" ]]; then bare="${bare}_$(date +%s)"; fi
    if timeout 120 git -c credential.helper= -c http.extraheader= -c core.askPass= \
        clone --bare --filter=blob:none --single-branch --branch "$branch" \
        "${prefixes[$i]}${repo}" "$bare" >"$log" 2>&1; then
        git --git-dir="$bare" cat-file -e "$commit^{commit}"
        git --git-dir="$bare" merge-base --is-ancestor "$commit" "refs/heads/$branch"
        selected="$bare"; break
    fi
    if grep -Eiq 'authentication failed|permission denied|HTTP[^0-9]*40[13]|repository not found' "$log"; then
        echo "Authorization failure; mirrors are not an authentication workaround" >&2; exit 3
    fi
done
[[ -n "$selected" ]] || { echo 'Direct GitHub and all four mirrors exhausted' >&2; exit 4; }
if [[ ! -e "$destination" ]]; then
    git clone --shared --no-checkout "$selected" "$destination"
    git -C "$destination" sparse-checkout init --cone
    git -C "$destination" sparse-checkout set agenticgen/src agenticgen/tests agenticgen/configs agenticgen/scripts agenticgen/docs/plans src
else
    [[ -d "$destination/.git" ]] || { echo 'Destination is not the expected repository' >&2; exit 5; }
    [[ -z "$(git -C "$destination" status --porcelain)" ]] || { echo 'Remote working tree is dirty' >&2; exit 6; }
    git -C "$destination" fetch "$selected" "$commit"
fi
git -C "$destination" checkout --detach "$commit"
[[ "$(git -C "$destination" rev-parse HEAD)" == "$commit" ]]
git -C "$destination" fsck --connectivity-only
printf '%s\n' "$commit" >"$cache/deployed_commit.txt"
echo "DEPLOYED $commit"
