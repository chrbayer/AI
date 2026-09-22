# bash completion for llmctl. llmctl works the candidates out itself
# (`llmctl _complete`), so this only hands over the words typed so far.
_llmctl() {
    local cur="${COMP_WORDS[COMP_CWORD]}" x
    local -a c words=() found
    mapfile -t c < <(llmctl _complete "${COMP_WORDS[@]:1:COMP_CWORD}" 2>/dev/null)
    COMPREPLY=()
    for x in "${c[@]}"; do
        if [[ "$x" == @files ]]; then
            compopt -o filenames
            mapfile -t found < <(compgen -f -- "$cur")
            COMPREPLY+=("${found[@]}")
        else
            words+=("$x")
        fi
    done
    mapfile -t found < <(compgen -W "${words[*]}" -- "$cur")
    COMPREPLY+=("${found[@]}")
}
complete -F _llmctl llmctl
