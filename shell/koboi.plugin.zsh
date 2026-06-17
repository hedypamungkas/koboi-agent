# koboi.plugin.zsh -- ZSH plugin for koboi-agent
#
# Provides the :koboi prefix command for querying koboi-agent from anywhere
# in the shell. Uses --print mode for JSON-lines output.
#
# Installation:
#   1. Copy/symlink to $ZSH_CUSTOM/plugins/koboi/
#   2. Add 'koboi' to plugins in .zshrc
#   3. Set KOBOI_CONFIG to your default config path (optional)
#
# Usage:
#   :koboi what is the capital of France?
#   :koboi -c configs/rag_agent.yaml search for Q2 revenue
#   echo "summarize this" | :koboi
#   :koboi --help

# Default config path (user can override via environment)
: ${KOBOI_CONFIG:=""}

# Find the koboi binary
if ! command -v koboi &>/dev/null; then
    # Try common locations
    for _koboi_bin in "$HOME/.local/bin/koboi" "$HOME/.cargo/bin/koboi" \
                      "./venv/bin/koboi" ".venv/bin/koboi"; do
        if [[ -x "$_koboi_bin" ]]; then
            export PATH="$(dirname "$_koboi_bin"):$PATH"
            break
        fi
    done
    unset _koboi_bin
fi

function :koboi() {
    local config=""
    local message=""
    local verbose=0
    local help=0

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -c|--config)
                config="$2"
                shift 2
                ;;
            -v|--verbose)
                verbose=1
                shift
                ;;
            -h|--help)
                help=1
                shift
                ;;
            --)
                shift
                message="$*"
                break
                ;;
            *)
                # Accumulate as message
                if [[ -n "$message" ]]; then
                    message="$message $1"
                else
                    message="$1"
                fi
                shift
                ;;
        esac
    done

    # Show help
    if [[ $help -eq 1 ]]; then
        echo "Usage: :koboi [options] <message>"
        echo ""
        echo "Options:"
        echo "  -c, --config <path>   YAML config file (default: \$KOBOI_CONFIG)"
        echo "  -v, --verbose         Show debug output"
        echo "  -h, --help            Show this help"
        echo ""
        echo "Examples:"
        echo "  :koboi what is the capital of France?"
        echo "  :koboi -c configs/rag_agent.yaml search for Q2 revenue"
        echo "  echo 'summarize this' | :koboi"
        return 0
    fi

    # Determine config
    if [[ -z "$config" ]]; then
        config="$KOBOI_CONFIG"
    fi

    # Build command
    local -a cmd=(koboi run)
    if [[ -n "$config" ]]; then
        cmd+=("$config")
    fi
    cmd+=(--print)
    if [[ $verbose -eq 1 ]]; then
        cmd+=(-v)
    fi

    # Handle piped input
    if [[ -z "$message" ]] && ! [[ -t 0 ]]; then
        message="$(cat)"
    fi

    if [[ -z "$message" ]]; then
        echo "Error: No message provided. Use :koboi --help for usage." >&2
        return 1
    fi

    cmd+=(-m "$message")

    # Execute and format output
    "${cmd[@]}" 2>/dev/null | while IFS= read -r line; do
        local type
        type=$(echo "$line" | command grep -o '"type":"[^"]*"' | head -1 | cut -d'"' -f4)
        case "$type" in
            text_delta)
                echo "$line" | command grep -o '"content":"[^"]*"' | head -1 | cut -d'"' -f4 | tr -d '\n'
                ;;
            complete)
                echo ""  # Newline after response
                local content
                content=$(echo "$line" | command grep -o '"content":".*"' | head -1 | sed 's/^"content":"//;s/"$//')
                if [[ -n "$content" ]]; then
                    echo "$content" | sed 's/\\n/\n/g'
                fi
                ;;
            error)
                echo "$line" | command grep -o '"error":"[^"]*"' | head -1 | cut -d'"' -f4 | sed 's/^/Error: /' >&2
                return 1
                ;;
            session_start|session_end|tool_call|tool_result|iteration)
                # Silently skip these event types
                ;;
            *)
                # Fallback: try to extract text content
                echo "$line" | command grep -o '"content":"[^"]*"' | head -1 | cut -d'"' -f4
                ;;
        esac
    done
}

# ZSH completion for :koboi
function _koboi_complete() {
    local -a configs
    # Complete config files
    if [[ -d configs ]]; then
        configs=(configs/*.yaml(N))
    fi
    local -a options=(
        '-c[Config file]:config file:('${configs[*]}')'
        '-v[Verbose output]'
        '-h[Show help]'
        '--config[Config file]:config file:('${configs[*]}')'
        '--verbose[Verbose output]'
        '--help[Show help]'
    )
    _arguments -s $options '*::message: '
}

# Only register completion if compdef is available
if (( $+functions[compdef] )); then
    compdef _koboi_complete ':koboi'
fi

# Aliases for convenience
alias koboi-ask=':koboi'
alias kq=':koboi'
