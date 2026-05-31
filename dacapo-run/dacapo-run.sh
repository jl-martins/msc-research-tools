#!/usr/bin/env bash

set -euo pipefail

readonly SYSFS_THP_ENABLED="/sys/kernel/mm/transparent_hugepage/enabled"
readonly SYSFS_CPU_GOVERNOR_PATTERN="/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"

# Default values
readonly DEFAULT_CONFIG_DIR="/usr/local/etc/dacapo-run"
readonly DEFAULT_JAVA_INSTALL_ROOT="/usr/lib/jvm"

readonly DEFAULT_JDKS=(temurin-8 temurin-11 temurin-17 temurin-21 temurin-25)
readonly DEFAULT_HEAP_MULTIPLIER=2
readonly DEFAULT_GC_OPTIONS=("-XX:+UseG1GC")

readonly DEFAULT_DACAPO_VERSION="23.11-MR2"
readonly DEFAULT_DACAPO_NICKNAME="chopin"
readonly DEFAULT_DACAPO_FORK=""
readonly DEFAULT_BM_DIR="."
readonly DEFAULT_BENCHMARKS=(
  batik biojava cassandra fop h2 h2o jme jython
  kafka luindex lusearch pmd spring tomcat zxing
)

readonly DEFAULT_IMPLEMENTATIONS=(loops streams)
readonly DEFAULT_INVOCATIONS_PER_BENCHMARK=20

# Environment variables
readonly CONFIG_DIR="${CONFIG_DIR:-$DEFAULT_CONFIG_DIR}"
readonly JAVA_INSTALL_ROOT="${JAVA_INSTALL_ROOT:-$DEFAULT_JAVA_INSTALL_ROOT}"
readonly DACAPO_VERSION="${DACAPO_VERSION:-$DEFAULT_DACAPO_VERSION}"
readonly DACAPO_NICKNAME="${DACAPO_NICKNAME:-$DEFAULT_DACAPO_NICKNAME}"
readonly DACAPO_FORK="${DACAPO_FORK:-$DEFAULT_DACAPO_FORK}"

# Command-line arguments
RUN_ID=""
OUT_DIR=""
LOG_FILE=""

JDKS=()
HEAP_MULTIPLIER=""
GC_OPTIONS=()

BM_DIR=""
BENCHMARKS=()
IMPLEMENTATIONS=()
INVOCATIONS_PER_BENCHMARK=""
DACAPO_ADDITIONAL_OPTIONS=()

# Configuration variables

## Mapping from benchmark name to minimum heap size in MiB
declare -A MIN_HEAP

# Cached values
readonly VERSION_PREFIX="$DACAPO_VERSION-$DACAPO_NICKNAME${DACAPO_FORK:+-$DACAPO_FORK}"

# Prerequisites
require_linux() {
    if [[ "$(uname)" != "Linux" ]]; then
        echo "This script must run on Linux" >&2
        exit 1
    fi
}

check_prerequisites() {
    require_linux

    local -a required_cmds=(awk date java realpath tee)
    local -a missing_cmds=()
    local cmd

    for cmd in "${required_cmds[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing_cmds+=("$cmd")
        fi
    done

    if [[ "${#missing_cmds[@]}" -gt 0 ]]; then
        echo "Missing prerequisite(s): ${missing_cmds[*]}."
        echo "Please ensure all required commands are installed and available in your system PATH."
        exit 1
    fi
}

# I/O
log() {
    local level datetime
    level="$1"
    datetime="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "$datetime [$level] ${*:2}" >&2
}

print_associative_array() {
    local -n a="$1"
    local open_indent_size="${2:-0}"
    local entry_indent_size="${3:-4}"
    local close_indent_size="${4:-0}"

    local open_indent="" entry_indent="" close_indent=""
    printf -v open_indent "%*s" "$open_indent_size" ""
    printf -v entry_indent "%*s" "$entry_indent_size" ""
    printf -v close_indent "%*s" "$close_indent_size" ""

    printf "%s{\n" "$open_indent"

    local key
    local first=1
    for key in "${!a[@]}"; do
        if (( first )); then
            first=0
        else
            printf ",\n"
        fi
        printf "%s'%s': '%s'" "$entry_indent" "$key" "${a[$key]}"
    done

    if (( !first )); then
        printf "\n"
    fi

    printf "%s}\n" "$close_indent"
}

print_variables() {
    cat <<MSG
Running DaCapo benchmarks with the following variables:
  - CONFIG_DIR=$CONFIG_DIR
  - JAVA_INSTALL_ROOT=$JAVA_INSTALL_ROOT
  - DACAPO_VERSION=$DACAPO_VERSION
  - DACAPO_NICKNAME=$DACAPO_NICKNAME
  - DACAPO_FORK=$DACAPO_FORK
  - RUN_ID=$RUN_ID
  - OUT_DIR=$OUT_DIR
  - LOG_FILE=$LOG_FILE
  - JDKS=${JDKS[*]}
  - HEAP_MULTIPLIER=$HEAP_MULTIPLIER
  - GC_OPTIONS=${GC_OPTIONS[*]}
  - BM_DIR=$BM_DIR
  - BENCHMARKS=${BENCHMARKS[*]}
  - IMPLEMENTATIONS=${IMPLEMENTATIONS[*]}
  - INVOCATIONS_PER_BENCHMARK=$INVOCATIONS_PER_BENCHMARK
  - DACAPO_ADDITIONAL_OPTIONS=${DACAPO_ADDITIONAL_OPTIONS[*]}
  - MIN_HEAP=$(print_associative_array MIN_HEAP 0 4 2)
MSG
}

die() {
    echo "Error: $*" >&2
    exit 1
}

# CLI
print_help() {
    cat <<HELP
Usage: dacapo-run [OPTIONS]

Run DaCapo benchmarks across multiple Java versions and implementations.

Options:
  --help                         show this help message and exit
  --run-id ID                    set the run id (default: YYYYMMDD_HHMMSS)
  --out-dir OUT_DIR              store the results in OUT_DIR
                                 (default: $HOME/dacapobench-results/<run-id>)
  --log-file LOG_FILE            write logs to LOG_FILE (default: <out-dir>/run.log)
  --jdks JDKS                    select the target JDKs as a single, space-separated
                                 string (default: '${DEFAULT_JDKS[*]}')
  --heap-multiplier N            set the Java heap size of each benchmark to N
                                 times its minimum value (default: $DEFAULT_HEAP_MULTIPLIER)
  --gc-opts OPTS                 configure the Java garbage collector options as a single,
                                 space-separated string (default: '${DEFAULT_GC_OPTIONS[*]}')
  --bm-dir DIR                   search for the JARs of the benchmark suite in DIR
                                 (default: .)
  --benchmarks BMS               run the specified benchmarks, provided as a single,
                                 space-separated string (default: '${DEFAULT_BENCHMARKS[*]}')
  --implementations IMPLS        run the specified implementations, provided as a single,
                                 space-separated string (default: '${DEFAULT_IMPLEMENTATIONS[*]}')
  --invocations N                invoke each benchmark N times per implementation and
                                 JDK, each time in a new Java process (default: $DEFAULT_INVOCATIONS_PER_BENCHMARK)
  --dacapo-additional-opts OPTS  provide additional options for DaCapo (default: None)

Environment variables:
  CONFIG_DIR                     configuration directory path (default: $DEFAULT_CONFIG_DIR)
  JAVA_INSTALL_ROOT              java installation root (default: $DEFAULT_JAVA_INSTALL_ROOT)
  DACAPO_VERSION                 dacapo benchmark version (default: $DEFAULT_DACAPO_VERSION)
  DACAPO_NICKNAME                dacapo benchmark nickname (default: $DEFAULT_DACAPO_NICKNAME)
  DACAPO_FORK                    dacapo benchmark fork (default: ${DEFAULT_DACAPO_FORK:-None})
HELP
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help)
                print_help
                exit 0
                ;;
            --run-id)
                [[ -n "${2:-}" ]] || die "--run-id requires an argument"
                RUN_ID="$2"
                shift 2
                ;;
            --out-dir)
                [[ -n "${2:-}" ]] || die "--out-dir requires an argument"
                OUT_DIR="$(realpath -m "$2")"
                shift 2
                ;;
            --log-file)
                [[ -n "${2:-}" ]] || die "--log-file requires an argument"
                LOG_FILE="$2"
                shift 2
                ;;
            --jdks)
                [[ -n "${2:-}" ]] || die "--jdks requires an argument"
                read -ra JDKS <<< "$2"
                shift 2
                ;;
            --heap-multiplier)
                [[ -n "${2:-}" ]] || die "--heap-multiplier requires an argument"
                HEAP_MULTIPLIER="$2"
                shift 2
                ;;
            --gc-opts)
                [[ -n "${2:-}" ]] || die "--gc-opts requires an argument"
                read -ra GC_OPTIONS <<< "$2"
                shift 2
                ;;
            --bm-dir)
                [[ -n "${2:-}" ]] || die "--bm-dir requires an argument"
                BM_DIR="$2"
                shift 2
                ;;
            --benchmarks)
                [[ -n "${2:-}" ]] || die "--benchmarks requires an argument"
                read -ra BENCHMARKS <<< "$2"
                shift 2
                ;;
            --implementations)
                [[ -n "${2:-}" ]] || die "--implementations requires an argument"
                read -ra IMPLEMENTATIONS <<< "$2"
                shift 2
                ;;
            --invocations)
                [[ -n "${2:-}" ]] || die "--invocations requires a positive number"
                INVOCATIONS_PER_BENCHMARK="$2"
                shift 2
                ;;
            --dacapo-additional-opts)
                [[ -n "${2:-}" ]] || die "--dacapo-additional-opts requires an argument"
                read -ra DACAPO_ADDITIONAL_OPTIONS <<< "$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1" >&2
                print_help >&2
                exit 2
                ;;
        esac
    done

    if [[ -z "$RUN_ID" ]]; then
        RUN_ID="$(date +%Y%m%d_%H%M%S)"
    fi

    : "${OUT_DIR:=$HOME/dacapobench-results/$RUN_ID}"
    : "${LOG_FILE:=$OUT_DIR/run.log}"
    : "${HEAP_MULTIPLIER:=$DEFAULT_HEAP_MULTIPLIER}"
    : "${BM_DIR:=$DEFAULT_BM_DIR}"
    : "${INVOCATIONS_PER_BENCHMARK:=$DEFAULT_INVOCATIONS_PER_BENCHMARK}"

    (( "${#JDKS[@]}" )) || JDKS=("${DEFAULT_JDKS[@]}")
    (( "${#GC_OPTIONS[@]}" )) || GC_OPTIONS=("${DEFAULT_GC_OPTIONS[@]}")
    (( "${#BENCHMARKS[@]}" )) || BENCHMARKS=("${DEFAULT_BENCHMARKS[@]}")
    (( "${#IMPLEMENTATIONS[@]}" )) || IMPLEMENTATIONS=("${DEFAULT_IMPLEMENTATIONS[@]}")
}

# Configuration
load_config() {
    local stats_file="$CONFIG_DIR/${VERSION_PREFIX}-stats.sh"

    if [[ -f "$stats_file" ]]; then
        # shellcheck disable=SC1090
        . "$stats_file"
    else
        echo "Warn: Stats file '$stats_file' not found." \
             "Java heap size will not be set." >&2
    fi
}

# Validation
validate_variables() {
    [[ -d "$CONFIG_DIR" ]] || die "CONFIG_DIR must be a directory"
    [[ -d "$JAVA_INSTALL_ROOT" ]] || die "JAVA_INSTALL_ROOT must be a directory"

    [[ -n "$DACAPO_VERSION" ]] || die "DACAPO_VERSION must not be empty"
    [[ -n "$DACAPO_NICKNAME" ]] || die "DACAPO_NICKNAME must not be empty"

    [[ -n "$RUN_ID" ]] || die "RUN_ID must not be empty"
    [[ -n "$LOG_FILE" ]] || die "LOG_FILE must not be empty"

    [[ "${#JDKS[@]}" -gt 0 ]] || die "JDKS must not be empty"
    [[ "$HEAP_MULTIPLIER" =~ ^[1-9][0-9]*$ ]] ||
        die "HEAP_MULTIPLIER must be a positive integer"

    [[ -d "$BM_DIR" ]] || die "BM_DIR must be a directory"
    [[ "${#BENCHMARKS[@]}" -gt 0 ]] || die "BENCHMARKS must not be empty"
    [[ "${#IMPLEMENTATIONS[@]}" -gt 0 ]] || die "IMPLEMENTATIONS must not be empty"
    [[ "$INVOCATIONS_PER_BENCHMARK" =~ ^[1-9][0-9]*$ ]] ||
        die "INVOCATIONS_PER_BENCHMARK must be a positive integer"
}

# Benchmark Execution
collect_sysinfo() {
    uname -a > "$OUT_DIR/uname.txt"

    if command -v lscpu >/dev/null; then
        lscpu > "$OUT_DIR/cpuinfo.txt"
    else
        log "WARN" "lscpu command not found. CPU information will not be recorded."
    fi

    if command -v free >/dev/null; then
        free -h > "$OUT_DIR/raminfo.txt"
    else
        log "WARN" "free command not found. RAM information will not be recorded."
    fi

    shopt -s nullglob
    # shellcheck disable=SC2206  # intentional globbing
    cpu_governor_files=($SYSFS_CPU_GOVERNOR_PATTERN)
    if [[ ${#cpu_governor_files[@]} -gt 0 ]]; then
        cat "${cpu_governor_files[@]}" > "$OUT_DIR/cpu_governors.txt" 2>/dev/null
    else
        log "WARN" "CPU scaling governor files ($SYSFS_CPU_GOVERNOR_PATTERN) not found." \
                   "CPU scaling governor will not be recorded."
    fi
    shopt -u nullglob

    if [[ -f "$SYSFS_THP_ENABLED" ]]; then
        cat "$SYSFS_THP_ENABLED" > "$OUT_DIR/thp_enabled.txt"
    else
        log "WARN" "THP configuration file ($SYSFS_THP_ENABLED) not found." \
                   "THP state will not be recorded."
    fi
}

collect_jvminfo() {
    local java_bin="$1"
    local jdk_name="$2"
    local java_out_dir="$OUT_DIR/$jdk_name"
    mkdir -p "$java_out_dir"
    "$java_bin" -version &> "$java_out_dir/java_version.txt"
}

get_java_major_version() {
    local java_bin="$1"
    "$java_bin" -version 2>&1 | awk -F '"' '
        /version/ {
            split($2, digits, ".");

            # If the first part is "1", it is legacy (e.g., 1.8 -> 8)
            # Otherwise, use the major version as-is (e.g., 17 -> 17)
            if (digits[1] == 1) {
                print digits[2];
            } else {
                print digits[1];
            }
        }
    '
}

get_java_options() {
    local java_major_version="${1:-0}"
    local benchmark="$2"
    local -n java_options_ref=$3
    local min_heap_size="${MIN_HEAP[$benchmark]:-}"

    java_options_ref=() # Clear any old values
    java_options_ref+=("${GC_OPTIONS[@]}")

    if [[ "$min_heap_size" =~ ^[1-9][0-9]*$ ]]; then
        local heap_size=$((min_heap_size * HEAP_MULTIPLIER))
        java_options_ref+=("-Xms${heap_size}m" "-Xmx${heap_size}m")
    else
        log "WARN" "Invalid minimum heap size for benchmark '$benchmark': '$min_heap_size'." \
                   "The heap size of this benchmark will be left unset."
    fi

    if [[ "$benchmark" = "cassandra" && "$java_major_version" -ge 17 ]]; then
        java_options_ref+=("-Djava.security.manager=allow")
    fi

    if [[ "$benchmark" = "h2o" && "$java_major_version" -gt 17 ]]; then
        java_options_ref+=("-Dsys.ai.h2o.debug.allowJavaVersions=$java_major_version")
    fi
}

run_single_benchmark() {
    local jdk_name="$1" java_bin="$2"
    # shellcheck disable=SC2178  # false positive
    local -n java_options_ref="$3"
    local jarfile="$4" bm="$5" impl="$6" bm_context="$7"
    local invocation_nr exit_code bm_out_dir

    # Remove any stale measurements from previous runs so they aren't attributed to new runs
    rm -f "measurements.csv"

    for (( invocation_nr = 1; invocation_nr <= INVOCATIONS_PER_BENCHMARK; invocation_nr++ )); do
        log "INFO" "Running ($invocation_nr/$INVOCATIONS_PER_BENCHMARK) $bm_context..."
        exit_code=0

        "$java_bin" \
            "${java_options_ref[@]}" \
            -jar "$jarfile" \
            --callback "org.dacapo.rapl.RAPLCallback" \
            "${DACAPO_ADDITIONAL_OPTIONS[@]}" \
            "$bm" || exit_code=$?

        if (( exit_code > 128 && exit_code < 193 )); then
            local signal=$(( exit_code - 128 ))
            log "INFO" "Terminated by SIG$(kill -l "$signal")"
            exit $exit_code
        elif (( exit_code != 0 )); then
            log "ERROR" "Failed ($invocation_nr/$INVOCATIONS_PER_BENCHMARK) $bm_context" \
                "exit_code=$exit_code"
            continue
        fi
    done

    if [[ -f "measurements.csv" ]]; then
        bm_out_dir="$OUT_DIR/$jdk_name/$impl/$bm"
        mkdir -p "$bm_out_dir"

        mv measurements.csv "$bm_out_dir"
        echo "${java_options_ref[*]}" > "$bm_out_dir/java_options.txt"

        log "INFO" "Done! Measurements stored at '$bm_out_dir/measurements.csv'"
    else
        log "WARN" "measurements.csv not found for $bm_context. Skipping..."
    fi
}

run_benchmarks() {
    local jdk_name java_bin java_major_version \
          bm impl jarfile bm_context
    # shellcheck disable=SC2034  # used by get_java_options
    local -a java_options

    pushd "$BM_DIR" >/dev/null
    collect_sysinfo
    print_variables
    for jdk_name in "${JDKS[@]}"; do
        java_bin="$JAVA_INSTALL_ROOT/$jdk_name/bin/java"
        [[ -x "$java_bin" ]] || { log "WARN" "$java_bin not found, skipping"; continue; }

        collect_jvminfo "$java_bin" "$jdk_name"
        java_major_version=$(get_java_major_version "$java_bin")
        for bm in "${BENCHMARKS[@]}"; do
            for impl in "${IMPLEMENTATIONS[@]}"; do
                get_java_options "$java_major_version" "$bm" java_options
                jarfile="dacapo-${VERSION_PREFIX}-${impl}.jar"
                bm_context="jdk=$jdk_name benchmark=$bm implementation=$impl"

                run_single_benchmark \
                    "$jdk_name" \
                    "$java_bin" \
                    java_options \
                    "$jarfile" \
                    "$bm" \
                    "$impl" \
                    "$bm_context"
            done
        done
    done
    popd >/dev/null
}

main() {
    check_prerequisites

    load_config
    parse_args "$@"
    validate_variables

    mkdir -p "$OUT_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"
    run_benchmarks 2>&1 | tee -i "$LOG_FILE"
}

main "$@"
