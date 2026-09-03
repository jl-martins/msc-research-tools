#!/usr/bin/env bash

set -euo pipefail

# Linux sysfs paths
readonly SYSFS_THP_ENABLED="/sys/kernel/mm/transparent_hugepage/enabled"
readonly SYSFS_CPU_GOVERNOR_PATTERN="/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"

# Default values
DEFAULT_CONFIG_DIR="/usr/local/etc/jbench-run"
DEFAULT_JAVA_INSTALL_ROOT="/usr/lib/jvm"

DEFAULT_JDKS=(temurin-8 temurin-11 temurin-17 temurin-21 temurin-25)
DEFAULT_HEAP_MULTIPLIER=2
DEFAULT_GC_OPTIONS=("-XX:+UseG1GC")

DEFAULT_ARTIFACT_DIR="."
DEFAULT_IMPLEMENTATIONS=()
DEFAULT_INVOCATIONS_PER_BENCHMARK=20

# Environment variables
readonly CONFIG_DIR="${CONFIG_DIR:-$DEFAULT_CONFIG_DIR}"
readonly JAVA_INSTALL_ROOT="${JAVA_INSTALL_ROOT:-$DEFAULT_JAVA_INSTALL_ROOT}"

# Command-line arguments
RUN_ID=""
OUT_DIR=""
LOG_FILE=""

JDKS=()
HEAP_MULTIPLIER=""
GC_OPTIONS=()

ARTIFACT_DIR="."
SUITE_NAME=""
SUITE_VERSION=""
SUITE_ADDITIONAL_OPTIONS=()
BENCHMARKS=()
IMPLEMENTATIONS=()
INVOCATIONS_PER_BENCHMARK=""

# Configuration variables

## Associative array from benchmark name to minimum heap size in MiB.
## Different values for different JDKs are not supported because
## the JDKs that we used in our research have basically identical values.
declare -A MIN_HEAP

# Other variables
SUITE_CONFIG_FILE=""

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
        echo "Missing required command(s): ${missing_cmds[*]}."
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
Running $SUITE_NAME benchmark suite with the following variables:
  - RUN_ID=$RUN_ID
  - OUT_DIR=$OUT_DIR
  - LOG_FILE=$LOG_FILE
  - CONFIG_DIR=$CONFIG_DIR
  - SUITE_CONFIG_FILE=$SUITE_CONFIG_FILE
  - JAVA_INSTALL_ROOT=$JAVA_INSTALL_ROOT
  - JDKS=${JDKS[*]}
  - HEAP_MULTIPLIER=$HEAP_MULTIPLIER
  - GC_OPTIONS=${GC_OPTIONS[*]}
  - SUITE_NAME=$SUITE_NAME
  - SUITE_VERSION=$SUITE_VERSION
  - SUITE_ADDITIONAL_OPTIONS=${SUITE_ADDITIONAL_OPTIONS[*]}
  - IMPLEMENTATIONS=${IMPLEMENTATIONS[*]}
  - BENCHMARKS=${BENCHMARKS[*]}
  - INVOCATIONS_PER_BENCHMARK=$INVOCATIONS_PER_BENCHMARK
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
Usage: jbench-run [OPTIONS]

Run multiple implementations of a Java benchmark suite across multiple JDKs.

Options:
  --help                         show this help message and exit
  --run-id ID                    set the run id (default: YYYYMMDD_HHMMSS)
  --out-dir OUT_DIR              store the results in OUT_DIR
                                 (default: $HOME/jbench-run-results/<suite-name>/<suite-version>/<run-id>)
  --log-file LOG_FILE            write logs to LOG_FILE (default: <out-dir>/run.log)
  --jdks JDKS                    select the target JDKs as a single,
                                 space-separated string
                                 (default: '${DEFAULT_JDKS[*]}')
  --heap-multiplier N            set each benchmark's Java heap size to N times
                                 its minimum value (default: $DEFAULT_HEAP_MULTIPLIER)
  --gc-opts GC_OPTS              configure the Java garbage collector options as a
                                 single, space-separated string (default: '${DEFAULT_GC_OPTIONS[*]}')
  --artifact-dir DIR             search for benchmark suite JARs in DIR
                                 (default: $DEFAULT_ARTIFACT_DIR)
  --suite-name SUITE_NAME        set the name of the benchmark suite (required)
  --suite-version SUITE_VERSION  set the version of the benchmark suite, without the
                                 implementation suffix (required)
  --suite-additional-opts OPTS   provide additional options for the benchmark suite
                                 (default: None)
  --benchmarks BMS               run the specified benchmarks, provided as a single,
                                 space-separated string
                                 (default: use the get_default_benchmarks function from
                                  the suite configuration script to get the list of
                                  benchmarks for each Java version)
  --implementations IMPLS        run the specified implementations, provided as
                                 a single, space-separated string
                                 (default: use DEFAULT_IMPLEMENTATIONS from the suite
                                  configuration script)
  --invocations N                invoke each benchmark N times per implementation and
                                 JDK, each time in a new Java process (default: $DEFAULT_INVOCATIONS_PER_BENCHMARK)

Environment variables:
  CONFIG_DIR                     configuration directory (default: $DEFAULT_CONFIG_DIR)
  JAVA_INSTALL_ROOT              java installation root (default: $DEFAULT_JAVA_INSTALL_ROOT)

Execution format:
  The current version of this script requires the benchmark suite to be packaged in a
  JAR file and support the following execution format:

  java -jar <jarfile> <suite_options> <benchmark>

Suite configuration script:
  Each version of a benchmark suite should have a configuration script at:

    <CONFIG_DIR>/<SUITE_NAME>/<SUITE_VERSION>-config.sh.

  The configuration should define the following elements:
    1. MIN_HEAP                 A global associative array mapping benchmark names to
                                their minimum heap sizes. This array is used to
                                determine the heap size based on the benchmark and the
                                --heap-multiplier option.
    2. get_default_benchmarks   A function that accepts a Java major version and a
                                nameref to an indexed array and populates the array with
                                the default benchmarks for that Java version.
                                This function is required to identify the default
                                benchmarks when the --benchmarks option is omitted.
                                Unlike --benchmarks, which only supports a fixed list of
                                benchmarks, get_default_benchmarks can return different
                                benchmarks for different Java versions. This is useful
                                when certain benchmarks require specific Java versions.

  In addition to these elements, the following variable and hooks are supported:
    1. DEFAULT_IMPLEMENTATIONS  An indexed array with the default values for the
                                --implementations option.
    2. extend_java_options      A function that receives a Java major version,
                                a benchmark name and a nameref to an indexed array and
                                fills the array with additional Java options for that
                                benchmark and Java version.
    3. extend_suite_options     A function that receives a nameref to an
                                indexed array and provides additional options for the
                                benchmark suite (e.g., a callback to measure the
                                execution time and energy consumption).
    4. pre_benchmark            A function that runs before the first invocation
                                of each benchmark, for every implementation and JDK.
    5. post_benchmark           A function that runs after the last invocation
                                of each benchmark, for every implementation and JDK.

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
            --artifact-dir)
                [[ -n "${2:-}" ]] || die "--artifact-dir requires an argument"
                ARTIFACT_DIR="$2"
                shift 2
                ;;
            --suite-name)
                [[ -n "${2:-}" ]] || die "--suite-name requires an argument"
                SUITE_NAME="$2"
                shift 2
                ;;
            --suite-version)
                [[ -n "${2:-}" ]] || die "--suite-version requires an argument"
                SUITE_VERSION="$2"
                shift 2
                ;;
            --suite-additional-opts)
                [[ -n "${2:-}" ]] || die "--suite-additional-opts requires an argument"
                read -ra SUITE_ADDITIONAL_OPTIONS <<< "$2"
                shift 2
                ;;
            --implementations)
                [[ -n "${2:-}" ]] || die "--implementations requires an argument"
                read -ra IMPLEMENTATIONS <<< "$2"
                shift 2
                ;;
            --benchmarks)
                [[ -n "${2:-}" ]] || die "--benchmarks requires an argument"
                read -ra BENCHMARKS <<< "$2"
                shift 2
                ;;
            --invocations)
                [[ -n "${2:-}" ]] || die "--invocations requires a positive number"
                INVOCATIONS_PER_BENCHMARK="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1" >&2
                print_help >&2
                exit 2
                ;;
        esac
    done

    [[ -n "$SUITE_NAME" ]] || die "--suite-name is required"
    [[ -n "$SUITE_VERSION" ]] || die "--suite-version is required"

    : "${RUN_ID:="$(date +%Y%m%d_%H%M%S)"}"
    : "${OUT_DIR:="$HOME/jbench-run-results/$SUITE_NAME/$SUITE_VERSION/$RUN_ID"}"
    : "${LOG_FILE:="$OUT_DIR/run.log"}"
    : "${HEAP_MULTIPLIER:="$DEFAULT_HEAP_MULTIPLIER"}"
    : "${ARTIFACT_DIR:="$DEFAULT_ARTIFACT_DIR"}"
    : "${INVOCATIONS_PER_BENCHMARK:="$DEFAULT_INVOCATIONS_PER_BENCHMARK"}"

    (( ${#JDKS[@]} )) || JDKS=("${DEFAULT_JDKS[@]}")
    (( ${#GC_OPTIONS[@]} )) || GC_OPTIONS=("${DEFAULT_GC_OPTIONS[@]}")
}

# Configuration
load_config() {
    readonly SUITE_CONFIG_FILE="$CONFIG_DIR/$SUITE_NAME/$SUITE_VERSION-config.sh"
    if [[ -f "$SUITE_CONFIG_FILE" ]]; then
        # shellcheck disable=SC1090
        . "$SUITE_CONFIG_FILE"
        (( ${#IMPLEMENTATIONS[@]} )) || IMPLEMENTATIONS=("${DEFAULT_IMPLEMENTATIONS[@]}")
    else
        echo "Warn: Config file '$SUITE_CONFIG_FILE' not found." \
             "The Java heap size will not be set and there will be no default" \
             "value for the --benchmarks option. For further information," \
             "please rerun the script with --help."
    fi
}

# Validation
validate_variables() {
    [[ -d "$CONFIG_DIR" ]] || die "CONFIG_DIR must be a directory"
    [[ -d "$JAVA_INSTALL_ROOT" ]] || die "JAVA_INSTALL_ROOT must be a directory"

    [[ -n "$RUN_ID" ]] || die "RUN_ID must not be empty"
    [[ -n "$LOG_FILE" ]] || die "LOG_FILE must not be empty"

    [[ "${#JDKS[@]}" -gt 0 ]] || die "JDKS must not be empty"
    [[ "$HEAP_MULTIPLIER" =~ ^[1-9][0-9]*$ ]] || die "HEAP_MULTIPLIER must be a positive integer"

    [[ -d "$ARTIFACT_DIR" ]] || die "ARTIFACT_DIR must be a directory"
    [[ "${#IMPLEMENTATIONS[@]}" -gt 0 ]] ||
        die "IMPLEMENTATIONS must not be empty." \
            "Please specify it with --implementations or define" \
            "the DEFAULT_IMPLEMENTATIONS array in '$SUITE_CONFIG_FILE'."

    [[ "$INVOCATIONS_PER_BENCHMARK" =~ ^[1-9][0-9]*$ ]] ||
        die "INVOCATIONS must be a positive integer"

    # SUITE_NAME and SUITE_VERSION are validated in parse_args because they are required
}

# Hook Execution
run_hook() {
    local hook_name="$1"
    shift
    if declare -f "$hook_name" >/dev/null 2>&1; then
        "$hook_name" "$@"
    fi
}

# System Information
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

# Benchmark Execution
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

    # Allow the benchmark configuration file to provide custom Java options
    local -a custom_options=()
    run_hook extend_java_options "$java_major_version" "$benchmark" custom_options
    java_options_ref+=("${custom_options[@]}")
}

get_suite_options() {
    local -n suite_options_ref=$1

    suite_options_ref=() # Clear any old values

    # Allow the benchmark configuration file to provide custom suite options
    local -a custom_options=()
    run_hook extend_suite_options custom_options
    suite_options_ref+=("${custom_options[@]}")

    # Append additional options from the command line
    suite_options_ref+=("${SUITE_ADDITIONAL_OPTIONS[@]}")
}

get_selected_benchmarks() {
    local java_major_version="$1"
    local -n benchmarks_ref=$2
    # shellcheck disable=SC2034  # false positive
    benchmarks_ref=("${BENCHMARKS[@]}")
}

select_benchmarks_getter() {
    if [[ ${#BENCHMARKS[@]} -gt 0 ]]; then
        echo "get_selected_benchmarks"
    elif declare -f get_default_benchmarks >/dev/null 2>&1; then
        echo "get_default_benchmarks"
    else
        log "ERROR" "No benchmarks were specified. Please use the --benchmarks" \
                    "options to specify a list of benchmarks or implement a" \
                    "get_default_benchmarks function in '$SUITE_CONFIG_FILE'"
        return 1
    fi
}

run_single_benchmark() {
    local java_bin="$1"
    # shellcheck disable=SC2178  # false positive
    local -n java_options_ref="$2"
    local jarfile="$3"
    # shellcheck disable=SC2178  # false positive
    local -n suite_options_ref="$4"
    local bm="$5" bm_context="$6" bm_out_dir="$7"
    local invocation_nr exit_code

    mkdir -p "$bm_out_dir"
    echo "${java_options_ref[*]}" > "$bm_out_dir/java_options.txt"
    echo "${suite_options_ref[*]}" > "$bm_out_dir/suite_options.txt"

    run_hook pre_benchmark
    for (( invocation_nr = 1; invocation_nr <= INVOCATIONS_PER_BENCHMARK; invocation_nr++ )); do
        log "INFO" "Running ($invocation_nr/$INVOCATIONS_PER_BENCHMARK) $bm_context..."
        exit_code=0

        "$java_bin" \
            "${java_options_ref[@]}" \
            -jar "$jarfile" \
            "${suite_options_ref[@]}" \
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

    run_hook post_benchmark "$bm_out_dir" "$bm_context"
}

run_benchmarks() {
    local jdk_name java_bin java_major_version \
          benchmarks_getter bm impl jarfile bm_context
    # shellcheck disable=SC2034  # used by get_java_options
    local -a java_options suite_options
    local -a benchmarks

    benchmarks_getter=$(select_benchmarks_getter)

    pushd "$ARTIFACT_DIR" >/dev/null
    collect_sysinfo
    print_variables
    get_suite_options suite_options

    for jdk_name in "${JDKS[@]}"; do
        java_bin="$JAVA_INSTALL_ROOT/$jdk_name/bin/java"
        [[ -x "$java_bin" ]] || { log "WARN" "$java_bin not found, skipping"; continue; }

        collect_jvminfo "$java_bin" "$jdk_name"
        java_major_version=$(get_java_major_version "$java_bin")
        if ! "$benchmarks_getter" "$java_major_version" benchmarks; then
            log "ERROR" "Failed to get benchmarks for Java $java_major_version"
            continue
        fi

        for bm in "${benchmarks[@]}"; do
            for impl in "${IMPLEMENTATIONS[@]}"; do
                get_java_options "$java_major_version" "$bm" java_options

                jarfile="${SUITE_NAME}-${SUITE_VERSION}-${impl}.jar"
                bm_context="jdk=$jdk_name benchmark=$bm implementation=$impl"
                bm_out_dir="$OUT_DIR/$jdk_name/$impl/$bm"

                run_single_benchmark \
                    "$java_bin" \
                    java_options \
                    "$jarfile" \
                    suite_options \
                    "$bm" \
                    "$bm_context" \
                    "$bm_out_dir"
            done
        done
    done
    popd >/dev/null
}

main() {
    check_prerequisites

    parse_args "$@"
    load_config
    validate_variables

    mkdir -p "$OUT_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"
    run_benchmarks 2>&1 | tee -i "$LOG_FILE"
}

main "$@"
