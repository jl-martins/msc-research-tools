#!/bin/sh

set -u

# If the script is running in zsh, force POSIX compatibility mode
[ -n "${ZSH_VERSION:-}" ] && emulate sh

# Memory values (MiB)
readonly SAFETY_MARGIN=500
readonly MIN_MEM=1

# Exit codes

## From https://github.com/openjdk/jdk11u/blob/jdk11.0.20.1/src/hotspot/share/utilities/debug.cpp#L344
readonly OOM_EXIT_CODE=3

# Default values
readonly DEFAULT_JAVA_INSTALL_ROOT="/usr/lib/jvm"

readonly DEFAULT_JDKS="temurin-8 temurin-11 temurin-17 temurin-21 temurin-25"
readonly DEFAULT_GC_OPTIONS="-XX:+UseG1GC"

# Environment variables
JAVA_INSTALL_ROOT="${JAVA_INSTALL_ROOT:-$DEFAULT_JAVA_INSTALL_ROOT}"

# Command-line arguments
OUT_DIR=""
MAX_MEM=""
JDKS=""
GC_OPTIONS=""
BENCHMARKS=""

# Positional arguments
JARFILE=""

# I/O
print_help() {
    cat <<EOF
Usage: $(basename "$0") [options] dacapo_jarfile

Options:
  --help                      show this help message and exit
  --out-dir OUT_DIR           store the results in OUT_DIR
                              (default: "$HOME/dacapo-min-heap/<jarname>")
  -m, --max-mem MAX_MEM       limit the heap size in MiB
                              (default: available memory - ${SAFETY_MARGIN})
  --jdks JDKS                 select the target JDKs as a single, space-separated string
                              (default: '$DEFAULT_JDKS')
  --gc-opts GC_OPTS           configure garbage collector options as a single,
                              space-separated string (default: '$DEFAULT_GC_OPTIONS')
  --benchmarks BMS            select benchmarks as a single, space-separated string
                              (default: all benchmarks in jar file)

Environment variables:
  JAVA_INSTALL_ROOT           java installation root (default: $DEFAULT_JAVA_INSTALL_ROOT)
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

# Validation
is_positive_integer() {
    case $1 in
        # Reject empty strings and any string containing non-digit characters
        ''|*[!0-9]*)
            return 1
            ;;
        # Accept strings containing at least one non-zero digit
        *[1-9]*)
            return 0
            ;;
        # Reject strings consisting only of zeros
        *)
            return 1
            ;;
    esac
}

# CLI
parse_args() {
    positional_count=0

    while [ $# -gt 0 ]; do
        case $1 in
            -h|--help)
                print_help
                exit 0
                ;;

            --out-dir)
                [ $# -ge 2 ] || die "Missing value for $1"
                OUT_DIR=$2
                shift 2
                ;;

            -m|--max-mem)
                [ $# -ge 2 ] || die "Missing value for $1"

                MAX_MEM=$2
                if ! is_positive_integer "$MAX_MEM"; then
                    die "The argument of $1 must be a positive number"
                fi

                shift 2
                ;;

            --jdks)
                [ $# -ge 2 ] || die "Missing value for $1"
                JDKS=$2
                shift 2
                ;;

            --gc-opts)
                [ $# -ge 2 ] || die "Missing value for $1"
                GC_OPTIONS=$2
                shift 2
                ;;

            --benchmarks)
                [ $# -ge 2 ] || die "Missing value for $1"
                BENCHMARKS=$2
                shift 2
                ;;

            -*)
                die "Unknown option: $1"
                ;;

            *)
                positional_count=$((positional_count + 1))

                if [ "$positional_count" -eq 1 ]; then
                    JARFILE=$1
                else
                    die "Unexpected positional argument: $1"
                fi

                shift
                ;;
        esac
    done

    if [ "$positional_count" -eq 0 ]; then
        printf "Error: dacapo_jarfile positional argument is missing\n" >&2
        print_help >&2
        exit 1
    fi

    [ -f "$JARFILE" ] || die "DaCapo jar file not found: $JARFILE"
    [ -n "$OUT_DIR" ] || OUT_DIR="$HOME/dacapo-min-heap/$(basename "$JARFILE" ".jar")"
    [ -n "$MAX_MEM" ] || MAX_MEM=$(get_default_max_mem) || exit $?
    [ -n "$JDKS" ] || JDKS="$DEFAULT_JDKS"
    [ -n "$GC_OPTIONS" ] || GC_OPTIONS="$DEFAULT_GC_OPTIONS"

    if [ -z "$BENCHMARKS" ]; then
        printf 'No benchmarks specified. Querying jar...\n'
        BENCHMARKS=$(java -jar "$JARFILE" -l) || die "Could not retrieve benchmark list"
        [ -n "$BENCHMARKS" ] || die "'java -jar $JARFILE -l' did not return any benchmarks"
    fi
}

get_default_max_mem() {
    available_mem=$(awk '/MemAvailable/ { print int($2 / 1024) }' /proc/meminfo) || {
        printf "Error: Failed to read available memory\n" >&2
        return 1
    }

    case $available_mem in
        ''|*[!0-9]*)
            printf "Error: Invalid memory value: %s\n" "$available_mem" >&2
            return 1
            ;;
    esac

    if [ "$available_mem" -le "$SAFETY_MARGIN" ]; then
        printf "Error: Insufficient memory: %s MiB. Need at least %d MiB.\n" \
               "$available_mem" "$((SAFETY_MARGIN + 1))" >&2
        return 1
    fi

    printf '%s\n' "$((available_mem - SAFETY_MARGIN))"
}

# System Information
collect_sysinfo() {
    uname -a > "$OUT_DIR/uname.txt" \
        || die "Failed to write system information to $OUT_DIR/uname.txt"
}

collect_jvminfo() {
    java_bin="$1"
    jdk_name="$2"
    jvm_out_dir="$OUT_DIR/$jdk_name"
    mkdir -p "$jvm_out_dir" \
        || die "Failed to create directory $jvm_out_dir"
    "$java_bin" -version > "$jvm_out_dir/java_version.txt" 2>&1 \
        || die "Failed to collect version info for $jdk_name"
}

save_gc_options() {
    printf '%s\n' "$GC_OPTIONS" > "$OUT_DIR/gc_opts.txt" \
        || die "Failed to write garbage collection options to $OUT_DIR/gc_opts.txt"
}

get_java_major_version() {
    java_bin="$1"
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

# Min Heap Search
get_java_options() {
    java_major_version="${1:-0}"
    benchmark="$2"
    mem_mb="$3"

    printf '%s ' \
        "$GC_OPTIONS" \
        "-Xms${mem_mb}m" \
        "-Xmx${mem_mb}m" \
        '-XX:+ExitOnOutOfMemoryError'

    if [ "$benchmark" = "cassandra" ] && [ "$java_major_version" -ge 17 ]; then
        printf '%s ' '-Djava.security.manager=allow'
    fi

    if [ "$benchmark" = "h2o" ] && [ "$java_major_version" -gt 17 ]; then
        printf '%s ' "-Dsys.ai.h2o.debug.allowJavaVersions=$java_major_version"
    fi

    printf '\n'
}

test_heap_size() {
    java_bin="$1"
    java_major_version="$2"
    benchmark="$3"
    mem_mb="$4"
    java_options="$(get_java_options "$java_major_version" "$benchmark" "$mem_mb")"

    # get_java_options outputs a whitespace-separated option list.
    # Intentional word splitting below.
    # shellcheck disable=SC2086
    java_output=$("$java_bin" $java_options -jar "$JARFILE" "$benchmark" 2>&1)
    exit_code=$?

    case "$java_output" in
        # Handles edge cases like running out of direct memory,
        # where the JVM does not exit with OOM_EXIT_CODE.
        *OutOfMemoryError*)
            return "$OOM_EXIT_CODE"
            ;;
        *)
            if [ "$exit_code" -ne 0 ] && [ "$exit_code" -ne "$OOM_EXIT_CODE" ]; then
                # Print Java output for troubleshooting
                printf "%s" "$java_output" >&2
            fi
            return "$exit_code"
            ;;
    esac
}

exp_search_upper_bound() {
    java_bin="$1"
    java_major_version="$2"
    benchmark="$3"
    current=16
    previous=$MIN_MEM

    while [ "$current" -le "$MAX_MEM" ]; do
        test_heap_size "$java_bin" "$java_major_version" "$benchmark" "$current"
        exit_code=$?

        if [ "$exit_code" -eq 0 ]; then
            printf '%s %s\n' "$previous" "$current"
            return 0
        elif [ "$exit_code" -eq "$OOM_EXIT_CODE" ]; then
            previous=$current
            current=$((current * 2))
        else
            return "$exit_code"
        fi
    done

    test_heap_size "$java_bin" "$java_major_version" "$benchmark" "$MAX_MEM"
    exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        printf '%s %s\n' "$previous" "$MAX_MEM"
        return 0
    else
        return "$OOM_EXIT_CODE"
    fi
}

binary_search_min_heap() {
    java_bin="$1"
    java_major_version="$2"
    benchmark="$3"
    low="$4"
    high="$5"
    result=-1

    while [ "$low" -le "$high" ]; do
        mid=$(( (low + high) / 2 ))

        test_heap_size "$java_bin" "$java_major_version" "$benchmark" "$mid"
        exit_code=$?

        if [ "$exit_code" -eq 0 ]; then
            result=$mid
            high=$((mid - 1))
        elif [ "$exit_code" -eq "$OOM_EXIT_CODE" ]; then
            low=$((mid + 1))
        else
            return "$exit_code"
        fi
    done

    if [ "$result" -eq -1 ]; then
        return "$OOM_EXIT_CODE"
    else
        printf '%s\n' "$result"
    fi
}

handle_benchmark_error() {
    benchmark="$1"
    exit_code="$2"
    csv_file="$3"

    if [ "$exit_code" -eq "$OOM_EXIT_CODE" ]; then
        printf '%s requires more than max memory: %s MiB\n' "$benchmark" "$MAX_MEM" >&2
    else
        printf '%s failed with an unexpected exit code: %s\n' "$benchmark" "$exit_code" >&2
    fi

    printf "%s,\n" "$benchmark" >> "$csv_file"
}

compute_min_heap_sizes() {
    for jdk_name in $JDKS; do
        java_bin="$JAVA_INSTALL_ROOT/$jdk_name/bin/java"
        [ -x "$java_bin" ] || { printf "WARN: %s not found, skipping\n" "$java_bin"; continue; }

        printf 'Calculating minimum heap size(s) with %s JDK...\n' "$jdk_name"

        collect_jvminfo "$java_bin" "$jdk_name"
        csv_file="$OUT_DIR/$jdk_name/min_heap.csv"
        printf '%s,%s\n' 'benchmark' 'min_heap_mb' > "$csv_file"

        java_major_version=$(get_java_major_version "$java_bin")
        for benchmark in $BENCHMARKS; do
            bounds=$(exp_search_upper_bound "$java_bin" "$java_major_version" "$benchmark") || {
                handle_benchmark_error "$benchmark" $? "$csv_file"
                continue
            }

            # shellcheck disable=SC2086
            set -- $bounds
            low=$1
            high=$2

            min_heap_size=$(binary_search_min_heap "$java_bin" "$java_major_version" "$benchmark" "$low" "$high")
            exit_code=$?

            if [ "$exit_code" -eq 0 ]; then
                printf '%s,%s\n' "$benchmark" "$min_heap_size" >> "$csv_file"
            else
                handle_benchmark_error "$benchmark" "$exit_code" "$csv_file"
            fi
        done

        printf 'Finished calculating minimum heap size(s) with %s JDK\n' "$jdk_name"
        printf "Results were written to '%s'\n" "$csv_file"
    done
}

main() (
    parse_args "$@"

    IFS=$(printf ' \t\n')
    mkdir -p "$OUT_DIR" || die "Could not create output directory: $OUT_DIR"
    collect_sysinfo
    save_gc_options

    compute_min_heap_sizes
)

main "$@"
