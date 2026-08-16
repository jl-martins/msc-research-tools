#!/usr/bin/env bash

# shellcheck disable=SC2034

# Nominal minimum heap size (MB) for default size configuration (with compressed pointers)
declare -rgA MIN_HEAP=(
    [batik]=185
    [biojava]=97
    [cassandra]=104
    [fop]=17
    [h2]=707
    [h2o]=58
    [jme]=29
    [jython]=27
    [kafka]=209
    [luindex]=38
    [lusearch]=21
    [pmd]=99
    [spring]=52
    [tomcat]=17
    [zxing]=75
)

# Default implementations of the benchmark suite
DEFAULT_IMPLEMENTATIONS=(loops streams)

# Populates an indexed array with all DaCapo benchmarks that support the
# Java Streams API and are compatible with the specified major version.
#
# Arguments:
#   $1 - Java major version (e.g., 21)
#   $2 - Name of the output array (nameref)
#
# Returns:
#   0 on success, 1 if the Java version is unsupported or invalid.
#
# Outputs:
#   Prints an error to stderr if the Java version is unsupported or invalid.
#
get_default_benchmarks() {
    local java_major_version="$1"
    local -n benchmarks_ref="$2"

    if [[ ! "$java_major_version" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid Java major version: '$java_major_version'" >&2
        return 1
    fi

    if (( java_major_version >= 8 && java_major_version <= 10 )); then
        benchmarks_ref=(batik fop h2o jme jython kafka pmd spring zxing)
    elif (( java_major_version >= 11 && java_major_version <= 23 )); then
        benchmarks_ref=(
            batik biojava cassandra fop h2 h2o jme jython
            kafka luindex lusearch pmd spring tomcat zxing
        )
    elif (( java_major_version >= 24 && java_major_version <= 25 )); then
        benchmarks_ref=(
            batik biojava fop h2 h2o jme jython kafka
            luindex lusearch pmd spring tomcat zxing
        )
    else
        echo "Unsupported Java major version: '$java_major_version'" >&2
        return 1
    fi
}

# Provides additional Java options for a specific Java version and benchmark.
#
# Arguments:
#   $1 - Java major version (e.g., 21)
#   $2 - Benchmark name (e.g., "cassandra")
#   $3 - Name of the output array (nameref). Separate from the main array of
#        Java options to ensure that this function cannot modify base options.
#
extend_java_options() {
    local java_major_version="$1"
    local benchmark="$2"
    local -n additional_options_ref=$3

    if [[ "$benchmark" = "cassandra" && "$java_major_version" -ge 17 ]]; then
        additional_options_ref+=("-Djava.security.manager=allow")
    fi

    if [[ "$benchmark" = "h2o" && "$java_major_version" -gt 17 ]]; then
        additional_options_ref+=("-Dsys.ai.h2o.debug.allowJavaVersions=$java_major_version")
    fi
}

# Provides additional options for the benchmark suite.
#
# Arguments:
#   $1 - Name of the output array (nameref). Separate from the main array of
#        suite options to ensure that this function cannot modify base options.
#
extend_suite_options() {
    # shellcheck disable=SC2178  # false positive
    local -n additional_options_ref=$1
    additional_options_ref+=(--callback "org.dacapo.rapl.RAPLCallback")
}

# Runs before the first invocation of each benchmark, for every
# implementation of the benchmark suite and selected JDK.
pre_benchmark() {
    rm -f "measurements.csv" # remove stale measurements (if any)
}

# Runs after the last invocation of each benchmark, for every
# implementation of the benchmark suite and selected JDK.
#
# Arguments:
#   $1 - The output directory
#   $2 - Benchmark context for logging
#
post_benchmark() {
    local bm_out_dir="$1"
    local bm_context="$2"

    if [[ -f "measurements.csv" ]]; then
        mv measurements.csv "$bm_out_dir"
        log "INFO" "Done! Measurements stored at '$bm_out_dir/measurements.csv'"
    else
        log "WARN" "measurements.csv not found for $bm_context. Skipping..."
    fi
}
