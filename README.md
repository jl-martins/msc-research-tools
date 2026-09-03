# MSc Research Tools

Custom research tools developed for my Master's Dissertation on the Impact of Code Rejuvenation on Energy Consumption.

The tools and their purposes are summarized below:

| Tool                  | Language | Purpose                                                                      |
| --------------------- | -------- | ---------------------------------------------------------------------------- |
| `bench-stats`         | python   | Aggregate results and computes non-parametric summary statistics.            |
| `dacapo-min-heap`     | python   | Find the minimum heap size required by DaCapo benchmarks in different JDKs.  |
| `format-comparisons`  | python   | Format effect-size comparisons into Excel workbooks.                         |
| `jbench-run`          | bash     | Run multiple implementations of a Java benchmark suite across multiple JDKs. |

The following sections explain how to install, use and uninstall each tool.

## Prerequisites

* **jbench-run**
  * OS: Linux
  * Dependencies: [Bash](https://www.gnu.org/savannah-checkouts/gnu/bash/manual/bash.html) (>= 4.3), [GNU Coreutils](https://www.gnu.org/software/coreutils/manual/html_node/index.html), [Java](https://www.oracle.com/java/)

* **dacapo-min-heap**
  * OS: Any POSIX-compliant operating system
  * Dependencies: [Java](https://www.oracle.com/java/), [Python](https://www.python.org/) (>= 3.14), [uv](https://docs.astral.sh/uv/)

* **bench-stats** and **format-comparisons**
  * OS: Cross-platform (Linux, macOS, Windows, etc.)
  * Dependencies: [Python](https://www.python.org/) (>= 3.14), [uv](https://docs.astral.sh/uv/)

## Installation

### GNU make

In the root directory of this project, you will find a [GNU Makefile](https://www.gnu.org/software/make/manual/html_node/index.html) that can be used to install every tool. All `make` commands in this README should be executed in that directory.

To install all tools to `/usr/local`, run:

```sh
sudo make install
```

The installation directories can be overridden with the following [standard GNU variables for installation directories](https://ftp.gwdg.de/pub/gnu/www/prep/standards/html_node/Directory-Variables.html):

* `prefix` - A prefix used in constructing the default values of the variables listed below (default: `/usr/local`)
* `exec_prefix` - A prefix used in constructing the default values of some of the variables listed below (default: `$(prefix)`)
* `bindir` - The directory for installing executable programs that users can run (default: `$(exec_prefix)/bin`)
* `sysconfdir` - The directory for installing configuration files (default: `$(prefix)/etc`)

For example, to install all tools to `$HOME/.local`, run:

```sh
make prefix="$HOME/.local" install
```

There are also individual installation targets for each tool. To install a specific tool, simply run:

```sh
sudo make <tool_name>
```

where `<tool_name>` is a placeholder for the name of the tool (e.g., `jbench-run`).

> [!NOTE]
> The installation directories can be overridden for all installation and uninstallation targets in the same way as for `make install`.

### Manual Installation

Although GNU make is the preferred installation method, each tool can be manually installed as follows.

#### jbench-run

```sh
cp jbench-run/jbench-run.sh <bindir>/jbench-run
cp -R jbench-run/config/. <sysconfdir>/jbench-run/
```

where `<bindir>` and `<sysconfdir>` are placeholders for the binary and configuration directories, respectively.

#### bench-stats, dacapo-min-heap and format-comparisons

```sh
uv tool install <tool_name>/ 
rm -rf <tool_name>/build <tool_name>/*.egg-info <tool_name>/__pycache__ # optional (removes Python build artifacts)
```

## Uninstallation

### GNU make

Each installation target has a corresponding uninstallation target.

To uninstall all tools from `/usr/local`, run:

```sh
sudo make uninstall
```

Custom installation directories can be specified in the same way as for installation targets. For example, if the tools were installed with `make prefix="$HOME/.local" install`, they can be uninstalled with:

```sh
make prefix="$HOME/.local" uninstall
```

Each tool also has its own uninstallation target, which can be invoked as follows:

```sh
sudo make uninstall-<tool_name>
```

### Manual Uninstallation

The manual uninstallation commands are essentially the opposite of the [manual installation commands](#installation--manual-instalation).

#### `jbench-run`

```sh
rm <bindir>/jbench-run
rm -rf <sysconfdir>/jbench-run/
```

#### `bench-stats`, `dacapo-min-heap` and `format-comparisons`

```sh
uv tool uninstall <tool_name>
```

## Usage Examples

```bash
# 1. Set the Java installation root. This is the directory where `jbench-run`
#    and `dacapo-min-heap` will search for JDKs. This example uses the default
#    installation root managed by mise (https://github.com/jdx/mise).
export JAVA_INSTALL_ROOT="$HOME/.local/share/mise/installs/java"

# 2. Select the target JDKs and garbage collector options.
#    Every JDK j must be installed at "$JAVA_INSTALL_ROOT/$j".
jdks="temurin-8 temurin-11 temurin-17 temurin-21 temurin-25"
gc_opts="-XX:+UseG1GC"

# 3. Set the name, version and implementations of the benchmark suite for the
#    following examples, along with the directory that contains the benchmark
#    suite JARs. We will use the loop-based and the stream-based implementation
#    from https://github.com/jl-martins/msc-dacapobench/tags
#
#    The JAR of every implementation i must be at
#    "$artifact_dir/$suite_name-$suite_version-$i.jar".
suite_name="dacapo"
suite_version="23.11-MR2-chopin-jlm"
implementations="loops streams"
artifact_dir="$HOME/benchmark-jars"

# 4. Find the minimum heap size of each benchmark for each selected implementation
#    and JDK, using the specified garbage collector options. The results will be
#    written to `$HOME/dacapo-min-heap/<jar_name>`, where <jar_name> is the name
#    of the JAR file without the extension.
for impl in $implementations; do
  dacapo-min-heap \
      --jdks="$jdks" \
      --gc-opts="$gc_opts" \
      "$artifact_dir/$suite_name-$suite_version-$impl.jar"
done

# 5.1. Open the suite configuration file in a text editor and update the `MIN_HEAP`
#      associative array according to the output of dacapo-min-heap.
#
#      Note that jbench-run only supports one `MIN_HEAP` array per suite name and
#      version because the values were mostly identical for all JDKs and
#      implementations used in my Master's Dissertation.
# 
#      The configuration file can implement several hooks to extend the base
#      functionality of jbench-run. All supported hooks are documented in the help
#      message of jbench-run.
#
#      The following examples assume that jbench-run is using the default hooks from
#      `jbench-run/config/dacapo/23.11-MR2-chopin-jlm-config.sh`, which use the
#      RAPLCallback from https://github.com/jl-martins/msc-dacapobench
#      to measure execution time and energy consumption.
sudoedit /usr/local/etc/jbench-run/dacapo/23.11-MR2-chopin-jlm-config.sh

# 5.2. Run the selected implementations of the benchmark suite across the selected
#      JDKs, once per execution scenario. Each benchmark will run with 2 times the
#      minimum heap size configured in point 5.1 and will use the specified garbage
#      collector options. jbench-run will invoke each benchmark 20 times per JDK and
#      implementation (--invocations 20); each invocation runs in a new Java process.
#
#      The "cold" scenario passes "-n 1" to the DaCapo benchmark harness to run a
#      single measurement iteration per Java process, whereas the "warm" scenario
#      passes "-n 5" to run 5 iterations in the same JVM (4 warm-up iterations
#      followed by a measurement iteration).
#
#      The results of each scenario are written to a dedicated output directory at
#      "$jbench_run_base_out_dir/<scenario>".
#
#      jbench-run must run as root to measure energy consumption with RAPL, so
#      JAVA_INSTALL_ROOT is passed explicitly to survive the environment reset
#      performed by sudo.
jbench_run_base_out_dir="$HOME/jbench-run-results/$suite_name/$suite_version"
declare -A scenario_to_suite_options=(
    [cold]="-n 1"
    [warm]="-n 5"
)
for scenario in "${!scenario_to_suite_options[@]}"; do
  sudo JAVA_INSTALL_ROOT="$JAVA_INSTALL_ROOT" jbench-run \
      --jdks "$jdks" \
      --gc-opts "$gc_opts" \
      --heap-multiplier 2 \
      --artifact-dir "$artifact_dir" \
      --suite-name "$suite_name" \
      --suite-version "$suite_version" \
      --suite-additional-opts "${scenario_to_suite_options[$scenario]}" \
      --implementations "$implementations" \
      --invocations 20 \
      --out-dir "$jbench_run_base_out_dir/$scenario"
done

# 5.3. Take ownership of the results, which were created by root, so that the
#      remaining steps can write their outputs to the same directory tree.
sudo chown -R "$USER" "$jbench_run_base_out_dir"

# 6. Generate benchmark statistics
bench-stats "$jbench_run_base_out_dir"

# 7. Format the comparisons from bench-stats into a styled Excel workbook
format-comparisons "$jbench_run_base_out_dir"
```

## Command-Line Reference

The complete command-line reference of each tool can be read with the `--help` option.
