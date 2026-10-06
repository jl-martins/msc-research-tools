# ==============================================================================
# CONFIGURATION
# ==============================================================================

prefix = /usr/local
exec_prefix = $(prefix)
bindir = $(exec_prefix)/bin
sysconfdir = $(prefix)/etc

binmod := 0755
sysconfmod := 0644

sh_tools := jbench-run
uv_tools := bench-stats dacapo-hot-loops dacapo-min-heap format-comparisons
tools := $(sh_tools) $(uv_tools)

uninstall_targets := $(addprefix uninstall-,$(tools))

export UV_TOOL_BIN_DIR ?= $(bindir)
export UV_TOOL_DIR ?= $(exec_prefix)/share/uv/tools
export UV_PYTHON_INSTALL_DIR ?= $(exec_prefix)/share/uv/python

.PHONY: all install uninstall clean $(tools) $(uninstall_targets)

# ==============================================================================
# LOGGING
# ==============================================================================

log_start := @printf "%s... "
log_done := @echo "Done"

# ==============================================================================
# TOOL DETECTION
# ==============================================================================

UV ?= $(shell \
	command -v uv 2>/dev/null || \
	sudo -u "$(SUDO_USER)" -i command -v uv 2>/dev/null || \
	sudo -u "$(SUDO_USER)" -i mise which uv 2>/dev/null \
)

ifeq ($(UV),)
  $(error $(strip \
    Could not locate 'uv'. \
    Please ensure it is available in the system PATH or \
    set the UV variable to the path of the uv binary. \
  ))
endif

# ==============================================================================
# INSTALLATION TARGETS
# ==============================================================================
all: install

$(bindir) $(sysconfdir):
	$(log_start) "Creating directory '$@'"
	@install -d "$@"
	$(log_done)

install: $(tools)
	@echo "Installation complete!"

jbench-run: | $(bindir) $(sysconfdir)
	$(log_start) "Installing $@"
	@install -m $(binmod) "$@/$@.sh" "$(bindir)/$@"
	@cp -R "$@/config/." "$(sysconfdir)/$@/"
	@find "$(sysconfdir)/$@/" -type f -exec chmod $(sysconfmod) {} +
	$(log_done)

$(uv_tools): | $(sysconfdir)
	$(log_start) "Installing $@"
	@$(UV) -q tool install "$@/"
	@if [ -d "$@/config" ]; then \
		cp -R "$@/config/." "$(sysconfdir)/$@/"; \
		find "$(sysconfdir)/$@/" -type f -exec chmod $(sysconfmod) {} +; \
	fi
	@rm -rf $@/build $@/*.egg-info $@/__pycache__
	$(log_done)

# ==============================================================================
# UNINSTALLATION TARGETS
# ==============================================================================
uninstall: $(uninstall_targets)
	@echo "Uninstallation complete!"

uninstall-jbench-run:
	$(log_start) "Uninstalling jbench-run"
	@rm -f "$(bindir)/jbench-run"
	@rm -rf "$(sysconfdir)/jbench-run"
	$(log_done)

$(addprefix uninstall-,$(uv_tools)): uninstall-%:
	$(log_start) "Uninstalling $*"
	@if $(UV) tool list | grep -q '^$* '; then \
		$(UV) -q tool uninstall $*; \
	fi
	@rm -rf "$(sysconfdir)/$*"
	$(log_done)
