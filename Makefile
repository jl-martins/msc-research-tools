# ==============================================================================
# CONFIGURATION
# ==============================================================================

prefix = /usr/local
exec_prefix = $(prefix)
bindir = $(exec_prefix)/bin
sysconfdir = $(prefix)/etc

binmod := 0755
sysconfmod := 0644

sh_tools := dacapo-min-heap dacapo-run
uv_tools := dacapo-stats format-comparisons
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

dacapo-min-heap: | $(bindir)
	$(log_start) "Installing $@"
	@install -m $(binmod) "$@/$@.sh" "$(bindir)/$@"
	$(log_done)

dacapo-run: | $(bindir) $(sysconfdir)
	$(log_start) "Installing $@"
	@install -m $(binmod) "$@/$@.sh" "$(bindir)/$@"
	@install -d "$(sysconfdir)/$@"
	@install -m $(sysconfmod) "$@"/config/* "$(sysconfdir)/$@/"
	$(log_done)

$(uv_tools):
	$(log_start) "Installing $@"
	@$(UV) -q tool install "$@/"
	@rm -rf $@/build $@/*.egg-info $@/__pycache__
	$(log_done)

# ==============================================================================
# UNINSTALLATION TARGETS
# ==============================================================================
uninstall: $(uninstall_targets)
	@echo "Uninstallation complete!"

uninstall-dacapo-min-heap:
	$(log_start) "Uninstalling dacapo-min-heap"
	@rm -f $(bindir)/dacapo-min-heap
	$(log_done)

uninstall-dacapo-run:
	$(log_start) "Uninstalling dacapo-run"
	@rm -f "$(bindir)/dacapo-run"
	@rm -rf "$(sysconfdir)/dacapo-run"
	$(log_done)

$(addprefix uninstall-,$(uv_tools)): uninstall-%:
	$(log_start) "Uninstalling $*"
	@if $(UV) tool list | grep -q '^$* '; then \
		$(UV) -q tool uninstall $*; \
	fi
	$(log_done)
