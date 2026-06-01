# ==============================================================================
# CONFIGURATION
# ==============================================================================

prefix = /usr/local
exec_prefix = $(prefix)
bindir = $(exec_prefix)/bin
sysconfdir = $(prefix)/etc

binmod := 0755
sysconfmod := 0644

tools := dacapo-min-heap dacapo-run
uninstall_targets := $(addprefix uninstall-,$(tools))

.PHONY: all install uninstall $(tools) $(uninstall_targets)

# ==============================================================================
# LOGGING
# ==============================================================================

log_start := @printf "%s... "
log_done := @echo "Done"

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
