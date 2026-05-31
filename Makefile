# ==============================================================================
# CONFIGURATION
# ==============================================================================

prefix = /usr/local
exec_prefix = $(prefix)
bindir = $(exec_prefix)/bin
sbindir = $(exec_prefix)/sbin
sysconfdir = $(prefix)/etc

binmod := 0755
sbinmod := 0700
sysconfmod := 0644

tools :=
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

$(bindir) $(sbindir) $(sysconfdir):
	$(log_start) "Creating directory '$@'"
	@install -d "$@"
	$(log_done)

install: $(tools)
	@echo "Installation complete!"

# ==============================================================================
# UNINSTALLATION TARGETS
# ==============================================================================

uninstall: $(uninstall_targets)
	@echo "Uninstallation complete!"
