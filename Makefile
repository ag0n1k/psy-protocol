LABEL = com.psy-protocol.bot
BOOT_DOMAIN = gui/$(shell id -u)
BOOT_TARGET = $(BOOT_DOMAIN)/$(LABEL)
PLIST = $(HOME)/Library/LaunchAgents/$(LABEL).plist

WATCHDOG_LABEL = com.psy-protocol.watchdog
WATCHDOG_PLIST = $(HOME)/Library/LaunchAgents/$(WATCHDOG_LABEL).plist

service-install:
	bash infra/macos/install.sh

service-uninstall:
	bash infra/macos/uninstall.sh

service-bootstrap:
	launchctl bootstrap $(BOOT_DOMAIN) $(PLIST)

service-bootout:
	launchctl bootout $(BOOT_TARGET)

service-start:
	launchctl start $(LABEL)

service-stop:
	launchctl stop $(LABEL)

service-restart:
	launchctl stop $(LABEL) && launchctl start $(LABEL)

service-status:
	launchctl list | grep $(LABEL) || echo "not loaded"

service-logs:
	tail -f logs/bot.stdout.log logs/bot.stderr.log

watchdog-install:
	bash infra/macos/install_watchdog.sh

watchdog-uninstall:
	launchctl unload $(WATCHDOG_PLIST) && rm -f $(WATCHDOG_PLIST)

watchdog-run:
	bash infra/macos/watchdog.sh

watchdog-status:
	launchctl list | grep $(WATCHDOG_LABEL) || echo "not loaded"

watchdog-logs:
	tail -f logs/watchdog.log
