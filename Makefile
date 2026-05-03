LABEL = com.psy-protocol.bot
BOOT_DOMAIN = gui/$(shell id -u)
BOOT_TARGET = $(BOOT_DOMAIN)/$(LABEL)
PLIST = $(HOME)/Library/LaunchAgents/$(LABEL).plist

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
