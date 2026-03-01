# Infra deploy

This directory contains scripts to deploy only Docker services from `docker-compose.yml`:

- `telegram-bot-api`
- `victoria-metrics`
- `grafana`

The Python bot is not deployed.

## Deploy to SSH server

```sh
./infra/deploy_ssh.sh user@server ~/psy-protocol-infra
```

If `.env` does not exist on server, script creates it from `infra/.env.server.example`
and stops so you can fill required credentials.
