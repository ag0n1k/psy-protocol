# Infra deploy

This directory contains scripts to deploy only Docker services from `docker-compose.yml`:

- `telegram-bot-api`
- `victoria-metrics`
- `grafana`

The Python bot is not deployed.

## Build archive locally

```sh
./infra/build_bundle.sh
```

Creates `infra/psy-protocol-infra-<timestamp>.tar.gz` with required files.
The bundle includes `docker-compose.yml` and `infra/monitoring/*` configs.

## Deploy to SSH server

```sh
./infra/deploy_ssh.sh user@server ~/psy-protocol-infra
```

If `.env` does not exist on server, script creates it from `infra/.env.server.example`
and stops so you can fill required credentials.
