#!/bin/bash
cd "$(dirname "$0")"
echo "Stopping Commodities Tracker..."
# --remove-orphans: also removes any containers not defined in the current
# docker-compose.yml, so nothing lingers behind.
docker compose down --remove-orphans
echo "Stopped."
