#!/bin/bash
cd "$(dirname "$0")"
echo "Starting Commodities Tracker..."
# Docker's layer cache makes this near-instant when nothing changed: a code
# change rebuilds only the final copy layer, and a requirements.lock change
# automatically triggers a full dependency reinstall — no --no-cache needed.
# (To force a truly fresh build after a base-image security update, run
# `docker compose build --pull --no-cache` manually.)
docker compose build
docker compose up -d
if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Container failed to start — see the Docker error above."
    exit 1
fi

echo "Waiting for the dashboard to come up..."
ready=0
for i in $(seq 1 60); do
    if curl -s -o /dev/null http://localhost:8057; then
        ready=1
        break
    fi
    sleep 1
done

if [ "$ready" -ne 1 ]; then
    echo ""
    echo "✗ Container started but the dashboard did not respond within 60 seconds."
    echo "  Check logs with: docker compose logs"
    exit 1
fi

open http://localhost:8057
echo ""
echo "Dashboard is running at http://localhost:8057"
echo "You can close this window."
