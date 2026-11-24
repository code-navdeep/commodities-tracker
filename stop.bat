@echo off
cd /d "%~dp0"
echo Stopping Commodities Tracker...
rem --remove-orphans: also removes any containers not defined in the current
rem docker-compose.yml, so nothing lingers behind.
docker compose down --remove-orphans
echo Stopped.
pause
