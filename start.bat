@echo off
cd /d "%~dp0"
echo Starting Commodities Tracker...
rem Docker's layer cache makes this near-instant when nothing changed: a code
rem change rebuilds only the final copy layer, and a requirements.lock change
rem automatically triggers a full dependency reinstall — no --no-cache needed.
rem (To force a truly fresh build after a base-image security update, run
rem `docker compose build --pull --no-cache` manually.)
docker compose build
docker compose up -d
if errorlevel 1 (
    echo.
    echo [FAILED] Container failed to start - see the Docker error above.
    pause
    exit /b 1
)

echo Waiting for the dashboard to come up...
for /l %%i in (1,1,60) do (
    curl -s -o nul http://localhost:8057
    if not errorlevel 1 goto ready
    timeout /t 1 /nobreak >nul
)
echo.
echo [FAILED] Container started but the dashboard did not respond within 60 seconds.
echo          Check logs with: docker compose logs
pause
exit /b 1
:ready

start http://localhost:8057
echo.
echo Dashboard is running at http://localhost:8057
pause
