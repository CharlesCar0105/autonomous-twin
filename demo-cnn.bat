@echo off
rem Demo : simulateur + pilote CNN (le reseau qui conduit par imitation)
rem Plus lent que le PID : il imite le pilote d'avant optimisation (normal).
cd /d "%~dp0"
start "Simulateur - gen_014" .venv\Scripts\python.exe -m simulator.main --server --circuit gen_014
timeout /t 3 /nobreak >nul
start "Pilote CNN + dashboard" .venv\Scripts\python.exe -m pilot.main --policy cnn --dashboard
