@echo off
rem Demo soutenance : simulateur + pilote PID (la config reglementaire du jour J)
rem Double-clic depuis la racine du repo. Fermer les 2 fenetres pour arreter.
cd /d "%~dp0"
start "Simulateur - gen_014" .venv\Scripts\python.exe -m simulator.main --server --circuit gen_014
timeout /t 3 /nobreak >nul
start "Pilote PID + dashboard" .venv\Scripts\python.exe -m pilot.main --dashboard
