@echo off
rem Demo : simulateur + pilote CNN (le reseau qui conduit par imitation)
rem Circuit gen_003, le plus roulant : le CNN y boucle proprement ses tours
rem (sur gen_014 il cale a l'epingle triple, comportement documente).
rem Plus lent que le PID : il imite le pilote d'avant optimisation (normal).
cd /d "%~dp0"
start "Simulateur - gen_003" .venv\Scripts\python.exe -m simulator.main --server --circuit gen_003
timeout /t 3 /nobreak >nul
start "Pilote CNN + dashboard" .venv\Scripts\python.exe -m pilot.main --policy cnn --dashboard
