@echo off
REM Pushes this folder to GitHub with the folder structure intact.
REM Replaces whatever is currently in the repository (fixes the flat upload).
cd /d "%~dp0"
git --version >nul 2>&1
if errorlevel 1 (
  echo Git is not installed. Install it from https://git-scm.com/download/win , then double-click this file again.
  pause
  exit /b 1
)
if exist .git rmdir /s /q .git
git init
git add -A
git -c user.name="Saurav Verma" -c user.email="sauravverma26m@users.noreply.github.com" commit -m "WAI project: pipeline, data, outputs, report"
git branch -M main
git remote add origin https://github.com/sauravverma26m-source/Solar-Resilience-SCM.git
git push -u origin main --force
echo.
echo Done. Open https://github.com/sauravverma26m-source/Solar-Resilience-SCM to check the folders.
pause
