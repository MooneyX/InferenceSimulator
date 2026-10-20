@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy-local.ps1"
set "deployExitCode=%ERRORLEVEL%"
if not "%deployExitCode%"=="0" pause
exit /b %deployExitCode%
