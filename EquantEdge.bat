@echo off
title EquantEdge Multi-Asset Quant Terminal
cd /d "%~dp0"
if exist "dist\EquantEdge.exe" (
    start "" "dist\EquantEdge.exe" --live
) else (
    start "" pythonw -m src.dashboard --live
)
exit
