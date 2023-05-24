@echo off

rem Check if the specific Python runtime folder exists
if exist "runtime\python.exe" (
    echo Running using the specific Python runtime.
    runtime/python.exe rvcgui.py --pycmd runtime/python.exe
pause

) else (
    echo Running using the system Python.
     python.exe rvcgui.py --pycmd python.exe
pause
)