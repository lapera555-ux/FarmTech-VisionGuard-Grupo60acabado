@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
 py -3.11 --version >nul 2>nul
 if not errorlevel 1 (
  set "PY=py -3.11"
 ) else (
  set "PY=py -3"
 )
) else (
 set "PY=python"
)
if not exist ".venv\Scripts\python.exe" (
 echo [1/3] Criando ambiente virtual Python...
 %PY% -m venv .venv
 if errorlevel 1 goto :erro
)
if not exist ".venv\.farmtech_ready" (
 echo [2/3] Instalando dependencias da aplicacao (Internet no primeiro uso)...
 ".venv\Scripts\python.exe" -m pip install -r operacao_rural\requirements_app.txt
 if errorlevel 1 goto :erro
 echo Dependencias instaladas.>".venv\.farmtech_ready"
) else (
 echo [2/3] Dependencias ja instaladas, inicializacao rapida.
)
echo [3/3] Abrindo painel local http://127.0.0.1:8501
echo Para sair, feche esta janela ou pressione CTRL+C.
".venv\Scripts\python.exe" -m streamlit run app_fazenda.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false --server.maxUploadSize 120
if errorlevel 1 goto :erro
goto :fim
:erro
echo.
echo ERRO: Confira se Python 3.10-3.12 esta instalado e ha Internet.
pause
:fim
endlocal
