@echo off
setlocal
set "ROOT=%~dp0"
set "RUNTIME=%ROOT%weka-runtime"
set "BUILD=%ROOT%.build"
set "CLASSPATH=%RUNTIME%\weka-stable-3.8.6.jar;%RUNTIME%\bounce-0.18.jar;%RUNTIME%\java-cup-runtime-11b-20160615.jar;%RUNTIME%\LibLINEAR.jar;%RUNTIME%\liblinear-java-1.96-SNAPSHOT.jar;%BUILD%"

where java >nul 2>nul || (echo Java was not found. Install a JDK 8+ and add it to PATH.& exit /b 1)
where javac >nul 2>nul || (echo javac was not found. Install a JDK, not only a JRE.& exit /b 1)
if not exist "%RUNTIME%\weka-stable-3.8.6.jar" (echo Run setup.ps1 first to download Weka dependencies.& exit /b 1)
if not exist "%RUNTIME%\LibLINEAR.jar" (echo Run setup.ps1 first to install the Weka LibLINEAR package.& exit /b 1)

if not exist "%BUILD%" mkdir "%BUILD%"
javac -encoding UTF-8 -cp "%CLASSPATH%" -d "%BUILD%" "%ROOT%SvmData.java" "%ROOT%SvmClassifierFactory.java" "%ROOT%ConvertToArff.java" "%ROOT%Train.java" "%ROOT%Evaluate.java"
if errorlevel 1 exit /b %errorlevel%

if /I "%~1"=="convert" goto convert
if /I "%~1"=="train" goto train
if /I "%~1"=="evaluate" goto evaluate
echo Usage:
echo   run.bat convert ^<input.jsonl^> ^<output.arff^>
echo   run.bat train ^<input.arff^> ^<output.model^>
echo   run.bat evaluate ^<input.arff^> ^<trained.model^> ^<results.json^>
exit /b 1

:convert
if "%~3"=="" (echo Usage: run.bat convert ^<input.jsonl^> ^<output.arff^>& exit /b 1)
java -Xmx8g -cp "%CLASSPATH%" ConvertToArff "%~2" "%~3"
exit /b %errorlevel%

:train
if "%~3"=="" (echo Usage: run.bat train ^<input.arff^> ^<output.model^>& exit /b 1)
java -Xmx8g -cp "%CLASSPATH%" Train "%~2" "%~3"
exit /b %errorlevel%

:evaluate
if "%~4"=="" (echo Usage: run.bat evaluate ^<input.arff^> ^<trained.model^> ^<results.json^>& exit /b 1)
java -Xmx8g -cp "%CLASSPATH%" Evaluate "%~2" "%~3" "%~4"
exit /b %errorlevel%
