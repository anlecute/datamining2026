@echo off
setlocal
set "ROOT=%~dp0"
set "RUNTIME=%ROOT%weka-runtime"
set "BUILD=%ROOT%.build"
set "WEKA_JAR=%RUNTIME%\weka.jar"
if exist "C:\Program Files\Weka-3-9-6\weka.jar" set "WEKA_JAR=C:\Program Files\Weka-3-9-6\weka.jar"
set "CLASSPATH=%WEKA_JAR%;%BUILD%"
if /I "%WEKA_JAR%"=="%RUNTIME%\weka.jar" set "CLASSPATH=%RUNTIME%\*;%BUILD%"
where javac >nul 2>nul || (echo Install JDK 11 or newer and add it to PATH.& exit /b 1)
if not exist "%WEKA_JAR%" (echo Run setup.ps1 first.& exit /b 1)
if not exist "%BUILD%" mkdir "%BUILD%"
javac -encoding UTF-8 -cp "%CLASSPATH%" -d "%BUILD%" "%ROOT%MnbData.java" "%ROOT%MnbFactory.java" "%ROOT%ConvertToArff.java" "%ROOT%train.java" "%ROOT%evaluate.java"
if errorlevel 1 exit /b 1
if /I "%~1"=="convert" if not "%~3"=="" goto convert
if /I "%~1"=="train" if not "%~3"=="" goto train
if /I "%~1"=="evaluate" if not "%~4"=="" goto evaluate
echo Usage: run.bat convert input.jsonl output.arff
echo        run.bat train input.arff output.model
echo        run.bat evaluate input.arff trained.model results.json
exit /b 1

:convert
java --add-opens=java.base/java.lang=ALL-UNNAMED -Duser.home="%TEMP%" -Xmx8g -cp "%CLASSPATH%" ConvertToArff "%~2" "%~3"
exit /b %errorlevel%
:train
java --add-opens=java.base/java.lang=ALL-UNNAMED -Duser.home="%TEMP%" -Xmx8g -cp "%CLASSPATH%" train "%~2" "%~3"
exit /b %errorlevel%
:evaluate
java --add-opens=java.base/java.lang=ALL-UNNAMED -Duser.home="%TEMP%" -Xmx8g -cp "%CLASSPATH%" evaluate "%~2" "%~3" "%~4"
exit /b %errorlevel%




