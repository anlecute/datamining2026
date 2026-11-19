$ErrorActionPreference = 'Stop'

$runtime = Join-Path $PSScriptRoot 'weka-runtime'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null

$downloads = @(
    @{
        Name = 'weka-stable-3.8.6.jar'
        Url = 'https://repo.maven.apache.org/maven2/nz/ac/waikato/cms/weka/weka-stable/3.8.6/weka-stable-3.8.6.jar'
    },
    @{
        Name = 'bounce-0.18.jar'
        Url = 'https://repo.maven.apache.org/maven2/nz/ac/waikato/cms/weka/thirdparty/bounce/0.18/bounce-0.18.jar'
    },
    @{
        Name = 'java-cup-runtime-11b-20160615.jar'
        Url = 'https://repo.maven.apache.org/maven2/com/github/vbmacher/java-cup-runtime/11b-20160615/java-cup-runtime-11b-20160615.jar'
    }
)

foreach ($artifact in $downloads) {
    $destination = Join-Path $runtime $artifact.Name
    if (-not (Test-Path -LiteralPath $destination -PathType Leaf)) {
        Write-Host "Downloading $($artifact.Name)"
        Invoke-WebRequest -Uri $artifact.Url -OutFile $destination
    }
}

$packageDirectory = Join-Path $env:USERPROFILE 'wekafiles\packages\LibLINEAR'
$libLinearJar = Join-Path $packageDirectory 'LibLINEAR.jar'
$linearCoreJar = Join-Path $packageDirectory 'lib\liblinear-java-1.96-SNAPSHOT.jar'
if (-not (Test-Path -LiteralPath $libLinearJar -PathType Leaf) -or
    -not (Test-Path -LiteralPath $linearCoreJar -PathType Leaf)) {
    $wekaJar = Join-Path $runtime 'weka-stable-3.8.6.jar'
    $bounceJar = Join-Path $runtime 'bounce-0.18.jar'
    $cupJar = Join-Path $runtime 'java-cup-runtime-11b-20160615.jar'
    $classpath = @($wekaJar, $bounceJar, $cupJar) -join [IO.Path]::PathSeparator
    & java -cp $classpath weka.core.WekaPackageManager -install-package LibLINEAR 1.9.8
    if ($LASTEXITCODE -ne 0) { throw 'Weka LibLINEAR package installation failed.' }
}

if (-not (Test-Path -LiteralPath $libLinearJar -PathType Leaf) -or
    -not (Test-Path -LiteralPath $linearCoreJar -PathType Leaf)) {
    throw "Weka LibLINEAR package files were not found under $packageDirectory"
}
Copy-Item -LiteralPath $libLinearJar -Destination (Join-Path $runtime 'LibLINEAR.jar') -Force
Copy-Item -LiteralPath $linearCoreJar -Destination (Join-Path $runtime 'liblinear-java-1.96-SNAPSHOT.jar') -Force

Write-Host 'Java/Weka dependencies are ready. Run .\run.bat convert, cv, or train.'
