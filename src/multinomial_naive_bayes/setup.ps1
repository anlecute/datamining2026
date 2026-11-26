$ErrorActionPreference = 'Stop'
$runtime = Join-Path $PSScriptRoot 'weka-runtime'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$localWeka = 'C:\Program Files\Weka-3-9-6\weka.jar'
if (Test-Path -LiteralPath $localWeka -PathType Leaf) {
    Write-Host 'Using the installed Weka 3.9.6; do not mix versions when reusing models.'
    exit 0
}
$artifacts = @(
    @{Name='weka.jar'; Url='https://repo.maven.apache.org/maven2/nz/ac/waikato/cms/weka/weka-stable/3.8.6/weka-stable-3.8.6.jar'},
    @{Name='bounce-0.18.jar'; Url='https://repo.maven.apache.org/maven2/nz/ac/waikato/cms/weka/thirdparty/bounce/0.18/bounce-0.18.jar'},
    @{Name='java-cup-runtime-11b-20160615.jar'; Url='https://repo.maven.apache.org/maven2/com/github/vbmacher/java-cup-runtime/11b-20160615/java-cup-runtime-11b-20160615.jar'}
)
foreach ($item in $artifacts) {
    $path = Join-Path $runtime $item.Name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        Write-Host "Downloading $($item.Name)"
        Invoke-WebRequest -Uri $item.Url -OutFile $path
    }
}
Write-Host 'Weka dependencies are ready.'



