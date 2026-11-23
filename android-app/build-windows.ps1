param(
    [string]$WorkRoot = (Join-Path $env:USERPROFILE '.codex\signai-android-native'),
    [string]$SdkRoot = $env:ANDROID_HOME,
    [string]$JavaHome = $env:JAVA_HOME
)
$ErrorActionPreference = 'Stop'
if (-not $SdkRoot) { $SdkRoot = $env:ANDROID_SDK_ROOT }
if (-not $SdkRoot -or -not (Test-Path -LiteralPath $SdkRoot)) { throw 'Pass -SdkRoot with the Android SDK directory.' }
if (-not $JavaHome -or -not (Test-Path -LiteralPath (Join-Path $JavaHome 'bin\java.exe'))) { throw 'Pass -JavaHome with a JDK 17 directory.' }
$taskWorkRoot = [IO.Path]::GetFullPath($WorkRoot)
if ($taskWorkRoot -match '[^\x00-\x7F]') { throw 'WorkRoot must use an ASCII-only path for Windows JVM test classpaths.' }
$source = $PSScriptRoot
$files = @()
foreach ($folder in @('app\src','gradle')) { $files += Get-ChildItem -LiteralPath (Join-Path $source $folder) -Recurse -File }
foreach ($name in @('settings.gradle.kts','build.gradle.kts','gradle.properties','gradlew','gradlew.bat','app\build.gradle.kts')) {
    $files += Get-Item -LiteralPath (Join-Path $source $name)
}
$fingerprint = (($files | Sort-Object FullName | ForEach-Object {
    $_.FullName.Substring($source.Length) + ':' + (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
}) -join "`n")
$sha = [Security.Cryptography.SHA256]::Create()
try { $id = ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($fingerprint)))).Replace('-','').Substring(0,16).ToLowerInvariant() }
finally { $sha.Dispose() }
$snapshot = Join-Path $taskWorkRoot ('source_' + $id)
New-Item -ItemType Directory -Path $snapshot -Force | Out-Null
foreach ($file in $files) {
    $relative = $file.FullName.Substring($source.Length).TrimStart('\')
    $target = Join-Path $snapshot $relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    if (-not (Test-Path -LiteralPath $target) -or (Get-FileHash -LiteralPath $target).Hash -ne (Get-FileHash -LiteralPath $file.FullName).Hash) {
        Copy-Item -LiteralPath $file.FullName -Destination $target
    }
}
$sdkForProperties = (Resolve-Path -LiteralPath $SdkRoot).Path.Replace('\','/')
Set-Content -LiteralPath (Join-Path $snapshot 'local.properties') -Value ('sdk.dir=' + $sdkForProperties) -Encoding ASCII
$env:JAVA_HOME = (Resolve-Path -LiteralPath $JavaHome).Path
$env:DEBUG = ''
$log = Join-Path $taskWorkRoot ('build_' + $id + '.log')
Push-Location -LiteralPath $snapshot
try {
    $ErrorActionPreference = 'Continue'
    & .\gradlew.bat assembleDebug testDebugUnitTest --console=plain *> $log
    $buildExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    Get-Content -LiteralPath $log -Tail 45
    if ($buildExit -ne 0) { throw "Build/tests failed, exit $buildExit. Log: $log" }
} finally { Pop-Location }
$artifacts = Join-Path $taskWorkRoot 'artifacts'
New-Item -ItemType Directory -Path $artifacts -Force | Out-Null
$apk = Join-Path $artifacts ('SIGN_AI_' + $id + '.apk')
Copy-Item -LiteralPath (Join-Path $snapshot 'app\build\outputs\apk\debug\app-debug.apk') -Destination $apk
Write-Output "APK: $apk"
Write-Output "JUnit XML: $(Join-Path $snapshot 'app\build\test-results\testDebugUnitTest')"
Write-Output 'Source repository files and existing APKs were preserved. No source directory was moved or deleted.'
