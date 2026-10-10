$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$web = Join-Path $root 'web'
$projectPath = ($root.TrimEnd('\', '/') + '\').Replace('/', '\').ToLowerInvariant()
$backendPath = $projectPath + '.venv\scripts\'
$frontendPath = $projectPath + 'web\node_modules\vite\bin\vite.js'

function Normalize-CommandLine($process) {
    if (-not $process -or -not $process.CommandLine) { return '' }
    return $process.CommandLine.Replace('/', '\').ToLowerInvariant()
}

function Stop-ProjectService([int]$port, [string]$role) {
    $owners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique)

    foreach ($ownerId in $owners) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$ownerId"
        if (-not $process) { continue }
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($process.ParentProcessId)"
        $command = Normalize-CommandLine $process
        $parentCommand = Normalize-CommandLine $parent

        if ($role -eq 'backend') {
            $isProject = $command.Contains('uvicorn') -and
                $command.Contains('simserver.app:create_app') -and
                ($command.Contains($backendPath) -or $parentCommand.Contains($backendPath))
        } else {
            $isProject = $command.Contains($frontendPath)
        }
        if (-not $isProject) {
            throw "端口 $port 被非本项目进程占用（PID $ownerId），未停止该进程。"
        }

        Write-Host "停止 $role 服务（PID $ownerId）..."
        Stop-Process -Id $ownerId -ErrorAction Stop
        if ($role -eq 'backend' -and $parent -and
            $parentCommand.Contains($backendPath) -and
            $parentCommand.Contains('uvicorn')) {
            Stop-Process -Id $parent.ProcessId -ErrorAction SilentlyContinue
        }
    }

    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        if (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) { return }
        Start-Sleep -Milliseconds 500
    }
    throw "端口 $port 未释放，请检查占用进程。"
}

function Wait-ForUrl([string]$url) {
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) { return }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "服务未就绪：$url"
}

$uv = Get-Command uv.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
if (-not $uv) { $uv = Join-Path $HOME '.local\bin\uv.exe' }
if (-not (Test-Path $uv)) { throw '找不到 uv.exe，请先安装 uv。' }

$managedNodes = Join-Path $HOME '.workbuddy\binaries\node\versions'
$nodeVersion = Get-ChildItem $managedNodes -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^22\.\d+\.\d+$' -and (Test-Path (Join-Path $_.FullName 'node.exe')) } |
    Sort-Object { [version]$_.Name } -Descending |
    Select-Object -First 1
if ($nodeVersion) {
    $nodeDir = $nodeVersion.FullName
} else {
    $nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
    if (-not $nodeCommand -or -not ((& $nodeCommand.Source --version) -match '^v22\.')) {
        throw '找不到 Node.js 22，请先安装或配置托管 Node.js 22。'
    }
    $nodeDir = Split-Path $nodeCommand.Source
}
$node = Join-Path $nodeDir 'node.exe'
$npm = Join-Path $nodeDir 'npm.cmd'
if (-not (Test-Path $npm)) { throw "找不到 npm.cmd：$npm" }

$originalPath = $env:PATH
$env:PATH = "$nodeDir;$env:PATH"
Push-Location $root
try {
    Stop-ProjectService 5173 'frontend'
    Stop-ProjectService 8000 'backend'

    Write-Host '同步 Python 依赖...'
    & $uv sync --locked
    if ($LASTEXITCODE -ne 0) { throw 'uv sync 失败。' }

    Write-Host '同步前端依赖...'
    Push-Location $web
    try {
        & $npm ci --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) { throw 'npm ci 失败。' }
    } finally {
        Pop-Location
    }

    Write-Host '启动后端和前端...'
    $backend = Join-Path $root '.venv\Scripts\uvicorn.exe'
    $vite = Join-Path $web 'node_modules\vite\bin\vite.js'
    Start-Process -FilePath $backend -ArgumentList @('simserver.app:create_app', '--factory', '--host', '127.0.0.1', '--port', '8000') -WorkingDirectory $root -WindowStyle Hidden | Out-Null
    Start-Process -FilePath $node -ArgumentList @("`"$vite`"", '--host', '127.0.0.1', '--port', '5173', '--strictPort') -WorkingDirectory $web -WindowStyle Hidden | Out-Null

    Wait-ForUrl 'http://127.0.0.1:8000/openapi.json'
    Wait-ForUrl 'http://127.0.0.1:5173/'
    $body = @{
        workload = @{ mode = 'synthetic'; synthetic = @{
            num_requests = 2; session_rate = 50.0; seed = 0
            prompt_len = @{ dist = 'lognormal'; mean = 16; sigma = 1.0 }
            turn_increment_len = @{ dist = 'fixed'; value = 4 }
            output_len = @{ dist = 'poisson'; mean = 4 }
        } }
        engine = @{ max_running_requests = 32; max_total_tokens = 65536; page_size = 64; pd_mutually_exclusive = $true }
        perf_model = @{ prefill_c1 = 0.05; prefill_c2 = 0.00001; decode_d1 = 8.0; decode_d2 = 0.002 }
    } | ConvertTo-Json -Depth 10 -Compress
    $response = Invoke-WebRequest -Uri 'http://127.0.0.1:5173/api/simulate' -Method Post -ContentType 'application/json' -Body $body -UseBasicParsing -TimeoutSec 20
    if ($response.StatusCode -ne 200) { throw '仿真接口验证失败。' }

    Write-Host '本地部署完成：'
    Write-Host '  页面：http://127.0.0.1:5173/'
    Write-Host '  API 文档：http://127.0.0.1:8000/docs'
} finally {
    Pop-Location
    $env:PATH = $originalPath
}
