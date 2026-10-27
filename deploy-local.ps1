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

function Get-ProjectService([int]$port, [string]$role) {
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

        [pscustomobject]@{ Process = $process; Parent = $parent }
    }
}

function Assert-NoActiveJobs {
    $backendServices = @(Get-ProjectService 8000 'backend')
    if ($backendServices.Count -eq 0) {
        Write-Host '本项目后端未运行，无需检查遗留任务状态。'
        return
    }

    $python = Join-Path $root '.venv\Scripts\python.exe'
    if (-not (Test-Path $python)) { throw '找不到本项目 Python，无法确认任务状态，未执行部署。' }
    Write-Host '只读检查本地任务状态...'
    & $python -X utf8 (Join-Path $root 'deploy-local-state.py')
    if ($LASTEXITCODE -ne 0) { throw '任务检查未通过，服务保持运行，未执行部署。' }
}

function Stop-ProjectService([int]$port, [string]$role) {
    foreach ($service in @(Get-ProjectService $port $role)) {
        $process = $service.Process
        $parent = $service.Parent
        $parentCommand = Normalize-CommandLine $parent
        Write-Host "停止 $role 服务（PID $($process.ProcessId)）..."
        Stop-Process -Id $process.ProcessId -ErrorAction Stop
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
    Assert-NoActiveJobs

    Stop-ProjectService 5173 'frontend'
    Stop-ProjectService 8000 'backend'

    Write-Host '同步 Python 依赖...'
    & $uv sync --locked
    if ($LASTEXITCODE -ne 0) { throw 'uv sync 失败。' }

    Write-Host '同步前端依赖...'
    Push-Location $web
    try {
        if (Test-Path 'node_modules') {
            & $npm install --no-save --no-audit --no-fund
            if ($LASTEXITCODE -ne 0) { throw 'npm install 失败。' }
        } else {
            & $npm ci --no-audit --no-fund
            if ($LASTEXITCODE -ne 0) { throw 'npm ci 失败。' }
        }
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
        num_instances = 1
        workload = @{ mode = 'synthetic'; session_rate = 50.0; seed = 0; synthetic = @{
            num_requests = 2
            prompt_len = @{ dist = 'lognormal'; mean = 16; sigma = 1.0 }
            turn_increment_len = @{ dist = 'fixed'; value = 4 }
            output_len = @{ dist = 'poisson'; mean = 4 }
        } }
        scheduler = @{ max_running_requests = 32; pd_mutually_exclusive = $true }
        inference = @{
            page_size = 64
            model = @{ name = 'toy'; num_layers = 2; hidden_size = 16; num_attention_heads = 2; num_kv_heads = 1; head_dim = 8; ffn_intermediate_size = 32; vocab_size = 16 }
            hardware = @{ name = 'toy'; memory_bytes_per_gpu = 10000000; effective_fp16_tflops = 1; effective_hbm_gb_s = 1 }
            storage = @{}
        }
    } | ConvertTo-Json -Depth 10 -Compress
    $response = Invoke-WebRequest -Uri 'http://127.0.0.1:5173/api/simulate' -Method Post -ContentType 'application/json' -Body $body -UseBasicParsing -TimeoutSec 20
    if ($response.StatusCode -ne 202) { throw '仿真任务提交失败。' }
    $jobId = ($response.Content | ConvertFrom-Json).id
    $deadline = [DateTime]::UtcNow.AddSeconds(60)
    do {
        $job = Invoke-RestMethod -Uri "http://127.0.0.1:5173/api/jobs/$jobId" -TimeoutSec 5
        if ($job.status -eq 'completed') { break }
        if ($job.status -eq 'failed' -or $job.status -eq 'cancelled') {
            throw "仿真任务 $($job.status)：$($job.error)"
        }
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $deadline)
    if ($job.status -ne 'completed') { throw "仿真任务超时：$jobId" }
    $result = Invoke-WebRequest -Uri "http://127.0.0.1:5173/api/jobs/$jobId/result" -UseBasicParsing -TimeoutSec 20
    if ($result.StatusCode -ne 200) { throw '仿真结果验证失败。' }

    Write-Host '本地部署完成：'
    Write-Host '  页面：http://127.0.0.1:5173/'
    Write-Host '  API 文档：http://127.0.0.1:8000/docs'
} finally {
    Pop-Location
    $env:PATH = $originalPath
}
