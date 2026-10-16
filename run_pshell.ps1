# Ejecutar: .\run_pshell.ps1
# Si PowerShell bloquea scripts (solo para este proceso):
# powershell -NoProfile -ExecutionPolicy Bypass -File .\run_pshell.ps1
# Python alternativo: .\run_pshell.ps1 -PythonExecutable C:\ruta\python.exe
# Revisar argumentos sin calcular: .\run_pshell.ps1 -PreviewOnly
[CmdletBinding()]
param(
    [string]$PythonExecutable = "",
    [switch]$PreviewOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# ============================================================================
# CONFIGURACION DEL CALCULO LOCAL
# ============================================================================
$Shell = "p"
$Protons = 0
$Neutrons = 2
$TotalJ = 0
# 1 = fundamental; 2 incluye primer excitado; 3 incluye segundo excitado.
$NStates = 2
# Cada modo ejecuta automaticamente las DOS variantes de beta.
# statevector_exact: sin ruido estadistico; full_quantum_circuit: con shots.
# both ejecuta ambos modos (cuatro familias).
$Mode = "statevector_exact"
# Exacta: beta_i = 2*abs(E_i-E_0).
# Aproximada: beta_i = abs(e_i-e_0+2*B), con diagonal ordenada y
# B = mayor suma absoluta de una fila fuera de la diagonal.
# El fundamental no usa penalizacion.
$MaxAdaptIterations = 20
$GradientTolerance = 1e-5
$EnergyImprovementTolerance = 1e-6
$NewThetaTolerance = 1e-5
# $null usa todo el pool; un entero limita los candidatos por iteracion.
$MaxCandidates = $null
$GradientMethod = "analytic"
$OptimizerMaxIterations = 1000
$MeasurementStrategy = "paper"
# Los shots solo se usan en full_quantum_circuit o both.
$EnergyShots = 500000
$OverlapShots = 500000
$Seed = 47433
$NJobs = 4
$ResultsDirectory = "vqd_results"
# Una carpeta nueva por ejecucion evita sobrescribir resultados anteriores.
$RunLabel = "local_${Shell}shell_${Protons}p${Neutrons}n_J${TotalJ}_${Mode}_seed${Seed}_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')"
# ============================================================================

$RepositoryRoot = $PSScriptRoot
$EntryPoint = Join-Path $RepositoryRoot "scripts\run_vqd.py"
$SourcePath = Join-Path $RepositoryRoot "src"
$OutputRoot = Join-Path $RepositoryRoot $ResultsDirectory
$RunDirectory = Join-Path $OutputRoot $RunLabel
$SummaryPath = Join-Path $RunDirectory "summary_${RunLabel}.csv"
$ReportPath = Join-Path $RunDirectory "vqd_results_${RunLabel}.txt"
if (-not (Test-Path -LiteralPath $EntryPoint -PathType Leaf)) {
    throw "Guarda run_pshell.ps1 en la raiz de VQD_nuclear. Falta scripts\run_vqd.py."
}

# Preferir el entorno activo o .venv. Evitar los alias de Microsoft Store.
if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $PythonCandidates = @()
    foreach ($EnvironmentRoot in @($env:VIRTUAL_ENV, $env:CONDA_PREFIX, (Join-Path $RepositoryRoot ".venv"))) {
        if (-not [string]::IsNullOrWhiteSpace($EnvironmentRoot)) {
            $PythonCandidates += Join-Path $EnvironmentRoot "Scripts\python.exe"
            $PythonCandidates += Join-Path $EnvironmentRoot "python.exe"
        }
    }
    $PythonCommands = @(Get-Command python -All -CommandType Application -ErrorAction SilentlyContinue)
    $PythonCandidates += @($PythonCommands | Where-Object { $_.Source -notlike '*\Microsoft\WindowsApps\*' } | ForEach-Object { $_.Source })
    # Instalacion local de este equipo, si no hay un entorno activo.
    $PythonCandidates += Join-Path $env:USERPROFILE "anaconda_py\python.exe"
    foreach ($Candidate in $PythonCandidates) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            $PythonExecutable = $Candidate
            break
        }
    }
}
if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    throw "Activa tu entorno Python o usa -PythonExecutable C:\ruta\python.exe."
}
$PythonCommand = Get-Command $PythonExecutable -CommandType Application -ErrorAction Stop
$PythonExecutable = $PythonCommand.Source

if ($Shell -notin @("p", "sd") -or $Mode -notin @("statevector_exact", "full_quantum_circuit", "both")) {
    throw "Revisa Shell y Mode."
}
if ($GradientMethod -notin @("analytic", "finite_difference") -or $MeasurementStrategy -notin @("paper", "qwc")) {
    throw "Revisa GradientMethod y MeasurementStrategy."
}
if ($Protons -lt 0 -or $Neutrons -lt 0 -or $NStates -lt 1) {
    throw "Los nucleones deben ser no negativos y NStates positivo."
}
if ($MaxAdaptIterations -lt 1 -or $OptimizerMaxIterations -lt 1 -or $NJobs -lt 1) {
    throw "Las iteraciones y NJobs deben ser positivos."
}
if ($null -ne $MaxCandidates -and $MaxCandidates -lt 1) {
    throw "MaxCandidates debe ser positivo o null."
}
if ($GradientTolerance -le 0 -or $EnergyImprovementTolerance -le 0 -or $NewThetaTolerance -le 0) {
    throw "Las tolerancias deben ser positivas."
}
if ($Mode -in @("full_quantum_circuit", "both") -and ($EnergyShots -lt 1 -or $OverlapShots -lt 1)) {
    throw "Los shots deben ser positivos."
}

# Evitar problemas con el separador decimal regional de Windows.
$InvariantCulture = [System.Globalization.CultureInfo]::InvariantCulture
$RunArguments = @(
    $EntryPoint,
    "--shell", $Shell,
    "--protons", "$Protons",
    "--neutrons", "$Neutrons",
    "--J", "$TotalJ",
    "--n-states", "$NStates",
    "--mode", $Mode,
    "--max-adapt-iter", "$MaxAdaptIterations",
    "--grad-tol", $GradientTolerance.ToString($InvariantCulture),
    "--optimizer-maxiter", "$OptimizerMaxIterations",
    "--energy-shots", "$EnergyShots",
    "--overlap-shots", "$OverlapShots",
    "--new-theta-tol", $NewThetaTolerance.ToString($InvariantCulture),
    "--energy-improvement-tol", $EnergyImprovementTolerance.ToString($InvariantCulture),
    "--gradient-method", $GradientMethod,
    "--measurement-strategy", $MeasurementStrategy,
    "--n-jobs", "$NJobs",
    "--n-workers", "$NJobs",
    "--qibo-backend", "numpy",
    "--seed", "$Seed",
    "--results-dir", $OutputRoot,
    "--run-label", $RunLabel
)
if ($null -ne $MaxCandidates) {
    $RunArguments += @("--max-candidates", "$MaxCandidates")
}

Write-Host "Python:     $PythonExecutable" -ForegroundColor Cyan
Write-Host "Sistema:    shell=$Shell, ${Protons}p, ${Neutrons}n, J=$TotalJ"
Write-Host "Estados:    $NStates por variante; modo=$Mode"
Write-Host "Betas:      exacta y aproximada, calculadas por estado"
Write-Host "ADAPT:      max=$MaxAdaptIterations, grad_tol=$GradientTolerance, COBYLA=$OptimizerMaxIterations"
Write-Host "Resultados: $RunDirectory"
if ($PreviewOnly) {
    Write-Host "Argumentos de Python:"
    $RunArguments | ForEach-Object { Write-Host "  $_" }
    return
}

# Importar src sin necesitar pip install -e .; restaurar el entorno al terminar.
$PreviousPythonPath = $env:PYTHONPATH
$StartTime = Get-Date
Push-Location $RepositoryRoot
try {
    $env:PYTHONPATH = $SourcePath
    if (-not [string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
        $env:PYTHONPATH += [System.IO.Path]::PathSeparator + $PreviousPythonPath
    }
    & $PythonExecutable @RunArguments
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "Python termino con codigo $ExitCode. Revisa el error anterior y las dependencias del entorno."
    }
}
finally {
    $env:PYTHONPATH = $PreviousPythonPath
    Pop-Location
}

if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
    throw "El calculo termino, pero falta el resumen esperado: $SummaryPath"
}
$Summary = @(Import-Csv -LiteralPath $SummaryPath)
foreach ($BetaMethod in @("exact", "approximate")) {
    $Title = if ($BetaMethod -eq "exact") { "BETA CON DIAGONALIZACION EXACTA" } else { "BETA APROXIMADA" }
    Write-Host "`n$Title" -ForegroundColor Cyan
    $Summary | Where-Object { $_.beta_method -eq $BetaMethod } |
        Select-Object method, state, beta, exact_energy, computed_energy, abs_error, n_layers |
        Format-Table -AutoSize | Out-Host
}
Write-Host "Estado 0 = fundamental; 1 = primer excitado; 2 = segundo excitado."
Write-Host "abs_error compara la energia calculada con la energia exacta del indice indicado."
Write-Host "Validacion por asignacion: $(Join-Path $RunDirectory "matched_validation_summary_${RunLabel}.csv")"
Write-Host "`nDuracion: $((Get-Date) - $StartTime)" -ForegroundColor Green
Write-Host "Resumen CSV: $SummaryPath"
Write-Host "Informe TXT: $ReportPath"
