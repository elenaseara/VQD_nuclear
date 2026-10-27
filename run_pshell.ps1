# Ejecutar desde la raiz de VQD_nuclear:
#   .\run_pshell.ps1 -PreviewOnly
#   .\run_pshell.ps1
#   .\run_pshell.ps1 -FullValidation
#   .\run_pshell.ps1 -SkipTests
#   .\run_pshell.ps1 -PythonExecutable "C:\ruta\python.exe"
# Si PowerShell bloquea scripts:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\run_pshell.ps1

[CmdletBinding()]
param(
    [string]$PythonExecutable = "",
    [switch]$PreviewOnly,
    [switch]$SkipTests,
    [switch]$FullValidation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Configuracion: He6, capa p, 0 protones y 2 neutrones de valencia, M=0.
# Cada ejecucion compara beta exacta y aproximada por estado.
# ---------------------------------------------------------------------------
$NuclearShell = "p"
$Protons = 0
$Neutrons = 2
$TotalJ = 0
$NStates = 2
$SimulationMode = "statevector_exact"
$GradientMethod = "analytic"
$SelectionCriterion = "full_vqd"
$EnableGradientDiagnostics = $true
$GradientTolerance = 1e-5
$EnergyImprovementTolerance = 1e-6
$NewThetaTolerance = 1e-5
$MaxCandidates = $null
$CobylaRhobeg = 0.3
$CobylaRhoend = 1e-8
$Seed = 47433
$NJobs = 1
$NWorkers = 1
$ResultsDirectory = "validation_local"

if ($FullValidation) {
    $MaxAdaptIterations = 20
    $OptimizerMaxIterations = 2000
}
else {
    $MaxAdaptIterations = 3
    $OptimizerMaxIterations = 150
}

$RepositoryRoot = $PSScriptRoot
$EntryPoint = Join-Path -Path $RepositoryRoot -ChildPath "scripts\run_vqd.py"
$SourcePath = Join-Path -Path $RepositoryRoot -ChildPath "src"
$OutputRoot = Join-Path -Path $RepositoryRoot -ChildPath $ResultsDirectory
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$InvariantCulture = [System.Globalization.CultureInfo]::InvariantCulture

if (-not (Test-Path -LiteralPath $EntryPoint -PathType Leaf)) {
    throw "Guarda run_pshell.ps1 en la raiz del repositorio VQD_nuclear."
}
if ($EnableGradientDiagnostics -and $GradientMethod -ne "analytic") {
    throw "Los diagnosticos requieren gradient-method analytic."
}
if ($CobylaRhobeg -le $CobylaRhoend -or $CobylaRhoend -le 0) {
    throw "COBYLA requiere rhobeg > rhoend > 0."
}

# Buscar Python, priorizando conda/venv activo.
if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $PythonCandidates = @()

    $EnvironmentRoots = @()
    if ($env:CONDA_PREFIX) { $EnvironmentRoots += $env:CONDA_PREFIX }
    if ($env:VIRTUAL_ENV) { $EnvironmentRoots += $env:VIRTUAL_ENV }
    $EnvironmentRoots += (Join-Path -Path $RepositoryRoot -ChildPath ".venv")

    foreach ($EnvironmentRoot in $EnvironmentRoots) {
        $PythonCandidates += (Join-Path -Path $EnvironmentRoot -ChildPath "Scripts\python.exe")
        $PythonCandidates += (Join-Path -Path $EnvironmentRoot -ChildPath "python.exe")
    }

    $PythonCommands = @(Get-Command -Name "python" -All -CommandType Application -ErrorAction SilentlyContinue)
    foreach ($PythonCommand in $PythonCommands) {
        if ($PythonCommand.Source -notlike "*\Microsoft\WindowsApps\*") {
            $PythonCandidates += $PythonCommand.Source
        }
    }

    foreach ($Candidate in $PythonCandidates) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            $PythonExecutable = $Candidate
            break
        }
    }
}

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    throw "No se ha encontrado Python. Activa pynuc o utiliza -PythonExecutable."
}

$ResolvedPython = Get-Command -Name $PythonExecutable -CommandType Application -ErrorAction Stop
$PythonExecutable = $ResolvedPython.Source

# Dos ejecuciones independientes, con los mismos parametros y semilla.
$RunConfigurations = @(
    [pscustomobject]@{
        Label = "he6_optimized_$Timestamp"
        UseModelSpace = $true
    },
    [pscustomobject]@{
        Label = "he6_full_circuit_$Timestamp"
        UseModelSpace = $false
    }
)

Write-Host "Python: $PythonExecutable" -ForegroundColor Cyan
Write-Host "Nucleo: He6; estados: $NStates por variante; modo: $SimulationMode"
Write-Host "Beta exacta: 2*abs(E_i-E_0); aproximada: abs(e_i-e_0+2*B)."
Write-Host "ADAPT: $MaxAdaptIterations iteraciones; COBYLA: $OptimizerMaxIterations evaluaciones"
Write-Host "Rutas: model-space y circuito completo"
Write-Host "Resultados: $OutputRoot"

# Construir los argumentos sin usar opciones eliminadas (--optimizer-tol,
# --pool-policy). El flag --no-model-space no recibe ningun valor.
$RunCommands = @()
foreach ($Configuration in $RunConfigurations) {
    $RunArguments = @(
        $EntryPoint,
        "--shell", $NuclearShell,
        "--protons", [string]$Protons,
        "--neutrons", [string]$Neutrons,
        "--J", [string]$TotalJ,
        "--n-states", [string]$NStates,
        "--mode", $SimulationMode,
        "--max-adapt-iter", [string]$MaxAdaptIterations,
        "--grad-tol", $GradientTolerance.ToString($InvariantCulture),
        "--optimizer-maxiter", [string]$OptimizerMaxIterations,
        "--cobyla-rhobeg", $CobylaRhobeg.ToString($InvariantCulture),
        "--cobyla-rhoend", $CobylaRhoend.ToString($InvariantCulture),
        "--new-theta-tol", $NewThetaTolerance.ToString($InvariantCulture),
        "--energy-improvement-tol", $EnergyImprovementTolerance.ToString($InvariantCulture),
        "--gradient-method", $GradientMethod,
        "--selection-criterion", $SelectionCriterion,
        "--n-jobs", [string]$NJobs,
        "--n-workers", [string]$NWorkers,
        "--qibo-backend", "numpy",
        "--seed", [string]$Seed,
        "--results-dir", $OutputRoot,
        "--run-label", $Configuration.Label
    )

    if ($EnableGradientDiagnostics) {
        # La version actual de argparse utiliza type=bool: NO pasar "False".
        $RunArguments += @("--gradient-diagnostics", "True")
    }
    if (-not $Configuration.UseModelSpace) {
        $RunArguments += "--no-model-space"
    }
    if ($null -ne $MaxCandidates) {
        $RunArguments += @("--max-candidates", [string]$MaxCandidates)
    }

    $RunCommands += [pscustomobject]@{
        Label = $Configuration.Label
        Arguments = $RunArguments
    }
}

if ($PreviewOnly) {
    if (-not $SkipTests) {
        Write-Host "`nTests: python -m pytest -q --basetemp .pytest_local_tmp" -ForegroundColor Cyan
    }
    foreach ($RunCommand in $RunCommands) {
        Write-Host "`nEjecucion: $($RunCommand.Label)" -ForegroundColor Cyan
        Write-Host ($PythonExecutable + " " + ($RunCommand.Arguments -join " "))
    }
    return
}

$PreviousPythonPath = $env:PYTHONPATH
$StartTime = Get-Date
$ResultsByRun = @()

Push-Location -LiteralPath $RepositoryRoot
try {
    $env:PYTHONPATH = $SourcePath
    if (-not [string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
        $env:PYTHONPATH += [System.IO.Path]::PathSeparator + $PreviousPythonPath
    }

    Write-Host "`nComprobando dependencias..." -ForegroundColor Cyan
    & $PythonExecutable -c "import sys, qibo, pynucshell, scipy; print('Python:', sys.version.split()[0]); print('Qibo:', qibo.__version__); print('PyNucShell:', pynucshell.__file__); print('SciPy:', scipy.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudieron importar todas las dependencias."
    }

    if (-not $SkipTests) {
        Write-Host "`nEjecutando pytest..." -ForegroundColor Cyan
        # Evitar la carpeta temporal global pytest-of-<otro usuario> en Windows.
        # --basetemp usa una carpeta dedicada bajo el repositorio.
        $PytestBaseTemp = Join-Path -Path $RepositoryRoot -ChildPath ".pytest_local_tmp"
        & $PythonExecutable -m pytest -q --basetemp $PytestBaseTemp
        if ($LASTEXITCODE -ne 0) {
            throw "Pytest ha detectado errores. Se detiene la validacion."
        }
    }

    foreach ($RunCommand in $RunCommands) {
        $RunLabel = $RunCommand.Label
        Write-Host "`nSimulacion: $RunLabel" -ForegroundColor Cyan

        & $PythonExecutable @($RunCommand.Arguments)
        if ($LASTEXITCODE -ne 0) {
            throw "La simulacion $RunLabel ha fallado."
        }

        $RunDirectory = Join-Path -Path $OutputRoot -ChildPath $RunLabel
        $SummaryPath = Join-Path -Path $RunDirectory -ChildPath "summary_${RunLabel}.csv"
        $JsonPath = Join-Path -Path $RunDirectory -ChildPath "raw_results_${RunLabel}.json"
        $DiagnosticsPath = Join-Path -Path $RunDirectory -ChildPath "gradient_diagnostics_${RunLabel}.csv"

        foreach ($RequiredFile in @($SummaryPath, $JsonPath)) {
            if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
                throw "Falta el archivo esperado: $RequiredFile"
            }
        }

        $Summary = @(Import-Csv -LiteralPath $SummaryPath)
        $ExpectedRows = 2 * $NStates * $(if ($SimulationMode -eq "both") { 2 } else { 1 })
        if ($Summary.Count -ne $ExpectedRows) {
            throw "Se esperaban $ExpectedRows filas en el resumen de $RunLabel; hay $($Summary.Count)."
        }

        Write-Host "`nResultados: $RunLabel" -ForegroundColor Green
        $Summary |
            Select-Object -Property method, beta_method, state, beta, computed_energy, exact_energy, abs_error, n_layers, stop_reason |
            Format-Table -AutoSize |
            Out-Host

        if (Test-Path -LiteralPath $DiagnosticsPath -PathType Leaf) {
            $Diagnostics = @(Import-Csv -LiteralPath $DiagnosticsPath)
            $RankingChanges = @($Diagnostics | Where-Object { $_.ranking_changed -eq "True" }).Count
            Write-Host "Diagnosticos: $($Diagnostics.Count); cambios de ranking: $RankingChanges"
        }
        else {
            Write-Warning "No se ha generado el CSV de diagnosticos."
        }

        $ResultsByRun += [pscustomobject]@{
            Label = $RunLabel
            Summary = $Summary
        }
    }

    # Comparar cada estado dentro de su modo y variante de beta.
    $OptimizedSummary = $ResultsByRun[0].Summary
    $FullCircuitSummary = $ResultsByRun[1].Summary
    $Comparison = @()

    foreach ($OptimizedRow in $OptimizedSummary) {
        $MatchingRows = @($FullCircuitSummary | Where-Object {
            $_.state -eq $OptimizedRow.state -and $_.beta_method -eq $OptimizedRow.beta_method -and
            $_.method -eq $OptimizedRow.method
        })
        if ($MatchingRows.Count -ne 1) {
            throw "No se encuentra un estado unico para state=$($OptimizedRow.state), beta=$($OptimizedRow.beta_method), method=$($OptimizedRow.method)."
        }
        $FullCircuitRow = $MatchingRows[0]
        $OptimizedEnergy = [double]::Parse($OptimizedRow.computed_energy, $InvariantCulture)
        $FullCircuitEnergy = [double]::Parse($FullCircuitRow.computed_energy, $InvariantCulture)

        $Comparison += [pscustomobject]@{
            Method = $OptimizedRow.method
            BetaMethod = $OptimizedRow.beta_method
            Beta = $OptimizedRow.beta
            State = $OptimizedRow.state
            Optimized = $OptimizedEnergy
            FullCircuit = $FullCircuitEnergy
            AbsoluteDifference = [math]::Abs($OptimizedEnergy - $FullCircuitEnergy)
        }
    }

    Write-Host "`nComparacion de energias:" -ForegroundColor Cyan
    $Comparison | Format-Table -AutoSize | Out-Host
    Write-Host "Validacion funcional completada. Las diferencias de energia no prueban equivalencia exacta de trayectorias." -ForegroundColor Green
    Write-Host "Duracion: $((Get-Date) - $StartTime)"
}
finally {
    $env:PYTHONPATH = $PreviousPythonPath
    Pop-Location
}
