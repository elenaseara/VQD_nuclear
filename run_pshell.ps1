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
# F18 se usa solo para perfilar la inicializacion y todas sus proyecciones.
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
$ComparisonTolerance = 1e-6

if ($FullValidation) {
    $NStates = 3
    $MaxAdaptIterations = 20
    $OptimizerMaxIterations = 2000
}
else {
    $MaxAdaptIterations = 3
    $OptimizerMaxIterations = 150
}

$RepositoryRoot = $PSScriptRoot
$EntryPoint = Join-Path -Path $RepositoryRoot -ChildPath "scripts\run_vqd.py"
$ProfileEntryPoint = Join-Path -Path $RepositoryRoot -ChildPath "scripts\profile_model_space.py"
$SourcePath = Join-Path -Path $RepositoryRoot -ChildPath "src"
$OutputRoot = Join-Path -Path $RepositoryRoot -ChildPath $ResultsDirectory
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$ValidationDirectory = Join-Path -Path $OutputRoot -ChildPath "projection_validation_$Timestamp"
$ProfilePath = Join-Path -Path $ValidationDirectory -ChildPath "projection_f18.json"
$ComparisonPath = Join-Path -Path $ValidationDirectory -ChildPath "comparison_he6.csv"
$ValidationSummaryPath = Join-Path -Path $ValidationDirectory -ChildPath "validation_summary.json"
$PytestBaseTemp = Join-Path -Path $ValidationDirectory -ChildPath "pytest_tmp"
$InvariantCulture = [System.Globalization.CultureInfo]::InvariantCulture

if (-not (Test-Path -LiteralPath $EntryPoint -PathType Leaf)) {
    throw "Guarda run_pshell.ps1 en la raiz del repositorio VQD_nuclear."
}
if (-not (Test-Path -LiteralPath $ProfileEntryPoint -PathType Leaf)) {
    throw "Falta scripts\profile_model_space.py. Actualiza el repositorio antes de validar."
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
    # Instalacion local de respaldo cuando no hay un entorno activado.
    $PythonCandidates += (Join-Path -Path $env:USERPROFILE -ChildPath "anaconda_py\python.exe")

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
Write-Host "Perfilado F18: 24 qubits, espacio fisico de 28 determinantes, todos los generadores"
Write-Host "Tolerancia de comparacion de energia sin shots: $ComparisonTolerance MeV"
Write-Host "Validacion: $ValidationDirectory"

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
        "--results-dir", $ValidationDirectory,
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

$ProfileArguments = @(
    $ProfileEntryPoint, "--shell", "sd", "--protons", "1", "--neutrons", "1",
    "--J", "0", "--all-generators", "--output", $ProfilePath
)

if ($PreviewOnly) {
    if (-not $SkipTests) {
        Write-Host "`nTests: python -m pytest -q -p no:cacheprovider --basetemp $PytestBaseTemp" -ForegroundColor Cyan
    }
    Write-Host "`nPerfilado F18:" -ForegroundColor Cyan
    Write-Host ($PythonExecutable + " " + ($ProfileArguments -join " "))
    foreach ($RunCommand in $RunCommands) {
        Write-Host "`nEjecucion: $($RunCommand.Label)" -ForegroundColor Cyan
        Write-Host ($PythonExecutable + " " + ($RunCommand.Arguments -join " "))
    }
    return
}

$PreviousPythonPath = $env:PYTHONPATH
$StartTime = Get-Date
$ResultsByRun = @()
$ValidationPassed = $false

Push-Location -LiteralPath $RepositoryRoot
try {
    New-Item -ItemType Directory -Path $ValidationDirectory -Force | Out-Null
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
        # Una carpeta nueva evita borrar resultados antiguos o temporales ajenos.
        & $PythonExecutable -m pytest -q -p no:cacheprovider --basetemp $PytestBaseTemp
        if ($LASTEXITCODE -ne 0) {
            throw "Pytest ha detectado errores. Se detiene la validacion."
        }
    }

    Write-Host "`nPerfilando F18 sin statevectors completos..." -ForegroundColor Cyan
    & $PythonExecutable @ProfileArguments
    if ($LASTEXITCODE -ne 0) {
        throw "El perfilado de F18 ha fallado."
    }
    $Profile = Get-Content -LiteralPath $ProfilePath -Raw | ConvertFrom-Json
    if (-not $Profile.available -or $Profile.full_state_callback_calls -ne 0 -or
        $Profile.n_qubits -ne 24 -or $Profile.physical_dimension -ne 28 -or
        $Profile.pool_size -lt 1 -or $Profile.projected_generators -ne $Profile.pool_size -or
        @($Profile.generator_projections).Count -ne $Profile.pool_size) {
        throw "F18 no cumple las condiciones de proyeccion esperadas. Revisa $ProfilePath"
    }
    $LeakingGenerators = @($Profile.generator_projections | Where-Object { -not $_.preserves_space })
    if ($LeakingGenerators.Count -ne 0) {
        throw "Hay generadores de F18 que no conservan el subespacio fisico. Revisa $ProfilePath"
    }
    Write-Host ("F18: motor={0:F3} s; contexto={1:F3} s; generadores={2}; memoria pico={3:F1} MiB" -f
        $Profile.model_space_seconds, $Profile.build_context_seconds, $Profile.projected_generators,
        ($Profile.process_peak_bytes / 1MB)) -ForegroundColor Green

    foreach ($RunCommand in $RunCommands) {
        $RunLabel = $RunCommand.Label
        Write-Host "`nSimulacion: $RunLabel" -ForegroundColor Cyan

        & $PythonExecutable @($RunCommand.Arguments)
        if ($LASTEXITCODE -ne 0) {
            throw "La simulacion $RunLabel ha fallado."
        }

        $RunDirectory = Join-Path -Path $ValidationDirectory -ChildPath $RunLabel
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
        foreach ($BetaMethod in @("exact", "approximate")) {
            $FamilyRows = @($Summary | Where-Object { $_.beta_method -eq $BetaMethod })
            if ($FamilyRows.Count -ne ($ExpectedRows / 2)) {
                throw "Falta la familia beta=$BetaMethod en $RunLabel."
            }
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

        $Difference = [math]::Abs($OptimizedEnergy - $FullCircuitEnergy)
        if ([double]::IsNaN($Difference) -or [double]::IsInfinity($Difference)) {
            throw "La comparacion contiene energias no finitas."
        }
        $DeterministicComparison = $OptimizedRow.method -eq "statevector"
        $Comparison += [pscustomobject]@{
            Method = $OptimizedRow.method
            BetaMethod = $OptimizedRow.beta_method
            Beta = $OptimizedRow.beta
            State = $OptimizedRow.state
            Optimized = $OptimizedEnergy
            FullCircuit = $FullCircuitEnergy
            AbsoluteDifference = $Difference
            DeterministicComparison = $DeterministicComparison
            WithinTolerance = $(if ($DeterministicComparison) { $Difference -le $ComparisonTolerance } else { $null })
        }
    }

    Write-Host "`nComparacion de energias:" -ForegroundColor Cyan
    $Comparison | Format-Table -AutoSize | Out-Host
    $Comparison | Export-Csv -LiteralPath $ComparisonPath -NoTypeInformation -Encoding UTF8
    $FailedComparisons = @($Comparison | Where-Object { $_.DeterministicComparison -and -not $_.WithinTolerance })
    $ValidationPassed = $FailedComparisons.Count -eq 0
    [pscustomobject]@{
        Passed = $ValidationPassed
        TestsExecuted = -not [bool]$SkipTests
        FullValidation = [bool]$FullValidation
        EnergyToleranceMeV = $ComparisonTolerance
        F18Profile = $ProfilePath
        He6Comparison = $ComparisonPath
        Runs = @($ResultsByRun | ForEach-Object { $_.Label })
        FailedEnergyComparisons = $FailedComparisons.Count
        Note = "La comparacion numerica no certifica convergencia fisica. El perfilado incluye seguimiento de memoria."
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ValidationSummaryPath -Encoding UTF8
    if (-not $ValidationPassed) {
        throw "Las energias sin shots difieren por encima de la tolerancia. Revisa $ComparisonPath"
    }
    Write-Host "Validacion funcional completada. Las pruebas de proyeccion comprueban tambien matrices, fuga y trayectorias." -ForegroundColor Green
    if (-not $FullValidation) {
        Write-Host "El modo rapido limita ADAPT a 3 iteraciones; no certifica convergencia de los excitados."
    }
    Write-Host "Resumen: $ValidationSummaryPath"
    Write-Host "Duracion: $((Get-Date) - $StartTime)"
}
finally {
    $env:PYTHONPATH = $PreviousPythonPath
    Pop-Location
}
