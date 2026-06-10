<#
.SYNOPSIS
    Run VPR placement on a target benchmark circuit and analyze the resulting HPWL.

.DESCRIPTION
    This script:
      1. Runs VPR with simulated annealing placement on a target circuit.
      2. Analyzes the generated .place file to compute HPWL.
    
    VPR command used:
        vpr <arch_file> <circuit>.blif --place --place_algorithm sa

.PARAMETER Circuit
    Circuit name (e.g., c2670, s1488, c5315).

.PARAMETER ArchFile
    Path to the VPR architecture XML file.

.PARAMETER BlifDir
    Directory containing the .blif netlist file (default: current directory).

.PARAMETER WorkDir
    Working directory where VPR runs and outputs are generated (default: current directory).

.PARAMETER VprPath
    Path to the VPR executable. If not provided, assumes 'vpr' is on PATH.

.PARAMETER AdditionalArgs
    Additional arguments to pass to VPR (e.g., '--seed 42').

.PARAMETER SkipVpr
    If set, skip running VPR and only analyze existing .place/.net files.

.EXAMPLE
    # Basic usage
    ./scripts/run_vpr_and_analyze.ps1 -Circuit c2670 -ArchFile arch/k6_N10.xml -BlifDir ./blif

.EXAMPLE
    # With custom VPR path and additional args
    ./scripts/run_vpr_and_analyze.ps1 -Circuit s1488 -ArchFile arch/k6_N10.xml -VprPath /tools/vpr/vpr -AdditionalArgs "--seed 1 --fix_pins random"
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$Circuit,

    [Parameter(Mandatory = $true)]
    [string]$ArchFile,

    [Parameter(Mandatory = $false)]
    [string]$BlifDir = ".",

    [Parameter(Mandatory = $false)]
    [string]$WorkDir = ".",

    [Parameter(Mandatory = $false)]
    [string]$VprPath = "vpr",

    [Parameter(Mandatory = $false)]
    [string]$AdditionalArgs = "",

    [Parameter(Mandatory = $false)]
    [switch]$SkipVpr
)

$ErrorActionPreference = "Stop"

# Resolve paths
$circuitBlif = Join-Path -Path $BlifDir -ChildPath "$Circuit.blif"
$placeFile = Join-Path -Path $WorkDir -ChildPath "$Circuit.place"
$netFile   = Join-Path -Path $WorkDir -ChildPath "$Circuit.net"

# Resolve script directory for the analysis script
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$analysisScript = Join-Path -Path $scriptDir -ChildPath "analyze_vpr_place.py"

Write-Host "=" * 70
Write-Host "  VPR Placement Runner"
Write-Host "  Circuit : $Circuit"
Write-Host "  Arch    : $ArchFile"
Write-Host "  Blif    : $circuitBlif"
Write-Host "  WorkDir : $WorkDir"
Write-Host "=" * 70

if (-not $SkipVpr) {
    # Validate .blif file exists
    if (-not (Test-Path -Path $circuitBlif -PathType Leaf)) {
        Write-Error "ERROR: .blif file not found: $circuitBlif"
        exit 1
    }

    # Validate architecture file exists
    if (-not (Test-Path -Path $ArchFile -PathType Leaf)) {
        Write-Error "ERROR: Architecture file not found: $ArchFile"
        exit 1
    }

    # Ensure working directory exists
    if (-not (Test-Path -Path $WorkDir -PathType Container)) {
        New-Item -Path $WorkDir -ItemType Directory -Force | Out-Null
    }

    # Build VPR command
    $vprCmd = "$VprPath $ArchFile $circuitBlif --place --place_algorithm sa"
    if ($AdditionalArgs) {
        $vprCmd += " $AdditionalArgs"
    }

    Write-Host ""
    Write-Host "Running VPR placement..."
    Write-Host "  Command: $vprCmd"
    Write-Host ""

    # Run VPR
    Push-Location -Path $WorkDir
    try {
        Invoke-Expression $vprCmd
        if ($LASTEXITCODE -ne 0) {
            Write-Error "VPR failed with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }
    finally {
        Pop-Location
    }

    Write-Host ""
    Write-Host "VPR placement completed successfully."
}

# Check if .place file was generated
if (Test-Path -Path $placeFile -PathType Leaf) {
    Write-Host ""
    Write-Host "Analyzing .place file..."
    
    # Build analysis command
    $analysisCmd = "python `"$analysisScript`" `"$Circuit`" --place_dir `"$WorkDir`""
    if ($AdditionalArgs -match "--verbose") {
        $analysisCmd += " --verbose"
    }
    
    Write-Host "  Command: $analysisCmd"
    Write-Host ""
    Invoke-Expression $analysisCmd
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "HPWL analysis failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
} else {
    Write-Warning "Warning: .place file not found at $placeFile"
    Write-Warning "Skipping HPWL analysis."
}

Write-Host ""
Write-Host "Done."
