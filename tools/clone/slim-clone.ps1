# Slim clone of the assembly techstack for one area (AT 164). PowerShell version of slim-clone.sh.
#
#   powershell -ExecutionPolicy Bypass -File slim-clone.ps1 <area> [-Folder <folder>] [-Url <repo url>] [-Ref <branch>]
#
# Needs only git and git-lfs. Partial clone (no file contents up front), sparse checkout of what
# every area needs plus the folders of <area>; LFS files come only for those folders.
# The areas and their folders are defined in tools/clone/areas.tsv, which the script reads from
# the clone itself (slim-clone.sh reads the same file).
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Area,
    [string]$Folder = "assembly-techstack",
    [string]$Url = "https://gitlab.kit.edu/kit/wbk/map/imk/sfb1574_kreislauffabrik/assembly-techstack.git",
    [string]$Ref = ""
)
$ErrorActionPreference = "Stop"

function Invoke-Git {  # a failing native command does not stop a script by itself
    & git @args
    if ($LASTEXITCODE -ne 0) { throw "git $($args -join ' ') failed (exit code $LASTEXITCODE)" }
}

if (Test-Path $Folder) {
    Write-Error "'$Folder' exists already; pick another folder or remove it."
}

$env:GIT_LFS_SKIP_SMUDGE = "1"  # the clone downloads no LFS files; the checkout below does, for the area only
$branch = @()
if ($Ref) { $branch = @("--branch", $Ref) }
Invoke-Git -c core.longpaths=true clone --filter=blob:none --no-checkout @branch $Url $Folder
$start = $PWD.Path
Set-Location $Folder
Invoke-Git config core.longpaths true

# Only this one file is downloaded before the area is known. A branch cut before the table existed
# does not have it: take the one on main.
$table = & git show HEAD:tools/clone/areas.tsv 2>$null
if ($LASTEXITCODE -ne 0) { $table = & git show origin/main:tools/clone/areas.tsv 2>$null }
if ($LASTEXITCODE -ne 0) {
    Set-Location $start
    Remove-Item -Recurse -Force $Folder
    Write-Host "Neither '$Ref' nor main has tools/clone/areas.tsv yet. Merge main into your branch, or clone without -Ref."
    exit 1
}
$rows = $table |
    Where-Object { $_ -and -not $_.StartsWith("#") } |
    ForEach-Object { , ($_ -split "`t") }
$data = $rows | Where-Object { $_[0] -ne "area" }
$every = $data | Where-Object { $_[0] -eq "*" }
$mine = $data | Where-Object { $_[0] -eq $Area -and $_[0] -ne "*" }
if (-not $mine) {
    Set-Location ..
    Remove-Item -Recurse -Force $Folder
    Write-Host "Unknown area '$Area'. Valid areas:"
    $data | Where-Object { $_[0] -ne "*" } | ForEach-Object { Write-Host ("  {0}  -  {1}" -f $_[0], $_[1]) }
    exit 1
}

function Cell($row, $i) { if ($row.Count -gt $i) { $row[$i] } else { "" } }
$sparse = @((Cell $every 2) -split " ") + @((Cell $mine 2) -split " ") | Where-Object { $_ }
$tests = Cell $mine 3

# Checkout downloads the LFS files of the sparse folders only (the LFS filter runs per file, in one batch);
# a later `git pull` does the same, so there is no separate `git lfs pull`.
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
Invoke-Git lfs install --local
Invoke-Git sparse-checkout set --cone @sparse
Invoke-Git checkout

Write-Host ""
Write-Host "Area '$Area' is in $(Get-Location)."
Write-Host "Next: uv sync; uv run pre-commit install"
Write-Host "Fast tests of your area:"
Write-Host "  uv run pytest -q -m `"not slow and not geom and not gpu`" $tests"
