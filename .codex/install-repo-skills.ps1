param(
  [string]$CodexSkillsRoot = "$HOME\.codex\skills"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourceRoots = @(
  (Join-Path $repoRoot ".codex\skills"),
  (Join-Path $repoRoot ".agents\skills")
)

New-Item -ItemType Directory -Force -Path $CodexSkillsRoot | Out-Null

foreach ($sourceRoot in $sourceRoots) {
  if (-not (Test-Path -LiteralPath $sourceRoot)) {
    continue
  }

  Get-ChildItem -LiteralPath $sourceRoot -Directory | ForEach-Object {
    $source = $_.FullName
    $dest = Join-Path $CodexSkillsRoot $_.Name

    if (-not (Test-Path -LiteralPath (Join-Path $source "SKILL.md"))) {
      Write-Warning "Skipping $source because SKILL.md was not found."
      return
    }

    if (Test-Path -LiteralPath $dest) {
      $existing = Get-Item -LiteralPath $dest -Force
      if ($existing.LinkType -eq "Junction" -and $existing.Target -eq $source) {
        Write-Output "OK $($_.Name) -> $source"
        return
      }

      Remove-Item -LiteralPath $dest -Recurse -Force
    }

    New-Item -ItemType Junction -Path $dest -Target $source | Out-Null
    Write-Output "Linked $($_.Name) -> $source"
  }
}

Write-Output "Restart Codex to pick up repo skills."
