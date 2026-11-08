# Refresh fields (table of contents, list of figures and tables, caption numbers) in docs/capstone_report.docx
# and export docs/capstone_report.pdf. Needs Microsoft Word. Usage: powershell -File scripts/finalize_report.ps1
# The progress log doubles as a short pause between steps; Word's hidden window can stall on export without it.
$root = Split-Path -Parent $PSScriptRoot
$docx = [System.IO.Path]::GetFullPath((Join-Path $root "docs\capstone_report.docx"))
$pdf = [System.IO.Path]::GetFullPath((Join-Path $root "docs\capstone_report.pdf"))
$log = Join-Path $env:TEMP "finalize_report.log"
function L($m) { "$(Get-Date -Format HH:mm:ss) $m" | Out-File -FilePath $log -Append -Encoding utf8 }
Remove-Item $log -ErrorAction SilentlyContinue
$word = New-Object -ComObject Word.Application
$word.Visible = $false; $word.DisplayAlerts = 0
L "word started"
$doc = $word.Documents.Open($docx, $false, $false)
L "opened"
$doc.Fields.Update() | Out-Null
L "fields updated"
foreach ($t in $doc.TablesOfContents) { $t.Update() }
L "toc updated"
foreach ($t in $doc.TablesOfFigures) { $t.Update() }
L "lists updated"
$doc.Repaginate()
L "repaginated; pages = $($doc.ComputeStatistics(2))"
$doc.ExportAsFixedFormat([string]$pdf, 17, $false)
L "pdf exported"
$doc.Save()
$doc.Close($false)
$word.Quit()
Write-Output "exported $pdf (log: $log)"
