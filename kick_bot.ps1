# Backup start for the GitHub trading bot, run by Windows Task Scheduler on weekday mornings.
# GitHub's own schedule sometimes skips or delays a start; this asks GitHub to start today's run
# unless one is already queued or running. Extra starts are harmless: they wait behind the day's
# run, then find the market shut and exit.
$repo = "rachankorada/stock-signals-bot"
$gh = "C:\Program Files\GitHub CLI\gh.exe"
$log = "C:\Users\korad\StockAnalyzerData\bot\kick.log"

function Log($msg) { Add-Content -Path $log -Value ("{0:yyyy-MM-dd HH:mm:ss}  {1}" -f (Get-Date), $msg) }

# After waking from sleep the network can take a minute to come back, so retry.
for ($try = 1; $try -le 8; $try++) {
    $today = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd")
    $json = & $gh run list -R $repo --workflow trading-day.yml --created $today --json status 2>$null
    if ($LASTEXITCODE -eq 0) {
        $active = @($json | ConvertFrom-Json | Where-Object { $_.status -ne "completed" })
        if ($active.Count -gt 0) { Log "a run is already $($active[0].status); nothing to do"; exit 0 }
        & $gh workflow run trading-day.yml -R $repo -f mode=day 2>$null
        if ($LASTEXITCODE -eq 0) { Log "started today's run"; exit 0 }
    }
    Start-Sleep -Seconds 30
}
Log "could not reach GitHub after 8 tries"
exit 1
