$COMMAND_PREFIX = "/"
$MAX_RESPONSE_LENGTH = 8000

Write-Host "Starting Lark-Claude Bridge..." -ForegroundColor Cyan
Write-Host "Listening for commands starting with: $COMMAND_PREFIX"
Write-Host ""

$tempFile = "$env:TEMP\lark_events_$PID.txt"

$proc = Start-Process -FilePath "lark-cli" -ArgumentList "event +subscribe --event-types im.message.receive_v1 --compact --quiet --as bot" -NoNewWindow -PassThru -RedirectStandardOutput $tempFile

Start-Sleep 2

if ($proc.HasExited) {
    Write-Host "Failed to start. Check permissions." -ForegroundColor Red
    exit 1
}

Write-Host "Bridge started (PID: $($proc.Id))" -ForegroundColor Green

try {
    Get-Content $tempFile -Wait | ForEach-Object {
        $line = $_
        $event = $line | ConvertFrom-Json -ErrorAction SilentlyContinue
        if (-not $event) { return }

        $content = $event.content
        $message_id = $event.message_id
        $sender_id = $event.sender_id

        if (-not $content -or -not $message_id) { return }

        if ($content.StartsWith($COMMAND_PREFIX)) {
            $command = $content.Substring($COMMAND_PREFIX.Length).Trim()
            $ts = Get-Date -Format "HH:mm:ss"

            Write-Host "[$ts] Command: $command" -ForegroundColor Yellow

            try {
                $response = claude -p $command 2>&1 | Out-String
                if (-not $response.Trim()) { $response = "No output" }
            } catch {
                $response = "Error: $_"
            }

            if ($response.Length -gt $MAX_RESPONSE_LENGTH) {
                $response = $response.Substring(0, $MAX_RESPONSE_LENGTH) + "... [truncated]"
            }

            $body = @{
                msg_type = "text"
                content = @{"text" = $response} | ConvertTo-Json -Compress
            } | ConvertTo-Json -Compress

            lark-cli api POST "/open-apis/im/v1/messages/$message_id/reply" --data $body --as bot --format data 2>$null | Out-Null

            Write-Host "[$ts] Replied" -ForegroundColor Green
        }
    }
} finally {
    Write-Host "Stopping..." -ForegroundColor Red
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Remove-Item $tempFile -ErrorAction SilentlyContinue
}
