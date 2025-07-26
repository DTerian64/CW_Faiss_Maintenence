param (
    [string]$envFile,
    [string]$resourceGroup,
    [string]$appName,
    [bool]$isFunctionApp = $true
)

if (-not $envFile) { $envFile = ".env" }

if (!(Test-Path $envFile)) {
    Write-Error "The .env file '$envFile' does not exist."
    exit 1
}

$settings = @()
Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    $pair = $_ -split '=', 2
    if ($pair.Length -eq 2) {
        $key = $pair[0].Trim()
        $value = $pair[1].Trim()
        $settings += "$key=$value"
    }
}

if ($isFunctionApp) {
    az functionapp config appsettings set `
        --name $appName `
        --resource-group $resourceGroup `
        --settings $settings
} else {
    az webapp config appsettings set `
        --name $appName `
        --resource-group $resourceGroup `
        --settings $settings
}

Write-Host "Environment variables from '$envFile' have been uploaded to the Azure App Service '$appName' in resource group '$resourceGroup'."
Write-Host "You can verify the settings in the Azure Portal under 'Configuration' for the App Service."
Write-Host "Note: Sensitive values are not displayed in the Azure Portal for security reasons."

#.\upload-env.ps1 -envFile ".env" -resourceGroup "aifoundry-rg" -appName "cw-products-trigger-faiss-update" -isFunctionApp $false
