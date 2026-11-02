# GitHub Actions normally checks out source with a detached HEAD.
# Windows PowerShell 5.1 can return $null for this case. Never call
# a string method on the absent result. The caller handles a blank branch.
function Get-Experiment025GitBranch {
    $name = & git branch --show-current
    if ($LASTEXITCODE -ne 0) {
        throw 'Cannot determine current Git branch'
    }
    if ($null -eq $name) {
        return ''
    }
    return ([string]$name).Trim()
}
