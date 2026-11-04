# Ollama setup for the HomeBench local runs M1-M8 (tests/homebench/variants/homebench.ecai.M*.yaml).
#
#   powershell -File tests/homebench/ollama/setup.ps1
#
# 1. Pulls the base models that are missing.
# 2. Creates the 16k-context copies from the Modelfile.* files next to this script.
#    qwen3-8b-16k and qwen2.5-coder-7b-16k were made by hand earlier (same num_ctx 16384)
#    and are only checked here.
# Idempotent: pull and create are no-ops when nothing changed.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

$bases = @(
    "erwan2/DeepSeek-R1-Distill-Qwen-1.5B",
    "erwan2/DeepSeek-R1-Distill-Qwen-7B",
    "qwen3:4b",
    "gemma3:1b"
)
foreach ($base in $bases) {
    ollama pull $base
}

foreach ($file in Get-ChildItem -Path $here -Filter "Modelfile.*") {
    $name = $file.Name -replace "^Modelfile\.", ""
    ollama create $name -f $file.FullName
}

foreach ($name in @("qwen3-8b-16k", "qwen2.5-coder-7b-16k")) {
    $params = (ollama show $name --parameters | Out-String)
    if ($params -notmatch "num_ctx\s+16384") {
        throw "$name is missing or has no num_ctx 16384; create it from the base model with PARAMETER num_ctx 16384"
    }
}

ollama list
