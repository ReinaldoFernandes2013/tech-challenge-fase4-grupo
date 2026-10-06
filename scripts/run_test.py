from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd: str) -> int:
    print(f"\n--- EXECUTANDO: {cmd} ---")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("LOGS / AVISOS:")
        print(result.stderr)

    if result.returncode != 0:
        print(f"ERRO: Comando falhou com código de saída {result.returncode}")
        sys.exit(result.returncode)

    return result.returncode


def main():
    py = sys.executable
    index_script = PROJECT_ROOT / "scripts" / "index_data.py"
    verify_script = PROJECT_ROOT / "scripts" / "verify_counts.py"

    print("=== Teste de Idempotência do Pipeline de Indexação ===")

    print("\n[RODADA 1] Indexação Inicial...")
    run_cmd(f'"{py}" "{index_script}"')

    print("\n[RODADA 1] Verificação de Consistência e Contagem...")
    run_cmd(f'"{py}" "{verify_script}"')

    print("\n[RODADA 2] Reexecução da Indexação (Validação de Idempotência)...")
    run_cmd(f'"{py}" "{index_script}"')

    print("\n[RODADA 2] Verificação Pós-Reexecução...")
    run_cmd(f'"{py}" "{verify_script}"')

    print("\n=== Sucesso: Idempotência Comprovada (Sem Duplicação de Documentos) ===")


if __name__ == "__main__":
    main()