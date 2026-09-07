# -*- coding: utf-8 -*-
"""
Script de Anonimização Criptográfica Irreversível para Dados Clínicos
Compatível com WinPython 2022 / Spyder (Ambiente Local Windows)
"""

import os
import pandas as pd
import hashlib
import secrets
import tkinter as tk
from tkinter import filedialog

# ==============================================================================
# 1. CONFIGURAÇÃO DE DIRETÓRIOS E ARQUIVOS (LOCAL WINDOWS)
# ==============================================================================
# Abre uma janela para você selecionar a pasta onde estão os dados
root = tk.Tk()
root.withdraw() # Oculta a janela principal do tkinter
root.attributes('-topmost', True)

print("Por favor, selecione a pasta onde estão os arquivos CSV...")
caminho_pasta = filedialog.askdirectory(title="Selecione a pasta dos dados do Doutorado")

# Caso deseje fixar o caminho sem abrir caixa de diálogo, comente as 4 linhas acima 
# e descomente a linha abaixo definindo o caminho no seu computador:
# caminho_pasta = r"C:\Users\SeuUsuario\Documents\Doutorado\Dados\Final"

if not caminho_pasta:
    raise ValueError("Nenhuma pasta foi selecionada. O processo foi cancelado.")

arquivo_entrada = os.path.join(caminho_pasta, 'Completo_2011_2024.csv')
arquivo_saida = os.path.join(caminho_pasta, 'Completo_Anon_2011_2024.csv')
arquivo_salt = os.path.join(caminho_pasta, 'salt_chave_secreta.txt')

# Verificar se o arquivo de origem existe
if not os.path.exists(arquivo_entrada):
    raise FileNotFoundError(f"O arquivo não foi encontrado no caminho:\n{arquivo_entrada}")

# ==============================================================================
# 2. CARREGAR O DATASET
# ==============================================================================
print("\nCarregando o arquivo original (pode levar alguns segundos)...")
# sep=None detecta automaticamente a vírgula ou ponto-e-vírgula do arquivo CSV
df = pd.read_csv(arquivo_entrada, sep=None, engine='python', encoding='utf-8-sig')

# Limpeza preventiva de espaços invisíveis no cabeçalho
df.columns = df.columns.str.strip()

# ==============================================================================
# 3. LOCALIZAR AS COLUNAS DINAMICAMENTE
# ==============================================================================
col_pedido = next((c for c in ['NumeroPedido', 'Numero Pedido', 'Nº Pedido', 'NoPedido'] if c in df.columns), None)
col_paciente = next((c for c in ['NomePaciente', 'Nome Paciente', 'Nome do Paciente'] if c in df.columns), None)
col_registro = next((c for c in ['Registro', 'NumeroRegistro', 'Nº Registro', 'NoRegistro', 'Prontuario', 'Prontuário'] if c in df.columns), None)

# Validação estrita
if not col_pedido or not col_paciente:
    print("\n[ERRO CRÍTICO] Não foram localizadas as colunas exatas de Pedido ou Paciente.")
    print("Colunas reais detectadas no arquivo:", df.columns.tolist())
    raise KeyError("Ajuste as listas de busca na seção 3 com a grafia exata contida no seu arquivo.")

print(f"-> Pedido detectado: '{col_pedido}'")
print(f"-> Paciente detectado: '{col_paciente}'")
print(f"-> Registro detectado: '{col_registro}'")

# ==============================================================================
# 4. GESTÃO SEGURA E PERSISTENTE DO SALT (LONGITUDINALIDADE DOS DADOS)
# ==============================================================================
# Tenta reaproveitar o mesmo SALT caso já tenha sido gerado anteriormente
if os.path.exists(arquivo_salt):
    with open(arquivo_salt, 'r', encoding='utf-8') as f:
        salt = f.read().strip()
    print("-> Chave Salt recuperada com sucesso do arquivo 'salt_chave_secreta.txt'.")
else:
    salt = secrets.token_hex(32)
    with open(arquivo_salt, 'w', encoding='utf-8') as f:
        f.write(salt)
    print("-> Nova chave Salt gerada e salva com segurança em 'salt_chave_secreta.txt'.")

def anonimizar_valor(valor, salt_key, sufixo_dominio=""):
    """
    Gera hash SHA-256 combinando o dado original com o Salt e um sufixo de contexto.
    Evita colisões entre prontuários e números de pedido coincidentes.
    """
    if pd.isna(valor) or str(valor).strip() == "":
        return valor
    
    texto_puro = f"{str(valor).strip()}_{sufixo_dominio}_{salt_key}"
    return hashlib.sha256(texto_puro.encode('utf-8')).hexdigest()

# ==============================================================================
# 5. APLICAR A ANONIMIZAÇÃO
# ==============================================================================
print("\nIniciando o processo de anonimização criptográfica SHA-256...")

df[col_pedido] = df[col_pedido].apply(lambda x: anonimizar_valor(x, salt, "PEDIDO"))
df[col_paciente] = df[col_paciente].apply(lambda x: anonimizar_valor(x, salt, "PACIENTE"))

if col_registro:
    df[col_registro] = df[col_registro].apply(lambda x: anonimizar_valor(x, salt, "REGISTRO"))
else:
    print("[ALERTA] Coluna de Registro/Prontuário não foi identificada. Apenas Pedido e Paciente foram anonimizados.")

# ==============================================================================
# 6. EXPORTAR O RESULTADO ANONIMIZADO
# ==============================================================================
print(f"\nSalvando o arquivo resultante em:\n{arquivo_saida}")
df.to_csv(arquivo_saida, index=False, encoding='utf-8-sig')

print("\n" + "="*80)
print("[SUCESSO] Processo concluído com êxito!")
print(f"Arquivo de saída: Completo_Anon_2011_2024.csv")
print(f"Total de registros processados: {df.shape[0]} linhas x {df.shape[1]} colunas.")
print("="*80)
