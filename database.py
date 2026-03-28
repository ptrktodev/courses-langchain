import sqlite3
from datetime import date

with sqlite3.connect("meu_banco.db") as conn:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM contas_a_pagar WHERE id = 1")
    conn.commit() # confirma/salva as operações
 

def create_transaction_unique(data: str, descr: str, destinat: str, valor: float, categoria: str, recorrencia: str) -> str:
    with sqlite3.connect("meu_banco.db") as conn:
        cursor = conn.cursor() # cria o ponteiro para executar comandos, é a caneta que escreve
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS contas_a_pagar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_vencimento DATE NOT NULL,
            descricao TEXT NOT NULL,
            destinatario TEXT NOT NULL,
            valor REAL NOT NULL,
            categoria TEXT NOT NULL,
            recorrencia TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)
        cursor.execute("""
            INSERT INTO contas_a_pagar (data_vencimento, descricao, destinatario, valor, categoria, recorrencia, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (data, descr, destinat, valor, categoria, recorrencia, 'A pagar'))
        conn.commit() # confirma/salva as operações

    conn.close() # fecha conexão com bd

# create_transaction_unique('06/05/2026', 'Vôlei com a galera', 'Arena Voolh', 15.90, 'Utilidade', 'Não')

'''with sqlite3.connect("meu_banco.db") as conn:
    cursor = conn.cursor()
    dias = 10
    cursor.execute("""
        SELECT * FROM contas_a_pagar
        WHERE data_vencimento >= date('now')
        AND data_vencimento <= date('now', ? || ' days')
    """, (f"+{dias}",))

    resultados = cursor.fetchall()  # agora sim tá na variável
    print(resultados)

conn.close()'''


