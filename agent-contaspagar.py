
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from dateutil.relativedelta import relativedelta
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from datetime import datetime, timedelta
from dotenv import load_dotenv
from rich.pretty import pprint
from dataclasses import dataclass
from datetime import date
import sqlite3
import time
import os

load_dotenv()
start = time.perf_counter()

@dataclass # gera automaticamente o __init__ 
class UserInfos:
    name: str
    age: int 
    city: str

api_key_google = os.environ['GOOGLE_API_KEY']
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash-lite",
    api_key=api_key_google,
)

date_current = date.today().isoformat()
system_prompt = f"""
Você é um assistente financeiro amigável e direto, especializado em registrar contas a pagar.
A data de hoje é: {date_current} (YYY-MM-DD)

## Sua única função
Registrar contas a pagar no banco de dados com os seguintes campos:
- Data de vencimento (formato YYYY-MM-DD)
- Descrição
- Destinatário
- Valor
- Categoria (inferida automaticamente por você)

### Categorias disponíveis
Infira a **categoria** com base na descrição e destinatário informados pelo usuário:
- `Moradia` — aluguel, condomínio, IPTU, reformas, manutenção do imóvel
- `Utilidades` — água, luz, gás, internet, telefone
- `Fornecedores` — compras para revenda, matéria-prima, prestadores de serviço
- `Alimentação` — mercado, feira, restaurante, delivery, hortifruti
- `Impostos e Taxas` — DAS, DARF, ISS, contador, taxas bancárias
- `Outros` — qualquer conta que não se encaixe nas categorias acima

###
- Confirme com o usuario antes de usar a tool `create_transaction_unique` ou `create_transaction_recurrence`.

## ATENÇÃO!

Ao utilizar as tools `get_due_bills` e `get_due_bills_today`, repasse ao usuário apenas as seguintes informações de cada conta:
- ID
- Descrição
- Destinatário
- Valor
- Data de vencimento

Não mencione outras colunas como status, data de criação, etc.

Ao utilizar a tool `value_total_by_category`, e receber os dados de retorno, 
reformate as informações de forma clara e amigável para o usuário.
Nunca retorne os dados crus como lista, dicionário ou algo vazio.

## Como se comportar

1. **Colete as informações** — se o usuário não informar todos os dados, pergunte os que faltam.
2. **Infira a categoria** — com base na descrição e destinatário, sugira a categoria automaticamente.
3. **Confirme antes de inserir** — sempre apresente um resumo e pergunte se está correto antes de chamar a tool correta
4. **Insira e confirme** — após a inserção, informe que foi salvo com sucesso.

## Exemplo de fluxo
Usuário: "preciso lançar uma conta"
Você: "Claro! Me passa os detalhes: descrição, destinatário, valor e data de vencimento."

Usuário: "aluguel, Imobiliária XYZ, 1500, vence dia 05/04/2026"
Você: "Confirma o lançamento abaixo?
- Descrição: Aluguel
- Destinatário: Imobiliária XYZ
- Valor: R$ 1.500,00
- Vencimento: 05/04/2026
- Categoria: Moradia 🏠"

Usuário: "sim"
Você: [chama a tool create_transaction_unique] → "Conta registrada com sucesso! ✅"

## Regras importantes
- Nunca insira sem confirmação explícita do usuário.
- Se o usuário passar a data em outro formato (ex: 05/04/2026), converta para YYYY-MM-DD antes de chamar a tool.
- Se tiver dúvida sobre a categoria, sugira a mais provável e deixe o usuário corrigir.
- Fora do escopo de contas a pagar, informe educadamente que não pode ajudar com isso.
"""

@tool
def get_info_user(runtime: ToolRuntime) -> str:
    """
    Retorna as informações do usuário atual (nome e idade e cidade).
    Use esta ferramenta sempre que precisar saber o nome, idade ou/e cidade do usuário,
    ou quando for necessário personalizar a resposta com os dados do usuário.
    """
    return f"O nome do usuário é {runtime.context.name}, a idade é {runtime.context.age} e ele mora em {runtime.context.city}."

@tool
def create_transaction_unique(data: str, descr: str, destinat: str, valor: float, categoria: str) -> str:
    """
    Insere uma conta a pagar no banco de dados local.
    Use esta ferramenta quando o usuário informar uma conta a pagar com data de vencimento,
    descrição, destinatário e valor. Retorna confirmação da inserção.

    Args:
        data: Data de vencimento no formato YYYY-MM-DD (ex: '2026-04-05')
        descr: Descrição da conta (ex: 'Aluguel', 'Conta de luz')
        destinat: Nome do destinatário ou empresa (ex: 'Imobiliária XYZ')
        valor: Valor da conta em reais (ex: 1500.00)
    """

    # valida se nenhum campo veio vazio
    if not all([data, descr, destinat, valor, categoria]):
        return "Erro: todos os campos são obrigatórios."

    if valor <= 0:
        return "Erro: valor deve ser maior que zero."

    with sqlite3.connect("meu_banco.db") as conn:
        cursor = conn.cursor() # cria o ponteiro para executar comandos, é a caneta que escreve
        data_base = datetime.strptime(data, "%Y-%m-%d")  # converte string → datetime

        cursor.execute("""
            INSERT INTO contas_a_pagar (data_vencimento, descricao, destinatario, valor, categoria, recorrencia, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (data_base.strftime("%Y-%m-%d"), descr.capitalize(), destinat.capitalize(), valor, categoria.capitalize(), 'Não', 'A pagar'))
        conn.commit() # confirma/salva as operações

    conn.close() # fecha conexão com bd
    return "Conta inserida com sucesso."

@tool
def create_transaction_recurrence(data: str, descr: str, destinat: str, valor: float, categoria: str, recurrence: int) -> str:
    """
    Insere contas a pagar recorrentes no banco de dados local.
    Use esta ferramenta quando o usuário informar uma conta a pagar que tenha parcelas/mensalidades com data de vencimento,
    descrição, destinatário e valor. Retorna confirmação da inserção.

    Args:
        data: Data de vencimento no formato YYYY-MM-DD
        descr: Descrição da conta
        destinat: Nome do destinatário
        valor: Valor da conta total
        categoria: Categoria da despesa
        recorr: Quantidade total de parcelas/mensalidades
    """

    # valida se nenhum campo veio vazio
    if not all([data, descr, destinat, valor, categoria, recurrence]):
        return "Erro: todos os campos são obrigatórios."

    if valor <= 0:
        return "Erro: valor deve ser maior que zero."
    
    with sqlite3.connect("meu_banco.db") as conn:
        cursor = conn.cursor() # cria o ponteiro para executar comandos, é a caneta que escreve
        data_base = datetime.strptime(data, "%Y-%m-%d")  # converte string → datetime
        valor_parcelado = round(valor / recurrence, 2)

        for i in range(1, recurrence + 1):
            descr_parcela = descr.capitalize() + f' {i}/{recurrence}'
            data_parcela = data_base + relativedelta(months=i - 1)

            cursor.execute("""
                INSERT INTO contas_a_pagar (data_vencimento, descricao, destinatario, valor, categoria, recorrencia, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (data_parcela.strftime("%Y-%m-%d"), descr_parcela, destinat.capitalize(), valor_parcelado, categoria.capitalize(), 'Sim', 'A pagar'))
            
        conn.commit() # confirma/salva as operações

    conn.close() # fecha conexão com bd
    return "Conta inserida com sucesso."

@tool
def get_due_bills(dias: int) -> list | str:
    """Retorna contas a pagar com vencimento nos próximos X dias."""
    if dias < 0:
        return "O parâmetro dias precisa ser maior que 0."
    
    with sqlite3.connect("meu_banco.db") as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM contas_a_pagar
            WHERE data_vencimento >= date('now')
            AND data_vencimento <= date('now', ? || ' days')
            AND status = 'A pagar'
        """, (f"+{dias}",))

        rows = cursor.fetchall()
        if not rows:
            return 'Nenhuma conta a pagar encontrada para hoje.'
        
        return [dict(row) for row in rows]

@tool
def get_due_bills_today() -> list | str:
    """Retorna as contas a pagar com o vencimento de hoje."""
    
    with sqlite3.connect("meu_banco.db") as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM contas_a_pagar
            WHERE data_vencimento >= date('now', 'localtime')
            AND data_vencimento <= date('now', 'localtime', ? || ' days')
            AND status = 'A pagar'
        """, (f"+{1}",))

        rows = cursor.fetchall()
        if not rows:
            return 'Nenhuma conta a pagar encontrada para hoje.'
        
        return [dict(row) for row in rows]

@tool
def get_bills_today() -> list | str:
    """Retorna as contas pagas da data de hoje."""
    
    with sqlite3.connect("meu_banco.db") as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM contas_a_pagar
            WHERE data_vencimento >= date('now', 'localtime')
            AND data_vencimento <= date('now', 'localtime', ? || ' days')
            AND status = 'Paga'
        """, (f"+{1}",))

        rows = cursor.fetchall()
        if not rows:
            return 'Nenhuma conta paga encontrada com data de hoje.'
        
        return [dict(row) for row in rows]

@tool
def update_today_status() -> str:
    """Marca todas as contas a pagar de hoje como 'Pagas'. Use esta tool quando o usuário quiser quitar todas as contas do dia de uma vez, sem especificar uma conta individual."""
    today_date = date.today().strftime("%Y-%m-%d")
    with sqlite3.connect("meu_banco.db") as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE contas_a_pagar
            SET status = ?
            WHERE data_vencimento = ?
        """, ('Paga', today_date))
        conn.commit()
        
        if cursor.rowcount == 0:
            return f"Nenhuma conta encontrada para a data de hoje."
        
        return f"{cursor.rowcount} conta(s) atualizada(s) para a data de hoje."

@tool
def update_status_by_id(id: int, status: str) -> str:
    """Marca uma conta como 'A pagar' ou 'Paga' dado seu ID. Use quando o usuário quiser alterar o status de uma conta específica."""
    if not 'A Pagar' or 'Paga' in status:
        return 'O argumento de status precisa ser A pagar ou Paga'
    
    with sqlite3.connect("meu_banco.db") as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE contas_a_pagar
            SET status = ?
            WHERE id = ?
        """, (status, id))
        conn.commit()
        
        if cursor.rowcount == 0:
            return f"Nenhuma conta encontrada."
        
        return f"{cursor.rowcount} conta(s) atualizada(s)."

@tool
def update_description_by_id(id: int, descr: str) -> str:
    """Atualiza a descrição de uma conta a pagar dado seu ID. Use quando o usuário quiser editar ou corrigir a descrição de uma conta específica."""
    with sqlite3.connect("meu_banco.db") as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE contas_a_pagar
            SET descricao = ?
            WHERE id = ?
        """, (descr, id))
        conn.commit()
        
        if cursor.rowcount == 0:
            return f"Nenhuma conta encontrada."
        
        return f"{cursor.rowcount} conta(s) atualizada(s)."

@tool
def update_recipient_by_id(id: int, dest: str) -> str:
    """Atualiza a coluna de destinatário de uma conta a pagar dado seu ID. Use quando o usuário quiser editar ou corrigir a coluna de destinatário de uma conta específica."""
    with sqlite3.connect("meu_banco.db") as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE contas_a_pagar
            SET destinatario = ?
            WHERE id = ?
        """, (dest, id))
        conn.commit()
        
        if cursor.rowcount == 0:
            return f"Nenhuma conta encontrada."
        
        return f"{cursor.rowcount} conta(s) atualizada(s)."

@tool
def update_value_by_id(id: int, value: float) -> str:
    """Atualiza a coluna de valor de uma conta a pagar dado seu ID. Use quando o usuário quiser editar ou corrigir a coluna de valor de uma conta específica."""
    with sqlite3.connect("meu_banco.db") as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE contas_a_pagar
            SET valor = ?
            WHERE id = ?
        """, (value, id))
        conn.commit()
        
        if cursor.rowcount == 0:
            return f"Nenhuma conta encontrada."
        
        return f"{cursor.rowcount} conta(s) atualizada(s)."
    
@tool
def update_category_by_id(id: int, categ: str) -> str:
    """Atualiza a coluna de categoria de uma conta a pagar dado seu ID. Use quando o usuário quiser editar ou corrigir a coluna de categoria de uma conta específica."""
    with sqlite3.connect("meu_banco.db") as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE contas_a_pagar
            SET categoria = ?
            WHERE id = ?
        """, (categ, id))
        conn.commit()
        
        if cursor.rowcount == 0:
            return f"Nenhuma conta encontrada."
        
        return f"{cursor.rowcount} conta(s) atualizada(s)."
        
@tool
def update_date_by_id(id: int, data: str) -> str:
    """Atualiza a coluna de data de uma conta a pagar dado seu ID. Use quando o usuário quiser editar ou corrigir a coluna de data de uma conta específica."""
    new_data = datetime.strptime(data, "%Y-%m-%d")  # converte string → datetime
    with sqlite3.connect("meu_banco.db") as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE contas_a_pagar
            SET data_vencimento = ?
            WHERE id = ?
        """, (new_data.strftime("%Y-%m-%d"), id))
        conn.commit()
        
        if cursor.rowcount == 0:
            return f"Nenhuma conta encontrada."
        
        return f"{cursor.rowcount} conta(s) atualizada(s)."

@tool
def delete_by_id(id: int) -> str:
    """Deleta uma linha dado seu ID. Use quando o usuário quiser deletar ou excluir a linha um registro específico."""
    with sqlite3.connect("meu_banco.db") as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            DELETE FROM contas_a_pagar
            WHERE id = ?
        """, (id,))
        conn.commit()
        
        if cursor.rowcount == 0:
            return f"Nenhuma conta encontrada."
        
        return f"{cursor.rowcount} conta(s) deletada(s)."

@tool
def value_total_by_category() -> list[dict]:
    """
    Retorna o valor total gasto por categoria a partir do banco de dados.
    Use esta tool quando o usuário perguntar sobre gastos por categoria,
    quanto foi gasto em cada categoria, ou quiser um resumo financeiro por categoria.
    O retorno é uma lista de dicionários com os campos 'categoria' e 'total'.
    """
    
    with sqlite3.connect("meu_banco.db") as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                categoria,
                SUM(valor) AS total
            FROM contas_a_pagar
            GROUP BY categoria
            ORDER BY total DESC
        """)

        rows = cursor.fetchall()
        if not rows:
            return 'Nenhuma conta paga encontrada com data de hoje.'
        
        return [{'categoria': row['categoria'], 'total': f'{row['total']:.2f}'} for row in rows]

tools_agent = [update_recipient_by_id, update_description_by_id, get_info_user, create_transaction_unique, 
               create_transaction_recurrence, get_due_bills, get_due_bills_today, 
               update_today_status, update_status_by_id, get_bills_today, update_value_by_id, update_category_by_id,
               update_date_by_id, delete_by_id, value_total_by_category]

agent = create_agent(
    model=llm,
    tools=tools_agent,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
    context_schema=UserInfos,
)

while True:
    input_text = str(input("User: "))
    if input_text.lower() == "exit":
        break
    response_ = agent.invoke(
        {"messages": [HumanMessage(input_text)]},
        {"configurable": {"thread_id": "1"}},
        context=UserInfos(name='Patrick', age=23, city='Porto Alegre')
    )
    pprint(response_)
    print(f"\nTempo: {time.perf_counter() - start:.4f}s")
