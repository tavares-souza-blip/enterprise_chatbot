from openai import OpenAI
import json

MODELO_IA_CLASSIFICADOR = "gpt-4o-mini"

def classificar_intencao(client, pergunta: str, historico: list) -> tuple[str, str]:
    """
    Retorna (tipo, produto_extraido).
    
    Tipos:
    - 'especifica_produto'  → máquina, equipamento ou categoria de produção
    - 'servico_curso'       → cursos, treinamentos, masterclasses, consultoria, aula, capacitação, capacitación, treinamento, consultoría, workshop, training, course
    - 'institucional'       → saudações, confirmações, perguntas sobre a empresa
    - 'consultivo'          → restrições de negócio sem produto (orçamento, espaço, público)
    - 'fora_de_escopo'      → completamente desconexo do ramo de vendas da Finamac (ramo de vendas da Finamac: gelados; vitrines para exposição de produtos, máquinas produtoras de picolés, sorvetes, gelato e gelo saborizado; peças de manutenção para máquinas; cursos para produção de produtos da área de gelado)
    - 'comparacao_produtos' → comparações diretas entre produtos ou serviços (ex: "Qual é melhor, a Turbo 8 ou a Pro 16?")
    - 'recomendacao_produtos' → recomendações de produtos ou serviços com base em critérios do usuário (ex: "Qual máquina é melhor para um espaço pequeno?")
    """

    contexto = (
        "\n".join([f"{m['role']}: {m['content']}" for m in historico[-4:]])
        if historico
        else "Sem histórico anterior."
    )

    prompt_sistema = """Você é o classificador semântico do assistente consultivo da Finamac.

DEFINIÇÃO DO ESCOPO DA FINAMAC (USE COMO REFERÊNCIA ABSOLUTA):
A Finamac atua em dois eixos de negócio:
1. EQUIPAMENTOS: Máquinas industriais e comerciais para produção de sorvete, picolé (paleta/ice pop), açaí, gelato, chocolate, misturadores, ultracongeladores e tanques de maturação. Inclui acessórios, peças e serviços técnicos relacionados.
2. EDUCAÇÃO E CONSULTORIA: Cursos presenciais, masterclasses, treinamentos técnicos, consultoria para abertura de negócios no setor de gelados. Qualquer menção a "curso", "aula", "capacitación", "treinamento", "masterclass" é escopo válido.

REGRA DE ESCOPO: Classifique como 'fora_de_escopo' APENAS assuntos completamente desconexos (automóveis, futebol, meteorologia, culinária geral não relacionada a gelados). Dúvidas sobre viabilidade de negócio, investimento inicial, tamanho de espaço e público-alvo são CONSULTIVAS e estão dentro do escopo.

REGRA DE CONSULTIVO: 
Quando o usuário estiver apenas descrevendo
necessidades, restrições ou objetivos. Terá como banco de informações 
apenas o conteúdo que consta no site da Finamac (https://finamac.com/). 
A invenção de catálogos, itens e qualquer conteúdo fora do ramo da Finamac
é severamente repreendido. 

Exemplos:
"Tenho orçamento baixo"
"Meu espaço é pequeno"
"Quero produzir 1000 picolés por dia"

→ consultivo

REGRA DE RECOMENDAÇÃO:

Classifique como recomendacao_produtos quando o usuário:

- pedir sugestões
- pedir indicações
- pedir opções
- pedir máquinas adequadas
- pedir lista de equipamentos
- pedir qual máquina atende determinada demanda

mesmo que mencione orçamento,
espaço ou capacidade produtiva.

Exemplos:

"Qual máquina você recomenda?"
"Liste equipamentos para produzir 10.000 picolés"
"Quais máquinas atendem essa demanda?"
"Me indique uma linha para produzir açaí"

→ recomendacao_produtos

REGRAS DE COMPARAÇÃO: 
1. Quando houver comparação entre dois modelos, use:
{
"tipo":"comparacao_produtos",
"produto_a":"...",
"produto_b":"..."
}
2. Quando houver apenas um equipamento:
{
"tipo":"especifica_produto",
"produto":"..."
}
3. Preserve exatamente números e nomes dos modelos.

EXTRAÇÃO DE ENTIDADE (campo "produto"):
- Se o usuário citar um modelo comercial com código (ex: "Turbo 8", "Pro 16"), extraia exatamente esse nome com dígitos intactos.
- Se citar um curso ou serviço (ex: "Clase magistral de helados"), extraia o nome do curso ou "curso" como palavra-chave.
- Se não houver entidade específica identificável, retorne "NONE".
- ZERO INFERÊNCIA: nunca invente um modelo que não foi citado.

NORMALIZAÇÃO DE IDIOMA: A pergunta pode vir em português, espanhol ou inglês. Classifique pela semântica, não pelo idioma.

EXEMPLOS (FEW-SHOT MULTILÍNGUES):

Exemplo 1:
Histórico: Sem histórico.
Mensagem: "Hola, deseo conocer cuál es la inversión para una máquina pequeña de hacer helados"
JSON: {"tipo": "consultivo", "produto": "sorvete"}

Exemplo 2:
Histórico: Sem histórico.
Mensagem: "Mi presupuesto es bajo y tengo un local de 2x2 metros para vender helados"
JSON: {"tipo": "consultivo", "produto": "NONE"}

Exemplo 3:
Histórico: Sem histórico.
Mensagem: "Quiero información sobre la Clase magistral de helados y paletas"
JSON: {"tipo": "servico_curso", "produto": "Clase magistral de helados"}

Exemplo 4:
Histórico: assistant: "Temos picoleteiras da linha Turbo disponíveis."
Mensagem: "Qual a capacidade da Turbo 8?"
JSON: {"tipo": "especifica_produto", "produto": "Turbo 8"}

Exemplo 5:
Histórico: Sem histórico.
Mensagem: "Obrigado, entendi!"
JSON: {"tipo": "institucional", "produto": "NONE"}

Exemplo 6:
Histórico: Sem histórico.
Mensagem: "Qual a diferença entre Turbo 8 e Turbo 25?"
JSON: {"tipo": "comparacao_produtos", "produto_a": "Turbo 8", "produto_b": "Turbo 25"}

Exemplo 7:
Histórico: Sem histórico.
Mensagem: "Quero uma recomendação de máquina para um espaço pequeno."
JSON: {"tipo": "recomendacao_produtos", "produto": "máquina pequena"}

Exemplo 8:
Histórico: Sem histórico.
Mensagem: "Quero produzir 10.000 picolés por dia"
JSON: {"tipo": "recomendacao_produtos", "produto": "picolés"}

Exemplo 9:
Histórico: Sem histórico.
Mensagem: "Quero saber sobre sorvete artesanal"
JSON: {"tipo": "especifica_produto", "produto": "sorvete"}"""

    prompt_usuario = f"""Histórico recente:
{contexto}

Mensagem atual: "{pergunta}"

Retorne OBRIGATORIAMENTE este JSON:
{{
  "tipo": "especifica_produto" | "servico_curso" | "consultivo" | "institucional" | "comparacao_produtos"  | "recomendacao_produtos" | "fora_de_escopo",
  "produto": "entidade extraída ou NONE"
}}"""

    try:
        resp = client.chat.completions.create(
            model=MODELO_IA_CLASSIFICADOR,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        dados = json.loads(resp.choices[0].message.content)
        tipo = dados.get("tipo", "institucional")
        produto = dados.get("produto", "NONE")

        # Garante que o tipo retornado é válido
        tipos_validos = {"especifica_produto", "servico_curso", "consultivo", "institucional", "comparacao_produtos", "recomendacao_produtos", "fora_de_escopo"}
        if tipo not in tipos_validos:
            tipo = "institucional"

        return tipo, produto

    except Exception:
        return "institucional", "NONE"