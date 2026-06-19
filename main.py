from dotenv import load_dotenv
import os
import sys
import time
import difflib
import re
#from classifier import classificar_intencao
from scraper import (
    buscar_produto, obter_produto, obter_preco,
    mapear_catalogo_maquinas, obter_produtos_da_colecao, http_client
)
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Centralizando o histórico de conversas global de forma limpa
historico_conversa = []
ultima_conversa = None
nome_usuario = None
modelos_listados = [] # guarda a lista exibida ao usuário com posição, nome e URL

CONTATO_COMERCIAL = "vendas@finamac.com.br ou pelo telefone +55 11 98846-5990. Se preferir, faça uma visita ao nosso Showroom em São Paulo para conhecer nossos produtos pessoalmente!"


# ─────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────

def efeito_digitar(texto):
    sys.stdout.write("\n")
    for letra in texto:
        sys.stdout.write(letra)
        sys.stdout.flush()
        time.sleep(0.01)


def formatar_preco_range(preco_float, idioma_usuario="pt"):
    inferior = preco_float * 0.80
    superior = preco_float * 1.20

    if idioma_usuario in ["en", "english"]:
        moeda = "USD"
        txt_inferior = f"{inferior:,.2f}"
        txt_superior = f"{superior:,.2f}"
        return f"between {moeda} {txt_inferior} and {moeda} {txt_superior}"
 
    elif idioma_usuario in ["es", "spanish"]:
        moeda = "USD"
        txt_inferior = f"{inferior:,.2f}"
        txt_superior = f"{superior:,.2f}"
        return f"entre {moeda} {txt_inferior} y {moeda} {txt_superior}"
 
    else:
        moeda = "R$"
        txt_inferior = f"{inferior:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        txt_superior = f"{superior:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"entre {moeda} {txt_inferior} e {moeda} {txt_superior}"

def escolher_melhor_produto(urls, nome_buscado):
    nome_lower = nome_buscado.lower()
    
    numeros_buscados = set(re.findall(r'\b\d+\b', nome_lower))
    if not numeros_buscados:
        numeros_buscados = set(re.findall(r'\d+', nome_lower))

    melhor_url = None
    melhor_score = -999
    
    penalidades = ["seal", "blade", "kit", "mold", "molde", "holder", "spare", "peca", "parts", "garantia", "warranty", "embalagem", "packaging", "accessory", "acessorio", "extended", "servico", "start up"]
    bonus = ["maquina", "machine", "producer", "freezer", "batch", "industrial", "picole", "artesanal", "descontinua", "produtora", "produção", "gelato", "vista", "vitrine", "expositor", "sorvete", "açaí", "acai", "chocolate", "mixer"]

    for url in urls:
        url_lower = url.lower().replace("_", "-").replace("/", "-")
        score = difflib.SequenceMatcher(None, nome_lower.replace(" ", "-"), url_lower).ratio() * 100
        
        for p in penalidades:
            if p in url_lower: 
                score -= 40
        for b in bonus:
            if b in url_lower: 
                score += 20

        numeros_na_url = set(re.findall(r'\b\d+\b', url_lower))
        if not numeros_na_url:
            numeros_na_url = set(re.findall(r'\d+', url_lower))

        if numeros_buscados:
            if numeros_buscados.intersection(numeros_na_url):
                score += 80  
            else:
                if numeros_na_url:
                    score -= 150  

        if score > melhor_score:
            melhor_score = score
            melhor_url = url
            
    return melhor_url

# ─────────────────────────────────────────
# FUNÇÕES DE CLASSIFICAÇÃO VIA IA
# ─────────────────────────────────────────

def detectar_referencia_posicional(pergunta):
    if not modelos_listados:
        return None

    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Sua única tarefa é identificar se o usuário mencionou uma ordem ou posição numérica "
                    "(ex: primeiro, segundo, terceiro, opção 1, número 2, o último, etc.) relacionada a uma lista anterior, "
                    "MESMO que a frase venha acompanhada de outras perguntas longas na mesma mensagem.\n"
                    "Responda SOMENTE com o número da posição (1, 2, 3...) ou a palavra 'ULTIMO'. "
                    "Se não houver nenhuma menção posicional evidente, responda estritamente 'NONE'.\n\n"
                    "EXEMPLOS:\n"
                    "'me fale sobre o primeiro, quais são as máquinas de picolés?' → 1\n"
                    "'qual a capacidade do segundo modelo?' → 2\n"
                    "'quero ver o último da lista' → ULTIMO\n"
                    "'quais máquinas fazem picolés?' → NONE"
                )
            },
            {"role": "user", "content": pergunta}
        ],
        temperature=0.0
    )

    resultado = resposta.choices[0].message.content.strip().upper()

    if "ULTIMO" in resultado:
        return len(modelos_listados) - 1

    try:
        # Extrai apenas os dígitos caso a IA retorne algo pontuado como "1."
        digitos = re.findall(r'\d+', resultado)
        if digitos:
            pos = int(digitos[0]) - 1
            if 0 <= pos < len(modelos_listados):
                return pos
    except ValueError:
        pass

    return None

def usuario_refina_lista(pergunta):
    if not modelos_listados:
        return False

    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "O usuário recebeu uma lista de máquinas e pode estar filtrando ou refinando.\n"
                    "Responda SOMENTE SIM ou NAO.\n\n"
                    "SIM: filtra, exclui, refina ou pede subcategoria da lista atual\n"
                    "NAO: muda completamente de assunto ou produto\n\n"
                    "EXEMPLOS:\n"
                    "'quero um industrial' → SIM\n"
                    "'que não seja a Popline' → SIM\n"
                    "'mostre só os maiores' → SIM\n"
                    "'quero saber de sorvete agora' → NAO\n"
                    "'qual o preço da Turbo 8?' → NAO"
                )
            },
            {"role": "user", "content": pergunta}
        ]
    )
    return resposta.choices[0].message.content.strip().upper() == "SIM"

def tipo_pergunta(pergunta, historico_ia=None):
    import json
    
    # Filtro rápido para respostas curtas de cortesia (evita gastar API e bloqueios errados)
    repostas_cortesia = ["ok", "obrigado", "obrigada", "valeu", "entendi", "perfeito", "tudo bem", "blz", "show"]
    if pergunta.lower().strip().rstrip(".") in repostas_cortesia:
        return "institucional", "NONE"

    contexto_historico = ""
    if historico_ia:
        ultimas_mensagens = historico_ia[-4:]
        contexto_historico = "\n".join([f"{m['role']}: {m['content']}" for m in ultimas_mensagens])

    prompt = f"""
    Você é o classificador de intenções do chatbot da Finamac.
    Analise a pergunta do usuário considerando o histórico recente.

    Histórico:
    {contexto_historico}

    Pergunta atual: "{pergunta}"

    Classifique em uma das opções abaixo no formato JSON:
    1. "especifica_produto": O usuário quer saber sobre uma máquina, modelo, preço ou detalhes de um equipamento específico da Finamac.
    2. "institucional": Saudações (olá, bom dia), agradecimentos (obrigado, ok), ou perguntas sobre a empresa Finamac em geral.
    3. "fora_de_escopo": Assuntos que não têm NENHUMA relação com gelados, sorvetes, picolés ou com a Finamac (ex: futebol, previsão do tempo).

    Responda EXCLUSIVAMENTE neste formato JSON:
    {{
        "tipo": "especifica_produto" ou "institucional" ou "fora_de_escopo",
        "produto": "Nome do produto ou NONE"
    }}
    """
    
    try:
        # Substitua pela sua chamada real de IA do main.py (ex: client.chat.completions...)
        resposta_bruta = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        ).choices[0].message.content
        
        dados = json.loads(resposta_bruta)
        return dados.get("tipo", "fora_de_escopo"), dados.get("produto", "NONE")
    except Exception:
        return "institucional", "NONE"
    
def extrair_produto(pergunta):
    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the exact product model name from the sentence.\n"
                    "Rules:\n"
                    "- Return ONLY the model name, capitalized correctly (ex: 'Turbo 8', 'Pro 4', 'Mixer 15').\n"
                    "- If the model name contains a number, always include it (ex: 'Turbo 8', not 'Turbo').\n"
                    "- If no specific model is mentioned, return exactly: NONE\n"
                    "- Never return brand names or generic category words.\n\n"
                    "EXAMPLES:\n"
                    "'qual o preço do modelo pro 4' → Pro 4\n"
                    "'tell me about turbo8' → Turbo 8\n"
                    "'I want an ice pop machine' → NONE\n"
                    "'mixer' → NONE"
                    
                )            
            },
            {"role": "user", "content": pergunta}
        ]
    )
    return resposta.choices[0].message.content.strip()


def usuario_confirmou_interesse(pergunta):
    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Analyze the phrase and respond ONLY with SIM or NAO.\n"
                    "SIM: user wants to see the products of that category.\n"
                    "NAO: user is changing the subject.\n\n"
                    "EXAMPLES:\n"
                    "'sim' → SIM\n'yes' → SIM\n'quiero ver' → SIM\n'no' → NAO"
                )
            },
            {"role": "user", "content": pergunta}
        ]
    )
    return resposta.choices[0].message.content.strip().upper() == "SIM"


def detectar_categoria(pergunta):
    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Identifique a categoria de máquina Finamac mais adequada para a frase do usuário.\n"
                    "Responda SOMENTE com uma das categorias abaixo ou NONE:\n\n"
                    "- ice-pops (picoleteira, picolé, paleta, popsicle, geladinho, gelo)\n"
                    "- ice-cream-batch-freezers (sorvete artesanal, mantecadeira)\n"
                    "- gelato-batch-freezers (gelato, sorvete italiano)\n"
                    "- ice-cream-and-acai-artisanal-packages (açaí, tigela gelada, frozen bowl)\n"
                    "- mixers-blenders (mixer, misturador, triturador, blender, polpa)\n"
                    "- chocolate-tempering-machines (chocolate, temperagem)\n"
                    "- blast-freezers (abatedor, resfriamento rápido, blast, ultra congelador, congelador)\n"
                    "- aging-tanks (tina, tanque, maturação, tanque de maturação, maturador)\n"
                    "- continuous-freezers-industrial-ice-cream (industrial, linha contínua, sorvete)\n"
                    "- ice-pop-industrial-machines (picoleteira, picolé industrial, linha automática)\n\n"
                    "EXEMPLOS:\n"
                    "'quero máquinas de picolé' → ice-pops"
                )
            },
            {"role": "user", "content": pergunta}
        ]
    )
    #st.write("DEBUG categoria =", categoria_detectada)
    resultado = resposta.choices[0].message.content.strip()
    return None if resultado == "NONE" else resultado


# ────────────────────────────────────────────────────
# CAMADA DE VALIDAÇÃO TÉCNICA REINTEGRADA
# ────────────────────────────────────────────────────

CAMPOS_TECNICOS = [
    "voltagem", "voltage", "kw", "consumo", "consumption",
    "produção", "production", "capacidade", "capacity",
    "peso", "weight", "dimensões", "dimensions",
    "picolés/hora", "popsicles/hour", "litros", "liters"
]

def validar_dados_produto(produto: dict) -> dict:
    if not isinstance(produto, dict):
        produto = {}
    descricao = produto.get("descricao", "") or ""
    ficha = produto.get("ficha_tecnica", {}) or {}
    
    tem_dados = ("[DADOS_INDISPONÍVEIS]" not in descricao and len(descricao) > 20) or len(ficha) > 0
    
    texto_para_analise = (descricao + " " + " ".join(ficha.keys()) + " " + " ".join(ficha.values())).lower()
    tem_ficha_tecnica = any(campo in texto_para_analise for campo in CAMPOS_TECNICOS)

    produto["_validacao"] = {
        "tem_dados": tem_dados,
        "tem_ficha_tecnica": tem_ficha_tecnica,
    }
    return produto

# ─────────────────────────────────────────
# FUNÇÕES DE RESPOSTA VIA IA
# ─────────────────────────────────────────

def perguntar_ia(pergunta, produto, nome, idioma, historico_atual):
    validacao = produto.get("_validacao", {})
    tem_dados = validacao.get("tem_dados", False)
    tem_ficha = validacao.get("tem_ficha_tecnica", False)

    if not tem_dados:
        restricao = (
            "REGRA ABSOLUTA DE ZERO INFERÊNCIA:\n"
            "Você NÃO pode:\n"
            "- Calcular valores a partir de outros dados presentes\n"
            "- Estimar com base em modelos similares\n"
            "- Usar conhecimento geral sobre máquinas para preencher lacunas\n"
            "- Dizer 'aproximadamente' ou 'em torno de' para dados técnicos\n\n"
            "Se o dado não estiver LITERALMENTE escrito nos dados abaixo, "
            "responda: 'Não localizei essa especificação no site da Finamac. "
            f"Para detalhes precisos, recomendo contatar: {CONTATO_COMERCIAL}'\n\n"
            "Você PODE:\n"
            "- Explicar o que é o produto\n"
            "- Listar as especificações que estão nos dados\n"
            "- Orientar o cliente comercialmente\n"
            "- Traduzir termos técnicos que já estejam nos dados"
        )
    elif not tem_ficha:
        restricao = (
            "Os dados coletados do site são parciais. "
            "Responda apenas o que estiver explicitamente escrito abaixo. "
            "Para dados técnicos não presentes, oriente o cliente ao departamento comercial."
        )
    else:
        restricao = (
            "Use SOMENTE os dados abaixo. Para qualquer dado não presente, "
            "oriente ao departamento comercial."
        )

    ficha_formatada = ""
    if "ficha_tecnica" in produto and isinstance(produto["ficha_tecnica"], dict):
        ficha_formatada = "\n".join([f"- {k}: {v}" for k, v in produto["ficha_tecnica"].items()])

    system_prompt = (
        f"Você é um consultor técnico da Finamac atendendo: {nome}.\n"
        f"Idioma obrigatório: {idioma}.\n\n"
        f"{restricao}\n\n"
        f"DADOS DO PRODUTO:\n"
        f"Título: {produto.get('titulo', '')}\n"
        f"Descrição Geral: {produto.get('descricao', '')}\n"
        f"Ficha Técnica Estruturada:\n{ficha_formatada}"
    )
    
    mensagens = (
        [{"role": "system", "content": system_prompt}]
        + historico_atual[-6:]
        + [{"role": "user", "content": pergunta}]
    )

    resposta = client.chat.completions.create(model="gpt-4o-mini", messages=mensagens)
    conteudo_resposta = resposta.choices[0].message.content
    
    #historico_atual.append({"role": "user", "content": pergunta})
    #historico_atual.append({"role": "assistant", "content": conteudo_resposta})
    return conteudo_resposta


def perguntar_ia_generico(pergunta, nome, catalogo_maquinas, idioma, historico_atual):
    lista = "\n".join([f"- {item}" for item in catalogo_maquinas])
    system_prompt = (
        f"Você é um consultor humanizado da Finamac, atendendo: {nome}.\n\n"
        f"Idioma obrigatório da resposta: {idioma}\n"
        "Se idioma = en, responda SOMENTE em inglês.\n"
        "Se idioma = es, responda SOMENTE em espanhol.\n"
        "Se idioma = pt, responda SOMENTE em português.\n\n"
        "REGRAS:\n"
        "1. Use a lista abaixo para sugerir grupos ou modelos — eles são REAIS, não inventados.\n"
        "2. Você PODE e DEVE citar os nomes da lista quando forem relevantes.\n"
        "3. Se o usuário pedir trituradores, associe com 'Mixers & Blenders' e cite os modelos dessa linha.\n"
        "4. Seja direto, cordial, e pergunte qual categoria ou modelo ele quer explorar.\n"
        "5. Responda OBRIGATORIAMENTE no mesmo idioma da última mensagem do usuário.\n\n"
        f"EQUIPAMENTOS E CATEGORIAS REAIS DO SITE:\n{lista}"
    )
    mensagens = [{"role": "system", "content": system_prompt}] + historico_atual[-6:] + [{"role": "user", "content": pergunta}]

    resposta = client.chat.completions.create(model="gpt-4o-mini", messages=mensagens)
    conteudo = resposta.choices[0].message.content
    
    historico_atual.append({"role": "user", "content": pergunta})
    historico_atual.append({"role": "assistant", "content": conteudo})
    return conteudo


def perguntar_ia_lista_colecao(pergunta, nome, titulo_colecao, produtos_lista, idioma, historico_atual):
    lista = "\n".join([f"{i+1}. {p['nome']}" for i, p in enumerate(produtos_lista)])

    system_prompt = (
        f"Você é um consultor da Finamac atendendo: {nome}.\n"
        f"Categoria consultada: {titulo_colecao}.\n\n"
        f"Idioma obrigatório da resposta: {idioma}\n"
        "Se idioma = en, responda SOMENTE em inglês.\n"
        "Se idioma = es, responda SOMENTE em espanhol.\n"
        "Se idioma = pt, responda SOMENTE em português.\n\n"
        "REGRAS:\n"
        "1. Apresente os modelos abaixo de forma clara e numerada.\n"
        "Quando o usuário pedir recomendações,\n"
        "escolha até 5 modelos da lista.\n"
        "Quando perguntar por modelos menores,\n"
        "maiores, industriais ou compactos,\n"
        "filtre usando os nomes disponíveis.\n"
        "NUNCA responda que só existe um modelo\n"
        "se houver mais de um item na lista.\n"
        "2. Esses modelos são REAIS — cite-os diretamente.\n"
        "3. NÃO mencione acessórios, peças, moldes ou embalagens.\n"
        "4. Pergunte qual deles o cliente quer conhecer em detalhes.\n"
        "5. Responda no mesmo idioma da última mensagem do usuário.\n\n"
        f"MODELOS DISPONÍVEIS:\n{lista}"
    )
    mensagens = [{"role": "system", "content": system_prompt}] + historico_atual[-6:] + [{"role": "user", "content": pergunta}]

    #st.write("DEBUG lista enviada para IA:")
    for p in produtos_lista:
        print(p["nome"])

    resposta = client.chat.completions.create(model="gpt-4o-mini", messages=mensagens)
    conteudo = resposta.choices[0].message.content
    
    historico_atual.append({"role": "user", "content": pergunta})
    historico_atual.append({"role": "assistant", "content": conteudo})
    return conteudo


def detectar_idioma(pergunta: str) -> str:
    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Detect the language of the sentence.\n"
                    "Reply ONLY with: pt, en, or es.\n"
                    "pt = Portuguese, en = English, es = Spanish.\n"
                    "Default to pt if uncertain."
                )
            },
            {"role": "user", "content": pergunta}
        ]
    )
    resultado = resposta.choices[0].message.content.strip().lower()
    return resultado if resultado in ["pt", "en", "es"] else "pt"

def resolver_url_colecao(http_client, slug_tentativa: str) -> str:
    """
    Tenta a URL com o slug fornecido. Se redirecionar, usa a URL final.
    Retorna a URL válida ou a original como fallback.
    """
    url = f"https://finamac.com/pt/collections/{slug_tentativa}"
    try:
        resp = http_client.get(url)  # follow_redirects=True já está no cliente
        if resp.status_code == 200:
            return str(resp.url)     # URL real após redirecionamento
    except Exception:
        pass
    return url  # fallback sem crash

# ─────────────────────────────────────────
# INICIALIZAÇÃO DO TERMINAL
# ─────────────────────────────────────────
if __name__ == "__main__":
    
    print("=== INICIALIZANDO AMBIENTE DE TESTES HTTPX ===")
    catalogo_oficial = mapear_catalogo_maquinas(http_client)
    
    efeito_digitar("Consultor Finamac: Olá! Bem-vindo à Finamac. Qual é o seu nome?")
    nome_usuario = input("\n 👤 ").strip()
    
    if nome_usuario.lower() == "sair":
        sys.exit()
    
    efeito_digitar(f"Consultor Finamac: Prazer em te conhecer, {nome_usuario}! Como posso ajudar no seu negócio de gelados hoje?")
    
    while True:
        pesquisa = input(f"\n 👤 {nome_usuario}: ").strip()
    
        if not pesquisa:
            continue
        
        if pesquisa.lower() in ["sair", "tchau", "até logo", "adeus", "bye", "goodbye", "adios"]:
            efeito_digitar("Consultor Finamac: Até logo! Qualquer dúvida, estamos à disposição.")
            break
        
        idioma_detectado = detectar_idioma(pesquisa)
        pesquisa_lower = pesquisa.lower()
    
        # ── BLINDAGEM CONTRA ACENTOS E ERROS DE ESCOPO ──────────────────────────
        termos_validos = [
            "maquina", "máquina", "maquinas", "máquinas", 
            "picole", "picolé", "picoles", "picolés", 
            "sorvete", "sorvetes", "gelato", "gelatos", 
            "chocolate", "chocolates", "pop", "ice cream", 
            "freezer", "mixer", "blast", "açaí", "acai",
            "picoleteira", "picoleteiras", "produtora"
        ]
        citou_termo_valido = any(t in pesquisa_lower for t in termos_validos)
    
        # ── PASSO -1: REFERÊNCIA POSICIONAL DETECTADA MAIS AGRESSIVAMENTE ───────
                    
        indice = detectar_referencia_posicional(pesquisa)
        if indice is not None:
            modelo_referenciado = modelos_listados[indice]
            url_alvo = modelo_referenciado.get("url", "")
            nome_alvo = modelo_referenciado.get("nome", "")
            
            # DRILL-DOWN AUTOMÁTICO: Se o item escolhido for uma Coleção/Grupo, extrai as máquinas internas!
            categoria_sub = detectar_categoria(nome_alvo)
            if categoria_sub or "/collections/" in url_alvo:
                slug = categoria_sub if categoria_sub else url_alvo.split("/collections/")[-1]
                url_colecao = f"https://finamac.com/pt/collections/{slug}"
                
                efeito_digitar(f"Consultor Finamac: Mapeando e abrindo os modelos ativos da categoria '{nome_alvo}'...")
                produtos_internos = obter_produtos_da_colecao(http_client, url_colecao)
                #st.write("DEBUG produtos encontrados =", len(produtos_internos))
                
                if produtos_internos:
                    resposta_ia = perguntar_ia_lista_colecao(
                        pesquisa, nome_usuario, nome_alvo, produtos_internos, idioma_detectado, historico_conversa
                    )
                    modelos_listados = produtos_internos
                    ultima_conversa = {
                        "titulo": nome_alvo,
                        "descricao": "Lista de modelos disponíveis nesta categoria: " + ", ".join([p["nome"] for p in produtos_internos]),
                        "url_original": url_colecao,
                        "tipo_schema": "lista_colecao",
                        "produtos": produtos_internos
                    }
                    ultima_conversa = validar_dados_produto(ultima_conversa)
                else:
                    resposta_ia = f"Não encontrei modelos ativos listados nesta categoria específica no momento. Para detalhes, consulte: {CONTATO_COMERCIAL}"
                
                efeito_digitar(resposta_ia)
                continue
            
            # Fluxo para quando for um produto individual único
            efeito_digitar(f"Consultor Finamac: Buscando especificações técnicas de '{nome_alvo}'...")
            produto = obter_produto(http_client, url_alvo)
            preco = obter_preco(http_client, url_alvo)
            produto = validar_dados_produto(produto) 
    
            if "descricao" not in produto:
                produto["descricao"] = ""

            if preco:
                range_preco = formatar_preco_range(preco, idioma_detectado)
                produto["descricao"] += (
                    f"\n\n[COMMERCIAL INFO]: Reference price estimate: {range_preco}. "
                    f"Never inform exact value. Contact for official quote: {CONTATO_COMERCIAL}"
                )
            else:
                produto["descricao"] += (
                    f"\n\n[COMMERCIAL INFO]: Price under consultation. Direct customer to: {CONTATO_COMERCIAL}"
                )
    
            ultima_conversa = produto
            resposta_ia = perguntar_ia(pesquisa, produto, nome_usuario, idioma_detectado, historico_conversa)
            efeito_digitar(resposta_ia)
            continue
        
        # ── PASSO 0: CONFIRMAÇÃO DE COLEÇÃO ──────────────────────────────────────
        if (
            ultima_conversa
            and ultima_conversa.get("tipo_schema") == "colecao"
            and usuario_confirmou_interesse(pesquisa)
        ):
            url_colecao = ultima_conversa.get("url_original")
            produtos_internos = obter_produtos_da_colecao(http_client, url_colecao)
            # st.write("DEBUG produtos encontrados =", len(produtos_internos))
    
            if produtos_internos:
                resposta_ia = perguntar_ia_lista_colecao(
                    pesquisa, nome_usuario,
                    ultima_conversa.get("titulo", ""), produtos_internos, idioma_detectado, historico_conversa
                )
                modelos_listados = produtos_internos
                ultima_conversa = {
                    "titulo": ultima_conversa.get("titulo", ""),
                    "descricao": "Lista de modelos: " + ", ".join([p["nome"] for p in produtos_internos]),
                    "tipo_schema": "lista_colecao",
                    "produtos": produtos_internos
                }
                ultima_conversa = validar_dados_produto(ultima_conversa)
            else:
                resposta_ia = f"Não consegui carregar os modelos dessa categoria, {nome_usuario}. Entre em contato: {CONTATO_COMERCIAL}"
    
            efeito_digitar(resposta_ia)
            continue
        
        # ── PASSO 0B: REFINAMENTO DENTRO DE LISTA JÁ EXIBIDA ────────────────────
        if (
            ultima_conversa
            and ultima_conversa.get("tipo_schema") == "lista_colecao"
            and usuario_refina_lista(pesquisa)
        ):
            lista_atual = ultima_conversa.get("produtos", modelos_listados)
            lista_numerada = "\n".join([f"{i+1}. {p['nome']}" for i, p in enumerate(lista_atual)])
    
            produto_refinamento = {
                "titulo": ultima_conversa.get("titulo", ""),
                "descricao": (
                    f"O cliente está refinando a busca dentro desta lista de modelos:\n{lista_numerada}\n\n"
                    f"Aplique o filtro solicitado e apresente apenas os modelos correspondentes. "
                    f"Pergunte qual deles ele quer conhecer em detalhes."
                ),
                "tipo_schema": "lista_colecao",
                "produtos": lista_atual
            }
            produto_refinamento = validar_dados_produto(produto_refinamento)
    
            resposta_ia = perguntar_ia(pesquisa, produto_refinamento, nome_usuario, idioma_detectado, historico_conversa)
            efeito_digitar(resposta_ia)
            continue
        
        # ── PASSO 1: INTENÇÃO GERAL ───────────────────────────────────────────────

        categoria_detectada = detectar_categoria(pesquisa)

        if categoria_detectada:
            url_colecao = resolver_url_colecao(
                http_client,
                categoria_detectada
            )

            produtos_internos = obter_produtos_da_colecao(
                http_client,
                url_colecao
            )

            if produtos_internos:

                modelos_listados = produtos_internos

                ultima_conversa = {
                    "titulo": categoria_detectada,
                    "tipo_schema": "lista_colecao",
                    "produtos": produtos_internos,
                    "url_original": url_colecao
                }

                resposta = perguntar_ia_lista_colecao(
                    pesquisa,
                    nome_usuario,
                    categoria_detectada,
                    produtos_internos,
                    idioma_detectado,
                    historico_conversa
                )

                efeito_digitar(resposta)
                continue

        produto_extraido = extrair_produto(pesquisa)
    
        if produto_extraido != "NONE":
            tipo = "NOVO_PRODUTO"

        else:
            tipo_bruto, produto_sugerido = tipo_pergunta(
                pesquisa,
                historico_conversa
            )

            MAPA_TIPO = {
                "especifica_produto": "NOVO_PRODUTO",
                "institucional": "CONTEXTO",
                "fora_de_escopo": "OUT_OF_SCOPE"
            }

            tipo = MAPA_TIPO.get(tipo_bruto, "CONTEXTO")

            categorias_genericas = {
                "sorvete",
                "gelato",
                "picolé",
                "acai",
                "açaí",
                "chocolate",
                "mixer"
            }

            if produto_sugerido.lower() in categorias_genericas:
                categoria_detectada = detectar_categoria(pesquisa)

                if categoria_detectada:
                    produto_extraido = categoria_detectada

            else:
                produto_extraido = (
                    produto_sugerido
                    if produto_sugerido != "NONE"
                    else extrair_produto(pesquisa)
                )

        # CORREÇÃO CRÍTICA: Se a IA errou o escopo mas o termo é legítimo, reclassificamos à força!
        #st.write("DEBUG tipo =", tipo)
        #st.write("DEBUG produto_extraido =", produto_extraido)
        if tipo == "OUT_OF_SCOPE" and citou_termo_valido:
            tipo = "CONTEXTO" if ultima_conversa else "NOVO_PRODUTO"
    
        if tipo == "CONTEXTO" and ultima_conversa:
            resposta_ia = perguntar_ia(pesquisa, ultima_conversa, nome_usuario, idioma_detectado, historico_conversa)
            efeito_digitar(resposta_ia)
            continue
        
        # ── PASSO 2: EXTRAÇÃO DE PRODUTO ESPECÍFICO ───────────────────────────────
        # Se você reativar o Passo 2 futuramente, certifique-se de envolver em validar_dados_produto()
        
        # ── PASSO 3: PRODUTO ESPECÍFICO IDENTIFICADO ──────────────────────────────
        if produto_extraido == "NONE":
            resposta = perguntar_ia_generico(
                pesquisa,
                nome_usuario,
                catalogo_oficial,
                idioma_detectado,
                historico_conversa
            )
            efeito_digitar(resposta)
            continue
        resultados = buscar_produto(http_client, produto_extraido)
    
        produtos_urls = resultados.get("produtos", [])
        colecoes_urls = resultados.get("colecoes", [])
    
        if not produtos_urls and not colecoes_urls:
            msg = {
                "en": f"I couldn't find results for '{produto_extraido}'. Contact us: {CONTATO_COMERCIAL}",
                "es": f"No encontré resultados para '{produto_extraido}'. Contacto: {CONTATO_COMERCIAL}",
                "pt": f"Não encontrei '{produto_extraido}' no site. Contato: {CONTATO_COMERCIAL}"
            }
            efeito_digitar(msg[idioma_detectado])
            continue
        
        if produtos_urls:
            url_chosen = escolher_melhor_produto(produtos_urls, produto_extraido)
            produto_bruto = obter_produto(http_client, url_chosen)
            preco = obter_preco(http_client, url_chosen)  # Unificado com os parâmetros corretos
            produto = validar_dados_produto(produto_bruto)
    
            if "descricao" not in produto:
                produto["descricao"] = ""

            if preco:
                range_preco = formatar_preco_range(preco, idioma_detectado)
                produto["descricao"] += (
                    f"\n\n[COMMERCIAL INFO]: Reference price estimate: {range_preco}. "
                    f"Never inform exact value. Prices vary by configuration. Contact for official quote: {CONTATO_COMERCIAL}"
                )
            else:
                produto["descricao"] += (
                    f"\n\n[COMMERCIAL INFO]: Price under consultation. Direct customer to: {CONTATO_COMERCIAL}"
                )
    
            ultima_conversa = produto
            print("\nULTIMA_CONVERSA DEFINIDA COMO:")
            print(ultima_conversa.get("titulo"))
            print(ultima_conversa.get("tipo_schema"))
    
        else:
            url_colecao = escolher_melhor_produto(colecoes_urls, produto_extraido)
            produto = {
                "titulo": produto_extraido,
                "descricao": "Categoria de equipamentos encontrada.",
                "url_original": url_colecao,
                "tipo_schema": "colecao"
            }
            produto = validar_dados_produto(produto)
            ultima_conversa = produto
    
        resposta_ia = perguntar_ia(pesquisa, produto, nome_usuario, idioma_detectado, historico_conversa)
        efeito_digitar(resposta_ia)