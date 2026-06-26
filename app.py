import streamlit as st
import os
import re
import difflib
from dotenv import load_dotenv
from openai import OpenAI
from classifier import classificar_intencao
from scraper import (
    buscar_produto, obter_produto, obter_preco,
    mapear_catalogo_maquinas, obter_produtos_da_colecao, http_client
)

load_dotenv()

st.set_page_config(page_title="Finamac Chatbot - Testes Internos", page_icon="🤖", layout="centered")

st.title("🤖 Finamac Chatbot — Ambiente de Testes")
st.caption("Ecossistema HTTPX estável com injeção de histórico unificado.")

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ENDERECOS_FINAMAC = {
    "BR": (
        "Avenida Nazaré 1657, São Paulo, SP, 04263-200 — Brasil",
        "https://maps.google.com/?q=Avenida+Nazaré+1657,+São+Paulo,+SP,+04263-200"
    ),
    "US": (
        "11231 NW 20th Street, Unit 130, Miami, FL 33172 — USA",
        "https://www.google.com/maps/search/?api=1&query=11231%20NW%2020th%20Street%2C%20Unit%20130%2C%20Miami%2C%20FL%2033172%2C%20USA"
    ),
}

ENDERECO_PADRAO = ENDERECOS_FINAMAC["BR"]  # fallback se não identificar o país

CONTATO_COMERCIAL = "vendas@finamac.com.br ou pelo telefone +55 (11) 98846-5990"

SHOWROOM_LINK = "[Showroom São Paulo](https://maps.google.com/?q=Avenida+Nazaré+1657,+São+Paulo,+SP,+04263-200)"

CONTATO_COMPLETO = f"{CONTATO_COMERCIAL} ou visite nosso {SHOWROOM_LINK}"

CATEGORIAS_SLUGS_VALIDAS = {
    "ice-pops": ["picole", "picoleteira", "pop", "paleta", "ice pop", "artisanal", "artesanal", "producer", "produtora", "automatic", "automática", "industrial", "production", "produção"],
    "ice-pop-unmolding-tanks": ["unmold", "desenformador", "demolder", "desmoldador", "ice pop unmold", "desenformador de picolé", "desmoldador de picolé"],
    "ice-pop-packaging-machines": ["pack", "embalagem", "maquina de embalagem", "embaladora", "wrapping machine"],
    "ice-pop-sealing-machines": ["sealing", "seladora", "ice pop packing", "embalagem de picolé", "seladora de picolé"],
    "stick-insertion-alignment-machines": ["sticks aligner", "stick insertion", "inserção de palito", "alignment", "alinhamento", "alinhador automático de palitos", "inseridor automático de palitos"],
    "artisan-ice-cream": ["artisanal ice cream", "sorvete artesanal", "artisanal", "artesanal"],
    "industrial-ice-cream": ["industrial ice cream", "sorvete industrial", "industrial", "incorpororator", "incorporadora", "continuous", "contínua", "production", "produção", "rotary", "envasadora"],
    "ice-cream-batch-freezers": ["ice cream", "sorvete", "batch freezer", "produtora", "premium", "gelato"],
    "gelato-batch-freezers": ["gelato", "produtora de gelato", "pasteurizer", "pasteurizador", "premium"],
    "flavored-ice-equipment": ["flavored ice", "gelo saborizado", "ice", "gelo", "flavoring", "saborização"],
    "instant-ice-cream": ["instant fresh ice cream", "fresh", "sorvete instantâneo"],
    "chocolate": ["chocolate", "temperadeira", "tempering", "vibration", "vibratória"],
    "mixers-blenders": ["mix", "mixer", "blender", "misturador", "liquidificador", "crusher", "máquina trituradora", "batedeiras", "ice cream", "sorvete", "ice pop", "picolé", "isothermal", "isotérmico", "flavoring", "saborização", "prepraration", "preparação"],
    "blast-freezers": ["ultracongelador", "blast freezer", "congelador rapido", "quick freezing"],
    "aging-tanks": ["tina", "tanque de maturacao", "maturation tank", "maturador", "maturation", "aging tank"],
    "soft-serve-and-instant-ice-cream-packages": ["instant ice cream","sorvete americano", "soft", "bombom", "drink", "cocktails", "frozen cocktails", "drink alcoólico"],
    "continuous-freezers-industrial-ice-cream": ["produtora continua", "continuous freezer", "industrial", "ice cream", "sorvete", "sorvete industrial"],
    "preventive-maintenance-kits": ["quarterly", "trimestral", "annual", "anual", "preventive", "maintenance", "preventiva", "manutenção", "kit", "garantia estendida", "extended warranty"],
    "ice-cream-and-ice-pop-display-case": ["vista", "display case", "vitrine", "vitrine para sorvete", "ice cream display case", "vitrine para picolé", "ice pop display case", "vitrine para gelato", "gelato display case"],
    "pasteurization-heat-treatment-equipment": ["pasteurizer", "pasteurizador", "heat treatment", "tratamento térmico", "pasteurization", "pasteurização", "homogeneizador", "homogenizer", "plants", "planta de pasteurização", "maturation tank", "tina de maturação", "aging tank", "tanque de maturação"],
    "cooling-towers": ["cooling tower", "torre de resfriamento", "resfriamento de água", "water cooling"],
    "finamac-courses": ["course", "courses", "curso", "cursos", "online", "practical", "prático", "training", "treinamento", "digital recipe book", "livro de receitas digital", "recipes", "receitas", "book", "livro"],
}

# ─────────────────────────────────────────
# LÓGICA DE NEGÓCIO E UTILITÁRIOS
# ─────────────────────────────────────────
def detectar_pais_usuario(pergunta: str, idioma: str) -> str:
    resposta = st.session_state.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Identifique o país de origem do usuário com base na mensagem e no idioma.\n"
                    "Responda SOMENTE com o código ISO do país (ex: BR, US, ES, MX, AR).\n"
                    "Se não for possível identificar com segurança, responda: BR\n\n"
                    "Exemplos:\n"
                    "mensagem em inglês americano → US\n"
                    "mensagem em português brasileiro → BR\n"
                    "mensagem em espanhol com 'acá' ou 'vos' → AR\n"
                    "mensagem em espanhol com 'acá' ou menção ao México → MX"
                )
            },
            {"role": "user", "content": f"Idioma detectado: {idioma}\nMensagem: {pergunta}"}
        ],
        temperature=0.0
    )
    return resposta.choices[0].message.content.strip().upper()

def montar_link_endereco(pais: str) -> str:
    endereco_label, endereco_url = ENDERECOS_FINAMAC.get(pais, ENDERECO_PADRAO)
    return f"[{endereco_label}]({endereco_url})"

def formatar_preco_range(preco_float, idioma_usuario="pt"):
    inferior = preco_float * 0.80
    superior = preco_float * 1.20
    if idioma_usuario in ["en", "english"]:
        return f"between USD {inferior:,.2f} and USD {superior:,.2f}"
    elif idioma_usuario in ["es", "spanish"]:
        return f"entre USD {inferior:,.2f} y USD {superior:,.2f}"
    else:
        txt_inferior = f"{inferior:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        txt_superior = f"{superior:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"entre R$ {txt_inferior} e R$ {txt_superior}"

def normalizar_modelo(pergunta):
    pergunta = pergunta.lower()

    pergunta = re.sub(
        r'([a-z]+)[\s\-_]*(\d+)',
        r'\1 \2',
        pergunta
    )

    return pergunta

def escolher_melhor_produto(urls, nome_buscado):
    nome_lower = nome_buscado.lower()
    numeros_buscados = set(re.findall(r'\b\d+\b', nome_lower))
    if not numeros_buscados:
        numeros_buscados = set(re.findall(r'\d+', nome_lower))

    melhor_url = None
    melhor_score = -999
    
    penalidades = [
        "pós", "seal", "blade", "kit", "mold", "molde", "holder", "spare", "peca", "parts",
        "garantia", "warranty", "start-up", "startup", "prevetinva", "preventive", "service", 
        "packaging", "extrator", "unmolding", "stick-insertion", "cooling-tower", "homogenizer",
        "chilling", "pasteurization", "incorporator", "filling", "tramontina", "cinta"
        ]
    bonus = [
        "maquina", "machine", "producer", "freezer", "batch", "industrial", "picole", 
        "artesanal", "artisanal", "sorvete", "gelato", "açaí", "acai", "chocolate", 
        "mixer", "blender", "misturador", "liquidificador", "batedeira", "temperadeira", 
        "ultracongelador", "blast freezer", "congelador rapido", "tanque de maturacao", 
        "maturador", "aging tank", "produtora continua", "industrial", "vista", "vitrine", 
        "display"
        ]

    for url in urls:
        url_lower = url.lower().replace("_", "-").replace("/", "-")
        score = difflib.SequenceMatcher(None, nome_lower.replace(" ", "-"), url_lower).ratio() * 100
        
        for p in penalidades:
            if p in url_lower: score -= 40
        for b in bonus:
            if b in url_lower: score += 20

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

# --- CLASSIFICADORES DE IA ---

def detectar_referencia_posicional(pergunta, modelos_listados):
    if not modelos_listados: return None
    resposta = st.session_state.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Responda SOMENTE de forma COMPACTA com o número posicional (1, 2, 3...) ou a palavra 'ULTIMO'. Se não houver ordem, retorne apenas 'NONE'."},
            {"role": "user", "content": pergunta}
        ],
        temperature=0.0
    )
    resultado = resposta.choices[0].message.content.strip().upper()
    digitos = re.findall(r'\d+', resultado)
    if digitos:
        pos = int(digitos[0]) - 1
        if 0 <= pos < len(modelos_listados):
            return pos
    return None

def usuario_refina_lista(pergunta, modelos_listados):
    if not modelos_listados: return False
    resposta = st.session_state.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "O usuário recebeu uma lista e está tentando aplicar um filtro (ex: 'quero o industrial', 'mostre o menor')? Responda SOMENTE SIM ou NAO."},
            {"role": "user", "content": pergunta}
        ],
        temperature=0.0
    )
    return resposta.choices[0].message.content.strip().upper() == "SIM"

def extrair_produto(pergunta):
    resposta = st.session_state.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": (
                    "Extraia o modelo específico da máquina mantendo dígitos intactos (ex: 'Turbo 8').\n"
                    "Se não houver modelo explícito na frase, responda apenas 'NONE'."
                )
            },
            {"role": "user", "content": pergunta}
        ],
        temperature=0.0
    )
    return resposta.choices[0].message.content.strip()

def usuario_confirmou_interesse(pergunta):
    resposta = st.session_state.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "O usuário aceitou ver os produtos da categoria sugerida? Responda APENAS SIM ou NAO."},
            {"role": "user", "content": pergunta}
        ],
        temperature=0.0
    )
    return resposta.choices[0].message.content.strip().upper() == "SIM"

def detectar_categoria(pergunta):
    opcoes_permitidas = ", ".join(CATEGORIAS_SLUGS_VALIDAS.keys())
    
    resposta = st.session_state.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""
                Você deve escolher APENAS um dos slugs abaixo.

                {opcoes_permitidas}

                REGRAS IMPORTANTES

                Quando o usuário pedir:

                - máquina de picolé
                - máquinas de picolé
                - produtora de picolé
                - produtoras de picolé
                - fabricar picolé
                - produzir picolé

                → responda SEMPRE:

                ice-pops

                Quando o usuário pedir:

                - embaladora
                - embalagem
                - embalar picolé

                → responda:

                ice-pop-packaging-machines

                Quando o usuário pedir:

                - seladora

                → responda:

                ice-pop-sealing-machines

                Quando o usuário pedir:

                - tanque de desmolde

                → responda:

                ice-pop-unmolding-tanks

                Quando o usuário pedir:

                - pasteurizador

                → responda:

                pasteurization-heat-treatment-equipment

                Se nenhuma categoria representar corretamente a intenção do usuário,
                responda apenas:

                NONE

                Responda SOMENTE com o slug.
                """            
            },
            {
                "role": "user",
                "content": pergunta
            }
        ],
        temperature=0.0
    )

    res = resposta.choices[0].message.content.strip()

    return res if res in CATEGORIAS_SLUGS_VALIDAS else None

# --- FUNÇÕES DE RESPOSTA DA IA ---

def perguntar_ia(pergunta, produto, nome, idioma, contato):
    ficha_json = produto.get("ficha_tecnica", {})
    tem_ficha = len(ficha_json) > 0
    preco_info = produto.get("preco", "Sob Consulta / Não listado publicamente")

    if tem_ficha:
        system_prompt = (
            f"Você é um consultor técnico e comercial especialista da empresa Finamac atendendo o cliente: {nome}.\n"
            f"Idioma obrigatório de resposta: {idioma}.\n\n"
            f"DIRETRIZ CRÍTICA DE CONFIABILIDADE (ZERO ALUCINAÇÃO):\n"
            f"1. Responda EXCLUSIVAMENTE com base nos dados técnicos abaixo.\n"
            f"2. Não invente nem estime valores ausentes — redirecione ao comercial.\n"
            f"3. Apresente os dados com bullet points e Markdown profissional, para facilitar a leitura do cliente, incluindo os links diretos enviados no JSON.\n"
            f"4. Ao mencionar o equipamento, inclua o link real do produto em formato Markdown, \n"
            f"5. Você é um consultor de vendas da Finamac. Nunca use respostas genéricas ou teóricas se houver produtos específicos na lista de {produto} ou {obter_produtos_da_colecao} fornecida no contexto. \n"
            "6. Se o contexto trouxer modelos específicos (ex: linha PP-60, PP-110, PP-200), você deve citar esses nomes comerciais exatos na resposta e explicar para que servem. \n"
            "7. Quando o produto selecionado for um combo, kit ou versão (como 'Versão Intermediária PLUS'), você não pode tratá-lo como um produto simples. É obrigatório abrir a descrição e listar para o cliente o que está incluso (ex: quantidade de formas, extratores, alinhadores e capacidade de produção)."
            f"Antes de declarar que um produto não existe no catálogo, você DEVE varrer todo o objeto JSON fornecido. Se o termo buscado pelo usuário (ex: pasteurizador) estiver listado nos títulos ou URLs do array {produto} ou {obter_produtos_da_colecao}, você deve citar esses modelos pelo nome, mesmo que a url escolhida pelo sistema traga uma ficha técnica de outro produto."
            f"ex: [{produto.get('titulo', 'Ver produto')}]({produto.get('url_original', '')}).\n\n"
            f"DADOS REAIS DO PRODUTO:\n"
            f"Equipamento: {produto.get('titulo', 'Não informado')}\n"
            f"Preço de Referência: {preco_info}\n"
            f"Especificações Técnicas (JSON): {ficha_json}\n"
            f"Link Original: {produto.get('url_original', 'https://finamac.com/pt')}\n"
            f"Contato autorizado: {contato}\n\n"
        )    
    else:
        system_prompt = (
            f"Você é um consultor técnico da Finamac atendendo: {nome}.\n"
            f"Idioma: {idioma}.\n\n"
            f"REGRA ABSOLUTA: As especificações detalhadas deste equipamento não estão disponíveis "
            f"no momento via scraping. NÃO invente dados técnicos. Se perguntado sobre potência, "
            f"peso ou capacidade, informe que não foi possível localizar e redirecione para: {contato}\n\n"
            f"Equipamento consultado: {produto.get('titulo', 'Não informado')}\n"
            f"URL: {produto.get('url_original', 'https://finamac.com/pt')}"
        )

    mensagens = (
        [{"role": "system", "content": system_prompt}]
        + st.session_state.historico_ia[-6:]
        + [{"role": "user", "content": pergunta}]
    )
    resposta = st.session_state.openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=mensagens, temperature=0.2
    )

    return resposta.choices[0].message.content

def perguntar_ia_generico(pergunta, nome, catalogo_maquinas, idioma, contato):
    lista = "\n".join([f"- {item['nome']}" for item in catalogo_maquinas])
    system_prompt = (
        f"Você é um consultor comercial da Finamac atendendo: {nome}.\nIdioma: {idioma}.\n\n"
        f"EQUIPAMENTOS REAIS DISPONÍVEIS NO SITE:\n{lista}\n\n"
        f"Oriente o usuário de forma educada e prestativa a buscar por um desses segmentos ou modelos específicos da nossa linha de fabricação."
        f"Contato autorizado: {contato}."
    )
    
    if "messages" not in st.session_state:
        st.session_state.messages = []        
    if "historico_ia" not in st.session_state:
        st.session_state.historico_ia = []    

    mensagens = [{"role": "system", "content": system_prompt}] + st.session_state.historico_ia[-6:] + [{"role": "user", "content": pergunta}]
    resposta = st.session_state.openai_client.chat.completions.create(model="gpt-4o-mini", messages=mensagens)

    return resposta.choices[0].message.content

def perguntar_ia_lista_colecao(pergunta, nome, titulo_colecao, produtos_lista, idioma, contato):
    lista = "\n".join([f"{i+1}. {p['nome']}" for i, p in enumerate(produtos_lista)])
    system_prompt = (
        f"Você é um consultor especializado da Finamac atendendo: {nome}.\nCategoria Atual: {titulo_colecao}.\nIdioma: {idioma}.\n\n"
        f"MODELOS DISPONÍVEIS NESTA CATEGORIA:\n{lista}\n\n"
        f"Apresente a lista para o usuário de forma organizada em Markdown. Diga clara e explicitamente que ele pode escolher um modelo digitando o nome completo do equipamento ou indicando pela posição numérica na lista (ex: 'o primeiro', 'opção 3')."
        f"Contato autorizado: {contato}"
        "Esses modelos são REAIS — cite-os diretamente, incluindo o link de cada um quando disponível. "
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []        
    if "historico_ia" not in st.session_state:
        st.session_state.historico_ia = []    

    mensagens = [{"role": "system", "content": system_prompt}] + st.session_state.historico_ia[-6:] + [{"role": "user", "content": pergunta}]
    resposta = st.session_state.openai_client.chat.completions.create(model="gpt-4o-mini", messages=mensagens)
    
    return resposta.choices[0].message.content

def detectar_idioma(pergunta: str) -> str:
    resposta = st.session_state.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Detect the language of the sentence.\n"
                    "Reply ONLY with: pt, en, or es."
                )
            },
            {"role": "user", "content": pergunta}
        ]
    )

    resultado = resposta.choices[0].message.content.strip().lower()

    if resultado not in ["pt", "en", "es"]:
        resultado = "pt"

    return resultado

def gerar_saudacao_personalizada(nome, perfil, interesse):

    if perfil == "Quero abrir meu negócio":

        if interesse == "Máquinas para sorvetes":
            return (
                f"Prazer em te conhecer, {nome}! "
                f"Vi que você pretende iniciar um negócio no setor de sorvetes. "
                f"Você já tem uma estimativa de produção ou ainda está avaliando as opções disponíveis?"
            )

        elif interesse == "Máquinas para picolés":
            return (
                f"Prazer em te conhecer, {nome}! "
                f"Vi que você deseja iniciar um negócio de picolés. "
                f"Você está pensando em uma produção artesanal ou já busca uma operação com maior escala?"
            )

        elif interesse == "Máquinas para açaí":
            return (
                f"Prazer em te conhecer, {nome}! "
                f"Vi que você quer abrir um negócio de açaí. "
                f"Você já tem experiência com o produto ou está começando do zero?"
            )

        elif interesse == "Cursos e treinamentos":
            return (
                f"Prazer em te conhecer, {nome}! "
                f"Vi que você tem interesse em cursos e treinamentos. "
                f"Está buscando capacitação para iniciar seu negócio ou aprimorar suas habilidades existentes?"
            )
        

    elif perfil == "Já produzo sorvetes":

        return (
            f"Prazer em te conhecer, {nome}! "
            f"Como você já atua com sorvetes, posso ajudá-lo a encontrar equipamentos para ampliar produtividade, qualidade ou capacidade de produção. "
            f"Qual é seu principal objetivo hoje?"
        )

    elif perfil == "Já produzo picolés":

        return (
            f"Prazer em te conhecer, {nome}! "
            f"Como você já produz picolés, posso ajudá-lo com expansão de produção, automação ou novos equipamentos. "
            f"Qual desafio você busca resolver?"
        )

    elif perfil == "Tenho fábrica de gelados":

        return (
            f"Prazer em te conhecer, {nome}! "
            f"Entendi que você já possui uma operação estruturada no setor de gelados. "
            f"Está buscando aumentar capacidade produtiva, modernizar equipamentos ou implantar novos processos?"
        )

    return (
        f"Prazer em te conhecer, {nome}! "
        f"Sou o consultor virtual da Finamac. Como posso ajudar no seu negócio de gelados hoje?"
    )

# ─────────────────────────────────────────
# INTERFACE E FLUXO DO STREAMLIT
# ─────────────────────────────────────────

if "inicializado" not in st.session_state:
    st.session_state.inicializado = False

if not st.session_state.inicializado:
    st.info("🤖 O chatbot está pronto para iniciar com o motor de busca HTTPX.")
    if st.button("Iniciar Servidor e Conectar Robô", type="primary"):
        with st.spinner("Mapeando catálogo estruturado do site institucional..."):
            st.session_state.catalogo_oficial = mapear_catalogo_maquinas(http_client)
            st.session_state.inicializado = True
            #st.rerun()
    st.stop()

if "messages" not in st.session_state: st.session_state.messages = []
if "historico_ia" not in st.session_state: st.session_state.historico_ia = []
if "ultima_conversa" not in st.session_state: st.session_state.ultima_conversa = None
if "modelos_listados" not in st.session_state: st.session_state.modelos_listados = []
if "nome_usuario" not in st.session_state: st.session_state.nome_usuario = None

if not st.session_state.nome_usuario:
    with st.form("formulario_vazio"):
        nome = st.text_input("Nome: *")

        email = st.text_input("Email: *")

        whatsapp = st.text_input("WhatsApp: *")

        perfil = st.selectbox("Qual sua situação atual? *", 
                               ["Selecione uma opção", 
                                "Quero abrir meu negócio", 
                                "Já produzo sorvetes",
                                "Já produzo picolés",
                                "Já trabalho com açaí",
                                "Tenho fábrica de gelados",
                                "Outro"])
        if perfil == "Selecione uma opção":  
            perfil = None

        if perfil == "Quero abrir meu negócio":
            perfil_prompt = """
        O usuário é iniciante no setor.

        Explique conceitos de forma simples.
        Evite excesso de termos técnicos.
        Priorize máquinas de entrada.
        Priorize orientação consultiva.
        Sempre responda às perguntas do usuário com base no conteúdo do site da Finamac (https://finamac.com/).
        """
        elif perfil in ["Já produzo sorvetes", "Já produzo picolés", "Já trabalho com açaí"]:
            perfil_prompt = """
        O usuário possui experiência prática.

        Use linguagem técnica moderada.
        Pode discutir produtividade, capacidade de produção e escalabilidade.
        Recomende máquinas adequadas ao nível de experiência do usuário.
        Sempre responda às perguntas do usuário com base no conteúdo do site da Finamac (https://finamac.com/).
        """
        elif perfil == "Tenho fábrica de gelados":
            perfil_prompt = """
        O usuário possui experiência industrial.

        Utilize linguagem técnica.
        Pode discutir produtividade, pasteurização, homogeneização,
        capacidade de produção e escalabilidade.
        Sempre responda às perguntas do usuário com base no conteúdo do site da Finamac (https://finamac.com/).
        """
        interesse = st.selectbox("O que você procura hoje? *",
                                ["Selecione uma opção",
                                "Máquinas para sorvetes",
                                "Máquinas para picolés",
                                "Máquinas para açaí",
                                "Cursos e treinamentos",
                                "Peças e manutenção"])
        if interesse == "Selecione uma opção":  
            interesse = None

        if interesse == "Máquinas para sorvetes":
            interesse_prompt = f"""
        O usuário demonstra interesse em 
        máquinas para produção de sorvetes.

        Pergunte para qual tipo de sorvete ele deseja produzir 
        (ex: artesanal, industrial, gelato, soft-serve) e 
        recomende máquinas adequadas a esse perfil, presentes
        nas {CATEGORIAS_SLUGS_VALIDAS} 
        """
            
        elif interesse == "Máquinas para picolés":
            interesse_prompt = f"""
        O usuário demonstra interesse em 
        máquinas para produção de picolés.

        Recomende máquinas adequadas a esse perfil, presentes
        nas {CATEGORIAS_SLUGS_VALIDAS} 
        """
            
        elif interesse == "Máquinas para açaí":
            interesse_prompt = f"""
        O usuário demonstra interesse em 
        máquinas para produção de açaí.

        Recomende máquinas adequadas a esse perfil, presentes
        nas {CATEGORIAS_SLUGS_VALIDAS} 
        """
                        
        elif interesse == "Cursos e treinamentos":
            interesse_prompt = f"""
        O usuário demonstra interesse em 
        cursos e treinamentos.

        Recomende os cursos adequados a esse perfil, presentes
        nas {CATEGORIAS_SLUGS_VALIDAS}
        """
            
        elif interesse == "Peças e manutenção":
            interesse_prompt = f"""
        O usuário demonstra interesse em 
        peças e manutenção.

        Recomende peças e serviços de manutenção adequados a esse perfil, presentes
        nas {CATEGORIAS_SLUGS_VALIDAS}
        """

        st.session_state.perfil_cliente = perfil
        st.session_state.interesse_cliente = interesse
        submitted = st.form_submit_button("Salvar")

    if submitted: 
        if not nome or not email or not whatsapp or not perfil or not interesse:
            st.error("Os campos são obrigatórios para iniciar o atendimento. Por favor, preencha todos os campos.")
            st.stop()

        else:
            with st.form("formulario"):
                st.write("Obrigado por fornecer suas informações! Agora você pode iniciar a conversa com o consultor virtual.")
                st.form_submit_button("Iniciar Conversa")
                st.session_state.nome_usuario = nome.strip()
                saudacao = gerar_saudacao_personalizada(st.session_state.nome_usuario, perfil, interesse)
                st.session_state.messages.append({"role": "assistant", "content": saudacao})
                #st.write("Digite sua pergunta sobre as máquinas Finamac no campo abaixo e pressione Enter.")
                #st.text_input("Digite sua pergunta:", key="pergunta_usuario")
                st.session_state.inicializado = True
                st.session_state.historico_ia.append({"role": "system", "content": f"Usuário: {nome.strip()}, Email: {email.strip()}, WhatsApp: {whatsapp.strip()}, Perfil: {perfil.strip()}, Interesse: {interesse.strip()}"})
                #st.form_submit_button("Obrigado! Agora você pode iniciar a conversa com o consultor virtual.") 

else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if pesquisa := st.chat_input("Digite sua pergunta sobre as máquinas Finamac..."):

        st.session_state.messages.append({"role": "user", "content": pesquisa})
        with st.chat_message("user"):
            st.markdown(pesquisa)

        pesquisa_lower = pesquisa.lower()
        idioma_detectado = detectar_idioma(pesquisa)
        pais_usuario = detectar_pais_usuario(pesquisa, idioma_detectado)
        link_endereco = montar_link_endereco(pais_usuario)

        CONTATO_COMPLETO = (
            f"vendas@finamac.com.br ou +55 11 98846-5990 "
            f"ou visite nossa unidade mais próxima: {link_endereco}"
        )

        resposta_gerada = ""

        with st.spinner("Consultando dados oficiais..."):

            texto_limpo = pesquisa.lower().strip()
            if "turbo" in texto_limpo and "80" in texto_limpo:
                prod_ext = "Turbo 80"
                indice = None
                interesse_confirmado = False
                refinando_lista = False
            else:
                prod_ext = extrair_produto(pesquisa)

                if prod_ext != "NONE":
                    indice = None
                    interesse_confirmado = False
                    refinando_lista = False
                else:
                    indice = detectar_referencia_posicional(pesquisa, st.session_state.modelos_listados)
                    interesse_confirmado = (st.session_state.ultima_conversa and 
                                            st.session_state.ultima_conversa.get("tipo_schema") == "colecao" and 
                                            usuario_confirmou_interesse(pesquisa))
                    refinando_lista = (st.session_state.ultima_conversa and 
                                       st.session_state.ultima_conversa.get("tipo_schema") == "lista_colecao" and 
                                       usuario_refina_lista(pesquisa, st.session_state.modelos_listados))

            # ─────────────────────────────────────────
            # ÁRVORE DE DECISÃO EXECUTÁVEL
            # ─────────────────────────────────────────

            if indice is not None:
                modelo_ref = st.session_state.modelos_listados[indice]
                produto = obter_produto(http_client, modelo_ref["url"])

                preco_num = obter_preco(http_client, modelo_ref["url"])
                if preco_num:
                    produto["preco"] = formatar_preco_range(preco_num, idioma_detectado)

                st.session_state.ultima_conversa = produto
                resposta_gerada = perguntar_ia(pesquisa, produto, st.session_state.nome_usuario, idioma_detectado, CONTATO_COMPLETO)

            elif interesse_confirmado:
                url_col = st.session_state.ultima_conversa.get("url_original")
                prod_internos = obter_produtos_da_colecao(http_client, url_col)
                if prod_internos:
                    resposta_gerada = perguntar_ia_lista_colecao(pesquisa, st.session_state.nome_usuario, st.session_state.ultima_conversa["titulo"], prod_internos, idioma_detectado, CONTATO_COMPLETO)
                    st.session_state.ultima_conversa = {"titulo": st.session_state.ultima_conversa["titulo"], "tipo_schema": "lista_colecao", "produtos": prod_internos}
                    st.session_state.modelos_listados = prod_internos
                else:
                    resposta_gerada = f"Não encontrei modelos ativos nesta categoria no momento. Contato Comercial: {CONTATO_COMPLETO}"

            elif refinando_lista:
                lista_at = st.session_state.ultima_conversa.get("produtos", st.session_state.modelos_listados)
                prod_ref = {"titulo": st.session_state.ultima_conversa["titulo"], "ficha_tecnica": {"Aviso": "O usuário está refinando a busca dentro desta sublista de modelos ativos."}, "tipo_schema": "lista_colecao", "produtos": lista_at}
                resposta_gerada = perguntar_ia(pesquisa, prod_ref, st.session_state.nome_usuario, idioma_detectado, CONTATO_COMPLETO)

            else:
                # Integração e Mapeamento Robustecido com o classifier.py
                tipo_bruto, produto_sugerido = classificar_intencao(
                    st.session_state.openai_client,
                    pesquisa,
                    st.session_state.get("historico_ia", [])
                )

                #st.write("DEBUG tipo_bruto =", tipo_bruto)
                #st.write("DEBUG produto_sugerido =", produto_sugerido)            

                MAPA_TIPO = {
                    "recomendacao_produtos": "CONSULTIVO",
                    "especifica_produto": "NOVO_PRODUTO",
                    "institucional": "CONTEXTO",
                    "consultivo": "CONSULTIVO",
                    "servico_curso": "CURSO",
                    "fora_de_escopo": "OUT_OF_SCOPE"
                }

                categorias_genericas = {
                    "sorvete",
                    "gelato",
                    "picolé",
                    "acai",
                    "açaí",
                    "chocolate",
                    "mixer"
                }

                #if produto_sugerido.lower() in categorias_genericas:
                if any(termo in pesquisa_lower for termo in ["pasteuriz", "sorvete", "gelato", "picol", "acai", "chocolate"]):
                    categoria_detectada = detectar_categoria(pesquisa)              
                    #st.write("\nDEBUG categoria_detectada =", categoria_detectada)
                    if categoria_detectada:
                        url_col = f"https://finamac.com/pt/collections/{categoria_detectada}"
                        prod_internos = obter_produtos_da_colecao(http_client, url_col)
                        if prod_internos:
                            resposta_gerada = perguntar_ia_lista_colecao(
                                pesquisa, st.session_state.nome_usuario,
                                categoria_detectada, prod_internos, idioma_detectado,
                                CONTATO_COMPLETO
                            )
                            st.session_state.ultima_conversa = {
                                "titulo": categoria_detectada,
                                "tipo_schema": "lista_colecao",
                                "produtos": prod_internos
                            }
                            st.session_state.modelos_listados = prod_internos
                        else:
                            resposta_gerada = (
                                f"Não encontrei modelos ativos nesta categoria no momento. "
                                f"Contato Comercial: {CONTATO_COMPLETO}"
                            )
                else:
                    produto_extraido = (
                        produto_sugerido
                        if produto_sugerido != "NONE"
                        else extrair_produto(pesquisa)
                    )

                if not resposta_gerada:
                    tipo = MAPA_TIPO.get(tipo_bruto, "CONTEXTO")

                    prod_ext = produto_sugerido
                    #st.sidebar.write("prod_ext:", prod_ext)

                    if produto_sugerido != "NONE":
                        prod_ext = produto_sugerido

                        if tipo_bruto == "especifica_produto":
                            tipo = "NOVO_PRODUTO"

                    if tipo == "OUT_OF_SCOPE":
                        resposta_gerada = f"Desculpe, mas sou um assistente virtual exclusivo da Finamac e só posso responder a questões técnicas e comerciais sobre os nossos equipamentos de sorvete, picolé, açaí, chocolate e refrigeração comercial. Se precisar de um atendimento com nosso setor comercial, entre em contato com {CONTATO_COMPLETO} Como posso ajudar no desenvolvimento do seu negócio hoje?"

                    elif tipo == "CURSO":
                        resposta_gerada = st.session_state.openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "Você é um consultor comercial da Finamac. "
                                        "Ajude o cliente a escolher o melhor curso disponível "
                                        "no site oficial da Finamac para sua situação específica "
                                        "que seja mais agradável para ele com base em interesses e "
                                        "objetivos, tipos de produção no ramo de gelados, "
                                        "público-alvo e tipo de produto produzido."
                                        "De maneira alguma invente ou faça suposições sobre cursos "
                                        "que não estão disponíveis e nem presentes no site. "
                                        "Considere que, juntamente dos cursos em módulos, têm também "
                                        "livro de receitas para diferentes preparações do ramo."
                                        f"\nContato comercial autorizado: {CONTATO_COMPLETO}"
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": pesquisa
                                }
                            ]
                        ).choices[0].message.content  

                    elif tipo == "CONSULTIVO":
                        resposta_gerada = st.session_state.openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "Você é um consultor comercial da Finamac. "
                                        "Ajude o cliente a escolher o equipamento ideal "
                                        "com base em orçamento, espaço físico, capacidade "
                                        "de produção, público-alvo e tipo de produto."
                                        f"\nContato comercial autorizado: {CONTATO_COMPLETO}"
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": pesquisa
                                }
                            ]
                        ).choices[0].message.content      

                    elif tipo == "CONTEXTO" and st.session_state.ultima_conversa is not None:
                        resposta_gerada = perguntar_ia(pesquisa, st.session_state.ultima_conversa, st.session_state.nome_usuario, idioma_detectado, CONTATO_COMPLETO)
                    elif tipo == "CONTEXTO" and st.session_state.ultima_conversa is None:
                        resposta_gerada = perguntar_ia_generico(pesquisa,st.session_state.nome_usuario, st.session_state.catalogo_oficial, idioma_detectado, CONTATO_COMPLETO)

                    elif prod_ext == "NONE":
                        cat = detectar_categoria(pesquisa)
                        if cat:
                            url_col = f"https://finamac.com/pt/collections/{cat}"
                            prod_internos = obter_produtos_da_colecao(http_client, url_col)
                            #st.write("DEBUG URL coleção =", url_col)
                            #st.write("DEBUG quantidade produtos =", len(prod_internos))
                            for p in prod_internos[:10]:
                                st.write(" -", p["nome"])
                            if prod_internos:
                                resposta_gerada = perguntar_ia_lista_colecao(pesquisa, st.session_state.nome_usuario, cat, prod_internos, idioma_detectado, CONTATO_COMPLETO)
                                st.session_state.ultima_conversa = {"titulo": cat, "tipo_schema": "lista_colecao", "produtos": prod_internos}
                                st.session_state.modelos_listados = prod_internos
                            else:
                                resposta_gerada = f"Nosso catálogo digital para esta categoria está passando por uma atualização rápida no momento. Por favor, consulte os dados técnicos diretamente pelo e-mail: {CONTATO_COMPLETO}"
                                st.session_state.modelos_listados = [] 
                        else:
                            resposta_gerada = perguntar_ia_generico(pesquisa,st.session_state.nome_usuario, st.session_state.catalogo_oficial, idioma_detectado, CONTATO_COMPLETO)
                            st.session_state.modelos_listados = [] 
                    else:
                        res_busca = buscar_produto(http_client, prod_ext)
                        #st.write(res_busca)
                        #st.write("DEBUG produto extraido =", produto_extraido)
                        p_urls, c_urls = res_busca.get("produtos", []), res_busca.get("colecoes", [])

                        if not p_urls and not c_urls:
                            resposta_gerada = f"Não encontrei nenhum equipamento correspondente a '{prod_ext}' no catálogo digital do site oficial. Contato Comercial: {CONTATO_COMPLETO}"
                            st.session_state.modelos_listados = []
                        else:
                            st.session_state.modelos_listados = [] 
                            if p_urls:
                                url_esc = escolher_melhor_produto(p_urls, prod_ext)
                                produto = obter_produto(http_client, url_esc)
                                #st.write("URL ESCOLHIDA:", url_esc)
                                #st.write(
                                #    "TITULO:",
                                #    produto.get("titulo")
                                #)
                                #st.write(
                                #    "QTD CAMPOS FICHA:",
                                #    len(produto.get("ficha_tecnica", {}))
                                #)
                                #st.write(
                                #    produto.get("ficha_tecnica", {})
                                #)

                                print(produto)

                                preco_num = obter_preco(http_client, url_esc)
                                if preco_num:
                                    produto["preco"] = formatar_preco_range(preco_num, idioma_detectado)

                                st.session_state.ultima_conversa = produto
                            else:
                                url_col = escolher_melhor_produto(c_urls, prod_ext)
                                produto = {"titulo": prod_ext, "ficha_tecnica": {}, "url_original": url_col, "tipo_schema": "colecao"}
                                st.session_state.ultima_conversa = produto

                            resposta_gerada = perguntar_ia(pesquisa, produto, st.session_state.nome_usuario, idioma_detectado, CONTATO_COMPLETO)

        st.session_state.messages.append({"role": "assistant", "content": resposta_gerada})
        st.session_state.historico_ia.append({"role": "user", "content": pesquisa})
        st.session_state.historico_ia.append({"role": "assistant", "content": resposta_gerada})

        # Força atualização visual sem perder estado
        placeholder = st.empty()
        with placeholder.container():
            st.chat_message("assistant").markdown(resposta_gerada)
