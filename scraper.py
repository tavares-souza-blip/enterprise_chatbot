import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re

# Cliente HTTP reutilizável com User-Agent de navegador para evitar bloqueios comerciais
http_client = httpx.Client(
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    },
    timeout=10.0,
    follow_redirects=True
)

def extrair_variantes(soup):

    variantes = {}

    variant_selects = soup.find("variant-selects")

    if not variant_selects:
        return variantes

    for fieldset in variant_selects.find_all("fieldset"):

        legenda = fieldset.find("legend")

        if not legenda:
            continue

        nome = legenda.get_text(" ", strip=True)

        valores = []

        for radio in fieldset.find_all("input", {"type": "radio"}):

            valor = radio.get("value", "").strip()

            if valor and valor not in valores:
                valores.append(valor)

        variantes[nome] = valores

    return variantes

def buscar_produto(http_client, nome_produto):
    busca = nome_produto.replace(" ", "+")
    url_busca = f"https://finamac.com/pt/search?q={busca}"
    
    resultados = {"produtos": [], "colecoes": []}
    try:
        resposta = http_client.get(url_busca)
        if resposta.status_code != 200:
            return resultados
        soup = BeautifulSoup(resposta.text, "html.parser")
        
        links_vistos = set()
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            
            # Normaliza para URL completa independente de como venha do HTML
            if href.startswith("/"):
                # Evita duplicar /pt/ se já vier no href
                if href.startswith("/pt/"):
                    url_completa = f"https://finamac.com{href}"
                else:
                    url_completa = f"https://finamac.com/pt{href}"
            else:
                url_completa = href            
            if url_completa in links_vistos:
                continue
                
            if "page=" in href or "policies" in href:
                continue
            
            # FILTRO CORRIGIDO: Verifica se a expressão existe na URL (funciona para absoluto e relativo)
            if "/products/" in href:
                links_vistos.add(url_completa)
                resultados["produtos"].append(url_completa)
            elif "/collections/" in href:
                links_vistos.add(url_completa)
                resultados["colecoes"].append(url_completa)
                
    except Exception as e:
        print(f"Erro ao buscar produto: {e}")
        
    return resultados

def obter_produto(http_client, url):
    """Extrai a Ficha Técnica de forma puramente ESTRUTURADA usando HTTPX."""
    try:
        resposta = http_client.get(url)
        soup = BeautifulSoup(resposta.text, "html.parser")

        titulo_tag = soup.find("h1")
        titulo = titulo_tag.text.strip() if titulo_tag else "Equipamento Finamac"
        variantes = extrair_variantes(soup)

        preco_extraido = _extrair_preco_do_soup(soup)
        preco_formatado = f"USD {preco_extraido:,.2f}" if preco_extraido else "Sob Consulta / Não listado publicamente"

        ficha_tecnica_estruturada = {}

        # Estratégia A: Tabelas HTML
        tabelas = soup.find_all("table")
        for tabela in tabelas:
            for linha in tabela.find_all("tr"):
                celulas = linha.find_all(["td", "th"])
                if len(celulas) == 2:
                    chave = celulas[0].get_text(strip=True).replace(":", "").strip()
                    valor = celulas[1].get_text(strip=True).strip()
                    if chave and valor and len(chave) < 40:
                        ficha_tecnica_estruturada[chave] = valor

        # Estratégia B: Divs de Descrição (Shopify Liquid standard)
        seletores_divs_tecnicas = [
            "div.product__description", "div.rte", "div.product-single__description",
            "div[class*='description']", "div[class*='tech']", "div.tabs-content"
        ]
        
        for seletor in seletores_divs_tecnicas:
            div_alvo = soup.select_one(seletor)
            if div_alvo:
                for elemento in div_alvo.find_all(["li", "p"]):
                    texto = elemento.get_text(strip=True)
                    if ":" in texto:
                        partes = texto.split(":", 1)
                        chave = partes[0].strip()
                        valor = partes[1].strip()
                        if 2 < len(chave) < 40 and valor:
                            ficha_tecnica_estruturada[chave] = valor

        # Fallback inteligente se o dicionário vier vazio
        if not ficha_tecnica_estruturada:
            conteudo_real = []
            for seletor in ["div.product__description", "div.rte"]:
                div_alvo = soup.select_one(seletor)
                if div_alvo:
                    for elemento in div_alvo.find_all(["p", "li"]):
                        t = " ".join(elemento.get_text().split()).strip()
                        if t and len(t) > 5:
                            conteudo_real.append(t)
            
            if conteudo_real:
                ficha_tecnica_estruturada["Informações Gerais"] = " | ".join(list(dict.fromkeys(conteudo_real))[:5])
            else:
                ficha_tecnica_estruturada["Status"] = "Especificações técnicas detalhadas indisponíveis no HTML desta página."

        return {
            "titulo": titulo,
            "preco_raw": preco_extraido,          # float ou None — main decide formatação
            "ficha_tecnica": ficha_tecnica_estruturada,
            "descricao": _ficha_para_texto(ficha_tecnica_estruturada),  # string para IA
            "url_original": url,
            "tipo_schema": "produto",
            "variantes": variantes
        }

    except Exception as e:
        return {
            "titulo": "Equipamento Finamac",
            "preco": "Erro ao extrair",
            "ficha_tecnica": {"Erro": f"Falha crítica na extração dos dados: {str(e)}"},
            "url_original": url,
            "tipo_schema": "produto",
            "variantes": []
        }
    
def _ficha_para_texto(ficha: dict) -> str:
    if not ficha:
        return "[DADOS_INDISPONÍVEIS]"
    return "\n".join([f"- {k}: {v}" for k, v in ficha.items()])

def _extrair_preco_do_soup(soup):
    try:
        seletores = [
            {"class": "price-item--regular"},
            {"class": "price-item price-item--regular"},
            {"class": "product__price"},
            {"class": "price"},
            {"itemprop": "price"},
            {"class": "price-item"},
        ]

        for seletor in seletores:
            elemento = soup.find(["span", "div", "p"], seletor)
            if elemento:
                texto_sujo = elemento.text.strip().upper()
                numeros_encontrados = re.findall(r"[\d.,]+", texto_sujo)
                if not numeros_encontrados:
                    continue
                
                texto = numeros_encontrados[0]
                try:
                    if "," in texto and "." in texto:
                        texto = texto.replace(",", "")
                    elif "," in texto and "." not in texto:
                        partes = texto.split(",")
                        if len(partes[-1]) == 2:
                            texto = texto.replace(".", "").replace(",", ".")
                        else:
                            texto = texto.replace(",", "")
                    
                    valor = float(texto)
                    if valor > 1:
                        return valor
                except ValueError:
                    continue
        return None
    except:
        return None

def obter_preco(http_client, url):
    try:
        resposta = http_client.get(url)
        soup = BeautifulSoup(resposta.text, "html.parser")
        return _extrair_preco_do_soup(soup)
    except:
        return None

def obter_produtos_da_colecao(http_client, url_colecao):
    """
    Consome os produtos da coleção garantindo o idioma pt-BR.
    """
    try:
        # 1. Garante que a URL que está entrando use o prefixo /pt/
        if "finamac.com/pt/" not in url_colecao:
            url_colecao = url_colecao.replace("finamac.com/", "finamac.com/pt/")
            
        url_limpa = url_colecao.split("?")[0].rstrip("/")
        url_json = f"{url_limpa}/products.json"
        resposta = http_client.get(url_json)
        
        if resposta.status_code == 200:
            dados = resposta.json()
            produtos = []
            
            # CORREÇÃO DE INDENTAÇÃO: Esta lista e a checagem abaixo devem ficar dentro do escopo do status 200
            RUIDO_HANDLES = [
                "seal", "blade", "kit", "mold", "holder", "spare", "parts",
                "service", "peca", "garantia", "packaging", "extrator",
                "unmolding", "stick-insertion", "chilling", "warranty",
                "tramontina"
            ]
            
            if "products" in dados:
                for prod in dados["products"]:
                    nome = prod.get("title", "Equipamento Finamac")
                    handle = prod.get("handle", "")
            
                    # Filtra acessórios pelo handle (equivale à URL)
                    if any(r in handle.lower() for r in RUIDO_HANDLES):
                        continue
                    
                    url_produto = f"https://finamac.com/pt/products/{handle}"
                    produtos.append({"nome": nome, "url": url_produto})
                return produtos

        return _fallback_html_colecao(http_client, url_limpa)
    except Exception:
        return _fallback_html_colecao(http_client, url_colecao.split("?")[0].rstrip("/"))


def _fallback_html_colecao(http_client, url_html):
    """
    Plano B via HTML: Varre a página da coleção capturando as tags traduzidas.
    """
    try:
        resposta = http_client.get(url_html)
        if resposta.status_code != 200:
            return []

        soup = BeautifulSoup(resposta.text, "html.parser")
        produtos = []
        urls_vistas = set()
        ruido_url = ["seal", "blade", "kit", "mold", "molde", "holder", "spare", "parts", "service", "acessorio", "peca"]

        for link in soup.find_all("a", href=True):
            href = link["href"]
            
            if "/products/" not in href:
                continue

            if any(r in href.lower() for r in ruido_url):
                continue

            # CORREÇÃO CRÍTICA: Assegura que links relativos montem a URL contendo obrigatoriamente o escopo /pt/
            if href.startswith("/"):
                if href.startswith("/pt/"):
                    url_completa = f"https://finamac.com{href}"
                else:
                    url_completa = f"https://finamac.com/pt{href}"
            else:
                url_completa = href

            if url_completa not in urls_vistas:
                urls_vistas.add(url_completa)
                
                nome_texto = link.get_text(strip=True)
                if len(nome_texto) < 3:
                    nome_texto = url_completa.split("/products/")[-1].replace("-", " ").title()

                produtos.append({"nome": nome_texto, "url": url_completa})
        return produtos
    except Exception:
        return []
        
def mapear_catalogo_maquinas(http_client):
    try:
        resposta = http_client.get("https://finamac.com/pt")
        soup = BeautifulSoup(resposta.text, "html.parser")

        catalogo = []  # lista de dicts com nome e slug
        urls_vistas = set()

        for link in soup.find_all("a", href=True):
            href = link["href"]
            texto = link.text.strip()

            if "/collections/" not in href or not texto or len(texto) < 3:
                continue

            # Extrai o slug limpo independente de /pt/ ou não
            slug = href.split("/collections/")[-1].rstrip("/").split("?")[0]
            url_completa = f"https://finamac.com/pt/collections/{slug}"

            if url_completa not in urls_vistas and slug not in ["all", "frontpage"]:
                urls_vistas.add(url_completa)
                catalogo.append({"nome": texto, "slug": slug, "url": url_completa})

        return catalogo if catalogo else _catalogo_fallback()

    except Exception:
        return _catalogo_fallback()

def _catalogo_fallback():
    slugs = [
        ("Ice Pop Machines", "ice-pops"),
        ("Ice Cream Batch Freezers", "ice-cream-batch-freezers"),
        ("Gelato Batch Freezers", "gelato-batch-freezers"),
        ("Mixers & Blenders", "mixers-blenders"),
        ("Chocolate Tempering Machines", "chocolate-tempering-machines"),
        ("Blast Freezers", "blast-freezers"),
        ("Açaí & Frozen Bowl Equipment", "acai-frozen-bowl-equipment"),
        ("Ice Pop Industrial Machines", "ice-pop-industrial-machines"),
        ("Ice Cream & Açaí Artisanal Packages", "ice-cream-and-acai-artisanal-packages"),
        ("Aging Tanks", "aging-tanks"),
    ]
    return [{"nome": n, "slug": s, "url": f"https://finamac.com/pt/collections/{s}"} for n, s in slugs]